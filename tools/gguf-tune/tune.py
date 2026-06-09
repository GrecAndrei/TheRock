#!/usr/bin/env python3
"""
gguf-tune orchestrator: extract GEMM shapes → Tensile benchmark → GPU profile.

Full pipeline:
  1. extract_shapes.py   → shapes JSON
  2. generate_tensile_config.py → benchmark YAML
  3. Tensile benchmark runner   → library logic YAML (best kernel per shape)
  4. parse_library_logic()      → profile JSON

The profile JSON can be committed to a community profile DB per (model, gfx) pair.

Usage:
  python3 tune.py model.gguf --gfx gfx1032 --output-dir /tmp/tune_out

Environment:
  ROCM_PATH  — path to ROCm installation (default: TheRock build/dist/rocm)
  THEROCK    — path to TheRock repo root (auto-detected from script location)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK = Path(os.environ.get("THEROCK", SCRIPT_DIR.parent.parent))
ROCM_PATH = Path(os.environ.get("ROCM_PATH", THEROCK / "build/dist/rocm"))

TENSILELITE_DIR = (
    THEROCK / "rocm-libraries/projects/hipblaslt/tensilelite"
)
TENSILE_BIN = TENSILELITE_DIR / "Tensile/bin/Tensile"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list, env: dict = None, cwd: Path = None, label: str = "") -> int:
    merged_env = {**os.environ, **(env or {})}
    tag = f"[{label}] " if label else ""
    print(f"{tag}$ {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, env=merged_env, cwd=cwd)
    return result.returncode


def check(rc: int, label: str) -> None:
    if rc != 0:
        print(f"ERROR: {label} failed (exit {rc})", file=sys.stderr)
        sys.exit(rc)


# ---------------------------------------------------------------------------
# Step 1: Extract shapes
# ---------------------------------------------------------------------------

def extract_shapes(model: Path, batch_sizes: list, shapes_json: Path) -> dict:
    if shapes_json.exists():
        print(f"[extract] Reusing {shapes_json}", flush=True)
        return json.loads(shapes_json.read_text())

    rc = run(
        [sys.executable, SCRIPT_DIR / "extract_shapes.py",
         str(model),
         "--batch-sizes", *[str(b) for b in batch_sizes],
         "--output", str(shapes_json)],
        label="extract",
    )
    check(rc, "extract_shapes")
    return json.loads(shapes_json.read_text())


# ---------------------------------------------------------------------------
# Step 2: Generate Tensile config YAML
# ---------------------------------------------------------------------------

def generate_config(shapes_json: Path, gfx: str, config_yaml: Path) -> None:
    if config_yaml.exists():
        print(f"[config] Reusing {config_yaml}", flush=True)
        return

    rc = run(
        [sys.executable, SCRIPT_DIR / "generate_tensile_config.py",
         str(shapes_json),
         "--gfx", gfx,
         "--output", str(config_yaml)],
        label="config",
    )
    check(rc, "generate_tensile_config")


# ---------------------------------------------------------------------------
# Step 3: Run Tensile benchmark
# ---------------------------------------------------------------------------

def run_tensile(config_yaml: Path, tensile_out: Path, gfx: str) -> None:
    """Invoke TensileLite's benchmark runner on the generated config."""

    if (tensile_out / "3_LibraryLogic").exists():
        print(f"[tensile] Reusing existing LibraryLogic in {tensile_out}", flush=True)
        return

    tensile_out.mkdir(parents=True, exist_ok=True)

    rocm = ROCM_PATH
    cxx   = rocm / "bin/amdclang++"
    asm   = rocm / "bin/amdclang++"
    bundler = rocm / "lib/llvm/bin/clang-offload-bundler"
    enumerator = rocm / "bin/rocm_agent_enumerator"

    env = {
        "ROCM_PATH": str(rocm),
        "PATH": f"{rocm}/bin:{rocm}/lib/llvm/bin:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{rocm}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "PYTHONPATH": str(TENSILELITE_DIR),
    }

    cmd = [
        sys.executable, str(TENSILE_BIN),
        str(config_yaml),
        str(tensile_out),
        "--gpu-targets", gfx,
        "--cxx-compiler", str(cxx),
        "--assembler", str(asm),
        "--offload-bundler", str(bundler),
        "--device-enumerator", str(enumerator),
        "--library-format", "msgpack",
    ]

    rc = run(cmd, env=env, cwd=TENSILELITE_DIR, label="tensile")
    check(rc, "Tensile benchmark")


# ---------------------------------------------------------------------------
# Step 4: Parse library logic → profile
# ---------------------------------------------------------------------------

def parse_library_logic(tensile_out: Path, shapes: dict, gfx: str) -> dict:
    """
    Parse TensileLite's 3_LibraryLogic YAML outputs into a compact profile.

    The logic files map ProblemSizes → kernel solution index + parameters.
    We extract the winning solution name per shape and record it.
    """
    logic_dir = tensile_out / "3_LibraryLogic"
    if not logic_dir.exists():
        raise FileNotFoundError(f"LibraryLogic not found at {logic_dir}")

    try:
        import yaml
    except ImportError:
        print("WARNING: PyYAML not available — returning raw file list only", file=sys.stderr)
        return {
            "gfx": gfx,
            "model": shapes["model"],
            "logic_files": [str(p) for p in logic_dir.glob("*.yaml")],
            "kernels": [],
        }

    kernels = []
    for logic_file in sorted(logic_dir.glob("*.yaml")):
        with open(logic_file) as f:
            logic = yaml.safe_load(f)

        # Logic file structure: list of [header, solutions, index_table, ...]
        # Header is index 0, solutions list is index 1, index table is index 2+
        if not isinstance(logic, list) or len(logic) < 3:
            continue

        solutions = logic[1]   # list of solution dicts
        size_map  = logic[2]   # list of [problem_size, solution_index] pairs

        for entry in size_map:
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            problem, sol_idx = entry[0], entry[1]
            if isinstance(sol_idx, int) and 0 <= sol_idx < len(solutions):
                sol = solutions[sol_idx]
                kernels.append({
                    "problem": problem,
                    "solution_index": sol_idx,
                    "solution_name": sol.get("SolutionNameMin", sol.get("SolutionName", f"sol_{sol_idx}")),
                    "source_file": logic_file.name,
                })

    return {
        "gfx": gfx,
        "model": shapes["model"],
        "architecture": shapes["architecture"],
        "batch_sizes": shapes["batch_sizes"],
        "unique_shapes": shapes["unique_shapes"],
        "logic_dir": str(logic_dir),
        "kernels": kernels,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end GEMM tuning pipeline for a GGUF model"
    )
    parser.add_argument("model", type=Path, help="Path to .gguf model file")
    parser.add_argument(
        "--gfx", default="gfx1032",
        help="Target GPU (default: gfx1032)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Working directory for all outputs (default: /tmp/gguf-tune/<model_stem>_<gfx>)"
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32],
        help="Token batch sizes to tune for (default: 1 4 8 16 32)"
    )
    parser.add_argument(
        "--skip-tensile", action="store_true",
        help="Skip the Tensile benchmark step (only extract + generate config)"
    )
    parser.add_argument(
        "--profile-out", type=Path, default=None,
        help="Write final profile JSON here (default: <output-dir>/profile.json)"
    )
    args = parser.parse_args()

    model = args.model.resolve()
    if not model.exists():
        print(f"ERROR: model file not found: {model}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir or Path("/tmp/gguf-tune") / f"{model.stem}_{args.gfx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    shapes_json  = out_dir / "shapes.json"
    config_yaml  = out_dir / f"{model.stem}_{args.gfx}_benchmark.yaml"
    tensile_out  = out_dir / "tensile_run"
    profile_json = args.profile_out or out_dir / "profile.json"

    print(f"=== gguf-tune ===")
    print(f"  Model:    {model}")
    print(f"  GPU:      {args.gfx}")
    print(f"  Output:   {out_dir}")
    print(f"  ROCM:     {ROCM_PATH}")
    print()

    # Step 1
    shapes = extract_shapes(model, args.batch_sizes, shapes_json)
    print(f"  Shapes:   {shapes['unique_shapes']} unique GEMM shapes extracted")

    # Step 2
    generate_config(shapes_json, args.gfx, config_yaml)
    print(f"  Config:   {config_yaml}")

    if args.skip_tensile:
        print("\n[skip-tensile] Stopping before Tensile benchmark run.")
        print(f"To run manually:\n  python3 {TENSILE_BIN} {config_yaml} {tensile_out} --gpu-targets {args.gfx}")
        return

    # Step 3
    print("\n=== Running Tensile benchmark (this takes a while) ===")
    run_tensile(config_yaml, tensile_out, args.gfx)

    # Step 4
    print("\n=== Parsing library logic → profile ===")
    profile = parse_library_logic(tensile_out, shapes, args.gfx)
    profile_json.write_text(json.dumps(profile, indent=2))
    print(f"  Profile:  {profile_json}")
    print(f"  Kernels:  {len(profile['kernels'])} kernel assignments recorded")

    print("\nDone.")


if __name__ == "__main__":
    main()

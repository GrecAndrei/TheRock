#!/usr/bin/env python3
"""
Profile database for gguf-tune results.

Layout on disk:
  profiles/
    <arch>/                     e.g. gfx1032/
      <model_arch>/             e.g. gemma4/
        baseline.json           rocBLAS SGEMM baseline timings
        tuned.json              TensileLite-tuned kernel assignments
        meta.json               GPU hardware info, date, contributor

Model arch (e.g. "gemma4") comes from the GGUF general.architecture field,
not the full model name — so all Q4_0/Q8_0/bf16 variants of the same model
share one profile directory since their GEMM shapes are identical.

Usage:
  # Save a baseline result
  python3 profile_db.py save-baseline /tmp/gemma4_baseline.json

  # Save a tuned result
  python3 profile_db.py save-tuned /tmp/tune_out/profile.json

  # List what's in the DB
  python3 profile_db.py list

  # Query best kernel for a shape
  python3 profile_db.py query gfx1032 gemma4 --M 32 --N 3840 --K 15360
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


DB_DIR = Path(__file__).resolve().parent / "profiles"


def db_path(arch: str, model_arch: str) -> Path:
    return DB_DIR / arch / model_arch


def save_baseline(baseline_json: Path) -> None:
    data = json.loads(baseline_json.read_text())

    arch = data.get("device", {}).get("gfx", "unknown")
    # model comes from shapes, pull arch from first result's dtype context
    # baseline JSON has "model" (full name) — we need model_arch separately
    model_arch = data.get("model_arch") or _guess_model_arch(data.get("model", ""))

    out_dir = db_path(arch, model_arch)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir.joinpath("baseline.json").write_text(json.dumps(data, indent=2))
    _write_meta(out_dir, arch, model_arch, data.get("device", {}))

    print(f"Saved baseline → {out_dir}/baseline.json")


def save_tuned(profile_json: Path) -> None:
    data = json.loads(profile_json.read_text())

    arch = data.get("gfx", "unknown")
    model_arch = data.get("architecture", "unknown")

    out_dir = db_path(arch, model_arch)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir.joinpath("tuned.json").write_text(json.dumps(data, indent=2))
    _write_meta(out_dir, arch, model_arch, {})

    print(f"Saved tuned profile → {out_dir}/tuned.json")


def _write_meta(out_dir: Path, arch: str, model_arch: str, device: dict) -> None:
    meta_path = out_dir / "meta.json"
    existing = {}
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())

    existing.update({
        "arch": arch,
        "model_arch": model_arch,
        "device": device,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    })
    meta_path.write_text(json.dumps(existing, indent=2))


def _guess_model_arch(model_name: str) -> str:
    name = model_name.lower()
    if "gemma" in name:
        return "gemma4" if "4" in name else "gemma"
    if "llama" in name:
        return "llama"
    if "mistral" in name:
        return "mistral"
    return model_name.split("/")[-1].split("_")[0] if model_name else "unknown"


def list_profiles() -> None:
    if not DB_DIR.exists():
        print("No profiles yet.")
        return

    found = False
    for arch_dir in sorted(DB_DIR.iterdir()):
        if not arch_dir.is_dir():
            continue
        for model_dir in sorted(arch_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            found = True
            has_baseline = (model_dir / "baseline.json").exists()
            has_tuned    = (model_dir / "tuned.json").exists()
            meta_path    = model_dir / "meta.json"
            updated = ""
            if meta_path.exists():
                updated = json.loads(meta_path.read_text()).get("updated", "")
            status = []
            if has_baseline: status.append("baseline")
            if has_tuned:    status.append("tuned")
            print(f"  {arch_dir.name}/{model_dir.name}  [{', '.join(status)}]  {updated}")

    if not found:
        print("No profiles yet.")


def query(arch: str, model_arch: str, M: int, N: int, K: int) -> None:
    profile_dir = db_path(arch, model_arch)

    tuned_path = profile_dir / "tuned.json"
    if tuned_path.exists():
        data = json.loads(tuned_path.read_text())
        for k in data.get("kernels", []):
            p = k.get("problem", [])
            # Tensile problem format: [M, N, batch, K]
            if len(p) >= 4 and p[0] == M and p[1] == N and p[3] == K:
                print(f"Tuned kernel: {k['solution_name']}")
                print(f"  Source: {k['source_file']}")
                return
        print(f"No tuned kernel found for M={M} N={N} K={K} in {arch}/{model_arch}")
    else:
        print(f"No tuned profile for {arch}/{model_arch}")

    baseline_path = profile_dir / "baseline.json"
    if baseline_path.exists():
        data = json.loads(baseline_path.read_text())
        for r in data.get("results", []):
            if r.get("M") == M and r.get("N") == N and r.get("K") == K:
                print(f"Baseline: {r.get('avg_ms')} ms  {r.get('tflops')} TFLOPS  {r.get('bandwidth_gbs')} GB/s")
                return


def main():
    parser = argparse.ArgumentParser(description="gguf-tune profile database")
    sub = parser.add_subparsers(dest="cmd")

    p_baseline = sub.add_parser("save-baseline", help="Save a baseline_bench.py result")
    p_baseline.add_argument("json", type=Path)

    p_tuned = sub.add_parser("save-tuned", help="Save a tune.py profile result")
    p_tuned.add_argument("json", type=Path)

    sub.add_parser("list", help="List all profiles")

    p_query = sub.add_parser("query", help="Look up kernel for a shape")
    p_query.add_argument("arch")
    p_query.add_argument("model_arch")
    p_query.add_argument("--M", type=int, required=True)
    p_query.add_argument("--N", type=int, required=True)
    p_query.add_argument("--K", type=int, required=True)

    args = parser.parse_args()

    if args.cmd == "save-baseline":
        save_baseline(args.json)
    elif args.cmd == "save-tuned":
        save_tuned(args.json)
    elif args.cmd == "list":
        list_profiles()
    elif args.cmd == "query":
        query(args.arch, args.model_arch, args.M, args.N, args.K)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inspect this workspace without starting inference or altering device state."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def revision(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    nodes = [Path("/dev/kfd"), *sorted(Path("/dev/dri").glob("renderD*"))]
    access = {
        str(path): {
            "exists": path.exists(),
            "read_write": os.access(path, os.R_OK | os.W_OK),
        }
        for path in nodes
    }
    gpu_visible = access["/dev/kfd"]["read_write"] and any(
        item["read_write"] for name, item in access.items() if name != "/dev/kfd"
    )
    models = [
        Path("/home/alex/Downloads/Qwen3.8-27B-UD-Q4_K_M.gguf"),
        Path("/home/alex/Downloads/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"),
    ]
    report = {
        "workspace": str(root),
        "kernel": platform.release(),
        "device_nodes": access,
        "gpu_boundary_accessible": gpu_visible,
        "device_identity_verified": False,
        "inference_tps": None,
        "note": "Device visibility is not proof of gfx1032 identity or successful dispatch.",
        "tools": {name: shutil.which(name) for name in ("cargo", "rustc", "cmake", "ninja", "clang++")},
        "sources": {
            name: revision(root / path)
            for name, path in (
                ("kyanite", "sources/kyanite"),
                ("llama.cpp", "references/llama.cpp"),
            )
        },
        "models": {
            str(path): {
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256_verified_by_doctor": False,
            }
            for path in models
        },
        "free_disk_bytes": shutil.disk_usage(root).free,
    }
    print(json.dumps(report, indent=2))
    return 2 if args.require_gpu and not gpu_visible else 0


if __name__ == "__main__":
    raise SystemExit(main())

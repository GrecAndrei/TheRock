#!/usr/bin/env python3
"""Generate an immutable source manifest and original worktree provenance."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def command(*args):
    return subprocess.check_output(args)


def source_hashes(path):
    hashes = {}
    for file in sorted(path.rglob("*")):
        relative = file.relative_to(path)
        if any(part in {".git", "__pycache__"} for part in relative.parts):
            continue
        if file.is_file():
            hashes[str(relative)] = hashlib.sha256(file.read_bytes()).hexdigest()
    return hashes


def main():
    root = Path(__file__).resolve().parent.parent
    out = root / "results/setup"
    repositories = {
        "original-kyanite": Path("/home/alex/rocmtry"),
        "original-inspection": Path("/home/alex/rocmtry-inspection"),
        "original-therock": root.parent,
        "original-llama": Path("/home/alex/llama.cpp"),
    }
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/record-setup.py",
        "network_refresh": "failed: terminal could not resolve github.com",
        "performance_claim": None,
        "repositories": {},
        "snapshots": {},
    }
    # Exclusive creation prevents replacing setup evidence on accidental reruns.
    with (out / "provenance.json").open("x") as manifest:
        for name, repo in repositories.items():
            scope = ["--", "examples/qwen38-native"] if name == "original-therock" else []
            status = command("git", "-C", str(repo), "status", "--short", *scope)
            diff = command("git", "-C", str(repo), "diff", "HEAD", "--binary", *scope)
            with (out / f"{name}.diff").open("xb") as file:
                file.write(diff)
            report["repositories"][name] = {
                "path": str(repo),
                "head": command("git", "-C", str(repo), "rev-parse", "HEAD").decode().strip(),
                "status": status.decode(),
                "diff_sha256": hashlib.sha256(diff).hexdigest(),
            }
        for relative in (
            "sources/kyanite", "sources/native-baseline", "references/llama.cpp",
            "fixtures", "results/setup/historical-kfd",
        ):
            report["snapshots"][relative] = source_hashes(root / relative)
        json.dump(report, manifest, indent=2, sort_keys=True)
        manifest.write("\n")
    print(f"Recorded {out / 'provenance.json'}")


if __name__ == "__main__":
    main()

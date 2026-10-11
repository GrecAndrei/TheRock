#!/usr/bin/env python3
"""Run native benchmarks under the managed process tool with durable lifecycle logs.

This wrapper does not detach or restart workers. Missing exit records (e.g. after
SIGKILL or host shutdown) remain unresolved, never benchmark successes.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from types import FrameType
from typing import TextIO


CONTROL_NAMES = (
    "QWEN_NATIVE_HYBRID",
    "QWEN_NATIVE_CPU_THREADS",
    "QWEN_NATIVE_CPU_Q6_PREDECODE",
    "QWEN_NATIVE_CPU_IQ4_TILE_GEMM",
    "QWEN_NATIVE_CPU_IQ4_PREDECODE",
    "QWEN_NATIVE_CPU_K_TILE_GEMM",
    "QWEN_NATIVE_SOA_SPLIT",
    "QWEN_NATIVE_DEVICE_CHECKPOINT",
    "QWEN_NATIVE_KV",
    "QWEN_NATIVE_Q8_TILED",
    "QWEN_NATIVE_Q4_TILED",
    "QWEN_NATIVE_PREFILL_CHUNK",
    "QWEN_NATIVE_PREFILL_PROFILE",
    "QWEN_NATIVE_PIN_MODEL",
    "QWEN_NATIVE_PINNED_STAGING",
    "QWEN_NATIVE_PINNED_STAGING_COARSE",
    "QWEN_NATIVE_SELECTOR_WEIGHT",
    "QWEN_NATIVE_ASYNC_ACTIVATIONS",
    "QWEN_NATIVE_GPU_PIPELINE",
    "QWEN_NATIVE_GPU_CHUNK_MB",
    "QWEN_NATIVE_Q8_PARTITIONS",
    "QWEN_NATIVE_HYBRID_FFN",
    "QWEN_NATIVE_UNFUSED_FFN",
    "QWEN_NATIVE_GPU_GATE",
    "QWEN_NATIVE_GPU_TOPK",
    "QWEN_NATIVE_DRAFT_N23",
)


@dataclass(frozen=True)
class Trial:
    binary: Path
    target: Path
    draft: Path
    corpus: Path
    prefix: Path
    runs: int = 3
    resident_gib: float = 1.0
    required_tps: float = 10.0

    def output(self, suffix: str) -> Path:
        return self.prefix.parent / (self.prefix.name + suffix)

    def command(self) -> list[str]:
        return [
            str(self.binary),
            "benchmark-128k",
            str(self.target),
            str(self.draft),
            str(self.output(".jsonl")),
            str(self.resident_gib),
            str(self.runs),
            str(self.corpus),
            str(self.required_tps),
        ]

    def validate(self) -> None:
        if self.runs < 1:
            raise ValueError("runs must be positive")
        if not math.isfinite(self.resident_gib) or not 0 <= self.resident_gib <= 2:
            raise ValueError("resident GiB must be finite and in [0, 2]")
        if not math.isfinite(self.required_tps) or self.required_tps <= 0:
            raise ValueError("required TPS must be finite and positive")
        for path in (self.binary, self.target, self.draft, self.corpus):
            if not path.is_file():
                raise FileNotFoundError(f"required input file not found: {path}")
        for suffix in (".jsonl", ".jsonl.prompt.tokens", ".log", ".lifecycle.jsonl"):
            if self.output(suffix).exists():
                raise FileExistsError(f"refusing to overwrite evidence: {self.output(suffix)}")


class Journal:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.started = time.monotonic()

    def emit(self, event: str, **fields: object) -> None:
        record = {
            "event": event,
            "utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - self.started,
            **fields,
        }
        self.stream.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def proc_command(pid: int) -> list[str]:
    raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    return [os.fsdecode(value) for value in raw.rstrip(b"\0").split(b"\0") if value]


def verified_signal(child: subprocess.Popen[bytes], command: list[str], number: int) -> bool:
    if child.poll() is not None:
        return False
    actual = proc_command(child.pid)
    if actual != command:
        raise RuntimeError(f"refusing signal to PID {child.pid}: command identity mismatch")
    child.send_signal(number)
    return True


def run_worker(*, command: list[str], log_path: Path, journal: Journal) -> int:
    pending_signals: list[dict[str, object]] = []
    previous_handlers = {}
    child: subprocess.Popen[bytes] | None = None
    with log_path.open("xb") as raw_log:
        try:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            def forward(number: int, frame: FrameType | None) -> None:
                # Avoid journal I/O from a reentrant signal handler.
                try:
                    forwarded = verified_signal(child, command, number)
                    pending_signals.append({"number": number, "forwarded": forwarded})
                except (OSError, RuntimeError) as error:
                    pending_signals.append({"number": number, "forwarded": False, "error": str(error)})

            for number in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[number] = signal.signal(number, forward)
            try:
                actual_command = proc_command(child.pid)
            except FileNotFoundError:
                actual_command = None  # A short-lived child can already be gone.
            journal.emit("worker_start", pid=child.pid, command=command, proc_command=actual_command)
            assert child.stdout is not None
            with child.stdout:
                for line in child.stdout:
                    raw_log.write(line)
                    raw_log.flush()
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
            return_code = child.wait()
        except BaseException as error:
            # Cleanup is the reason to catch broadly; never turn exceptions
            # into success. Identity mismatches refuse signals, not guess.
            if child is not None:
                verified_signal(child, command, signal.SIGTERM)
                child.wait()
            journal.emit("launch_error" if child is None else "runner_error", error=str(error))
            raise
        finally:
            for number, previous in previous_handlers.items():
                signal.signal(number, previous)
            if child is not None and child.stdout is not None:
                child.stdout.close()
            for record in pending_signals:
                journal.emit("runner_signal", **record)
            if child is not None and child.poll() is not None:
                journal.emit(
                    "worker_exit",
                    returncode=child.returncode,
                    signal=-child.returncode if child.returncode < 0 else None,
                )
    return return_code


def run_trial(trial: Trial) -> int:
    trial.validate()
    trial.prefix.parent.mkdir(parents=True, exist_ok=True)
    with trial.output(".lifecycle.jsonl").open("x", encoding="utf-8") as stream:
        journal = Journal(stream)
        journal.emit(
            "runner_start",
            pid=os.getpid(),
            command=trial.command(),
            executable_sha256=file_sha256(trial.binary),
            controls={name: os.environ.get(name) for name in CONTROL_NAMES},
            note="Exit code alone is not proof of physical-context correctness or target TPS.",
        )
        return_code = run_worker(command=trial.command(), log_path=trial.output(".log"), journal=journal)
        journal.emit("runner_exit", returncode=return_code)
        # Preserve failed requirement exits and actual signal numbers. The
        # native JSONL remains authoritative for runs/summary/token evidence.
        return return_code if return_code >= 0 else 128 - return_code


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "target", "draft", "corpus", "output-prefix"):
        parser.add_argument(f"--{name}", type=Path, required=True, help=f"Path for {name}")
    parser.add_argument("--runs", type=int, default=3, help="Independent full prefills/runs (default 3)")
    parser.add_argument("--resident-gib", type=float, default=1.0, help="Packed weight residency GiB")
    parser.add_argument("--required-tps", type=float, default=10.0, help="Recorded committed-token TPS threshold")
    args = parser.parse_args(argv)
    trial = Trial(
        binary=args.binary.resolve(),
        target=args.target.resolve(),
        draft=args.draft.resolve(),
        corpus=args.corpus.resolve(),
        prefix=args.output_prefix.resolve(),
        runs=args.runs,
        resident_gib=args.resident_gib,
        required_tps=args.required_tps,
    )
    return run_trial(trial)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

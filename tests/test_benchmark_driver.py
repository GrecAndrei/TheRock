"""Lifecycle tests use CPU-only subprocesses; no model/GPU benchmark is run."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_driver as driver


class DriverTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root / "input"
        self.input.write_bytes(b"fixture, not a model")
        self.trial = driver.Trial(
            binary=Path(sys.executable).resolve(),
            target=self.input,
            draft=self.input,
            corpus=self.input,
            prefix=self.root / "trial",
        )

    def worker(self, program: str) -> tuple[int, list[dict[str, object]]]:
        journal_path = self.root / "journal.jsonl"
        with journal_path.open("x", encoding="utf-8") as stream:
            code = driver.run_worker(
                command=[sys.executable, "-c", program],
                log_path=self.root / "raw.log",
                journal=driver.Journal(stream),
            )
        records = [json.loads(line) for line in journal_path.read_text().splitlines()]
        return code, records

    def test_lossless_log_and_success_exit(self) -> None:
        code, records = self.worker("import os; os.write(1, b'raw\\xff\\n'); os.write(2, b'error stream\\n')")
        self.assertEqual(code, 0)
        self.assertEqual((self.root / "raw.log").read_bytes(), b"raw\xff\nerror stream\n")
        self.assertEqual(records[0]["event"], "worker_start")
        self.assertEqual(records[-1]["returncode"], 0)
        self.assertIsNone(records[-1]["signal"])
        self.assertFalse(self.trial.output(".jsonl").exists())

    def test_failed_exit_is_not_success(self) -> None:
        code, records = self.worker("import sys; print('partial prefill'); sys.exit(7)")
        self.assertEqual(code, 7)
        self.assertEqual(records[-1]["returncode"], 7)
        self.assertFalse(self.trial.output(".jsonl").exists())

    def test_killed_worker_has_negative_return_and_signal(self) -> None:
        code, records = self.worker("import os, signal; os.kill(os.getpid(), signal.SIGKILL)")
        self.assertEqual(code, -signal.SIGKILL)
        self.assertEqual(records[-1]["signal"], signal.SIGKILL)

    def test_launch_error_is_durable(self) -> None:
        path = self.root / "journal.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            with self.assertRaises(FileNotFoundError):
                driver.run_worker(
                    command=[str(self.root / "missing-executable")],
                    log_path=self.root / "raw.log",
                    journal=driver.Journal(stream),
                )
        record = json.loads(path.read_text())
        self.assertEqual(record["event"], "launch_error")
        self.assertFalse(self.trial.output(".jsonl").exists())

    def test_identity_mismatch_refuses_signal(self) -> None:
        command = [sys.executable, "-c", "import signal; signal.pause()"]
        with subprocess.Popen(command, stdout=subprocess.DEVNULL) as child:
            try:
                with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                    driver.verified_signal(child, ["not-this-worker"], signal.SIGTERM)
                self.assertIsNone(child.poll())
            finally:
                driver.verified_signal(child, command, signal.SIGTERM)
                child.wait()
        self.assertEqual(child.returncode, -signal.SIGTERM)

    def test_evidence_is_never_overwritten(self) -> None:
        for suffix in (".jsonl", ".jsonl.prompt.tokens", ".log", ".lifecycle.jsonl"):
            with self.subTest(suffix=suffix):
                path = self.trial.output(suffix)
                path.write_text("preserve", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    self.trial.validate()
                self.assertEqual(path.read_text(), "preserve")
                path.unlink()

    def test_invalid_thresholds_and_residency_fail_before_launch(self) -> None:
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(required_tps=value):
                with self.assertRaisesRegex(ValueError, "TPS"):
                    replace(self.trial, required_tps=value).validate()
        for value in (-1.0, 2.1, float("nan"), float("inf")):
            with self.subTest(residency=value):
                with self.assertRaisesRegex(ValueError, "GiB"):
                    replace(self.trial, resident_gib=value).validate()
        with self.assertRaises(ValueError):
            replace(self.trial, runs=0).validate()

    def test_explicit_threshold_and_paths_in_command(self) -> None:
        self.assertEqual(self.trial.command()[1], "benchmark-128k")
        self.assertEqual(self.trial.command()[-1], "10.0")
        self.assertEqual(self.trial.command()[4], str(self.trial.output(".jsonl")))
        self.assertEqual(
            driver.file_sha256(self.input),
            hashlib.sha256(b"fixture, not a model").hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()

# Native engine rebuild: 10 generated tokens/s at physical 128K

Active goal: `mv2tg60t-s338ww`. This supersedes the paused 20 tok/s goal.
The model assets, frozen corpus, canonical tokenizer, native C++20/HIP
requirement, complete history, single-stream greedy correctness and evidence
contract are unchanged. This is an engine architecture task, not a flag-only
performance claim.

## Verified starting state

- Latest canonical short-fixture throughput: **3.50 generated tok/s**. Its
  physically populated prompt is **four tokens**, despite a 131336-slot capacity.
- **No completed physically populated 131072-token generation benchmark exists.**
- On 2026-10-10, both old canonical PIDs 54444/54479 were absent, no canonical
  benchmark command appeared in the OS process list, and its JSONL contained
  only metadata. Its last log checkpoint is **16384/131072 at 970.2 s**.
  Old managed ID `proc_4ec3` is absent from the current process registry and its
  old `/tmp` process logs were not found. **Exit reason is unknown.** Do not
  infer a crash, cancellation, success, or continued execution from this.
- Earlier CPU experiment manifests described work as concurrent with run-1
  prefill based on a stale lifecycle assumption. That overlap is **not verified**.
  The component timings remain uncontrolled-host diagnostics, not idle-host or
  end-to-end evidence. Prior files are preserved, not rewritten as clean runs.
- Canonical literal-text tokenizer parity remains independently verified:
  **78/78 fixtures and all 1,150,789 full frozen corpus IDs**, with the reference
  bound to supplied-GGUF vocabulary, merges and added IDs.
- Current CPU suites: **285 bit-exact fixtures** (51 Q6 predecode, 53 IQ4
  tile-GEMM, 53 IQ4 expanded predecode, 128 Q4/Q5_K tile-GEMM); CPU CTest passes.
  K formats preserve their original interleaved `p, p+32` FMA order. The K
  experiment uses the existing packed cache, 8 KiB worker-local scratch and a
  transposed batch input; it is opt-in, not yet a native performance claim.
  The combined IQ4/K experimental native fixture passed greedy equivalence
  (32 outputs, 11 cycles, 21/77 acceptance) but measured **2.72 tok/s** on four
  physical prompt tokens. This is not an improvement claim; both experimental
  tile-GEMM flags stay **off** in the baseline.
- Expanded IQ4 predecode remains **off**: +112% cache storage and slower in all
  three gate-matrix timing pairs. The cache-free IQ4 gate experiment improved
  three component pairs, but native correctness/performance must be gated.

## Baseline lifecycle preparation

The pending native edits build successfully and reject all five invalid
threshold probes with the expected `REQUIRED_TPS` errors before loading models.
`benchmark_driver.py` runs a foreground child inside the managed process group,
never detaches/restarts it, tees native bytes to a raw log, and fsyncs a journal
of exact commands, executable hash, worker identity, normal/nonzero/signal exit.
It refuses signals on command-identity mismatches and refuses evidence overwrite.
A missing exit record is explicitly unresolved, not success.

**Eight CPU-only lifecycle tests pass**, including lossless logs, launch failure,
nonzero exit, SIGKILL worker reporting, identity refusal and preservation. They
are not GPU/model performance tests. Black is unavailable in the active Python
installation; its failed formatting attempt is retained separately and did not
prevent the subsequent unit/bytecode tests or `git diff --check` from passing.

A fresh baseline uses three independent prefills with the original Q8/control
configuration and experimental IQ4/K flags disabled. Immutable executable,
source archive/diff, full asset/corpus hashes and settings are prepared before
launch; new files do not replace the old interrupted attempt.

The fresh baseline was launched at **2026-10-10 20:23:20 UTC** as
`proc_e6c6`, `qwen-10tps-physical-128k-baseline-three-runs`. At launch, its
journal observed managed Python PID **112752**, native child PID **112764**
and the child's exact command line. These are historical identities, not
proof of future survival; check the managed lifecycle/OS before signals.

- Executable SHA256: `e6e486d81c742b1f7d25ac5cd5cf88b5aae5125768cc6a05ca5d46669df86eb3`.
- Records: `goal-10tps/baseline-default-v1.jsonl`, `.jsonl.prompt.tokens`,
  `.log`, `.lifecycle.jsonl`. Immutable inputs are in `baseline-v1-*`.
- Initial loadavg **0.19**. Metadata confirms 131072 prompt tokens,
  FNV64 **0x171866bf66be32a9**, three runs, eight excluded warmups, 256 outputs,
  threshold **10**, Q8 KV, chunk 512, one stream, all new tile-GEMM flags off.
- Current PCRE2 reports **10.49** (historical audit used 10.48); ICU **78.3**.
  `cmp` verifies **all prompt-token file bytes** against the previously
  canonical-reference-certified prompt. See `baseline-v1-prompt-identity.log`.
- At the startup audit, JSONL had metadata only: **zero completed generation
  runs**. No throughput/architectural speedup claim follows from launch.
- Repeatable prefill/run/failure watches notify the parent. No overlapping
  GPU tests or CPU performance diagnostics; generation must stay isolated.
- Setting the automatic run cap to **0** unexpectedly paused the goal
  controller. The earlier claim that it would only suppress checkpoint churn
  was wrong; the managed benchmark itself was not stopped. The user changed
  `.pi/pi-goal-x-settings.json` to **10000000** and resumed the goal; active
  status is confirmed. **Preserve that user setting**; do not remove/reset it
  at benchmark completion. No pending automatic-settings restoration remains.

## Ordered execution

1. Finish validation of pending edits and record a real full-context baseline.
   Use a durable managed runner recording command, identities, exit status and
   raw output. A missing exit record must remain explicitly unresolved.
2. Use measured baseline phases to design weight streaming/residency,
   CPU/GPU work scheduling, FFN batching, full-context attention and speculative
   verification/commit changes. Retain validated arithmetic where appropriate;
   do not silently alter precision, model computation or context.
3. Gate changes against the canonical tokenizer, numerical/component oracles,
   real-prefill parity, rollback/commit, session lifetime and native greedy
   target-only/speculative equivalence. Fixtures are not full-context TPS.
4. Freeze executable/source/settings; run **three independent physical 131072
   prefills**, **eight excluded warmups**, **256 committed outputs per run**,
   isolated generation. Record each TPS, arithmetic mean, TTFT, acceptance,
   hashes/controls and raw IDs. Success requires mean **>=10.0 generated
   target tokens/s** with correctness gates satisfied. Report shortfalls honestly.

## Evidence and compatibility

Historical records stay under `benchmark-results/goal-20tps/`.
New rebuild records go under `benchmark-results/goal-10tps/`.
Reuse the unchanged `goal-20tps/fixed-project-corpus.txt`; do not regenerate it
from the now-modified working tree.

`benchmark-128k` gains an explicit final `[REQUIRED_TPS]` argument. Default **20**
is retained for compatibility with old evidence. New goal invocations supply
**10** and record that threshold in metadata/summary. Physical prompt/output
counts, warmup exclusions and minimum three-run success criteria do not change.

No commit, push, PR or external posting is authorized. Preserve unrelated work
and processes. The final funny evidence-backed retrospective waits until the
entire task concludes.

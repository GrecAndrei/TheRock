# 20 generated tok/s at genuine 128K: historical work

> **Superseded by the engine-rebuild goal targeting 10 tok/s.** See
> [`GOAL_10TPS_ENGINE_REBUILD.md`](GOAL_10TPS_ENGINE_REBUILD.md).
> On 2026-10-10 the canonical worker was absent; JSONL has metadata only and
> the last prefill checkpoint is 16384/131072 at 970.2 s. Exit cause is unknown.
> Earlier references below to a running worker/concurrent prefill were based
> on a stale lifecycle assumption; the asserted CPU diagnostic overlap is
> not verified. Component timings are still not clean end-to-end evidence.

**No completed 128K performance result or 20 tok/s claim exists yet.** This is a
checkpoint, not the final report. The user requested a funny, formatted retrospective
only after the entire goal is finished; defer that report.

## Contract

Keep the supplied Qwen3.8-27B UD Q4_K_M and exact DFlash2 assets, native
C++20/TheRock HIP execution, all 131,072 real prompt tokens, deterministic
single-stream greedy output, and target-only/speculative correctness. Measure
256 committed output tokens after eight excluded warm-up tokens, three runs,
per-run throughput and arithmetic mean, TTFT, acceptance, controls and raw
outputs. Never count candidates or a zero-filled cache as generated throughput.
The goal is 20 tok/s, not the old 10 tok/s gate.

Work tracking is the operator's local goal `mv1ab6w8-ict3dg`; no external issue or
PR has been opened. This change is device/kernel tuning, a resource-leak fix,
tokenizer algorithm optimization and benchmark tooling. Validation so far is
local gfx1032 only, not cross-device certification.

## Environment and preserved baseline

- Source base: `ef774ed4d411f8e3fb41014a923939d2e4de866d`.
- Branch: `users/grecandrei/qwen38-native-gfx1032`.
- AMD RX 6650 XT, gfx1032, 16 CUs, 7.98 GiB VRAM; i5-14600K, 20 logical CPUs;
  approximately 31 GiB RAM. The GPU also serves the desktop.
- SDK: `/home/alex/TheRock/build/dist/rocm`; build with
  `./examples/qwen38-native/build.sh`.
- Raw evidence: `benchmark-results/goal-20tps/` (untracked local artifacts).
- Original binary preserved as `baseline-executable`, SHA256
  `d172666e47830f695ed68ceb1aa243227b6f21a3c3fa81ed617d92fa697a1bf8`.
- Existing untracked `PERFORMANCE_PLAN.md`, benchmark records and `.pi/` were
  not discarded. Do not reset/fetch submodules or delete unrelated processes.

## Implemented and checked

### Device-resident speculative recurrent snapshots (opt-in)

`QWEN_NATIVE_DEVICE_CHECKPOINT=1` copies convolution and gated-delta matrices
on the verifier's compute stream, restores them there, and replays captured
accepted-prefix factors without CPU round trips. Requires about 150 MiB extra
VRAM; default remains the pre-existing host path pending broader validation.
Logical KV rollback preserves the populated target history.

Same 4-token fixture, 131336 allocated capacity (NOT populated 128K), 32 real
generated tokens, 11 cycles, 21/77 accepted proposals:

| Execution | Real generated tok/s | Time | Oracle |
| --- | ---: | ---: | --- |
| Preserved baseline | 1.01 | 31.59 s | match |
| Device checkpoint | 3.39 | 9.43 s | match |

These are short-fixture results, not the goal benchmark. Default Q8 KV and
1.0 GiB target weight residency were used. DFlash profiling showed about
26 ms layers + 15 ms logits per cycle. All prefix lengths 1–8 passed the
extended `target-rollback-test` at 8192 capacity; rollback and DFlash commit
also passed at 131336 and FP16 4096 capacities.

### Session workspace leak

`NativeTargetSession` now destroys its `recurrent_device_`. Added
`target-session-lifetime-test TARGET.gguf CONTEXT`, which primes the HIP
allocator then measures three create/destroy cycles, allowing 8 MiB allocator
bookkeeping but not the recurrent workspace.

- Untouched base plus reproduction driver: **FAIL**, first extra session
  retained 148,897,792 bytes.
- Fixed build: **PASS**, 2,097,152 bytes retained consistently across all three
  sessions, no growth.
- Reproduction build lives outside the repo at
  `/tmp/qwen38-20tps-baseline.m9M5B3/examples/qwen38-native/`.

### Headless teacher-forced prompt prefill

Prompt batches no longer calculate unused vocabulary predictions except for
the last prompt batch. All model layers and KV/recurrent state updates still
execute; feature downloads finish before injection/buffer reuse. No prompt
positions are skipped. `target-prefill-test` compared the final anchor,
subsequent eight predictions and all five feature vectors exactly, including
a partial final prompt batch: **PASS** at 131336 capacity (29 actual tokens).

### Quadratic BPE merge loop

A whole-document input exposed repeated full-vector pair scans/erases after
every merge. Replaced production merging with a versioned linked-list and
priority queue, preserving minimum merge rank and leftmost tie breaking.
The slow implementation remains only as a test oracle.

`tokenizer-self-test TARGET.gguf`: **73 fixtures PASS**, exact legacy token
pieces plus byte-lossless encode/decode. Includes ties, whitespace, numbers,
code, Unicode, arbitrary bytes and deterministic random inputs. This proves
algorithm equivalence, **not** complete official regex/pretokenizer parity;
that pre-existing quality gap remains open.

### Benchmark evidence gate

`benchmark-128k TARGET DFLASH OUTPUT.jsonl [RESIDENT_GIB] [RUNS] [PROMPT.txt]`
now accepts supplied text and refuses repetition/padding of insufficient input.
Its template reserves space for a fixed technical-review request. Passing
requires supplied text, at least three complete runs and mean >=20 tok/s;
legacy repeated filler cannot pass. Records include KV/checkpoint/draft controls,
min/max rates, prefill duration and model-initialization-inclusive TTFT. A
short-input rejection test passed. No completed summary exists yet.

## Non-cherry-picked physical benchmark in progress

Frozen corpus: 411 tracked project Markdown/Python/CMake sources, ordered
lexically, with source headings, 4,624,757 bytes. Snapshot:
`benchmark-results/goal-20tps/fixed-project-corpus.txt`, SHA256
`dc3184025b4a1a9b9f72d25940f8a5775db37a512ac625221bff632d47f92872`.

The first attempt was terminated by us before model prefill: the quadratic
legacy tokenizer was still using a CPU core after several minutes. The empty
log and manifest are retained; it is **not** a performance result.

The tokenizer-fixed attempt uses an immutable executable and source diff:

- Executable: `benchmark-results/goal-20tps/tokenizer-fixed-run-executable`.
- Manifest/diff: `tokenizer-fixed-run.manifest`, `tokenizer-fixed-run.diff`.
- Output: `fixed-corpus-tokenizer-fixed.jsonl`, `.jsonl.prompt.tokens`, `.log`.
- Process originally launched as PID **23839**, now intentionally terminated
  after its first 8192 physical tokens took **780.0 seconds**. The manifest
  records why it was stopped to validate genuine larger weight reuse. It did
  not complete 128K or produce an output-throughput result.
- Settings: HYBRID=1, CPU_THREADS=20, Q8_TILED=warp16,
  DEVICE_CHECKPOINT=1, KV=q8, target residency 1.0 GiB, three independent
  complete physical prefills.
- Metadata confirms exactly 131072 tokens; prompt FNV64
  `0x28d338ff840b0356`; initial load average 0.44.
- This attempt produced metadata and a partial prefill log, **not a completed
  generation run**. Preserve these files as failed/partial evidence.

Do not run concurrent GPU performance tests while its decode is timed. Changes
may be compiled independently, but document interference with prefill/TTFT.
Never replace this corpus because acceptance is poor or omit unsuccessful runs.

## Layer-major implementation and current physical run

Implemented `NativeTargetSession::prefill_layer_major`: tokens advance through
one target layer as a complete chunk before proceeding to the next layer.
`ScopedWeightResidency` stages the unchanged packed matrices once per layer,
without evicting or consuming the persistent decode cache budget. Recurrence
and causal KV updates remain ordered in eight-token groups; hidden states are
retained between layers. Temporary weights are freed after their work completes.
This is genuine layer-major reuse, not repeated whole-model calls.

`QWEN_NATIVE_PREFILL_CHUNK=512` enables this experimental path in the benchmark;
the default remains 8. `QWEN_NATIVE_PREFILL_PROFILE=1` adds diagnostic phase
synchronization; never enable it for the final run.

All-GPU comparisons (explicit GPU_GATE=1, HYBRID/HYBRID_FFN unset), starting
from a 17-token populated prefix:

| Capacity / compared tokens | Ordinary GPU prefill | Layer-major prefill | Predictions, logits, features |
| --- | ---: | ---: | --- |
| 8192 / 9 | 2.362 s | 1.957 s | bit-exact |
| 8192 / 64 | 9.441 s | 4.494 s | bit-exact |
| FP16 512 / 65 | 10.613 s | 4.894 s | bit-exact |
| Q8 8192 / 513 (crosses chunk boundary) | 76.660 s | 30.203 s | bit-exact |

These are **prefill equivalence fixtures, not generation performance**. All
feature relative RMSE values were zero. The diagnostic 64-token phase profile
measured recurrent 0.793 s, attention 0.225 s, FFN 2.339 s; FFN is the dominant
remaining short-context GPU phase. Its reference was also bit-exact. Normal
hybrid decode regression after extracting the shared layer executor: 32 tokens,
11 cycles, 21/77 acceptance, oracle match, **3.42 generated tok/s / 9.35 s**.

The layer-major **legacy-tokenizer attempt was interrupted** after the
independent canonical tokenizer audit found a correctness defect:

- Verified actual worker PID **34083** before sending SIGTERM; watcher
  **proc_8df0** subsequently exited. Its successful exit is not benchmark success.
- Last physical checkpoint: **32,768 tokens / 2034.4 s**. No generation run or
  128K throughput result completed.
- Preserve `layer-major-run-executable`, `.manifest`, `.diff`, and every
  `fixed-corpus-layer-major.*` record, including old prompt hash
  `0x28d338ff840b0356` and the interruption explanation in the manifest.
- The frozen corpus and model assets are unchanged. The replacement below
  corrects tokenization; it does not select different source text.

## Exact CPU Q6 predecode experiment

`QWEN_NATIVE_CPU_Q6_PREDECODE=1` is opt-in (only the exact string `1` enables it).
It stores signed six-bit codes and precomputed FP32 `d * scale`, retaining the
original coefficient values and FMA order in single and dual packed CPU paths.
Q6 tile size rises from **1680 to 2560 bytes (+52%)**; other formats are unchanged.

- New CPU CTest target: **51 bit-exact fixtures**, including an independent
  scalar Q6 oracle, partial rows, 1/4 threads, widths 256/512/5120/17408,
  guarded outputs, cache-key changes, cache bytes/clear, mixed dual paths,
  and invalid dimensions. CTest passes; the initial HIP-loader failure is
  retained in `cpu-predecode-loader-failure.log`. SDK BUILD_RPATH fixed loading.
- Initial real-tensor selection `blk.0.ffn_down.weight` failed because it is
  **IQ4_XS**, not Q6_K. Retain `cpu-predecode-real-tensor-diagnostic.log`.
- Actual Q6 tensor `blk.63.ffn_down.weight`, 17408 columns / 5120 rows,
  20 threads, 8 iterations per timing: original/predecoded pairs in ms
  **4.14717/3.25921**, **3.67397/3.14899**, **3.34443/3.13188**; all bit-exact.
  These concurrent-prefill CPU component timings are diagnostic, not idle-host
  performance and not generated-token throughput.
- Native 32-output regression with this option: oracle match, 11 cycles,
  acceptance **21/77**, **3.50 tok/s / 9.15 s**, four prompt tokens and capacity
  131336. This is **not** physically populated 128K.
- Actual base-layer FFN tensor counts: **IQ4_XS 85**, **Q5_K 50**, **Q4_K 38**,
  IQ3_S 4, IQ4_NL 6, Q3_K 7, Q6_K 2. Q6 is not the main FFN format.

## Opt-in IQ4_XS CPU 8x8 microkernel

`QWEN_NATIVE_CPU_IQ4_TILE_GEMM=1` is a second opt-in CPU experiment, aimed at
85 of the 192 base-layer FFN matrices. It transposes the eight input batches,
decodes one 256-column block into an **8 KiB worker-local FP32 scratch tile**,
and accumulates across batches rather than across output rows. Packed weight
storage is unchanged. Coefficients and ascending-column FMA order are preserved;
no activation quantization, model changes, or context changes are introduced.
The input transpose additionally uses up to about **544 KiB** at width 17408.

- **53 additional bit-exact fixtures** pass against both original packed and
  generic scalar paths: widths 256/512/5120/17408, partial rows, 1/4 workers,
  guarded outputs, cache reuse/unchanged bytes, IQ4/IQ4, IQ4/Q4 and IQ4/Q6 dual
  combinations. Existing **51 Q6 fixtures still pass**. CPU-only CTest succeeds.
- Real `blk.0.ffn_down.weight` (17408 x 5120, 20 workers, 8 iterations)
  original/tile-GEMM paired timings in ms: **2.34289/2.49375**,
  **3.09124/2.65777**, **3.03402/2.42226**. All outputs are bit-exact.
  One pair is slower. Preserve every pair; do not claim a consistent speedup,
  idle-host performance, or generated-token throughput from these diagnostics.
- Evidence: `build-iq4-cpu-self-test.log`, `iq4-cpu-ctest.log`,
  `iq4-cpu-self-test.log`, `iq4-real-tensor-component-diagnostic.log`.
- Real gate orientation `blk.0.ffn_gate.weight` (5120 x 17408, 20 workers,
  8 iterations): original/tile-GEMM pairs in ms **2.57172/1.81997**,
  **2.14339/1.73746**, **2.28598/2.03806**; all bit-exact. Every gate pair is
  faster, unlike the mixed down-matrix result, but these are still concurrent
  host diagnostics—not an idle-host or end-to-end improvement claim. Preserve
  `iq4-gate-gemm-diagnostic.log`; native model parity is still pending.
- Source inspection found that full inline down-weight prefetch makes
  `pipeline_project_batch8_hybrid` choose the full GPU path. Consequently,
  CPU down-matrix component speed is not necessarily on the current native
  critical path. The gate/up case is relevant to CPU hybrid work. Do not change
  prefetch routing without separate numeric/model gates; no routing fix has
  been applied from inspection alone.
- This prototype is **absent from the immutable running benchmark**; no GPU
  validation was run concurrently. Native inference parity and idle-host
  component timing remain pending before selecting any default.
- CPU-only compilation/tests and component diagnostics during run-1 prefill are
  disclosed in `canonical-run.manifest`; TTFT includes this host interference.
  Timed generation must remain isolated from all additional performance tests.

## IQ4_XS expanded-code predecode: tested, not selected

A separate opt-in experiment, `QWEN_NATIVE_CPU_IQ4_PREDECODE=1`, replaces
per-column nibble lookup and scale extraction with cached signed nonlinear
codes and FP32 group-scale products. Coefficients/FMA order remain exact.
It takes precedence over IQ4 tile-GEMM and expands IQ4 cache tiles from
**1088 to 2304 bytes (+112%)**. It is **off by default**, absent from the frozen
benchmark, and not selected for performance use on the basis of these results.

- **53 additional bit-exact fixtures** pass: packed/scalar oracle, partial rows,
  widths 256/512/5120/17408, 1/4 workers, old/new cache keys and byte accounting,
  combined flag precedence, dual IQ4/IQ4 and mixed IQ4/Q4/Q6 paths. Together
  with 51 Q6 and 53 tile-GEMM fixtures, **157 fixtures pass**; CPU CTest passes.
- Down tensor (17408 x 5120): original/predecoded ms **4.21260/2.90422**,
  **2.13999/2.82804**, **2.27386/3.05871**. Two of three pairs are slower.
- Gate tensor (5120 x 17408): **2.21552/2.45719**, **2.09496/2.62699**,
  **2.24481/2.62700**. All three pairs are slower. All outputs are bit-exact;
  20 workers, 8 iterations, alternating order, packing/warmup excluded.
- Measurements were during run-1 prefill, not idle-host measurements or
  generated-token rates. Increased traffic has no demonstrated useful trade-off.
- Retain `build-iq4-predecode-cpu.log`, `iq4-predecode-cpu-ctest.log`,
  `iq4-predecode-cpu-self-test.log`, `iq4-predecoded-real-tensor-diagnostic.log`,
  `iq4-gate-predecode-diagnostic.log`, `cpu-iq4-predecode-source.diff`, and
  `cpu-iq4-predecode-self-test-source.cpp`.
- Native build verification **passed** (`proc_d7b4`, 7 s), without executing
  GPU tests. `build-native-cpu-experiments.log` and the frozen
  `cpu-experiments-native-executable` / `.sha256` are retained for later native
  parity gates. The running canonical executable/model/corpus were not replaced.
  This successful build does not constitute new native inference validation.

## Canonical tokenizer correction and replacement physical benchmark

Downloaded only the official **Qwen3.5-family tokenizer JSON**, not any weights.
The supplied asset declares `qwen35`; all **248044 normal vocabulary tokens**,
**247587 ordered merge ranks**, and **26 added-token IDs** bind exactly to its
GGUF metadata. Source URL/hash are retained beside the reference JSON.

The legacy whole-document BPE matched only **11/14** independent fixtures:
repeated spaces were incorrectly grouped and combining characters lacked NFC.
These failures are preserved in `official-tokenizer-parity-audit.log`; an earlier
audit-output-parser error is preserved separately. Implemented ICU NFC followed
by the reference Unicode split pattern via PCRE2, then existing byte BPE.
Invalid UTF-8 diagnostics retain the old byte-lossless fallback; unknown pieces
and unsupported pretokenizer families now fail instead of silently dropping text.
Chat control IDs are explicit; source-document special markers are literal text.

Corrected evidence:

- **78/78 independent reference fixtures** match, including seeded Unicode cases.
- **Every one of 1,150,789 literal-text corpus token IDs matches** the reference.
- Existing 73 BPE fixtures still pass; three canonical regressions are permanent.
- CPU CTest and native speculative oracle checks pass. A fresh 513-token
  chunk-boundary layer-major test passes with bit-exact features/predictions.
- Native dependencies: PCRE2 **10.48**, ICU **78.3**, C++20 unchanged. Version and
  CPU-predecode controls are recorded in benchmark metadata.

The replacement genuine three-run benchmark **was launched**, managed directly
as **proc_4ec3**, `qwen128k-canonical-three-run-benchmark` (not a tail watcher).
It is **not currently running** and produced no completed generation records:

- Actual native worker PID **54479**, managed wrapper PID **54444**. Verify
  the exact `/proc/PID/cmdline` before any future signal; no unrelated processes.
- Frozen binary `canonical-run-executable`; manifest/diff `canonical-run.*`.
- Records `fixed-corpus-canonical.jsonl`, `.jsonl.prompt.tokens`, `.log`.
- Prompt: **131072 canonical token IDs**, new FNV64 **0x171866bf66be32a9**.
- Same corpus SHA256 **dc3184025b4a1a9b9f72d25940f8a5775db37a512ac625221bff632d47f92872**.
- Q8 KV, 1.0 GiB residency, HYBRID=1, CPU_THREADS=20, Q8_TILED=warp16,
  DEVICE_CHECKPOINT=1, PREFILL_CHUNK=512, CPU_Q6_PREDECODE=1. Three independent
  full prefills; eight excluded warmups; 256 committed outputs per run.
- Model SHA256s, executable/corpus/reference hashes and library versions are in
  the manifest. Metadata exists; **zero completed generation runs** at launch.
- No overlapping GPU tests. Follow managed notifications and actual JSONL run
  records/summary; no readiness polling. **20 tok/s remains unproven.**

## C++26 question

Tested the actual SDK's AMD Clang 23 with `-std=c++20` versus `-std=c++2c` on
`cpu_matvec.cpp`, identical `-O3 -march=native -mtune=native -mavx2 -mfma -mf16c`.
The math-loop generated assembly was unchanged; differences were exception
cleanup, RTTI and related string-label metadata. Artifacts `cpu-matvec-*.s`
and `compiler-*.log` preserve this comparison. This is **code-generation
inspection, not measured generation throughput**. Keep the project C++20;
changing the standard flag alone is not a demonstrated optimization.

## What did not help / remaining work

Seven capacity-only verifier tuning settings were recorded, not presented as
output performance. Splits 70/80%, 12/16 workers and increased residency were
not enough; warmed passes remained 0.78–0.88 s versus about 0.75–0.78 s with
the default 90% split and 20 workers. Logs `tune-*.log` retain every setting.

The eight-position verifier cannot deliver 20 generated tok/s at a ~0.75 s
pass, even with perfect acceptance, before adding full-history attention.
Real acceptance was 27% on the preserved fixture. This is a serious feasibility
constraint, not proof that all possible implementations are impossible.

Next priorities:

1. Consume the actual physical 128K result; do not infer success from metadata.
2. Layer-major prefill now exists and passes exact partial/chunk-boundary
   fixtures. Next investigate larger FFN projection kernels that amortize
   dequantization and activation loads across more than eight queries, rather
   than only reusing staged weights. Preserve the original FMA/reduction order
   if possible to retain exact all-GPU parity; no wide FFN kernel is implemented
   yet. Any experiment must be opt-in and independently tested.
3. Profile long-context attention and weight-streaming costs, and investigate
   correctness-preserving ways to reduce verifier cost/increase useful work.
   Avoid concurrent GPU workloads during timed decode; retain all attempts.
4. Canonical tokenizer parity now covers the full frozen literal-text corpus;
   retain that audit while broadening independent inference correctness coverage.
5. Only claim completion with three recorded real runs meeting the objective.
   Otherwise state the actual shortfall and hardware constraints honestly.

All builds used `-Wall -Wextra -Wpedantic -Werror`; `git diff --check` passed.
No commit, push, external post, governor change or unrelated process termination
has been performed.

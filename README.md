# qwen38-native

This is the isolated timeout/prefill repair branch. Current changes and commands
are documented in [`TIMEOUT_PREFILL_FIX.md`](TIMEOUT_PREFILL_FIX.md): reclaimable
model sources, shared chunked chat/benchmark prefill and exact DFlash feature
capture. Historical observations below remain historical evidence.

`qwen38-native` is a standalone C++20/HIP chat and test executable for the
official Qwen3.8-27B UD Q4_K_M target and DFlash2 Q4_K_M drafter on gfx1032.
It parses GGUF v3 and executes the model directly. It does **not** build, link,
or invoke llama.cpp, ggml, or another inference runtime.

> **Current status:** interactive deterministic chat works and accepts a
> 131,072-token cache capacity, but the project performance goal is not yet
> met. The historical, pre-canonical-tokenizer Q4 short-prompt result was
> **4.82 generated tokens/s**, not physically populated 128K. The latest Q8
> four-prompt-token regression with device checkpoints and exact CPU Q6
> predecode is **3.50 tok/s**, also not full-context performance. Canonical
> tokenization now matches every frozen-corpus token ID. The previous canonical
> worker is absent; its records stop at partial prefill, with no full-context
> generation result. The active goal is an engine rebuild targeting **10 tok/s**;
> see [`GOAL_10TPS_ENGINE_REBUILD.md`](GOAL_10TPS_ENGINE_REBUILD.md).

See [`PERFORMANCE_PLAN.md`](PERFORMANCE_PLAN.md) for the baseline, correctness
history, paper mapping, experiments, raw measurements, and remaining work.

## Required model files

The locally tested official assets are:

```text
/home/alex/Downloads/Qwen3.8-27B-UD-Q4_K_M.gguf
/home/alex/Downloads/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

The target is a 15.33 GiB mixed-quantization GGUF, not a uniformly Q4 tensor
file. The native dispatcher handles its F32, IQ3_S, IQ4_NL, IQ4_XS, Q3_K,
Q4_K, Q5_K, Q6_K, and Q8_0 tensors. IQ3_S now executes directly from its
official 110-byte blocks with the canonical 512-entry D4 lattice, packed
high bits, signs, and odd subblock scales; the former IQ3_S-to-Q8_0 runtime
requantization has been removed. The source GGUF bytes are never modified.

## Build

The tokenizer requires Unicode-enabled PCRE2 and ICU (ICU 60+), plus
`pkg-config`. On Debian/Ubuntu, install `libpcre2-dev libicu-dev pkg-config`.
The tested versions are PCRE2 10.48 and ICU 78.3. These are Unicode/text
libraries, not inference runtimes; no Python tokenizer is used during inference.

From the TheRock checkout root:

```bash
./examples/qwen38-native/build.sh
```

The executable is written to:

```text
examples/qwen38-native/build/qwen38-native
```

Inspect the selected device with:

```bash
./examples/qwen38-native/build/qwen38-native device
```

The measured development system is an AMD Radeon RX 6650 XT (`gfx1032`, 16
CUs, 7.98 GiB) with an Intel i5-14600K.

## Interactive chat

Recommended current one-liner:

```bash
QWEN_NATIVE_HYBRID=1 QWEN_NATIVE_KV=q4 ./examples/qwen38-native/build/qwen38-native chat /home/alex/Downloads/Qwen3.8-27B-UD-Q4_K_M.gguf /home/alex/Downloads/Qwen3.8-27B-DFlash2-Q4_K_M.gguf 131072 3.0
```

Syntax:

```text
qwen38-native chat TARGET.gguf DFLASH.gguf [CONTEXT] [RESIDENT_GIB]
```

- `CONTEXT` is the target cache capacity. It does not fabricate or prefill that
  many tokens.
- `RESIDENT_GIB` is the target GPU weight-cache budget. `1.0` leaves room for
  the 4.25 GiB exact Q8 target KV cache and the resident official DFlash2
  matrices on this 8 GiB card.
- Generation is deterministic greedy, one stream. Seven DFlash proposals are
  verified with the anchor in the checkpoint-native eight-position block.
- Only target-confirmed tokens are counted and displayed as generated tokens.
  Draft proposals are never counted as output throughput.
- The checkpoint's `xhigh` reasoning system turn is added. Reasoning text is
  hidden by default.

Interactive commands:

| Command | Effect |
|---|---|
| `/reset` | Clear target and DFlash conversation state. |
| `/think` | Toggle display of reasoning text. |
| `/quit` | Exit. |

The first prompt includes model initialization, first-use CPU packing, and
prefill. It can be much slower than steady-state speculative cycles. The CLI
prints proposal time, target verification/commit time, accepted draft prefix,
context occupancy, and generated-token throughput to distinguish these costs.

## What is implemented

### Target

- Native GGUF metadata/tensor parsing and canonical `qwen35` tokenization:
  NFC normalization, Unicode pretokenization, and byte-level BPE. Ordinary source
  text treats textual special-token markers literally; chat control IDs are
  inserted explicitly. Invalid UTF-8 retains a byte-lossless diagnostic fallback.
  Unsupported pretokenizer families and unknown BPE pieces fail explicitly.
- Token embedding, RMSNorm, all 64 target layers, final norm, Q6_K output head,
  and device argmax.
- Forty-eight gated-delta recurrent layers with dynamic convolution, corrected
  recurrent V-head tiling, L2 normalization, and FP32 recurrent matrix state.
- Sixteen causal full-attention layers with Qwen3.5 MRoPE and an authoritative
  device-resident Q8_0 K/V cache.
- Exact per-query causal lengths for speculative verification.
- Transactional recurrent, convolution, attention, and feature state. A
  rejected suffix is discarded and the accepted prefix is committed without a
  second 64-layer replay.
- Native mixed-quant GPU kernels plus exact AVX2/FMA/F16C batch-eight CPU
  decoders and a persistent row pool (24-worker capacity, 20 active by default
  on the i5-14600K while HIP is concurrent).

### DFlash2

- Official eight-position block: one anchor and seven proposals.
- Official mask token `248070`, top-k 16 selector lattice, and target feature
  capture after zero-based layers `[5, 19, 33, 47, 61]`.
- Five noncausal draft attention layers with the corrected symmetric
  2048-token window and full 128-dimensional Qwen3 RoPE.
- Batched target-feature encoding/injection for committed tokens only.
- Official quantized draft matrices resident by default. The older N23 draft
  conversion is now opt-in because it is not the checkpoint's exact execution
  path.
- Greedy accepted-prefix verification plus target bonus token, with target and
  DFlash caches kept transactionally aligned.

## Correctness evidence

Canonical tokenizer audit binds all 248044 ordinary vocabulary tokens,
247587 merge ranks and 26 added-token IDs in the official Qwen3.5-family JSON
exactly to the supplied asset's `qwen35` metadata. The corrected native tokenizer
matches **78/78 independent fixtures** and **all 1,150,789 frozen literal-text
corpus token IDs**. Three whitespace/NFC regressions are retained in
`tokenizer-self-test`; the earlier 11/14 failure and interrupted legacy run are
preserved under `benchmark-results/goal-20tps/`. This certifies text tokenization,
not whole-model numerical parity with an external inference implementation.


The following checks currently pass:

- Sequential committed-Q8 target execution and pipelined batch-eight execution
  return the same token sequence for tested fixtures and 131072-capacity state.
- `target-rollback-test` restores a rejected speculative suffix.
- `dflash-commit-test` keeps target position and DFlash injected-cache length in
  lockstep.
- `speculative-equivalence` matched 16 target-only greedy tokens exactly over
  seven DFlash cycles; 10 of 49 draft proposals were accepted.
- Corrected Q6_K output projection tests have maximum errors around
  `2.4e-6` (single) and `3.1e-6` (batch eight).
- Native IQ3_S batch-eight GPU execution agrees with an independently written
  scalar CPU decoder within `8.11e-6` on the real
  `blk.11.ffn_gate.weight`; direct streamed execution measured 4.045 ms and
  9.47 GB/s. Removing the old Q8 conversion changed deterministic target
  fixture tokens, confirming that direct IQ3_S is correctness-critical.
- Tested ASCII tokenizer fixtures agree with the official `tokenizer.json`.
- A current chat fixture produces the coherent deterministic reply
  `Hello! How can I help you today?`.

These checks establish native pipeline consistency, not full model
certification. Remaining correctness work includes an independent external
numerical oracle for the complete hot path, dedicated fixtures for every mixed
quant format and recurrent/attention component, authoritative DFlash parity
coverage, and Unicode regex/pretokenization validation.

Useful commands:

```bash
BIN=./examples/qwen38-native/build/qwen38-native
TARGET=/home/alex/Downloads/Qwen3.8-27B-UD-Q4_K_M.gguf
DFLASH=/home/alex/Downloads/Qwen3.8-27B-DFlash2-Q4_K_M.gguf

$BIN target-compare-batch "$TARGET" 131072 8678 1.0
$BIN target-rollback-test "$TARGET" 131072 0 1.0
# Exercise every accepted-prefix length with device-resident snapshots.
for prefix in {1..8}; do
  QWEN_NATIVE_DEVICE_CHECKPOINT=1 QWEN_NATIVE_HYBRID=1 \
    $BIN target-rollback-test "$TARGET" 8192 0 0.5 "$prefix"
done
$BIN target-session-lifetime-test "$TARGET" 8192
$BIN tokenizer-self-test "$TARGET"
$BIN target-prefill-test "$TARGET" 131336 1.0
# Layer-major tests compare against unchanged all-GPU microbatch arithmetic.
env -u QWEN_NATIVE_HYBRID -u QWEN_NATIVE_HYBRID_FFN \
  QWEN_NATIVE_GPU_GATE=1 $BIN target-layer-major-test "$TARGET" 8192 513 1.0
$BIN dflash-commit-test "$TARGET" "$DFLASH" 131072 0 1.0
$BIN speculative-equivalence "$TARGET" "$DFLASH" 131072 \
  151644,8678,198,248068 16 1.0
```

## Current performance

All figures below are local observations, not portable guarantees.

| Measurement | Current observation | Qualification |
|---|---:|---|
| Q4 KV speculative chat (3.0 GiB residency) | about 4.82 generated tok/s | Real committed tokens; short active history. |
| Q8 KV speculative chat (1.0 GiB residency) | about 3.79 generated tok/s | Real committed tokens; short active history. |
| Synthetic 131K target batch, Q4 KV (4.0 GiB) | about 1.09 s | Zero-filled Q4 cache; target-only, no DFlash. |
| Synthetic 131K target batch, Q8 KV (2.0 GiB) | about 1.10 s | Zero-filled Q8 cache; target-only, no DFlash. |
| Synthetic full-window Q8 attention portion | about 0.40 s/16-layer batch | Phase attribution is partially asynchronous. |
| Synthetic full-window Q4 attention portion | about 0.44 s/16-layer batch | DP4A scoring; ALU-bound on RDNA2 VALU. |
| Synthetic full-window FFN, Q4 at 4.0 GiB | about 0.27 s | More weight residency from halved KV cache. |
| Synthetic full-window FFN, Q8 at 2.0 GiB | about 0.37 s | Max Q8 residency (DFlash fit limited). |
| Exact resident DFlash proposal | about 0.10–0.12 s/cycle | Seven draft proposals. |
| Exact CPU SoA Q4_K/Q5_K | about 22–24 GB/s | Real model tensors, batch eight. |
| Exact CPU SoA Q6_K/IQ4_XS | about 19 GB/s | Real model tensors, batch eight. |

Q4 KV attention is ~10% slower than Q8 at 131K (ALU-bound: nibble extraction
overhead outweighs bandwidth savings on RDNA2). The net win comes from halving
KV cache VRAM (1.12 vs 2.12 GiB), which allows 3× higher weight residency in
speculative chat mode (3.0 vs 1.0 GiB with DFlash present), yielding 27%
faster FFN and 27% higher end-to-end throughput. At the current approximately
1.09 s synthetic full-window target pass plus 0.10–0.12 s draft cost, the
20 generated-token/s requirement remains unmet. Actual DFlash acceptance is
prompt dependent and has included `0/7`, `1/7`, `5/7`, and `7/7` cycles.

## Performance controls

These are implementation experiments, not model-quality switches:

| Variable | Meaning/status |
|---|---|
| `QWEN_NATIVE_HYBRID=1` | Exact CPU/GPU output-row sharding using reusable CPU SoA packs. Recommended for current local testing. |
| `QWEN_NATIVE_DEVICE_CHECKPOINT=1` | Opt-in device-resident speculative recurrent snapshots and prefix restoration; avoids CPU round-trips. Uses approximately 150 MiB of additional VRAM. Tested across prefixes 1–8 and target-only/speculative equivalence on gfx1032; not a completed 128K throughput result. |
| `QWEN_NATIVE_PREFILL_CHUNK=512` | Experimental benchmark-only layer-major prefill, staging each layer once for up to 512 teacher-forced tokens. Requires extra temporary VRAM for one layer's weights. All-GPU predictions, logits and feature values matched bit-exactly at partial/chunk-boundary fixtures on gfx1032; full 128K throughput remains unproven. Default is 8. |
| `QWEN_NATIVE_PREFILL_PROFILE=1` | Diagnostic synchronization/timing of layer-major recurrent, attention and FFN phases. Current source enables it only for literal `1`; old frozen executables treated any defined value (including `0`) as enabled. The baseline-v1 prefill/TTFT includes that diagnostic overhead; its decode path does not. Disable for final clean runs. |
| `QWEN_NATIVE_CPU_Q6_PREDECODE=1` | Experimental exact CPU Q6_K coefficient/scale predecode. Preserves the original FMA order; 51 CPU fixtures and a real target tensor matched bit-exactly. Uses 52% more memory for Q6_K CPU cache entries only. This is not a demonstrated 128K generated-token speedup. |
| `QWEN_NATIVE_CPU_IQ4_TILE_GEMM=1` | Opt-in IQ4_XS CPU 8x8 microkernel: transposes batch inputs and decodes one 256-column block into 8 KiB worker-local scratch. Preserves coefficients and ascending-column FMA order, without expanding the packed matrix cache. 53 scalar/packed parity fixtures passed; no end-to-end speedup is established. Absent from the old frozen canonical benchmark. |
| `QWEN_NATIVE_CPU_IQ4_PREDECODE=1` | Experimental exact IQ4_XS nonlinear-code/scale predecode. Expands its CPU cache tiles from 1088 to 2304 bytes (+112%); takes precedence over IQ4 tile-GEMM. 53 additional bit-exact fixtures pass. Down-matrix diagnostics were slower in two of three pairs and gate-matrix diagnostics slower in all three; no end-to-end gain is established. Disabled by default and absent from the old frozen canonical benchmark. |
| `QWEN_NATIVE_CPU_K_TILE_GEMM=1` | Opt-in Q4_K/Q5_K CPU coefficient-tile GEMM; preserves interleaved `p,p+32` FMA order. 128 additional bit-exact fixtures pass. Combined IQ4/K short native fixture measured 2.72 tok/s, not a speedup claim; both stay off in baseline-v1. |
| `QWEN_NATIVE_ASYNC_ACTIVATIONS=1` | New opt-in activation D2H stream and producer/copy events. CPU jobs wait for completed copies while the host queues GPU shards; buffer reuse waits for job completion. Worker failure/lifetime CPU tests pass; HIP integration/numerical/performance gates remain pending. Default off, absent from baseline-v1. |
| `QWEN_NATIVE_SOA_SPLIT=N` | Override selected CPU row percentage (`10`–`90`) for supported major formats. Tuning aid. |
| `QWEN_NATIVE_CPU_THREADS=N` | Select `1`–`24` active CPU workers. Default `20`; isolated CPU microbenchmarks can prefer `24`, while concurrent HIP generally does not. |
| `QWEN_NATIVE_GPU_PIPELINE=1` | Chunked transfer/compute overlap for GPU row shards, enabled by default in current source (metadata records `1`). `0` selects the non-pipelined comparison path. Earlier neutral/slower diagnostics are not a full-context conclusion. |
| `QWEN_NATIVE_Q8_TILED=warp16` | Selected Q8 long-attention launch shape. `warp`, `warp12`, `warp24` and `warp16lds` remain comparisons. New opt-in `warp16packed` stages codes and FP32 scales in 9 KiB LDS instead of 32 KiB; CPU tile tests pass, GPU parity/speed remain untested. Absent from baseline-v1. |
| `QWEN_NATIVE_Q8_PARTITIONS=N` | Override long-attention partition count, clamped to the allocated maximum of **32** in current source. The older claim of 64 was stale. |
| `QWEN_NATIVE_Q4_TILED=gqa48` | Experimental Q4-KV 48-query GQA/LDS attention kernel. It keeps one 16-token K/V tile per `(KV head, partition)` workgroup. Opt-in while numerical parity coverage is expanded; the `warp16` Q4 path remains the default. |
| `QWEN_NATIVE_HYBRID_HEAD=1` | Experimental CPU/GPU output-head split; can exceed VRAM at larger residency budgets. |
| `QWEN_NATIVE_HYBRID_FFN=1` | Experimental fused FFN hybrid path; measured slower than the selected path. |
| `QWEN_NATIVE_GPU_GATE=1` | Disable the default CPU recurrent gate schedule and return it to GPU. Measured slower. |
| `QWEN_NATIVE_DRAFT_N23=1` | Experimental approximate draft repack. Off by default; target verification still protects committed output. |
| `QWEN_NATIVE_KV=q4` | Lower-precision KV experiment, retaining physical history. At benchmark capacity 131336, K+V storage alone is about **2.25 GiB** versus **4.26 GiB** for Q8. Earlier 27% gains were short/precanonical fixtures, not physical-128K evidence. Not selected for the revised goal; requires long-context correctness/quality gates. |
| `QWEN_NATIVE_KV=n23` or `n23k` | Experimental lower-bit target KV. Both changed deterministic outputs and are rejected for correctness claims. |
| `QWEN_NATIVE_TRACE=1` | Enable verbose target/DFlash diagnostics. |
| `QWEN_NATIVE_CASCADE_TRACE=1` | Diagnose an experimental second stale DFlash block. Initial full-accept cycles missed the required bonus token, so this is not a decode mode. |

## Benchmarks and diagnostics

```bash
BIN=./examples/qwen38-native/build/qwen38-native
TARGET=/home/alex/Downloads/Qwen3.8-27B-UD-Q4_K_M.gguf

# Real tensor CPU/GPU paths and numerical deltas.
$BIN bench-matvec "$TARGET" blk.3.ffn_gate.weight 10

# Stateful target verifier over batches of eight.
QWEN_NATIVE_HYBRID=1 $BIN target-verify-batch "$TARGET" 131072 248068 4 1.0

# Physical 131072-entry zero-filled Q8 scan, repeated three times.
QWEN_NATIVE_HYBRID=1 QWEN_NATIVE_Q8_TILED=warp16 \
  $BIN target-long-window-bench "$TARGET" 131072 248068 1.0

# Isolated cache and recurrent kernels.
$BIN bench-kv4 131072 5 64
$BIN bench-gdn 100
```

`target-long-window-bench` deliberately labels itself **zero-filled KV timing
only**. It is useful for kernel profiling, but it does not satisfy the final
benchmark contract.

## Deterministic 128K benchmark harness

The native harness now constructs and records an exactly 131,072-token ChatML
prompt from tokenizer-produced paragraph segments, performs an eight-generated-
token warm-up outside the timed interval, and measures exactly 256 committed
greedy tokens. Its context allocation includes the 128K prompt plus warm-up and
measured continuation, so no prompt token is dropped. It records hardware,
model sizes/paths, environment controls, prompt and output hashes, raw output
token IDs, TTFT, phase times, acceptance, each run, and the arithmetic mean as
JSONL. A companion file preserves all 131,072 prompt token IDs. Supply a
fixed text corpus large enough for the context; the harness refuses to repeat
or pad an undersized corpus. It selects the leading corpus tokens that fit
alongside the ChatML template and a fixed technical-review question.

```bash
BIN=./examples/qwen38-native/build/qwen38-native
TARGET=/home/alex/Downloads/Qwen3.8-27B-UD-Q4_K_M.gguf
mkdir -p examples/qwen38-native/benchmark-results
PROMPT=/path/to/fixed-corpus.txt
QWEN_NATIVE_HYBRID=1 QWEN_NATIVE_Q8_TILED=warp16 \
QWEN_NATIVE_DEVICE_CHECKPOINT=1 QWEN_NATIVE_KV=q8 \
  $BIN benchmark-128k "$TARGET" \
  /home/alex/Downloads/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  examples/qwen38-native/benchmark-results/rx6650xt.jsonl 1.0 3 "$PROMPT" 10
```

The command warns when initial system load is above the clean-run threshold,
checks that every run emits the same greedy token sequence, and exits with an
error when the mean is below the explicit `REQUIRED_TPS` (10.0 in this example),
fewer than three runs complete, or no supplied text corpus is used. Omit the
final argument to retain the historical 20.0 threshold; the selected threshold
is recorded in metadata and summary. The legacy repeated-filler
workload remains available without `PROMPT.txt` for diagnostics, but cannot
pass this gate. Each run records throughput; the summary also records its
minimum and maximum, not just its mean. At current
speed, physical 128K prefill is a multi-hour operation per clean run. A smoke
start validated prompt size `131072`, prompt FNV64
`0x4158e53b26398dc3`, JSON metadata, and the companion prompt-token record; the
full three-run command has not completed.

## Final benchmark contract: not yet satisfied

Completion requires all of the following in one reproducible result:

1. An actually tokenized 131,072-token chat prompt, not merely allocated cache
   capacity or synthetic zero-filled history.
2. Deterministic greedy sampling, one stream, warm-up excluded.
3. Exactly 256 real generated target tokens per timed run.
4. Three clean timed runs on the local RX 6650 XT/i5-14600K.
5. Each decode rate, arithmetic mean, TTFT, and DFlash acceptance/draft
   statistics recorded separately.
6. Mean decode throughput of at least **10.0 generated tokens/s** for the active
   rebuild goal (the original 20.0 goal is paused).
7. Target-only/speculative equivalence and all quality/dependency checks still
   passing.

The harness is implemented, but its complete three-run measurement remains
unfinished because the current executor is below the required throughput.
Draft proposals must not be substituted for generated tokens, and a fake 128K
occupancy result must not be reported.

## Legacy external baseline

[`../qwen38-dflash`](../qwen38-dflash/) records an older llama.cpp-based
comparison. It is not built, linked, invoked, or used as a runtime dependency
by this native executable and is outside the pure-TheRock completion path.

# Candidate ports and promotion gates

Reference pin: llama.cpp `5ecbe1ac17ec0484c5b44af0bd580cdc9c428ed4`.
Fresh upstream pin: `966baae76b578d5d3dd4c179a75a35c80ff75577`, checked out in
`references/llama-upstream` on 2026-10-11. Keep the two references explicit in
every future port and measurement rather than treating them as the same source.
No candidate below is proven faster on this RX 6650 XT by this setup.
No algorithm has been ported yet.

| Candidate | Reference source | Required evidence before adoption |
| --- | --- | --- |
| GGUF metadata and mixed quantization semantics | `ggml/src/gguf.cpp`, `ggml/src/ggml-quants.c` | Independent real-tensor decoder parity for every format in the supplied GGUF; malformed-input guards |
| Canonical tokenizer | `src/llama-vocab.cpp` | All frozen ordinary-text IDs and special-token rules; retain native Unicode fixtures |
| Qwen recurrent/full-attention graph | `src/models/qwen35.cpp`, `src/models/qwen3next.cpp` | Component and complete-model numerical oracle, causal tails, real-prefill boundaries |
| DFlash2 convolution/selector and draft graph | `src/models/dflash.cpp`, `common/speculative.cpp` | Official metadata compatibility, feature taps, window semantics, exact commit/rollback and greedy equivalence |
| CPU dequantization and batched projection | `ggml/src/ggml-cpu` | Quantization parity, correct arithmetic order, measured bandwidth and full inference benefit |
| Quantized GPU matvec/matmul tiling | `ggml/src/ggml-cuda`, `ggml/src/ggml-hip` | Reimplement in Kyanite's target/compiler path; verify gfx1032 instruction legality, tails, silicon parity and end-to-end win |
| Attention tiling and partition/reduction | `ggml/src/ggml-cuda`, `ggml/src/ggml-hip` | Full real KV history, causal parity, occupancy/LDS measurement and whole generation wall time |
| Residency and transfer scheduling | `src/llama-model-loader.cpp`, `ggml/src/ggml-backend.cpp` | Actual transfer accounting, memory ownership, bounded lifetime and measured hybrid critical path |

Algorithm provenance must include the exact source pin and source paths. When
copying or adapting licensed source, preserve copyright/license attribution and
record the destination and changes. Kyanite's direct-KFD runtime remains central;
reference engines and HIP kernels are development oracles outside its shipped
dependency graph.

For each future experiment record: hypothesis, source/destination, correctness
fixtures, device result, workload, baseline/candidate distributions, resource
cost, acceptance decision and failure evidence. Record `unmeasured`, `rejected`
or `promoted` explicitly. Compiler support, modeled cost and upstream popularity
do not constitute a promotion gate.

## Native scheduling experiments, 2026-10-11

The isolated HIP/C++ diagnostic and optional staging buffers are original local
scheduling work, not an upstream algorithm port or a Kyanite runtime dependency.
See `DECODE_PROGRESS.md` for frozen inputs, actual output IDs and phase evidence.
Bounded coherent and noncoherent host staging both passed exact native parity
and produced the baseline's measured outputs, but achieved only 1.11 and 2.11
tokens/s versus the 3.18 baseline interval. Both are **rejected for promotion**
and default off. The controlled real-prompt diagnostic is retained as evidence
infrastructure, not as a claimed engine speedup.

## Draft-head calibration

The user authorized head/drafter/target changes. `MODEL_EXPERIMENTS.md` retains
the selector scales 0.0, 0.5 and 1.0, conditional acceptance/recall diagnostics,
and an independent C++ holdout. All tested head variants matched their real
prompt's 256 committed target IDs. Half-strength reduced verification count on
one prompt but not the holdout. It is **not promoted**; scale 1.0 remains default.
No target weights changed and no head was retrained. New target-model variants
require separately identified quality evidence under `BENCHMARK_CONTRACT.md`.

## FFN transfer scheduling

Optional profiling separates CPU gate/up jobs, host down-copy submission and a
GPU-event down dependency span; it changes synchronization and is diagnostic
only. The half-CPU down-row candidate failed the unchanged strict feature/state
gate and is **not promoted**. The separate async activation replay passed strict
parity but had no measured throughput benefit over its same-build baseline.
Its default remains off. See `FFN_SCHEDULING.md` for exact scope and failed logs.

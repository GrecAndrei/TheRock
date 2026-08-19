# qwen38-native

This is the native Qwen3.8-27B/DFlash2 inference engine for gfx1032. It is a
standalone executable built with TheRock's HIP SDK. It does not link to or call
llama.cpp, ggml, or another inference runtime.

The implementation is being brought up in independently testable layers. It
parses GGUF v3 directly, maps real target tensors, and has a native raw-token
target decoder with persistent recurrent state, convolution history, causal
attention cache, MRoPE, final logits, and greedy generation. It does not link
to or call an inference runtime.

```bash
./build.sh
build/qwen38-native device
build/qwen38-native inspect ~/Downloads/Qwen3.8-27B-Q4_K_M.gguf
build/qwen38-native bench-matvec ~/Downloads/Qwen3.8-27B-Q4_K_M.gguf \
    blk.0.ffn_gate.weight 100
build/qwen38-native bench-ffn ~/Downloads/Qwen3.8-27B-Q4_K_M.gguf 0
build/qwen38-native bench-gdn
build/qwen38-native show-token ~/Downloads/Qwen3.8-27B-Q4_K_M.gguf 0
build/qwen38-native target-generate ~/Downloads/Qwen3.8-27B-Q4_K_M.gguf 131072 0 6
```

`target-generate` intentionally accepts raw token IDs while the model executor
is validated. `CONTEXT` is the enabled capacity and the command reports active
cache entries separately. The optional final argument is the resident target
weight cache in GiB (default `6`, `0` disables it). It is not a text/chat CLI
yet, and DFlash2 proposal/acceptance scheduling is not yet connected to this
reference decoder. The default 6 GiB weight mode and a 2.25 GiB device N23KV4
128K cache cannot coexist on an 8 GiB RX 6650 XT; this reference decoder keeps
its growing FP16 cache in host memory for that reason. The N23KV4 device cache
is validated by its own benchmark below, but has not yet been wired into this
raw-token executor.

## N23Q4

`N23Q4` is a gfx1032-specific 5-bit repack of Q4_K. A wave32 produces four
rows, its eight-lane subgroups use RDNA2 `v_dot4_i32_i8`, and scale/bias data is
pre-expanded so the kernel avoids Q4_K's packed-scale decoding. The repack
preserves the original asymmetric four-bit codes; only the derived FP16
scale/bias and Q8 activation introduce a small numerical delta.

Real-weight measurements on the RX 6650 XT show that no single layout wins:

| Tensor | Direct Q4_K | N23Q4 | N23Q4 relative RMSE |
|---|---:|---:|---:|
| `blk.0.attn_gate.weight` (17.7 MB) | 0.062 ms / 286 GB/s | 0.057 ms / 346 GB/s | 0.374% |
| `blk.0.ffn_gate.weight` (50.1 MB) | 0.231 ms / 217 GB/s | 0.302 ms / 184 GB/s | 0.374% |

For the eight-token verifier used by DFlash2, weight reuse and RDNA2 DP4A change
the result: N23Q4 runs the 50.1 MB FFN projection in 0.520 ms instead of 0.992
ms for direct Q4_K (1.91x), and the smaller projection in 0.171 ms instead of
0.226 ms. The batch-8 output delta is about 0.45% relative RMSE.

The model executor will therefore dispatch single-token large FFNs directly
from Q4_K, use N23Q4 for smaller single-token projections, and use N23Q4 for
the eight-token DFlash verification path. Thresholds will be retuned using
whole-layer benchmarks.

## Target execution primitives

The direct native dispatcher now covers every K-quant format present in the
two supplied models: Q4_K, Q5_K, and Q6_K. Each path has a CPU reference check
against the mapped model tensor, including its eight-token form. For example,
the RX 6650 XT measures the target's Q5_K `blk.0.ssm_out.weight` at 0.074 ms
with a maximum error of `2.26e-6`, and its Q6_K
`blk.0.attn_qkv.weight` at 0.176 ms with a maximum error of `5.96e-7`.

`bench-ffn` is the first multi-projection target execution path: it performs
both input projections, SiLU gating, and the down projection for eight real
target states. It validates the final down-projection against the GPU hidden
state. On target layer 0:

| Eight-token FFN route | GPU pass | Proposed tokens/s | Down-projection check |
|---|---:|---:|---:|
| Direct Q4_K + Q4_K + Q6_K | 3.017 ms | 2,652 | `1.21e-6` |
| N23Q4 gate/up + Q6_K down | 2.081 ms | 3,844 | `9.83e-7` |

The latter route includes a GPU-side Q8 activation quantizer, N23Q4 wave32
DP4A gate/up projections, SiLU multiplication, and the native Q6_K down
projection. Its N23Q4 repack happens once when a verifier tensor is admitted
to the GPU weight cache, never once per proposed block.

The 48 recurrent target layers use a fused gated-delta state update. The
`bench-gdn` command drives the exact 48 x 128 x 128 FP16-state topology and
checks it against a CPU recurrence with identical FP16 state rounding. It
measures 0.153 ms per recurrent layer on this device with maximum error
`3.96e-6`.

`target-generate` is the semantic target checkpoint. Each step fetches the
real Q4_K embedding, executes all 64 live layers, updates all 48 gated-delta
states and all 16 causal attention caches, applies final norm plus the Q6_K
head, and returns a greedy next token. The default 6 GiB resident cache holds
the output head, all FFN-down matrices, and as many recurrent QKV matrices as
fit; remaining target matrices stream through HIP. On this machine, a six-step
128K-capacity run has measured roughly `0.99` s for a warm step (about
`1.0 tok/s`). That is an end-to-end target number, not a component benchmark.
It is bounded mainly by streaming the target's remaining weights over PCIe and
does not meet a 10 tok/s target.

For 128K context, the target has only 16 full-attention layers; its other 48
layers use fixed-size recurrent state. The `N23KV4` cache format stores the
full-attention K/V at 4 bits plus per-group scales, leaving substantially more
VRAM for weights than Q8 K/V. The built-in `plan-128k` command measures the
exact model metadata and reports 4.25 GiB for target Q8 KV versus 2.25 GiB for
N23KV4, reclaiming 2.00 GiB for target weights on this card.

The real 131,072-token N23KV4 key-score benchmark reaches 212 GB/s and 2.14
ms per full-attention layer for one query. Its eight-query verifier reads each
cached key once and completes in 6.86 ms/layer, 2.50x faster than launching the
single-query pass eight times. CPU-vs-GPU error is below `5e-7`.

The fused online-softmax and V path never writes the 8 x 24 x 128K score
matrix to VRAM. Its gfx1032 tile shares each K/V load across eight queries.
With 64 context partitions, a fully populated 128K cache takes 25.90 ms per
full-attention layer (about 414 ms across all 16 layers), with CPU-vs-GPU error
near `2e-9`. This is deliberately reported separately from “128K capacity with
a short prompt”: allocated context capacity does not mean 128K cache entries
are read during every short-prompt decode.

The previous llama.cpp integration remains under `../qwen38-dflash` only as a
correctness and throughput baseline while the native path is completed.

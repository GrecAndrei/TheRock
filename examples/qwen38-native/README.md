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
build/qwen38-native find-token ~/Downloads/Qwen3.8-27B-Q4_K_M.gguf Hello
build/qwen38-native target-generate ~/Downloads/Qwen3.8-27B-Q4_K_M.gguf 131072 0 6
build/qwen38-native target-verify-batch ~/Downloads/Qwen3.8-27B-Q4_K_M.gguf 131072 0 3 7
build/qwen38-native dflash-propose ~/Downloads/Qwen3.8-27B-Q4_K_M.gguf \
    ~/Downloads/Qwen3.8-27B-DFlash2-Q4_K_M.gguf 131072 0 7.3
```

`target-generate` intentionally accepts raw token IDs while the model executor
is validated. `CONTEXT` is the enabled capacity and the command reports active
cache entries separately. The optional final argument is the resident target
weight cache in GiB (default `7`, `0` disables it). It is not a text/chat CLI
yet. DFlash2 now has native feature capture, encoder injection, five-layer
non-causal block decoding, selector-lattice greedy proposals, and a native
batch-8 target verifier. `dflash-verify` and `dflash-verify-prompt` report an
accepted prefix rather than inflating the draft rate into generated tok/s; the
continuous accept/rollback scheduler is still being completed. The default 7
GiB weight mode and a 2.25 GiB device N23KV4 128K cache cannot coexist on an
8 GiB RX 6650 XT. The executor therefore keeps compressed N23KV4 history in
GPU-mapped system RAM and only its FP16 hot-tail ring in VRAM.

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

The target verifier stays on the original quantized target weights. N23Q4 is
used only in the DFlash draft path, where target verification makes its small
activation-quantization delta safe: a proposal is never accepted merely
because the draft produced it.

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

The 48 recurrent target layers use a fused gated-delta state update for the
batch verifier. `bench-gdn` drives the 48 x 128 x 128 FP16-state topology; on
this device it measures about 0.034 ms per recurrent layer. Its FP16 state is
an optimization path and remains subject to end-to-end oracle validation.

`target-generate` is the semantic target checkpoint. Each step fetches the
real Q4_K embedding, executes all 64 live layers, updates all 48 gated-delta
states and all 16 causal attention caches, applies final norm plus the Q6_K
head, and returns a greedy next token. The default 7 GiB resident cache holds
the output head, all FFN-down and recurrent QKV matrices, and as many FFN
gate/up matrices as fit; remaining target matrices stream through HIP. On this machine, a six-step
128K-capacity run has measured roughly `0.99` s for a warm step (about
`1.0 tok/s`). That is an end-to-end target number, not a component benchmark.
It is bounded mainly by streaming the target's remaining weights over PCIe and
does not meet a 10 tok/s target.

`target-verify-batch` is an end-to-end, stateful target execution over eight
candidate tokens: recurrent state advances in order, while each target matrix
is transferred once per block. Its FFN keeps gate/up/SiLU intermediates on the
device, and each recurrent layer uploads the complete eight-token delta-state
input once before issuing eight ordered state updates. On the reference card,
stable warm blocks measure about 6.7 candidate tok/s with a 7 GiB target-weight
cache; run-level rates are lower when the streamed-weight path stalls. That is
not a generated-tok/s claim; generation speed additionally depends on DFlash
acceptance and rollback correctness.

For 128K context, the target has only 16 full-attention layers; its other 48
layers use fixed-size recurrent state. The `N23KV4` cache format stores the
full-attention K/V at 4 bits plus per-group scales, leaving substantially more
VRAM for weights than Q8 K/V. At 128K, Q8 KV would occupy 4.25 GiB and N23KV4
history occupies 2.25 GiB, but the latter is mapped from system RAM rather than
competing with the 7 GiB resident weight cache.

The real 131,072-token N23KV4 key-score benchmark reaches about 208 GB/s and
2.18 ms per full-attention layer for one query. Its eight-query verifier reads
each cached key once and completes in about 7.25 ms/layer. CPU-vs-GPU error is
below `5e-7`.

The fused online-softmax and V path never writes the 8 x 24 x 128K score matrix
to VRAM. One workgroup handles one verifier token and KV head, so all warps
share the same causal limit and barrier schedule. At 128K, 64 partitions make
each partition about 2.25 MiB across K/V for one layer. The grid clusters all
eight verifier queries over a partition so their reads can be reused through
the RX 6650 XT's 32 MiB Infinity Cache; that scheduling change reduced the
correct fused kernel from 54.27 to 49.27 ms/layer. CPU-vs-GPU error is near
`2e-9`. The device FP16 ring allocates 16 MiB across all 16 attention layers;
the active 128-token hot window is 8 MiB. This is deliberately reported
separately from “128K capacity with a short prompt”: allocated capacity does
not mean 128K entries are read during every short-prompt decode.

## Chat

Interactive chat using DFlash speculative decoding for 10 tok/s generation:

```bash
build/qwen38-native chat ~/Downloads/Qwen3.8-27B-UD-Q4_K_M.gguf \
    ~/Downloads/Qwen3.8-27B-DFlash2-Q4_K_M.gguf 4096 6.39
```

Arguments: `chat TARGET.gguf DFLASH.gguf [CONTEXT] [GIB]`. The DFlash draft
model proposes 7 tokens per block and the target verifies all 8 in one
pipelined batch. Accepted tokens are emitted immediately; rejected suffixes are
rewound so the attention cache stays clean. The model generates `<think>...</think>`
reasoning before each response (hidden by default; `/think` to toggle).

Commands: `/quit`, `/reset` (clear context), `/think` (show/hide reasoning).

The previous llama.cpp integration remains under `../qwen38-dflash` only as a
correctness and throughput baseline while the native path is completed.

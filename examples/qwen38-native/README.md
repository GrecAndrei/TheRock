# qwen38-native

This is the native Qwen3.8-27B/DFlash2 inference engine for gfx1032. It is a
standalone executable built with TheRock's HIP SDK. It does not link to or call
llama.cpp, ggml, or another inference runtime.

The implementation is being brought up in independently testable layers. It
currently validates the TheRock runtime, parses GGUF v3 directly, and runs two
validated quantized matrix-vector kernels against tensors mapped from the real
16 GB target model.

```bash
./build.sh
build/qwen38-native device
build/qwen38-native inspect ~/Downloads/Qwen3.8-27B-Q4_K_M.gguf
build/qwen38-native bench-q4 ~/Downloads/Qwen3.8-27B-Q4_K_M.gguf \
    blk.0.ffn_gate.weight 100
```

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
matrix to VRAM. With eight queries sharing each K/V read
and 64 context partitions, a fully populated 128K cache takes 25.90 ms per
full-attention layer (about 414 ms across all 16 layers), with CPU-vs-GPU error
near `2e-9`. This is deliberately reported separately from “128K capacity with
a short prompt”: allocated context capacity does not mean 128K cache entries
are read during every short-prompt decode.

The previous llama.cpp integration remains under `../qwen38-dflash` only as a
correctness and throughput baseline while the native path is completed.

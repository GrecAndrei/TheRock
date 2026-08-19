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

The model executor will therefore dispatch smaller projections to N23Q4 and retain
direct Q4_K for large FFN matrices. The thresholds will be retuned using full
layer and speculative batch benchmarks.

For 128K context, the target has only 16 full-attention layers; its other 48
layers use fixed-size recurrent state. The planned `N23KV4` cache stores the
full-attention K/V at 4 bits plus per-group scales, leaving substantially more
VRAM for weights than Q8 K/V.

The previous llama.cpp integration remains under `../qwen38-dflash` only as a
correctness and throughput baseline while the native path is completed.

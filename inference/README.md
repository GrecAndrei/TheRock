# TheRock Inference Engine

Custom RDNA2-native inference runtime for running GGUF models on AMD GPUs without
`HSA_OVERRIDE_GFX_VERSION` or system ROCm dependencies beyond the HIP runtime.

## Performance

**Gemma 4 12B (Q4_0) on RX 6650 XT (gfx1032, 8 GB VRAM):**

| Metric | Value |
|--------|-------|
| Decode throughput | 30.7 tokens/sec |
| Per-token latency | 32.6 ms |
| Effective bandwidth | 200 GB/s (71% of 280 GB/s peak) |
| VRAM usage | 7.14 GB |
| Prefill throughput | ~290 tokens/sec |

Compared to llama.cpp on the same hardware: ~18-22 t/s with `HSA_OVERRIDE_GFX_VERSION` hack.

## Architecture

Single-file standalone runner (`tools/run_gguf.hip`) with all kernels inline.
No external BLAS, no framework dependencies. Loads real GGUF files via mmap.

### Key design decisions

- **Wave32 throughout**: all GEMV kernels use one wavefront per output row
- **Infinity Cache aware**: activation vectors (7.5 KB) stay IC-resident while
  weight matrices stream from VRAM at near-peak bandwidth
- **Fused kernels**: post-norm + residual add + scale in one launch; head-norm + RoPE
  fused; GPU embedding lookup with scale factor baked in
- **Precomputed RoPE tables**: cos/sin tables uploaded at load time, no trig in the hot path
- **GPU-only decode path**: embedding dequant, all layers, lm_head GEMV, and argmax
  all run on device — only a single 4-byte DtoH for the final token ID

### Kernel inventory

| Kernel | Purpose |
|--------|---------|
| `kern_gemv_q4_0` | Q4_0 dequant-dot GEMV (weight layers) |
| `kern_gemv_q6k` | Q6_K dequant-dot GEMV (lm_head / embedding) |
| `kern_rmsnorm` | 256-thread RMSNorm with LDS wave reduction |
| `kern_fused_postnorm_residadd` | Post-norm + residual add + layer scale |
| `kern_headnorm_rope` | Per-head RMSNorm + RoPE rotation (fused) |
| `kern_attn_decode` | Single-query flash-decoding (online softmax) |
| `kern_gelu_mul` | GeLU activation × up projection |
| `kern_embed_lookup` | Q6_K row dequant on GPU with scale factor |
| `kern_argmax_single` | GPU argmax over vocabulary (262144 elements) |

## Build

Requires TheRock's HIP compiler (built from source):

```bash
cd /path/to/TheRock/inference
../build/dist/rocm/bin/hipcc -O3 --offload-arch=gfx1032 \
  -I include tools/run_gguf.hip \
  -L ../build/dist/rocm/lib -lamdhip64 -w \
  -o tools/run_gguf
```

## Run

```bash
LD_LIBRARY_PATH=../build/dist/rocm/lib \
  ./tools/run_gguf /path/to/model.gguf "Your prompt here"
```

No `HSA_OVERRIDE_GFX_VERSION` needed. Runs natively on gfx1032.

## Model support

Currently supports:
- **Gemma 4 12B** (Q4_0 quantization, QAT variant)
- Architecture: 48 layers, hidden=3840, FFN=15360, GQA with sliding window attention
- Two layer types: SWA (head_dim=256, 8 KV heads) and Global (head_dim=512, 1 KV head)

## Roadmap

- [ ] Q4_R2: custom RDNA2-native quantization (separated scales, IC-line aligned, 4.125 bpw)
- [ ] Proper BPE tokenizer (currently byte-level fallback)
- [ ] Sliding window attention masking
- [ ] Multi-token speculative decoding
- [ ] Support for other GGUF model architectures

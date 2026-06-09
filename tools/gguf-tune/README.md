# gguf-tune

Extract GEMM shapes from GGUF models and tune GPU kernels for them using TensileLite.

## What it does

LLM inference is dominated by matrix multiplications between the input token batch and each model weight. The shapes of those GEMMs are fully determined by the model architecture and the batch size — and they repeat identically every forward pass.

`gguf-tune` extracts all unique GEMM shapes from a GGUF model, runs TensileLite's
auto-tuning benchmark to find the optimal kernel parameters for your GPU, and saves
the resulting profile so the best kernels can be pre-selected at runtime.

## Tools

| Script | Purpose |
|--------|---------|
| `extract_shapes.py` | Read GGUF header, emit unique (M,N,K,dtype) shapes as JSON |
| `generate_tensile_config.py` | Convert shape JSON to TensileLite benchmark YAML |
| `baseline_bench.py` | Benchmark default rocBLAS SGEMM per shape (pre-tuning baseline) |
| `tune.py` | Orchestrator: runs the full pipeline end-to-end |

## Quick start

```bash
# Extract shapes only (fast — reads header, not weights)
python3 tools/gguf-tune/extract_shapes.py model.gguf --summary

# Full tuning pipeline (slow — compiles and benchmarks GPU kernels)
python3 tools/gguf-tune/tune.py model.gguf --gfx gfx1032 --output-dir /tmp/tune_out

# Dry run: stop before the Tensile benchmark step
python3 tools/gguf-tune/tune.py model.gguf --gfx gfx1032 --skip-tensile

# Baseline benchmark (requires built rocBLAS in ROCM_PATH)
ROCM_PATH=/path/to/rocm python3 tools/gguf-tune/baseline_bench.py shapes.json
```

## Requirements

- Python 3.10+
- Built TheRock with `hipBLASLt` enabled for your GPU target
  (`-DTHEROCK_AMDGPU_FAMILIES=gfx1032` or similar)
- PyYAML (`pip install pyyaml`) for profile parsing
- `ROCM_PATH` environment variable pointing to your ROCm installation
  (defaults to `TheRock/build/dist/rocm`)

## How GEMM shapes are determined

For a weight matrix stored as `[N, K]` (GGUF row-major), inference computes:

```
output[M, N] = input[M, K] @ weight.T[K, N]
```

Where `M` is the token batch size (1 for single-token autoregression, larger for
prefill). `gguf-tune` generates shapes for configurable batch sizes (default: 1, 4, 8,
16, 32) to cover both decode and prefill regimes.

Quantized weights (Q4_0, Q8_0, etc.) are dequantized before the GEMM — TensileLite
sees fp16 inputs with mixed-precision accumulation.

## Output profile

After a tuning run, `profile.json` contains:

```json
{
  "gfx": "gfx1032",
  "model": "gemma-4-12b-it",
  "architecture": "gemma4",
  "unique_shapes": 30,
  "kernels": [
    {
      "problem": [1, 3840, 1, 15360],
      "solution_name": "Cijk_Ailk_Bljk_HHS_...",
      "source_file": "gfx1032_inference_HH.yaml"
    },
    ...
  ]
}
```

This profile can be committed to a community database indexed by `(model_name, gfx)`
so other users with the same GPU can skip the tuning step.

## Long-term vision

- Community profile database: tuning runs from many GPUs → shared kernel selection
- TheRock-native inference engine that loads profiles at startup
- Per-model, per-GPU kernel selection without requiring a full Tensile install

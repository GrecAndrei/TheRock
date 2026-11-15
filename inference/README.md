# rxinfer — Pure-HIP LLM Inference Engine (`gfx1201`)

`rxinfer` is a modular, config-driven pure-HIP inference engine targeting AMD RDNA4 (`gfx1201`, Radeon RX 9070 XT) with zero external inference-framework dependencies. It supports standard dense transformers (Qwen3, Llama, Mistral) and hybrid linear-attention + full-attention architectures (Qwen3.5 / Qwen3.8 / Qwen3-Next with Gated DeltaNet + gated D=256 attention).

## Build

Configure and build as a standalone CMake + HIP project against a ROCm installation:

```bash
cmake -S inference -B inference/build -G Ninja \
  -DCMAKE_PREFIX_PATH=/path/to/dist/rocm \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201
cmake --build inference/build -j
```

## Verification & Benchmarks

All suites run directly on `gfx1201`:

```bash
# Tokenizer, sampler, ChatML/Llama3 multi-turn formatting, safetensors/GGUF/RXR4 I/O
./inference/build/rxinfer_test_io

# GPU kernel unit tests (RMSNorm+1, SwiGLU, R4 embedding, 248k argmax, F16/FP8/FP4 attn, GDN)
./inference/build/rxinfer_test_kernels

# Head-to-head R4_GFX12_G64 vs GGUF Q4_0 GEMV/GEMM microbenchmarks
./inference/build/rxinfer_bench_quant

# End-to-end prefill/decode/verify/multi-chunk/multi-turn parity + 27B benchmark
./inference/build/rxinfer_bench_e2e
```

Verified RX 9070 XT (`gfx1201`) Qwen3.8-27B synthetic benchmark (`64` layers: `48` GDN + `16` FullAttn `D=256`, `R4_GFX12_G64`, `FP4` KV):

- **VRAM**: `13.76 GiB` total (`13.34 GiB` weights, `365.6 MB` state, `65.0 MB` scratch)
- **Prefill (512 tok)**: `373.7 ms` (`1370.2 tok/s`)
- **Decode (`hipGraph`, `M=1`)**: `24.32 ms/tok` (`41.12 tok/s`)
- **Verify (`hipGraph`, `M=4`)**: `25.99 ms/step` (`153.90 tok/s` @ 100% accept)
- **Verify (`hipGraph`, `M=8`)**: `27.15 ms/step` (`294.61 tok/s` @ 100% accept)
- **Cold-L3 M=1 `N=17408, K=5120`**: `78.1 us` (`606.7 GB/s`) vs `GGUF Q4_0` `109.2 us` (`459.6 GB/s`, `1.40x` faster)

## CLI & Interactive Chat Mode

`rxinfer` keeps the model weights, KV cache, and GDN recurrent states resident in VRAM across turns:

```bash
# Interactive multi-turn chat from a HuggingFace safetensors directory, .rxr4, or .gguf
./inference/build/rxinfer --model /path/to/model_dir -i

# Interactive chat with custom sampling and KV cache quantization
./inference/build/rxinfer --model /path/to/model.rxr4 -i \
  --kv-dtype fp4 --temp 0.7 --top-k 40 --top-p 0.95 --repeat-penalty 1.05

# Convert a HuggingFace safetensors directory to a fast-loading .rxr4 checkpoint
./inference/build/rxinfer --model /path/to/model_dir --save-rxr4 /path/to/model.rxr4

# Single-shot prompt
./inference/build/rxinfer --model /path/to/model_dir --chat --prompt "Explain Wave32 WMMA."

# Synthetic 27B interactive dry-run (using an external tokenizer.json)
./inference/build/rxinfer --model synthetic --tokenizer /path/to/tokenizer.json --kv-dtype fp4 -i
```

### Interactive Slash Commands

- `/help`, `/?` — show interactive commands
- `/reset`, `/clear` — zero KV cache & GDN recurrent state and start a new conversation
- `/stats` — show context usage, VRAM breakdown, sampler path, and session/last-turn TPS
- `/think [on|off]` — toggle `<think>` mode in ChatML turns
- `/system <text>` — update system prompt and reset conversation state
- `/temp`, `/top-k`, `/top-p`, `/min-p`, `/repeat-penalty`, `/max-tokens`, `/seed` — inspect or adjust sampling live
- `/bench` — run prefill/decode/verify microbenchmarks and reset state
- `/quit`, `/exit`, `/q` — exit (`Ctrl-C` interrupts generation; trailing `\` continues a multi-line message)

## Directory Layout

- `include/rxinfer/` — public headers (`arena`, `attention`, `common`, `config`, `engine`, `gdn`, `loader`, `model`, `ops`, `quant`, `sampler`, `tensor`, `tokenizer`)
- `src/` — engine, loaders (Safetensors / GGUF v2-v3 / `.rxr4` / synthetic), tokenizer, sampler, and CLI
- `src/kernels/` — `gfx1201` HIP kernels (`quant_r4_gfx12`, `quant_q4_gguf`, `norm_rope_ops`, `attention_gfx12`, `gdn_gfx12`)
- `bench/` — unit tests (`test_io.cpp`, `test_kernels.hip`), probes (`wmma_probe.hip`), and benchmarks (`bench_quant.hip`, `bench_e2e.hip`)

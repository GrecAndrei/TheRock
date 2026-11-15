# rxinfer — Pure-HIP LLM Inference Engine (`gfx1201`)

`rxinfer` is a modular, config-driven pure-HIP inference engine targeting AMD RDNA4 (`gfx1201`, Radeon RX 9070 XT) with zero external inference-framework dependencies. It supports standard dense transformers (Qwen3, Llama, Mistral) and hybrid linear-attention + full-attention architectures (Qwen3.5 / Qwen3.8 / Qwen3-Next with Gated DeltaNet + gated `D=256` attention) up to **200,000+ token contexts** via tiered VRAM + mapped GTT paged KV caching.

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
# Tokenizer, sampler, MarkdownStreamFormatter, ChatML/Llama3, safetensors/GGUF/RXR4 I/O
./inference/build/rxinfer_test_io

# GPU kernel unit tests (RMSNorm+1, SwiGLU, R4 embedding, 248k argmax, F16/FP8/FP4 attn, Tiered VRAM+GTT, 200k RoPE, GDN)
./inference/build/rxinfer_test_kernels

# Head-to-head R4_GFX12_G64 vs GGUF Q4_0 GEMV/GEMM microbenchmarks
./inference/build/rxinfer_bench_quant

# End-to-end prefill/decode/verify/multi-chunk/LCP/GTT/checkpoint parity + 27B 200k benchmark
./inference/build/rxinfer_bench_e2e
```

Verified RX 9070 XT (`gfx1201`) Qwen3.8-27B synthetic benchmark (`64` layers: `48` GDN + `16` FullAttn `D=256`, `R4_GFX12_G64`, `FP4` KV):

- **VRAM (200k context)**: `14.12 GiB` total (`13.34 GiB` weights, `0.36 GiB` GDN state, `0.42 GiB` VRAM KV + `3.01 GiB` mapped GTT KV)
- **Prefill (512 tok)**: `~1370 tok/s` (WMMA prefill)
- **Decode (`hipGraph`, `M=1`)**: `~40 tok/s` (`~25 ms/tok`)
- **Verify (`hipGraph`, `M=4`)**: `~154 tok/s` @ 100% accept
- **Cold-L3 M=1 `N=17408, K=5120`**: `78.1 us` (`606.7 GB/s`) vs `GGUF Q4_0` `109.2 us` (`459.6 GB/s`, `1.40x` faster)

## CLI & Interactive Chat Mode

In interactive mode (`-i`), `rxinfer` automatically enables:

1. **Large Context (`--max-seq-len auto` up to `200,000` tokens)** and **automatic KV dtype selection** (`f16` -> `fp8` -> `fp4`) with seamless VRAM + `hipHostMallocMapped` GTT KV spillover.
2. **Automatic Prompt Caching**:
   - In-memory Longest Common Prefix (LCP) KV reuse + pinned system-prompt GDN checkpoints across `/reset`, `/undo`, `/retry`, and periodic `4096`-token grid GDN snapshots for fast long-context branching.
   - Optional bounded LRU persistent `.rxprompt` disk cache via `--auto-cache` (or `--prompt-cache auto`, capped at `6 GiB` and preserving `>= 8 GiB` free disk), or explicit `--prompt-cache <path>`.
3. **Streaming Terminal Markdown Formatting** (`--format` auto-enabled on TTY, override with `--no-format` / `--raw`, or `/format [on|off]`): renders `#`–`######` headings, `**bold**`, `*italic*`, `***bold-italic***`, `~~strike~~`, bullets/numbered lists, `> blockquotes`, horizontal rules, `` `inline code` ``, ```` ```fences``` ````, and dimmed `<think>...</think>` blocks with UTF-8 boundary buffering and clean ANSI SGR state handling.

```bash
# Interactive multi-turn chat (auto 200k FP4 context on 27B, auto TTY Markdown formatting, in-memory LCP + checkpoints)
./inference/build/rxinfer --model /path/to/model.gguf -i

# Interactive chat with persistent LRU disk prompt caching + system prompt
./inference/build/rxinfer --model /path/to/model.gguf -i --auto-cache --system "You are a helpful coding assistant."

# Reuse shared document prefix across separate CLI runs automatically
./inference/build/rxinfer --model /path/to/model.gguf --auto-cache --chat \
  --prompt-file large_codebase.txt --prompt "Summarize the architecture."

# Single-shot prompt with custom sampling and KV cache quantization
./inference/build/rxinfer --model /path/to/model.rxr4 --chat --prompt "Explain Wave32 WMMA." \
  --kv-dtype fp8 --temp 0.7 --top-k 40 --top-p 0.95 --repeat-penalty 1.05

# Convert a HuggingFace safetensors directory to a fast-loading .rxr4 checkpoint
./inference/build/rxinfer --model /path/to/model_dir --save-rxr4 /path/to/model.rxr4
```

### Interactive Slash Commands & Multi-line Input

- `/help`, `/?` — show interactive commands
- `/file <path> [prompt]` — attach a file/codebase into context (send immediately if `[prompt]` is provided)
- `/paste` or `"""` — start multi-line block input (end with `/end` or `"""`; TTY bracketed paste and trailing `\` continuation are also supported)
- `/undo [N]` — rewind conversation by `N` turns using cached snapshots
- `/retry` — rewind the last turn and regenerate the response
- `/reset`, `/clear` — reset active position while preserving in-memory LCP & pinned system-prompt checkpoints
- `/save-cache [path]`, `/load-cache <path>`, `/cache`, `/clear-cache` — manage `.rxprompt` prompt caches and in-memory GDN snapshots
- `/format [on|off]` — toggle or set streaming ANSI Markdown formatting
- `/think [on|off]` — toggle `<think>` mode in ChatML turns
- `/system <text>` — update system prompt and reset conversation state
- `/stats` — show context usage, VRAM/GTT KV pool breakdown, snapshots, sampler path, and session/last-turn TPS
- `/temp`, `/top-k`, `/top-p`, `/min-p`, `/repeat-penalty`, `/max-tokens`, `/seed` — inspect or adjust sampling live
- `/bench` — run prefill/decode/verify microbenchmarks and reset state
- `/quit`, `/exit`, `/q` — exit (`Ctrl-C` interrupts generation)

## Directory Layout

- `include/rxinfer/` — public headers (`arena`, `attention`, `common`, `config`, `engine`, `gdn`, `loader`, `markdown`, `model`, `ops`, `quant`, `sampler`, `tensor`, `tokenizer`)
- `src/` — engine, loaders (Safetensors / GGUF v2-v3 / `.rxr4` / synthetic), tokenizer, sampler, Markdown stream formatter, and CLI
- `src/kernels/` — `gfx1201` HIP kernels (`quant_r4_gfx12`, `quant_q4_gguf`, `norm_rope_ops`, `attention_gfx12`, `gdn_gfx12`)
- `bench/` — unit tests (`test_io.cpp`, `test_kernels.hip`), probes (`wmma_probe.hip`), and benchmarks (`bench_quant.hip`, `bench_e2e.hip`)

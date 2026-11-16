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

By default, generation length is **unlimited** (`--max-tokens 0`, bounded only by EOS/stop tokens or context size; pass `--max-tokens N` or `/max-tokens N` to cap).

In interactive mode (`-i`), `rxinfer` automatically enables:

1. **Large Context (`--max-seq-len auto` up to `200,000` tokens)** and **automatic KV dtype selection** (`f16` -> `fp8` -> `fp4`) with seamless VRAM + `hipHostMallocMapped` GTT KV spillover.
2. **Automatic Prompt Caching**:
   - In-memory Longest Common Prefix (LCP) KV reuse + pinned system-prompt GDN checkpoints across `/reset`, `/undo`, `/retry`, and periodic `4096`-token grid GDN snapshots for fast long-context branching.
   - Optional bounded LRU persistent `.rxprompt` disk cache via `--auto-cache` (or `--prompt-cache auto`, capped at `6 GiB` and preserving `>= 8 GiB` free disk), or explicit `--prompt-cache <path>`.
3. **Interactive Terminal UI & Streaming Markdown** (`--format` auto-enabled on TTY, override with `--no-format` / `--raw`, or `/format [on|off]`):
   - Zero-dependency POSIX `termios` line editor with `/command` and file-path `Tab` completion, inline ghost suggestions, persistent command history, bracketed paste, and multi-line input (`Alt+Enter`, trailing `\`, or `/paste` / `"""`).
   - Progress bar during `.rxr4`/GGUF/safetensors weight loading, prefill spinner, boxed session/stats cards, and 256-color streaming Markdown (`#`–`######` headings, `**bold**`, `*italic*`, `~~strike~~`, bullets/numbered lists, `> blockquotes`, `` `inline code` ``, ```` ```fences``` ````, and `╭─ thinking` / `│ ` / `╰───` rails for `<think>...</think>` blocks).
4. **Tool Calling & Reasoning Preservation**:
   - Supports OpenAI-format tool definitions (`--tools <file.json>`), built-in workspace tools (`--builtin-tools`: `read_file`, `list_dir`, `search_text`, `write_file`, `run_command`), interactive/CLI confirmation for mutating tools (`--auto-approve` or `/approve [on|off]`), and custom JSON tool execution (`--tool-exec <cmd>`).
   - Follows Qwen3.5 / ChatML XML `<tools>` / `<tool_call>` / `<tool_response>` and Llama3 `<|python_tag|>` formats, and **serves prior `<think>` reasoning back** across both multi-round tool calls and multi-turn chats (`--preserve-thinking` default on; `--no-preserve-thinking` strips prior-turn `<think>` on new user queries while keeping reasoning across tool calls).

```bash
# Interactive multi-turn chat (auto 200k FP4 context on 27B, auto TTY Markdown formatting, in-memory LCP + checkpoints)
./inference/build/rxinfer --model /path/to/model.gguf -i

# Interactive coding agent with built-in workspace tools + reasoning enabled
./inference/build/rxinfer --model /path/to/model.rxr4 -i --builtin-tools --think

# Single-shot tool-calling run (non-interactive)
./inference/build/rxinfer --model /path/to/model.rxr4 --builtin-tools --auto-approve \
  --prompt "Inspect README.md and summarize the build commands."

# Interactive chat with persistent LRU disk prompt caching + system prompt
./inference/build/rxinfer --model /path/to/model.gguf -i --auto-cache --system "You are a helpful coding assistant."

# Reuse shared document prefix across separate CLI runs automatically
./inference/build/rxinfer --model /path/to/model.gguf --auto-cache --chat \
  --prompt-file large_codebase.txt --prompt "Summarize the architecture."

# Single-shot prompt with custom sampling and KV cache quantization
./inference/build/rxinfer --model /path/to/model.rxr4 --chat --prompt "Explain Wave32 WMMA." \
  --kv-dtype fp8 --temp 0.7 --top-k 40 --top-p 0.95 --repeat-penalty 1.05

# Convert a HuggingFace safetensors directory or GGUF to a fast-loading .rxr4 checkpoint
# (Loading foo.rxr4 auto-discovers foo.tokenizer.json, foo.gguf, or tokenizer.json in the same folder)
./inference/build/rxinfer --model /path/to/model_dir_or_file.gguf --save-rxr4 /path/to/model.rxr4
```

## OpenAI-Compatible HTTP Server (`--serve`)

`rxinfer` includes a zero-dependency POSIX HTTP/1.1 + SSE server compatible with OpenAI chat, streaming, and function/tool-calling clients:

```bash
./inference/build/rxinfer --model /path/to/model.rxr4 --serve --host 127.0.0.1 --port 8080
```

- Endpoints: `GET /health` (`/v1/health`), `GET /v1/models`, `POST /v1/chat/completions` (blocking JSON + `stream: true` SSE with `stream_options.include_usage`).
- Supports `tools` / `functions`, `tool_choice` (`auto`, `none`, `required`, or `{"type":"function","function":{"name":...}}`), `reasoning_content` (both blocking `message.reasoning_content` and streaming `delta.reasoning_content`), `delta.tool_calls`, `max_completion_tokens` / `max_tokens`, and `chat_template_kwargs` (`enable_thinking`, `preserve_thinking`, `reasoning_effort`).
- Automatically preserves live KV + GDN prefix checkpoints across turns and recovers prior reasoning + exact assistant turn tokens for multi-step OpenAI tool loops (even if a client omits `reasoning_content` on follow-up `tool` messages).

### Interactive Slash Commands & Multi-line Input

- `/help`, `/?` — show interactive commands
- `/file <path> [prompt]` — attach a file/codebase into context (send immediately if `[prompt]` is provided)
- `/paste` or `"""` — start multi-line block input (end with `/end` or `"""`; TTY bracketed paste, `Alt+Enter`, and trailing `\` continuation are also supported)
- `/undo [N]` — rewind conversation by `N` turns (and structured tool history) using cached snapshots
- `/retry` — rewind the last turn and regenerate the response
- `/reset`, `/clear` — reset active position while preserving in-memory LCP & pinned system-prompt checkpoints
- `/tools [builtin|load <path.json>|clear]` — inspect active tools or enable/load/clear tools mid-session
- `/approve [on|off]` — toggle auto-approval for mutating built-in tools (`write_file`, `run_command`)
- `/history` — inspect structured message, reasoning, and tool-call history
- `/save-cache [path]`, `/load-cache <path>`, `/cache`, `/clear-cache` — manage `.rxprompt` prompt caches and in-memory GDN snapshots
- `/format [on|off]` — toggle or set streaming ANSI Markdown formatting
- `/think [on|off]` — toggle `<think>` mode in ChatML turns
- `/system <text>` — update system prompt and reset conversation state
- `/stats` — show context usage, VRAM/GTT KV pool breakdown, snapshots, sampler path, and session/last-turn TPS
- `/temp`, `/top-k`, `/top-p`, `/min-p`, `/repeat-penalty`, `/max-tokens`, `/seed` — inspect or adjust sampling live (`0` = unlimited tokens)
- `/bench` — run prefill/decode/verify microbenchmarks and reset state
- `/quit`, `/exit`, `/q` — exit (`Ctrl-C` interrupts generation)

## Directory Layout

- `include/rxinfer/` — public headers (`arena`, `attention`, `common`, `config`, `engine`, `gdn`, `loader`, `markdown`, `model`, `ops`, `quant`, `sampler`, `server`, `tensor`, `tokenizer`, `tools`, `tui`)
- `src/` — engine, loaders (Safetensors / GGUF v2-v3 / `.rxr4` / synthetic), tokenizer, sampler, Markdown stream formatter, TUI, tool parser/executor, OpenAI HTTP server, and CLI
- `src/kernels/` — `gfx1201` HIP kernels (`quant_r4_gfx12`, `quant_q4_gguf`, `norm_rope_ops`, `attention_gfx12`, `gdn_gfx12`)
- `bench/` — unit tests (`test_io.cpp`, `test_kernels.hip`), probes (`wmma_probe.hip`), and benchmarks (`bench_quant.hip`, `bench_e2e.hip`)

# Qwen3.8-27B + DFlash2 on gfx1032 — legacy external baseline

> **Scope notice:** this directory records the earlier llama.cpp-based baseline.
> It is retained for historical comparison only and is not built, linked, or
> invoked by the pure-TheRock engine. Current native implementation status,
> commands, correctness evidence, and limitations are documented in
> [`../qwen38-native/README.md`](../qwen38-native/README.md) and
> [`../qwen38-native/PERFORMANCE_PLAN.md`](../qwen38-native/PERFORMANCE_PLAN.md).
> The external path does not satisfy the native project's dependency boundary.

This directory captures the fastest historical external-runtime configuration
measured on the local RX 6650 XT 8 GB and i5-14600K system. Its figures must not
be combined with or reported as native-engine results.

## Historical result

The best deterministic 132-token run produced **7.93 real generated tokens/s** with 104 of 192 draft tokens accepted (54.17%). The command uses the official Q4_K_M target and official DFlash2 Q4_K_M head.

```bash
./examples/qwen38-dflash/run-qwen38-dflash.sh "Your prompt"
```

The launcher defaults match the winning run: 20 target GPU layers, the complete draft model on GPU, 16 CPU threads, flash attention, seven draft tokens, and greedy sampling. Paths can be changed with `LLAMA_CPP_DIR`, `THEROCK_DIR`, `DOWNLOADS_DIR`, `TARGET_MODEL`, and `DRAFT_MODEL`.

### Friendly 128K test CLI

`qwen128k` enables a 131,072-token context, Q8 KV caches, automatic VRAM fitting, colored output, and a compact statistics card after every run:

```bash
./examples/qwen38-dflash/qwen128k --tokens 128 "Explain how a B-tree works"
```

Run it without a prompt for an interactive `Prompt>` field. Use `--help` for options. The 128K allocation leaves fewer GPU layers available, so short-prompt throughput is lower than the 1K-context speed configuration; the validated 64-token test reached 3.25 generated t/s.

Required local assets:

- `/home/alex/Downloads/Qwen3.8-27B-Q4_K_M.gguf`
- `/home/alex/Downloads/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`
- `/home/alex/llama.cpp` at DFlash2 commit `5ecbe1ac17ec0484c5b44af0bd580cdc9c428ed4`
- TheRock ROCm libraries under `/home/alex/TheRock/build`

## Why this historical path is below 20 t/s

Correct speculative decoding still evaluates every proposed token with the 27B target. In the best run, target evaluation consumed 15.51 seconds for 243 evaluated tokens, including prompt processing. Removing all DFlash overhead would cap this workload at roughly 9.5 generated tokens/s. The trained DFlash2 block contains only seven proposals, so increasing verification width is not valid; forcing a 15-token draft reduced acceptance to 33.53% and throughput to 5.32 t/s.

The tested lower-bit targets did not help on RDNA2. IQ2_XXS reached 5.78 t/s, Q3_K_S reached 6.29 t/s, and Q2_K_XL had lower verifier throughput than Q4. Forced hipBLAS, forced MMQ, CPU affinity, alternate tensor placement, compact batch buffers, and a smaller requantized draft head also lost to the default Q4 configuration. Full measurements are in `benchmarks.csv`.

Twenty correct single-stream tokens/s therefore requires materially faster target hardware, a smaller target model, or a future target kernel improvement. It cannot be reached by changing current llama.cpp flags on this 8 GB card.

## Historical benchmark hygiene

Pause large-file indexing and other heavy background I/O before measuring. KDE Baloo began indexing the downloaded GGUF files and temporarily reduced the same run to 4.24 t/s. The launcher warns when Baloo is active but does not alter desktop services.

The unused `blk.64.*` warnings are expected: block 64 is the target model's separate MTP layer and is not executed by the base Qwen3.8 model or this DFlash2 path.

# Timeout and physical prefill repair

This branch preserves the official model bytes, quantization, Q8 KV, full target
history, DFlash window, greedy sampling and xhigh chat system instruction.
The original source and executable remain in `sources/native-baseline` and
`build/native-hip` at the enclosing workspace level.

## Changes

- Model mappings remain ordinary reclaimable file mappings by default.
  Previously both complete GGUF mappings were unconditionally registered with
  HIP: about 17 GiB before roughly 6.3 GiB of CPU packed-weight copies. The
  working set competed with the rest of this 32 GiB machine. Whole-model
  registration is now an explicit `QWEN_NATIVE_PIN_MODEL=1` diagnostic option;
  the ordinary CPU source bytes and model computation are unchanged.
- Chat and the final 128K benchmark share one physical-prefill executor.
  Default chunks are 512 positions; `QWEN_NATIVE_PREFILL_CHUNK=8` selects the
  microbatch reference schedule. Each layer's weights are staged once per
  chunk and reused across its ordered eight-position work units.
- Layer-major prefill captures all five DFlash feature taps when required.
  Only the final 2048 prompt positions need drafter injection; every target
  prompt position is still processed and retained. Chunk/tail alignment does
  not truncate target history or inject rejected/synthetic positions.
- Teacher-forced prompt positions omit unused vocabulary predictions. The
  final 1–8 positions produce the one initial generation prediction.
- `prefill-corpus` reads actual supplied frozen token IDs, reports physical
  prompt counts/checksum, initialization, prefill, continuation IDs and cache
  occupancy. It never pads a short file, calls capacity populated history, or
  certifies the final TPS goal. Its short continuation may stop at EOS.
- Benchmark metadata and the journal include the model registration control.

There is no claim that removing registration alone explains every observed
stall. The first cold diagnostic and subsequent warm runs had different cache
and host conditions. The original injection attribution also includes waits,
copies and allocation work; it is not evidence of 75 seconds of encoder math.

## Verification

Four CPU CTests and eight Python lifecycle tests pass. On the RX 6650 XT, the
1-, 17- and 513-position layer-major comparison cases match reference predicted
IDs, logits, subsequent state and **every captured prompt feature tap bit for
bit**. These are all-GPU schedule parity gates, not an external full-model
oracle or generation benchmarks. The 513-position measured phase took 30.23
seconds versus 84.12 seconds on its reference schedule.

The enclosing workspace retains real-request and frozen-corpus measurements.
The first fixed chat produced a full answer, but editing its shell runner while
it was executing broke the runner's final status record. That run is retained
as such; it is not given a fabricated exit code. A stable-runner replay checks
completion independently. The xhigh reasoning text remains part of committed
generation and is displayed in diagnostic input.

## Commands from the enclosing workspace

```sh
cmake -S sources/native-engine -B build/native-engine -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/home/alex/TheRock/build/dist/rocm \
  -DCMAKE_CXX_COMPILER=/home/alex/TheRock/build/dist/rocm/bin/hipcc
cmake --build build/native-engine --parallel 2
bash scripts/run-chat-diagnostic.sh 131072 1.0 360
```

The frozen prompt file is already bound to canonical tokenizer evidence. A
real-prefix diagnostic, separate from the final three-run certification, is:

```sh
QWEN_NATIVE_HYBRID=1 QWEN_NATIVE_DEVICE_CHECKPOINT=1 \
QWEN_NATIVE_CPU_THREADS=20 QWEN_NATIVE_CPU_Q6_PREDECODE=1 \
QWEN_NATIVE_KV=q8 QWEN_NATIVE_Q8_TILED=warp16 \
  build/native-engine/qwen38-native prefill-corpus \
  /home/alex/Downloads/Qwen3.8-27B-UD-Q4_K_M.gguf \
  /home/alex/Downloads/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  fixtures/canonical-128k.tokens 3073 131072 512 1.0
```

Source fixtures and these diagnostics do not establish 10 generated tokens/s
at physical 128K. The existing three-prefill, warmup and committed-output
evidence contract remains required for that claim.

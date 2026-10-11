# Measured native engine execution redesign

This is a development design/evidence ledger, **not a completed 128K throughput
claim**. The ≥10 tok/s goal still requires three independent physical prefills,
correctness gates and committed-output timing on the frozen corpus.

## Evidence that can currently guide architecture

The frozen baseline has reported run-1 physical prefill **49152/131072 at
3440.2 s**. No generation run or arithmetic mean has completed in the evidence
observed so far. Its original `QWEN_NATIVE_PREFILL_PROFILE=0` is a bug: the old
executable tests environment presence, not value, and therefore synchronizes
prefill phases. Its prefill/TTFT must be labelled diagnostic, not clean production
prefill timing. Decode phase profiling is off.

Five consecutive 512-token chunk profiles inspected during partial prefill were:

| Chunk | Recurrent seconds | Attention seconds | FFN seconds |
| --- | ---: | ---: | ---: |
| 1 | 6.4 | 9.6 | 21.4 |
| 2 | 6.5 | 8.9 | 20.8 |
| 3 | 7.2 | 9.8 | 21.1 |
| 4 | 6.4 | 8.8 | 20.4 |
| 5 | 7.1 | 9.6 | 22.4 |

These rounded, synchronized wall intervals are partial-prefill diagnostics, not
full-context generation measurements. They support work on FFN batching and
attention, but do **not** establish decode routing/split choices: layer-major
prefill forces GPU math; hybrid decode has a different critical path.

Static capacity accounting at 131336 slots is **4572594176 bytes (~4.26 GiB)**
for the target's Q8 K+V arrays alone. Full physical history remains intact. The
DFlash feature window of 2048 is not a target-context truncation. Current
attention allocation supports at most **32 partitions**, not the previously
documented 64.

## Implemented prototype: activation handoff scheduling

Opt-in `QWEN_NATIVE_ASYNC_ACTIVATIONS=1`, default **off**. No model coefficient,
CPU/GPU row split, FMA order, proposal policy or accepted-token accounting changes.

Previously, hybrid projection/grouped QKV/recurrent gate/FFN paths synchronized
the compute stream and performed a blocking activation D2H copy before the host
could submit CPU work and queue GPU shards. The prototype instead uses a separate
activation-copy stream, avoiding the weight-H2D queue as an activation-delivery
queue, with producer/copy events:

```text
compute:    normalize -> input-ready -> GPU shards -> GPU-complete
activation:              wait-ready -> D2H -> copy-complete
CPU worker:                                  wait-copy -> CPU rows -> job-complete
host:       queue transfer + job + GPU work, then await GPU and CPU completion
combine:    upload completed CPU rows -> existing scatter/FFN/residual work
```

The source activation is read-only through each paired job. A generic projection
aliasing its output slot additionally waits for the D2H event before GPU writes.
The CPU worker explicitly selects the pipeline's device before waiting for its
copy. Existing pinned input/output buffers are retained: additional resources
are one stream and two non-timing events, not additional tensor buffers. Actual
DMA overlap, CPU-worker HIP overhead and end-to-end benefit remain unmeasured.

`SingleJobWorker` rejects overlapping/empty submissions, preserves unobserved
exceptions and propagates them through `wait()`. Completion releases captures
before returning buffer ownership. Device-pipeline destruction drains accepted
CPU work and pending streams before freeing buffers/events. Factory ownership is
RAII so partially initialized pipelines are cleaned up after allocation errors.
These are lifetime/error-path repairs, not measured speedup claims.

## Implemented prototype: Q8 attention temporary representation

Opt-in `QWEN_NATIVE_Q8_TILED=warp16packed`, not selected in the baseline.
It retains the existing query/GQA mapping, causal limits, global Q8 codes/scales,
softmax/reduction order and full history. Sixteen query waves cooperatively stage
16 K/V tokens as exact signed codes and FP32 scales rather than expanded FP32
coefficients. Temporary LDS decreases **32768 -> 9216 bytes** per block. Reading a
coefficient repeats its code-times-scale multiplication; lower LDS pressure may
or may not outweigh that work. No precision reduction is introduced by staging.

CPU tests cover coefficient/FMA bits (all int8 values and representative finite
scales), unique cooperative scale ownership, 1/7/15/16-token tile tails, all 192
GQA query mappings and causal partition coverage including 131072 history slots.
This is structural/arithmetic evidence, **not device numerical parity**.

## Verification/commit measurement boundaries

New source records `checkpoint_host_seconds`, `verification_wall_seconds` and
`commit_wall_seconds` in addition to historical `target_seconds`. The historical
field retains verification-plus-commit semantics, excluding checkpoint setup.
No extra GPU synchronization is inserted for these host-clock subdivisions.
An asynchronous checkpoint's device-copy cost can land in verification wall
time; checkpoint host time is not claimed to be GPU checkpoint latency. All
clock/commit overhead remains inside committed-generation wall time.

New metadata records effective prefill profiling and the activation option.
The human TTFT line now includes initialization, matching the existing JSONL
initialization-inclusive TTFT field. This does not rewrite old records.

## Gates and selection policy

`rebuild-prototypes-v1-executable` and its complete source archive are immutable
snapshots under `benchmark-results/goal-10tps/`. TheRock HIP compilation succeeded;
**4/4 CPU CTests and 8/8 Python lifecycle tests** passed. The native build used
at most two compiler jobs during baseline prefill (2026-10-11 00:03:18–24 +03:00).
That host activity is disclosed; it was not a GPU test or CPU performance test,
and it did not replace/modify the frozen benchmark executable.

A new `target-engine-variant-test` prepares sequential-session comparisons of
reference `warp16`/synchronous activation delivery against the requested async
and/or LDS candidate, using the same CPU/GPU routing. It requires exact predicted
IDs, top logits and five feature arrays. It forces Q8, requires long capacity,
restores child-process settings after session teardown, and rejects a no-op or
unsupported variant. Its repeated-text physical fixture is deliberately labelled
**not generation throughput**. GPU execution is deferred until the baseline
worker has definitively exited; compilation/CPU guards alone do not pass it.

Still required before selecting either prototype:

1. Device parity on causal tails, multiple partitions and genuinely long history.
2. Real-prefill boundary equivalence; rollback and every accepted prefix 1–8.
3. Session lifetime and native target-only/speculative greedy equivalence.
4. Isolated resource/latency/end-to-end measurements; reject slower paths.
5. Canonical tokenizer/frozen-prompt identity and three final independent physical
   prefills with eight excluded warmups and 256 committed outputs each.

## Remaining architecture decisions

- **Streaming/residency:** instrument the actual decode critical path. Full inline
  down-weight prefetch can make routing choose GPU-only down projection; CPU-down
  microbenchmarks alone cannot justify changing that policy. Keep source weights
  and packed cache ownership stable across transfers.
- **CPU/GPU FFN:** expanded IQ4 is currently slower and memory-heavy; IQ4/K tiled
  experiments are off after a slower combined native fixture. Q6 is rare in base
  FFNs. Consider wider GPU FFN work and coefficient reuse for real prefill after
  numerical gates; teacher-forced vocabulary elision is already preserved.
- **Long attention:** measure packed LDS versus selected warp16, including its
  register/occupancy and repeated-multiplication costs; do not assume less LDS
  guarantees speed.
- **Speculation:** use full-context acceptance and measured verification/commit
  boundaries, not candidate counts or four-token fixture TPS, to prioritize work.

No route/precision change or unvalidated prototype is promoted to a default.
The full-context generation baseline, architecture selection and final three-run
≥10 tok/s demonstration remain outstanding.

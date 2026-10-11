# FFN transfer/scheduling experiments, 2026-10-11

The goal remains active. No result here certifies 10 tokens/s or physical-128K
generation. The original model files and original pending native edits are intact.

## Optional diagnostic instrumentation

`QWEN_NATIVE_FFN_PROFILE=1` instruments the hybrid target FFN. It records host
activation staging, CPU gate/up job duration, GPU-shard submission, down-weight
prefetch submission, GPU and CPU waits, and a GPU-event down dependency span.
The latter includes any remaining wait for prefetched weights, projection and
residual completion; it is **not isolated kernel time**. CPU work overlaps other
intervals; the reported durations cannot simply be added.

The profile synchronizes the down endpoint, changing available overlap. It is
for diagnosis only and defaults off in performance comparisons. Counters are
reset after the eight warmup outputs and read only after CPU jobs complete.
Event/buffer destruction still drains accepted workers and streams first.

The 64-output real-prompt profile used 101 physical prompt tokens and 131072
capacity. It matched the earlier baseline's first 64 measured IDs. It required
13 cycles and 51 accepted of 83 proposals; this high-acceptance short interval
is not comparable to the fixed 256-output interval. Observed attribution:

| Measurement | Seconds | Scope |
| --- | ---: | --- |
| CPU gate/up jobs | 3.49263185 | 728 fused-dual FFNs, overlaps GPU/submission work |
| Down-weight prefetch host submission | 2.97496467 | Host API interval, includes blocking/staging behavior |
| GPU down dependency span | 1.2368592 | 832 FFNs, including remaining transfer dependency |
| Activation staging host interval | 0.550840849 | Fused-dual path only |
| GPU gate/up submission host interval | 0.679734906 | Not GPU kernel time |
| CPU gate/up remaining wait | 0.328015791 | After GPU/submission work |

The instrumentation passed strict GPU parity: 17-token prefix, 32 following
positions, identical predictions, bit-exact logits/features and every captured
prompt tap. Five CPU CTests and eight Python lifecycle tests passed.

## Down-projection row-shard candidate: rejected by strict parity

Existing full inline prefetch makes a whole down projection use the GPU-only
path. The opt-in `QWEN_NATIVE_HYBRID_FFN_DOWN=1` instead prefetches only a GPU
row shard for supported Q4_K/Q5_K/Q6_K/IQ4_XS tensors, while reusing the existing
CPU packed projection for the other output rows. Unsupported/resident tensors
retain their original path. `QWEN_NATIVE_FFN_DOWN_CPU_PERCENT` defaults to 50,
accepts integers 1–99, and aligns CPU rows to eight. No model quantization changes.

The prefetched GPU shard is consumed from the separately synchronized inline
buffer and scattered to its original output rows. CPU output rows are copied
to their original positions. This does not reduce token count or omit weights.

The 50% candidate's existing strict parity fixture **failed**: predictions
matched, but features/logits were not bit-exact and prompt feature arrays
differed. Reported feature relative RMSE: 0.000908711. Wrapper exit 1 is retained;
assertions were not loosened. This flag remains off and is **not promoted**.
CPU/GPU arithmetic repartition needs independent component/state investigation
before it can be treated as an original-checkpoint-compatible optimization.
No generation-throughput result is claimed for this failed candidate.

## Asynchronous activation transfer replay

The existing `QWEN_NATIVE_ASYNC_ACTIVATIONS=1` path is now selectable by the
frozen real-benchmark runner, independently of attention LDS changes and the
repaired model-mapping policy. It passed the same strict 17+32 position parity
fixture, including all prompt taps. Its 256-output performance comparison must
be completed and recorded before any default is changed.

The async replay completed 256 outputs in 80.3295052 seconds (3.18687386 tokens/s).
The same-build ordinary-copy replay took 80.0362851 seconds (3.19854926 tokens/s).
Both used 101 physical prompt tokens, 131072 capacity, eight excluded warmup
outputs and 93 cycles with 163/638 accepted proposals. Both matched all 256 IDs
and hash `0xf3ebf90500393c67`, returned wrapper/logger exit 0, and ran with FFN
profiling and down repartition disabled. These single intervals do not establish
a benefit for async activation copies; default remains **off**.

During the ordinary replay's prefill, a read-only sysfs sample at
2026-10-11T18:00:11+03:00 reported 8096858112 bytes VRAM used of 8573157376 total.
It is one live sample, not a peak/average or HIP allocation guarantee. Any next
residency experiment must retain headroom for transient physical-prefill layers
and lazy device checkpoints, preserve the original resident tensor placements,
and record actual added bytes rather than just a requested budget.

Exact evidence is retained in `results/ffn-*-gates-*`,
`results/async-activation-gates-*` and `results/decode-*`. Compact raw records
are tracked in `records/ffn-scheduling`. Defaults remain unchanged; Kyanite is still a
separate direct-KFD engine, with no HIP runtime dependency added by this work.

Code commit: `77684584ce1eb8c53901b598b4860de9b55caf1d`.
Measured/default-rebuilt executable SHA256:
`4bf931e05e74aa03b53fd0103d989bfa4880b651a76b31bb099599bd95c9f419`.
Frozen source archive SHA256 for the two 256-output comparisons:
`e060d430064fefb6caabd2e37c609de530f814940f659e3566efb13de55b11ab`.

## Next exact-residency experiment

Preserve the original resident tensor set first, then add a bounded set of
unchanged down matrices after the drafter is resident. Use `hipMemGetInfo` to
cap additions and reserve transient-prefill/checkpoint headroom. Tighten the
cache budget to the actual populated bytes afterward, so unused requested room
cannot silently move other CPU tensors to the GPU. Record requested and actual
added bytes, free-memory observations and exact output/state checks. This is a
design for the next experiment, **not implemented or measured yet**.

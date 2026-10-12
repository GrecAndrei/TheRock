# Genuine 32-position FFN prefill

This is an opt-in prototype, `QWEN_NATIVE_WIDE_PREFILL=1`, default off. It is not
an increase to a chunk-count label: each projection kernel computes 32 input
rows in one launch, sharing each decoded weight across those rows.

The eight quantized formats used by this target have native 32-position GPU
projections. Per-position accumulation order and wave reduction order mirror
the existing eight-position implementation. The old kernels remain unchanged.
`wide-projection-test` compares each new projection to four old calls on actual
GGUF weights, not a mock model or a claimed generation benchmark.

## Verified component gate, 2026-10-12

All 45 real-weight fixtures passed on gfx1032: Q8_0, Q3_K, Q4_K, Q5_K, Q6_K,
IQ4_NL, IQ3_S and IQ4_XS, at model input widths 5120/17408 where present and
output-row boundaries 1, 17 and 33. All 32 outputs and surrounding host guards
were bit-exact. Tested executable SHA256:
`858b01c3627feec42038d93bacb3caabd012f421c5e3f7247d66eccc16b8a56c`.
This certifies the exercised projection components, not complete-model parity.

## Model integration

Layer-major prefill retains ordered eight-position recurrent and full-attention
updates. Within each layer, it completes those updates before pointwise FFN
work, then runs FFN in 32-position groups. The layer input sequence is already
known; a layer's FFN output does not feed its own attention recurrence at the
next position. Final FFN output becomes input to the next layer as before.

RMSNorm, SiLU and residual arithmetic reuse existing kernels with larger grids;
the new FFN projections share the unchanged packed weights staged for that
layer. Extra padded rows are pointwise-only, never advance context occupancy,
and never become generated tokens. All five DFlash taps are captured after FFN
at valid physical positions. Work completes before staged weights are released;
the separate small workspace drains its stream before freeing buffers.

## Full-model gates and rejected first layouts

The initial single-wave implementation kept 32 accumulators per thread. Its
fully unrolled quant-block loop spilled registers: Q4_K/Q5_K used 128 VGPRs
with 11 spills, IQ4_XS used 128 with 221 spills. Its 513-position layer-major
fixture took 110.772 seconds. Disabling that loop's unrolling removed spills
and reduced the candidate to 65.1178 seconds. Both matched the microbatch
reference's predictions, logits, features and all prompt taps bit for bit.
The fully unrolled version also passed 1/17/33-position tail fixtures.

Those comparisons used **token-major microbatch-8** references (84.0076 and
83.9619 seconds). They do not establish an improvement over the existing
layer-major batch-8 prefill path. Both layouts are rejected for promotion.

The next kernel uses four waves per output row, eight positions per wave, with
a cooperatively decoded quant block in LDS shared across all 32 positions.
All eight formats again passed all 45 projection fixtures. This rolled-loop
layout uses 33–37 VGPRs, no private storage or spills. The new
`target-wide-prefill-test` compares **layer-major batch-8 against layer-major
batch-32**, with the same physical prompt and original attention/recurrent
updates, independently of the earlier token-major comparison.

On gfx1032, the rolled LDS version matched predictions, logits, features and
every prompt tap bit for bit for 1/17/33/513 positions after a 17-position
prefix. The 513-position direct comparison took 30.3591 seconds for batch-8
versus 31.9039 seconds for batch-32: no demonstrated improvement. At tails
1/17/33, batch-8 took 1.52886/2.37661/3.22457 seconds and batch-32 took
2.39869/2.66411/4.11705 seconds. Extra padded FFN positions do not advance
context or enter the counted prompt. All wrappers/loggers exited zero.

Rolled LDS executable SHA256:
`18f072ddabaa788d2f6f6dc76c1679a8a0be1df17ea8e14ada5eef3809567588`.
Frozen source SHA256:
`f6961015f4005829903f1e1e515ab16c4637ed11ac3ecf0b4688ba0c4c9cdf45`.
Original evidence is retained in `results/wide-prefill-lds-gates-z9r3D9`.
The subsequent LDS-loop-unrolling candidate passed the 45 component fixtures
and strict 513-position comparison but regressed: batch-8 took 30.4111 seconds
versus 45.215 for wide prefill. Diagnostic FFN host spans on the 512-position
chunk were 19.06 versus 32.2904 seconds; these spans are not isolated kernel
times. Its 34–42 VGPRs had no spills, demonstrating that removing spills alone
does not prove performance. This candidate is rejected and the rolled LDS
prototype is retained with the experiment still off by default.

Unrolled LDS executable SHA256:
`5c2ec0f47a18cc699a4854f18fe94cf120eced7c0494b378f6c33daee12333b3`.
Frozen source SHA256:
`ecbb5cf72263fdf35a60a93f6f6240650f65277fb797c645231074050a98b645`.
Evidence: `results/wide-prefill-lds-unrolled-S7LrAq`. Compact raw logs for the
component and model gates, resources and exits are in `records/wide-prefill`.

No variant is promoted. These gates are not generated-token benchmarks or
physically populated 128K histories, and 10 generated tokens/s is unproven.

## Default-path real-prompt replay

After retaining the rolled LDS prototype, a same-build default-path replay
completed 256 measured outputs after eight excluded warmup outputs. It used
101 physical prompt tokens and 131072 capacity, not 128K occupied history.
Prefill took 11.886011 seconds and measured decode 81.5304221 seconds
(3.13993223 committed tokens/s). It matched all previous 256 target IDs/hash
`0xf3ebf90500393c67`, with 93 cycles and 163/638 accepted proposals.
Both wrapper and logger exited zero. The bounded output interval is not a
completed free-running answer or a quality evaluation.

The wide-prefill replay also completed normally and matched every measured ID,
with the same 93 cycles and 163/638 acceptance. Its prefill took 9.5871047
seconds and decode 81.3121407 seconds (3.14836134 committed tokens/s).
The difference in decode is about 0.27%, not evidence of a speedup; decode
arithmetic is unchanged. The candidate's shorter small-prompt prefill interval
does not override the 513-position regression or establish a robust TTFT win:
the default ran first and initialization was 2.2116116 versus 0.313299284
seconds, so host/cache/driver warmth was not identical. No default changes.

Real replay executable SHA256:
`18f072ddabaa788d2f6f6dc76c1679a8a0be1df17ea8e14ada5eef3809567588`.
Both replay source archives SHA256:
`3484cc35b49200704449a87e0afb172985ce24de3a43feb99d7111fb050420b9`.
Native source commit: `5d5f3804fb8152372a5120add2ed1aaf606afb64`.
Five CTests and eight Python lifecycle tests passed on the retained build.
The source, original models and Kyanite runtime boundaries remain isolated.

These are single intervals, not performance distributions. Full physical-128K
generation, an independent full-model oracle, broad reasoning-quality evaluation
and the 10 committed-token/s goal remain outstanding.

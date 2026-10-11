# Model and DFlash-head experiments, 2026-10-11

The user explicitly authorizes head, drafter and target-model changes. Existing
GGUF assets stay intact. Engine-only, draft/head-only and changed-target results
are separate tracks; the original-checkpoint evidence is never relabelled as
evidence for a different model. Target variants require explicit quality results.

## Selector calibration ablation

The existing selector ranks each top-16 token with its draft logit plus a
predecessor-conditioned bilinear correction. Its default scale is 1.0. The
benchmark runner now exposes `QWEN_NATIVE_SELECTOR_WEIGHT` and records it.
Scale 0 disables the correction in scoring; it does not change model bytes or
the target verifier. This is a runtime head ablation, not a newly trained model.

The option is finite/nonnegative, fully parsed and owned by each drafter session
instead of a process-static cached value. Both GPU-top16 and CPU-logit selectors
use the same scale. The default scoring operations are unchanged.

New benchmark metrics record accepted-prefix histograms, acceptance by draft
position, conditional top-16 target-token recall and first-rejection rank.
They label only the accepted prefix and first rejected position. Later verifier
logits have already consumed an incorrect draft prefix and are not ground-truth
labels for the real greedy trajectory. Final partial blocks are counted by their
actual proposed length; anchor-only cycles do not become draft cycles.

Five CPU CTests (including exhaustive prefix/rank fixtures and invalid option
cases) and eight Python lifecycle tests passed before GPU measurements.

## Source-level audit scope

The original pinned llama.cpp DFlash reference
`5ecbe1ac17ec0484c5b44af0bd580cdc9c428ed4`,
`src/models/dflash.cpp` and `common/speculative.cpp`, uses the same additive
selector scoring and greedy predecessor walk. The supplied GGUF feature indices
`[6,20,34,48,62]` are layer-input positions, matching this engine's captures after
zero-based layers `[5,19,33,47,61]`. Its convolution layout also matches the
reference grouping. These observations are not independent full-model numerical
certification. No reference code was copied or changed.

## First observed baseline

The instrumentation replay generated the same 256 measured token IDs as the
previous baseline, hash `0xf3ebf90500393c67`, in 84.3351788 seconds (3.0355067
tokens/s). Prompt: 101 physical tokens; capacity: 131072; warmup: eight excluded
committed outputs; wrapper/logger exit 0. It still required 93 verification
cycles with 163 of 638 draft proposals accepted.

At draft position one, the target's correct token was in the top 16 on 89 of 93
conditional attempts, while the selected token matched on 62. Across 87 first
rejections, only 12 correct tokens were outside the top 16. Thus selection has
measurable headroom on this workload, but this is not an oracle TPS estimate or
proof that any particular rescaling helps. Candidate measurements and independent
holdout prompts must follow before a default is changed.

The selector-off ablation completed with exactly the same 256 output IDs and
the same 93 cycles/163 accepted proposals. Its interval was 83.7026632 seconds
(3.0584451 tokens/s). First-position acceptance rose from 62/93 to 68/93, but
later acceptance redistributed and total verification work did not improve.
The sub-one-percent wall-time difference is not a demonstrated optimization;
the default remains 1.0.

The half-strength correction (`0.5`) completed the identical 256-token sequence
in 79.3533238 seconds (3.22607785 tokens/s), with 91 cycles and 165 accepted of
624 proposals. Two fewer verifications is a real difference on this prompt;
the entire wall-time difference is not attributed solely to it. This is a
candidate, not a promoted default or a sustained-speed distribution.
The independent holdout prompt reviews C++ string-view lifetime safety; its
baseline and candidate must be measured separately and their output IDs compared.

The holdout baseline physically processed 140 prompt tokens and generated 256
measured outputs in 71.1137478 seconds (3.59986652 tokens/s), with 82 verification
cycles and 174 accepted proposals. Output hash: `0x9dae7dd9a0d4b5ae`. This is
another short-context interval, not a completed-answer quality score.

Its half-strength candidate generated the identical 256 IDs in 70.5072437
seconds (3.63083262 tokens/s), again with 82 cycles, 174 accepted and 554 proposed
tokens. It did not reduce verification work on the holdout. The 0.86% timing
difference is not a demonstrated general speedup; default scale stays **1.0**.

| Prompt / head scale | Physical prompt tokens | Decode seconds | Committed tokens/s | Verification cycles |
| --- | ---: | ---: | ---: | ---: |
| CPU/GPU explanation / 1.0 | 101 | 84.3351788 | 3.0355067 | 93 |
| CPU/GPU explanation / 0.0 | 101 | 83.7026632 | 3.0584451 | 93 |
| CPU/GPU explanation / 0.5 | 101 | 79.3533238 | 3.22607785 | 91 |
| C++ holdout / 1.0 | 140 | 71.1137478 | 3.59986652 | 82 |
| C++ holdout / 0.5 | 140 | 70.5072437 | 3.63083262 | 82 |

Each is a single 256-output interval after eight excluded outputs, with explicit
exit 0 and no early EOS. These are not repeated-run distributions. All variants
matched their prompt's baseline token array and independently recomputed hash.
Model files were not modified or retrained in this experiment. The measurable
selection headroom motivates collecting genuine-prefix labelled logits and
corrections for learned calibration; it does not justify using the verifier as
an uncounted draft oracle. FFN scheduling and genuine wide prefill remain major
engine work, and 10 tokens/s has not been demonstrated.

Artifacts are append-only `results/decode-selector-*` directories. Each retains
the executable/source snapshot, hashes, command, JSON including every committed
output ID, and explicit exits. Compact exact evidence is tracked in
`records/head-calibration`. Native code commit:
`20337067862d4652e434b2a2bb277d8be445e7d5`; tested executable SHA256:
`681a3a02f783d51fb126dfb7a76c05a47c509284b4884da5674a3a992db8bd9b`.
Holdout prompt SHA256:
`293e4ae952f5c2863040549c8fbd43855bc12367ce05303f2ca4dba8bfd7a000`.
Neither this short-context diagnostic nor model
modification permission establishes physical-128K performance.

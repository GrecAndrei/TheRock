# Real-prompt decode diagnostic

`benchmark-real TARGET.gguf DRAFT.gguf PROMPT.txt CONTEXT [OUTPUTS] [CHUNK] [RESIDENT_GIB]`
uses the same canonical chat template and xhigh system instruction as interactive
chat. It physically processes the prompt, excludes eight committed warmup outputs,
then measures the requested number of committed greedy outputs. It records every
measured token ID, output hash, proposal acceptance and phase wall attribution.
Early EOS fails the requested-length test; proposals are never counted as outputs.

The workspace runner freezes the executable and a source archive before launch,
records their hashes, the input hash, command, raw JSON and both pipeline exits.
It uses a 360-second bound. Do not edit the runner during a foreground request.
The diagnostic is not physical-128K certification: 131072 is capacity, while the
current real chat prompt contains 101 physically processed tokens.

## Baseline and rejected candidate, 2026-10-11

The frozen baseline generated 256 measured outputs in 80.5666386 seconds:
3.17749387 generated tokens/s. Seven-token drafting proposed 638 tokens, accepted
163, and required 93 verification cycles. Verification occupied 75.7441242
seconds; FFN phase attribution was 39.8047217 seconds. Phase labels attribute
host wall intervals in an overlapping pipeline, not independent GPU kernel time.
The separate complete 1101-output chat's 5.45 tokens/s is a different interval,
not a contradictory result or a controlled optimization speedup.

Optional `QWEN_NATIVE_PINNED_STAGING=1` uses three bounded host staging buffers
(270.94 MiB total for this model), rather than registering the entire GGUF.
Before reuse, the host waits for the previous copy's completion event. Device
buffer reuse still waits for its last consumer. Both streams are drained before
resources are released. No quantization, projection arithmetic or row split is
changed. This flag defaults to **off**.

The coherent allocation candidate passed bit-exact predictions, logits, features
and every prompt feature tap, but generated the identical 256 outputs in
229.69454 seconds: 1.11452366 tokens/s. It is **rejected for promotion**.
`QWEN_NATIVE_PINNED_STAGING_COARSE=1` selects noncoherent allocations and system-
release copy events for an additional experiment; it has no effect when staging
is disabled and is not a recommendation. This candidate also passed bit-exact
parity, then generated the identical 256 outputs in 121.479986 seconds:
2.10734302 tokens/s. It is also **rejected for promotion**. Correctness alone
never promotes a flag. These are single diagnostic runs, not distributions or
proof of a unique underlying cache/DMA slowdown mechanism.

A baseline replay with both staging flags off in the candidate binary generated
the same 256 outputs in 92.8776443 seconds (2.75631453 tokens/s). Host variability
is visible; 3.18 is not a stable sustained-speed promise. Four CPU CTests and
eight Python lifecycle tests passed after that replay.

## Head diagnostics

The real benchmark now records per-position acceptance, conditional top-16
recall, accepted-prefix histograms and first-rejection rank. Labels stop at the
first rejection; logits after a wrong speculative prefix are not true greedy
labels. Anchor-only and short-tail cycles retain their actual proposal counts.

`QWEN_NATIVE_SELECTOR_WEIGHT` is finite/nonnegative and captured per drafter
session (default 1.0). It scales only the bilinear selector correction, not the
target model or verifier. The workspace runner exposes and records the option.
Changing a draft head must pass committed-target-ID comparisons; changing the
target itself is a separately identified quality/performance experiment.
Five CPU CTests and eight lifecycle tests pass with the new instrumentation.

Raw evidence belongs to the workspace `results/decode-*` directories and the
tracked evidence ledger. No result here demonstrates 10 tokens/s, completed
physical-128K prefill, or independent complete-model numerical certification.

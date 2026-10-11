# Controlled real decode progress, 2026-10-11

The timeout/prefill repair remains the working baseline. The complete original
request produced 1101 committed outputs at 5.45 generated tokens/s. A newly
fixed-length diagnostic measures a different interval: eight excluded warmup
outputs followed by 256 committed outputs, with all output IDs retained.

Every run below physically prefilled the same 101-token chat prompt, used the
same xhigh instruction, target/drafter files, Q8 KV, 1.0 GiB target residency
budget, greedy single-stream decoding and 131072 **capacity**. None populated
128K history. The measured outputs contain real model reasoning, not synthetic
kernel operations or unverified draft proposals.

| Experiment | Prefill seconds | 256-output decode seconds | Generated tokens/s | Decision |
| --- | ---: | ---: | ---: | --- |
| Baseline | 11.67897 | 80.5666386 | 3.17749387 | Retain |
| Baseline replay, staging disabled in candidate binary | 10.1370648 | 92.8776443 | 2.75631453 | Retain; host variability visible |
| Bounded coherent host staging | 15.361 | 229.69454 | 1.11452366 | Rejected for promotion; default off |
| Bounded noncoherent host staging | 13.0813502 | 121.479986 | 2.10734302 | Rejected for promotion; default off |

All four runs returned wrapper/logger exit 0, generated identical 256-token
arrays with hash `0xf3ebf90500393c67`, proposed 638 draft tokens, accepted 163
(25.55%), and required 93 target verification cycles. These are individual
diagnostics, not controlled throughput distributions. The baseline replay
rules out presenting the best single interval as stable sustained performance.
The whole-chat 5.45 and first-256
3.18 results are kept separate; acceptance and cost vary through an answer.

Baseline verification occupied 75.7441242 seconds; existing FFN phase wall
attribution was 39.8047217 seconds. Checkpoint host submission was only
0.007800321 seconds, although device checkpoint work can fall within later
waits. Overlapping phase labels are not isolated GPU kernel timings. There is
no unique proven cache/DMA explanation for the staging regressions.

Both staging variants passed physical parity fixtures: 17-position prefix,
32 following tokens, matching predictions, bit-exact logits/features and all
prompt feature taps. The buffers total 270.94 MiB and wait for previous DMA
completion before host overwrite. No whole-model pinning was reintroduced.
Neither correctness pass justifies enabling a slower variant.
Four CPU CTests and eight Python lifecycle tests also passed after the replay.

## Reproduce and inspect

From this workspace, after building the isolated native engine:

```sh
bash scripts/run-real-benchmark.sh baseline 256
QWEN_NATIVE_PINNED_STAGING=1 bash scripts/run-real-benchmark.sh coherent 256
QWEN_NATIVE_PINNED_STAGING=1 QWEN_NATIVE_PINNED_STAGING_COARSE=1 \
  bash scripts/run-real-benchmark.sh coarse 256
```

The runner freezes the executable and source archive, records hashes, command,
raw JSON, time bounds and exits. Original snapshots remain in ignored
`results/decode-*`; compact exact raw evidence is tracked in
`records/decode-diagnostic`. The fixed prompt is
`fixtures/real-benchmark-prompt.txt`, SHA256
`e178890cd5131a6bed792e9923d4fffc7883a3bdaf8769d281daefb3509cad4b`.
Models retain the hashes in the timeout/prefill repair records. Original dirty
native edits and the Kyanite runtime remain untouched.

Next performance work should focus on measured FFN critical-path costs and real
wide-batch prefill, with exact numerical and output gates. Current 512-position
prefill staging still executes eight-position projection units; it is not a
wide matrix-multiply implementation. Prefill speed and generated-token speed
are distinct metrics. **10 generated tokens/s is not demonstrated**, and the
unchanged physical-128K protocol still governs final certification.

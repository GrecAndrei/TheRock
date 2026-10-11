# Timeout repair and prefill evidence, 2026-10-11

The repaired HIP/C++ engine completed the original real request within the
unchanged 360-second bound. The stable runner returned 0 and retained its exit
record. The model's xhigh instruction, prompt, model files, Q8 KV, weight budget,
greedy decoding and one-stream policy were preserved. No Kyanite inference or
physical-128K generation speed is claimed.

## Observations and cause assessment

The original cold run physically prefilled 101 tokens in 91.4 seconds, reported
75.6 seconds in one injection phase and timed out without an answer. A later
warm baseline prefilled the same prompt in 16.3 seconds and generated reasoning
normally until its separate 180-second diagnostic bound. That rules out a
deterministic deadlock for this input and exposes strong cache/host sensitivity.

Source inspection found unconditional registration of both entire GGUF model
mappings with HIP. A live baseline snapshot showed approximately 17 GiB of
model-file RSS and 6.6 GiB of anonymous process memory on this 32 GiB host.
System swap use was substantial. The engine also ran unnecessary vocabulary
projections after every teacher-forced chat microbatch. Those are concrete
working-set and prefill defects; they are repaired in the isolated engine.

The injection phase label contains waits, transfers and allocations, so it
does not prove a 75-second arithmetic kernel. The evidence does not identify
one unique driver operation responsible for every original cold-run stall.
Warm/cold figures are retained separately and are not treated as a controlled
speedup ratio attributable solely to the code change.

## Verified behavior

| Workload | Observation | Scope |
| --- | --- | --- |
| Original cold chat | 91.4 s prefill; 360 s timeout | No completed answer or sustained TPS result |
| Warm baseline chat | 16.3 s prefill; 180 s diagnostic timeout | Reasoning was progressing |
| Stable fixed chat, identical real input | 10.8 s prefill; 1101 committed output tokens; 5.45 generated tok/s; exit 0 | Complete answer, reasoning included; 101 physical prompt tokens |
| Fixed request wall time | 214 s from runner start/end records | Includes initialization, prefill and generation; input hashing occurred before start |
| Real frozen-corpus prefix | 3073 physical tokens in 197.675439 s, 15.5456845 prompt tok/s; exit 0 | 131072 capacity is not populated history |
| Corpus continuation | 8 committed outputs, then normal EOS; target occupancy 3081; drafter occupancy 2048 | Short diagnostic, not sustained decode certification |
| 513-position schedule parity fixture | 84.1153 s reference vs 30.2282 s layer-major phase | All-GPU arithmetic fixture, not a real-generation benchmark |

The corpus source is the copied canonical 128K token file, SHA256
`1ac073a8c65b8fa11b336e412574701a124ec9e2fa6aeb52578909c1c8d0a0de`.
The selected real prefix has FNV64 integer `3020550682829038446`; it was not
repeated or padded. All target history was processed. Only the drafter's
documented 2048-token window limits feature injection.

Four CPU CTests and eight Python lifecycle tests passed. Physical GPU parity
fixtures for 1, 17 and 513 positions matched predicted IDs, logits, following
state and all five captured prompt feature arrays bit for bit. These compare
the two native all-GPU schedules; full external-model certification remains a
separate requirement.

## Implementation and artifacts

- Native repository: `sources/native-engine`, branch `engine/timeout-prefill-fix`.
- Code commit: `b01570281cc32c25306512de4dd5acefa040679d`.
- Documentation commit: `d51fb997f2ac806c5eb54e4b66051818ac990f6f`.
- Tested executable SHA256:
  `2f91e6d05178093b301d82f2961b8021284f1c0d785656d120a9154856d6bd1a`.
- Raw logs, commands, exit records and input hashes: `records/prefill-repair`.
- Run script: `scripts/run-chat-diagnostic.sh`, defaulting to the repaired binary.
- Build and detailed source contracts: `sources/native-engine/TIMEOUT_PREFILL_FIX.md`.

The first fixed request also produced a complete answer and printed 5.49 tok/s,
but the shell runner was edited while it was executing, corrupting its final
status parsing. Its raw output is preserved under
`first-fixed-chat-runner-error.raw.log`; no exit 0 was invented for it. The stable
rerun above is the completed-request result used for the conclusion.

The original repositories and pending native edits remain untouched. Kyanite
remains an independent compute-stack track. Prefill still needs substantial
further acceleration, and **10 generated tokens/s at physical 128K has not been
demonstrated**. The unchanged three-prefill correctness/evidence protocol still
governs that goal.

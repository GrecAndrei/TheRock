# Guarded extra down-weight cache

The opt-in `QWEN_NATIVE_EXTRA_DOWN_MB` requests integer MiB in 0–1024, default
zero. After the target and drafter have their original resident tensors, the
engine observes HIP free memory, reserves 512 MiB for transient prefill and lazy
checkpoints, and adds only unchanged down-projection matrices. Requested unused
room is removed from the budget afterward, preventing unrelated CPU tensors
from silently changing backend. Actual added bytes and matrices are reported.

The same helper is used by real chat, the short real benchmark, corpus prefill
and the physical-128K benchmark. Unit fixtures cover option parsing and a free
memory allowance clipped above, at and below the reserve. The guard is not a
guarantee against unrelated external GPU allocations changing free memory.

## Evidence from 2026-10-11

The strict 17+32-position fixture actually added two matrices (97,484,800 bytes),
then matched predictions, logits, features and all prompt taps bit for bit. The
fixture fails if a requested candidate adds no matrices, so an unexercised no-op
cannot be presented as a parity success.

| Real-prompt interval | Physical prompt tokens | Measured outputs | Decode seconds | Generated tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Same-build default | 101 | 256 | 80.0476762 | 3.19809409 |
| Extra-cache request 128 MiB | 101 | 256 | 81.2458438 | 3.15093041 |

Both used eight excluded warmup outputs, one greedy stream, 131072 capacity,
the original models and exactly the same 256 output IDs/hash
`0xf3ebf90500393c67`. Both exited normally. The candidate loaded two matrices,
97,484,800 bytes, from a 134,217,728-byte allowance. HIP reported 880,803,840 free
bytes before addition. Its prefill took 19.9029687 seconds versus the default's
10.5305748; these single runs do not identify a unique cold/cache/driver cause.

This candidate has **no demonstrated speed benefit** and stays default off.
The runs are not distributions, physical-128K histories or final-goal evidence.
Measured executable SHA256:
`27e0d97d733871fd526d99329fdf41a82e3565e419c21b73cdbac0db3c4b2c3e`.
Frozen source archive SHA256:
`5a9f72e90df9267164b212676b00d10566210c3fd5d86f4a0aceb642497834e6`.
Original snapshots remain in `results/decode-extra-down-*`; compact exact logs
are tracked in `records/guarded-cache`.

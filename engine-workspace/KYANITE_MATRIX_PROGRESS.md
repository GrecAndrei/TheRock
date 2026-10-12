# Guarded LDS matrix checkpoint, 2026-10-12

Kyanite now compiles and dispatches a restricted FP32 tiled multiplication on
the RX 6650 XT: A[8,8] * B[8,32] = C[8,32]. This is a correctness milestone,
not measured inference throughput or general matrix readiness.

Four 8x8 workgroups each use 512 bytes of private LDS and two wave32s.
A workgroup barrier separates cooperative input staging from dot products;
output rows 0..3 require B values staged by the other wave. Each output uses
eight separate FP32 products and seven additions, not fused multiply-adds.
The fixture tests row pitches [11,37,39], input preservation, row padding and
eight-float guards on both sides of every allocation. All bytes match an
independent CPU dot-product reference, including a non-exact decimal pattern.

The second LDS tile starts at byte 256, exposing the old inline-only constant
limit. The gfx1032 leaf now has a typed literal move with independently checked
encoding; U64 constants materialize both dwords without truncation. Pending
memory-result overwrite hazards include this instruction. LLVM 23 is only an
external development oracle, absent from the shipped compile/runtime path.
No HIP/HSA userspace, COMGR, rocBLAS or llama.cpp dependency was introduced.

## Evidence

- Kyanite commit: `224109564721d06bd78eb3a5ed3d193e3403de3b`.
- Executable SHA256:
  `8c6134a23005e55fcbf35d5f8b7f58122ffd5a613734b46e51e1e825a4c45211`.
- Committed source archive SHA256:
  `f45dd41468757ae6018005b27a7f7927d5de136456f9b57324eaf6f249845748`.
- Earlier working-source archive SHA256:
  `87eeac05e8cdae4339f4480354fc726f4f2374f10a4faac90ce8d383b9575076`.
- Linux 6.18.55-1-lts; gfx100302, KFD node 1/GPU ID 40264;
  Rustc 1.98.1, debug build, locked/offline Cargo, two build jobs.

Queue/no-op/fill/vector-add/reduction/matrix ran separately in that order on
one frozen executable, three teardown/reinstall repetitions each. The matrix
patterns were binary fractions, non-exact decimals and repeated binary fractions.
After commit, rebuilding produced the identical executable and matrix passed
three more repetitions. Every wrapper/logger exited zero. Captured phase
journals contain no entries. Waits were five seconds with 30-second external
process bounds. The installed event mailbox remains pinned until process exit.

Host format/Clippy/workspace and doc tests passed, as did the 285 native CPU
fixtures and eight Python lifecycle tests. The matrix interpreter checks 1/3/4
workgroups with both numerical patterns; removing its barrier reports a race.
Machine tests check literal encodings, U64 high bits and pending-load overwrite
wait insertion. These host tests are not extra physical performance samples.

Raw artifacts: `results/kyanite-matrix-gBcSHM`. Compact logs, exits, journals,
hashes, oracle outputs and final host checks: `records/kyanite-matrix`.
Previous failed dispatch attempts remain retained under the preceding dispatch
checkpoint. No matrix attempt was discarded or labeled as a throughput run.

## Remaining work

Only eight rows, K=8 and complete eight-column tiles are covered. General K
loops, tail/mask handling, quantized matrix operators, exceptional/denormal
floating modes, larger dispatch stress and real inference remain pending.
The general matrix capability stays unimplemented; this fixture does not
silently enable it. Original target/DFlash weights are unchanged.

The latest native short-context replay remains about 3.14 committed generated
tokens/s; this checkpoint does not change that number. Physical-128K inference
and the 10 generated tokens/s goal remain unproven.

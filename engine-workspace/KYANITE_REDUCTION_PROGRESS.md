# Cross-wave LDS reduction checkpoint, 2026-10-12

Kyanite's six-stage 64-invocation workgroup sum now passes on the RX 6650 XT
through its own compiler and production direct-KFD runtime. This is a small
correctness gate, not inference throughput or Phase 3 completion.

The generic DSL/IR adds unsigned XOR for lane-partner indexing. KYIR is now
version 3; old versions are explicitly rejected. The gfx1032 leaf lowers U64
XOR as two dword instructions without truncating its high half. Encoding is
checked independently with LLVM 23 as an external development oracle, not a
runtime/compiler dependency. Source/IR tests reject float XOR, check integer
high bits and precedence, and canonical-round-trip the new operation.

The kernel uses one private 256-byte LDS tile per workgroup and six butterfly
stages. Eleven workgroup barriers separate initialization, reads and overwrites.
Each invocation writes a distinct output slot containing its group's sum:
four logical group sums are replicated into 256 output positions. These are
not 256 independently reduced groups, generated tokens or model outputs.

## Evidence

- Code commit: `083c4e28f834107c6450b06320359d85decf18d0`.
- Executable SHA256:
  `7df4ad801234f85b5c091b64635e128ec808ac071d139d0113ddf62e00573653`.
- Committed source archive SHA256:
  `a9f26f92a67125f343034ccc5b47c14ff3d3bb19ec8938d0e566907ac39196fb`.
- Working-source archive before final documentation:
  `d22caf184370a10d35040fd0cadbd05c75ee5f3c9dd13ff54bc7a7adac4b3355`.
- Linux 6.18.55-1-lts, gfx100302, KFD node 1/GPU ID 40264; Rustc 1.98.1,
  debug build, locked/offline, two build jobs. Linked libraries: libc/libgcc_s.

Physical fixture: four 64-thread groups, two wave32s per group. Input element
`i` is `i*0.25-16`, giving distinct group sums. Every output was compared
bit-for-bit to an independent serial sum; inputs and eight-float guards on both
sides were unchanged. Three completion/teardown/reinstall repetitions passed
before commit, then three more on the committed build with the same executable
hash. All wrappers/loggers returned zero; phase journals had no entries.
Waits were five seconds, with 30-second outer process bounds.

Queue/no-op/fill/vector-add also passed three repetitions each on this same
build before reduction. Full host format/Clippy/workspace and doc tests passed,
plus 285 native-baseline CPU fixtures and eight Python lifecycle tests.
The host scheduled interpreter checks 1/3/4 groups and three numerical patterns,
including non-exact decimal inputs. Removing a read-before-overwrite barrier
triggers a race diagnostic. Decimal/high-bit checks are host evidence, not
additional physical numerical coverage.

Raw evidence: `results/kyanite-reduction-NCqDIM`; compact logs, exits, journals,
oracle records and host checks: `records/kyanite-reduction`. The installed event
mailbox remains pinned until process exit, as documented for the prior gates.

## Next gate and goal status

Tiled matrix multiplication, broader dispatch stress and inference operators
remain pending. General reductions, tails and IEEE exceptional/denormal modes
are not certified by this fixture. Original target/DFlash weights are unchanged.
Latest native replay remains about 3.14 committed generated tokens/s; this work
does not change it. Physical-128K generation and 10 generated tokens/s remain
unproven. See `KYANITE_DISPATCH_PROGRESS.md` for the preceding queue repairs
and retained failed hardware attempts.

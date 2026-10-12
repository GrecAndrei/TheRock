# Production Kyanite silicon dispatch, 2026-10-12

Kyanite executes its own compiled no-op, fill and vector-add through the
production direct-KFD dispatcher on the RX 6650 XT. This is correctness evidence,
not inference throughput. No HIP, HSA userspace, LLVM AMDGPU, COMGR, rocBLAS or
llama.cpp is linked or invoked by the gate.

## Repairs

Four independent exact BOs replace combined storage: 4096-byte ring, index and
EOP, plus 11190272-byte save/debug allocation. Logical CWSR is 11157504 bytes,
control stack 16384 and debugger reservation 32768. The OS-observing example
provides typed topology; gfx1032 geometry lives in its target leaf. Save memory
and its 40-byte header are initialized before queue creation. Queue destruction
precedes reverse storage cleanup. Failed destruction retains retry state;
indeterminate creation retains storage through Drop until process teardown.

Queue-only BOs require GPU execute permission, without CPU executable mapping.
Normal buffers remain non-executable. Kernel text now uses target-provided
256-byte absolute-address alignment, checked padded allocation and relocation.

## Committed silicon gates

Source: `709a5f491b82e616b87dabbdb8e5f80410951889`.
Executable SHA256:
`66e287f79770ed4d42aad2035def724154f305c9fa585556f64f3231cb7f82d4`.
Committed source archive SHA256:
`1b71115b675e545e39988d560bc522bc702ee81f0c8c907df2e3259e5c02a206`.
Rustc 1.98.1, debug/locked/offline, two build jobs; Linux 6.18.55-1-lts,
KFD node 1, GPU ID 40264, gfx target 100302.

Each phase ran in a separate process with a 30-second outer bound, five-second
completion waits and three create/close/reclose cycles. Every wrapper/logger
exit was zero. Journals captured after each phase had no entries.

| Gate | Exercised result |
| --- | --- |
| Queue | Production create/map/close/reclose, initial producer index zero |
| No-op | Actual GPU completion and explicit teardown |
| Fill | One 64-thread workgroup, 64 exact values of 3.25, intact guards |
| Vector-add | Four 64-thread workgroups, 256 exact outputs, unchanged inputs/guards |

Buffer gates compare whole byte arrays, including eight-float guards on both
sides. Vector inputs are exactly representable binary fractions. Out-of-bounds
host writes are rejected. These fixtures do not certify arbitrary dimensions,
tails, reductions, LDS/barriers, matrix multiplication or model math. The
installed event mailbox remains pinned until process exit: this KFD boundary
does not expose a mailbox uninstall operation.

Original evidence: `results/kyanite-committed-dispatch-WB59P4`.
Compact logs/exits/hashes/journals: `records/kyanite-dispatch/committed`.
Full host checks passed formatting, warnings-denied Clippy, workspace tests/doc
tests, including 117 Linux-KFD tests. Native baseline checks passed 285 CPU
quantization fixtures and eight Python lifecycle tests. Host evidence is in
`records/kyanite-dispatch/host-checks`; it is not GPU throughput evidence.

## Retained failures

- `kyanite-dispatch-gates-nuYXry`: queue passed, no-op timed out; journal CPF
  permission fault at ring address `0x1000000000`.
- `kyanite-executable-ring-9dw2B0`: executable ring alone still timed out;
  journal had no entries. No fault did not establish completion.
- `kyanite-aligned-dispatch-aPE3Vr`: aligned text and executable ring still
  timed out; CPF permission fault at EOP `0x1000002000`.
- `kyanite-queue-exec-dispatch-xamm8g`: all queue BOs executable plus aligned
  text passed all four gates. A vector-add invocation used the wrong topology
  root (`.../topology/nodes`) and failed before GPU access; its log and the
  corrected successful invocation are both retained.

Queue mapping rules were checked against Linux v6.18 `kfd_queue.c`, GPU PTE
permissions against `amdgpu_amdkfd_gpuvm.c`, and entry alignment against LLVM's
AMDGPU ABI documentation. Local ROCr executable allocator behavior was inspected
at rocm-systems commit `d34cbb640961dcd0c0aa6c1c3c1cfa5d4becfa31`. These are
development references, not shipped dependencies or copied algorithm ports.

## Remaining work

Next: reduction with LDS/barriers, tiled matrix multiplication and broader
dispatch/error stress before inference operators. Native generation remains
about 3.14 committed tokens/s on the latest 256-output replay; this checkpoint
does not change it. Full physical-128K generation, broad quality evaluation and
10 generated tokens/s remain unproven. Original target/DFlash weights unchanged.

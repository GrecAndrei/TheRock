# RX 6650 XT engine workspace

Kyanite is the engine under development. The target is sustained, single-stream
Qwen3.8-27B generation on gfx1032 at >=10 committed target tokens/s, with real
prompt processing and recorded output. Setup does not establish that result.

## Sources and ownership

| Path | Role | Starting revision |
| --- | --- | --- |
| `sources/kyanite` | Independent Git clone, branch `engine/rx6650xt-10tps`; expand this engine | `867ea43cfb67db9782b17814020a24593d9e4508` plus preserved local edits |
| `sources/native-baseline` | Copy of the pending C++/HIP engine, including uncommitted changes; correctness and performance baseline | TheRock `ef774ed4d411f8e3fb41014a923939d2e4de866d` plus working files |
| `sources/native-engine` | Independent Git repository for HIP/C++ timeout and prefill fixes | Branch `engine/timeout-prefill-fix`; preserved source snapshot plus repairs |
| `references/llama.cpp` | Independent, detached development reference and comparison engine | `5ecbe1ac17ec0484c5b44af0bd580cdc9c428ed4` |
| `references/llama-upstream` | Fresh upstream reference, detached worktree sharing the reference clone's objects | `966baae76b578d5d3dd4c179a75a35c80ff75577` |
| `fixtures` | Copied frozen project corpus and canonical 131072-token prompt | Existing benchmark assets, not regenerated text |
| `results/setup/historical-kfd` | Copied historical hardware probe, including failure | From `/home/alex/rocmtry-inspection/hardware-results` |
| `build`, `results` | New build products and append-only experiment evidence | Separate from previous work |

The clones have independent Git objects and the original upstream URLs. Original
repositories, existing build directories, benchmark evidence and controller
settings are preserved. Models stay at their existing paths under
`/home/alex/Downloads`; no 16 GiB model copy is made. The local Kyanite bringup
crate and the inspection checkout's hardware-smoke example are carried forward
and remain visible as pending changes in the new clone.

The first restricted-session fetch failed on 2026-10-11. After host access was
enabled, both repositories were fetched successfully. rocmtry's upstream main
still points to `867ea43`; current llama.cpp master was checked out separately at
`966baae76`. The earlier DFlash2 reference remains available. Use
`bash scripts/refresh-references.sh` to update remote refs without changing the
pinned checkouts. See `HARDWARE_PROGRESS.md` for the subsequent hardware probe.

## Assessment

Kyanite has substantial host-tested context, IR, compiler, direct-KFD resource,
model-format and planning infrastructure. The README understates some compiler
progress: straight-line bringup kernels compile, but `inference-plan` explicitly
does not execute graphs, and the target reports software matrix execution as
unimplemented. It is not yet a complete model inference engine.

The historical 2026-10-11 hardware probe discovered gfx1032 and passed physical
VM acquisition and three GTT allocation/map/readback/cleanup cycles. AQL queue
creation then failed with `AMDKFD_IOC_CREATE_QUEUE: EINVAL`. No kernel was
dispatched. Memory lifecycle timing is not GPU throughput. The first engineering
gate is to diagnose queue parameters against the actual driver contract, then
prove bounded repeated kernel dispatch and cleanup. The error alone does not
identify the faulty parameter.

The native engine's latest documented canonical four-token fixture is 3.50
generated tokens/s. The earlier 4.82 Q4 result predates canonical tokenization.
Neither is physical-128K evidence. The latest baseline log found during setup
ends during prefill after its 49152-token progress marker; JSONL contains metadata
only and lifecycle records contain starts without an exit. Completion and exit
reason remain unresolved. This terminal has an isolated process view, so it
cannot establish whether that host worker is alive.

The initial restricted terminal could not see GPU devices. Host access is now
enabled: `/dev/kfd` and both DRM render nodes are accessible, and the native HIP
device query identifies an RX 6650 XT with gfx1032 and 7.98 GiB VRAM. A new direct
KFD queue probe passed three physical lifecycle repetitions. The production
dispatcher still needs the proven mapping/save-area arrangement integrated;
kernel execution and model inference in Kyanite remain unproven.

The 15.33 GiB target weights exceed the card's approximately 8 GiB VRAM before
KV state and drafter storage. Weight residency, transfers, CPU/GPU partitioning,
prefill batching and exact speculative verification will matter. A new runtime
alone is not evidence of a speedup. No 10 TPS feasibility guarantee is made.

## Development sequence

1. Run the host checks and inspect source provenance. Resolve direct-KFD queue
   creation on the real device, then exercise repeated empty/write-index/vector
   add dispatch with bounded waits, readback and teardown.
2. Establish real baseline generation with the existing native engine and the
   pinned llama.cpp reference using identical assets and documented settings.
   Keep 128K certification separate from practical short/medium prompt tests.
3. Implement missing Kyanite execution and model operators. Use the port ledger
   to select reference algorithms; preserve typed contexts, owned resources and
   gfx1032 behavior in its target leaf.
4. Gate each optimization on independent numerical fixtures and actual device
   measurements. Promote only measured winners. Keep failures and regressions.
5. Run the final protocol in `BENCHMARK_CONTRACT.md`. A failed or incomplete run
   remains a shortfall, even when a kernel or burst exceeds 10 tokens/s.

Kyanite's `AGENTS.md` governs its sources. Its runtime stays independent of HIP,
HSA userspace, LLVM AMDGPU, COMGR, rocBLAS, MIOpen, PyTorch and llama.cpp. Reference
code may inform ports without introducing those runtime dependencies. There are
no engine algorithm ports in this setup change.

## Commands

From this directory:

```sh
python3 scripts/doctor.py
bash scripts/check-host.sh
```

`doctor.py --require-gpu` returns a failure when the required device nodes are
unavailable. `check-host.sh` uses offline Cargo, new output directories, saved
logs, host-only tests and a small compiler job limit. These checks do not start
GPU workloads. Native host tests default to `clang++`, matching the native
engine's Clang build assumptions; `QWEN_HOST_CXX` can select another compatible
compiler. Initial provenance is saved in `results/setup/provenance.json`.

After the native HIP build, `bash scripts/run-chat-diagnostic.sh` runs the saved
real request with a 360-second bound and retains hashes, raw output and wrapper
exit status even on timeout. Optional arguments are context capacity, requested
resident GiB and timeout seconds. This diagnostic has no final-goal success
classifier. Use the existing journaled benchmark driver for final certification.

For the repaired native HIP engine:

```sh
cmake -S sources/native-engine -B build/native-engine -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/home/alex/TheRock/build/dist/rocm \
  -DCMAKE_CXX_COMPILER=/home/alex/TheRock/build/dist/rocm/bin/hipcc
cmake --build build/native-engine --parallel 2
```

Configure a fresh llama.cpp comparison build from the pinned source using its
`docs/build.md` HIP instructions and `GPU_TARGETS=gfx1032`. Record all options and
the resulting executable hash. Existing external binaries are not assumed to
match this source pin. Do not overlap GPU comparisons or CPU performance tests
with a generation benchmark.

The diagnostic runner defaults to `build/native-engine/qwen38-native`.
`QWEN_NATIVE_DIAGNOSTIC_BINARY` can select the preserved baseline explicitly.
Never edit a shell runner while its foreground request is executing.

The completed timeout/prefill repair results are in
[`TIMEOUT_PREFILL_RESULTS.md`](TIMEOUT_PREFILL_RESULTS.md). The same 360-second
request now completes with its exit record; physical corpus prefill and exact
feature/state parity are recorded separately from generated-token throughput.

The next controlled decode measurements and rejected staging experiments are
in [`DECODE_PROGRESS.md`](DECODE_PROGRESS.md). Run
`bash scripts/run-real-benchmark.sh baseline 256` for the fixed real-prompt
diagnostic; it never classifies short-context results as final-goal success.

The primary-source literature shortlist and concrete experimental gates are in
[`RESEARCH_IDEAS.md`](RESEARCH_IDEAS.md). NVIDIA kernel speedups are not assumed
to transfer to gfx1032, and approximate prompt/KV skipping is excluded.

Model/head experimentation was explicitly authorized on 2026-10-11. The added
track, selector diagnostics and measured ablations are documented in
[`MODEL_EXPERIMENTS.md`](MODEL_EXPERIMENTS.md); original models remain intact.

FFN attribution, a failed down-shard parity gate and the independent async-copy
replay are in [`FFN_SCHEDULING.md`](FFN_SCHEDULING.md). Both scheduling candidates
remain off; instrumentation results are not presented as throughput wins.

The extra-cache comparison is in [`GUARDED_CACHE_RESULTS.md`](GUARDED_CACHE_RESULTS.md).
The genuine 32-position projection/FFN prefill prototype and its gates are in
[`WIDE_PREFILL_PROGRESS.md`](WIDE_PREFILL_PROGRESS.md). Both default off until
their applicable correctness and performance evidence supports promotion.

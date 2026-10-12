# Matrix gate provenance

Original executable and source archives remain in ignored
`results/kyanite-matrix-gBcSHM`; this directory retains compact raw evidence.
Interpretation and scope: `../../KYANITE_MATRIX_PROGRESS.md`.

Build from the outer workspace:

```sh
CARGO_TARGET_DIR=/home/alex/TheRock-qwen38/engine-workspace/build/kyanite-host \
CARGO_BUILD_JOBS=2 cargo build --locked --offline -p kyanite-bringup \
  --example hardware_dispatch --manifest-path engine-workspace/sources/kyanite/Cargo.toml
```

Run phases separately in order: queue, noop, fill, vector-add, reduction, matrix:

```sh
timeout --signal=TERM --kill-after=5s 30s RESULTS/executable \
  /sys/class/kfd/kfd/topology /dev/kfd /dev/dri 4096 PHASE
```

Each process performs three repetitions. Wrapper and tee logger exits are
recorded from PIPESTATUS. Kernel journals were inspected after every phase
from `start.txt`; any failure or fault would stop progression. After commit,
the rebuilt executable was byte-compared with the frozen one, and matrix ran
again; see `matrix-committed*`, `committed-start.txt` and its journal.

`inputs.sha256` identifies the earlier working snapshot; `committed-inputs.sha256`
identifies `git archive HEAD`. They are distinct archives; neither is silently
substituted for the other. `committed-revision.txt` gives the code revision.

The independent development oracle command was:

```sh
printf 'v_mov_b32 v3, 0x100\nv_mov_b32 v255, 0xdeadbeef\n' | \
  /home/alex/TheRock/build/dist/rocm/llvm/bin/llvm-mc \
    -triple=amdgcn-amd-amdhsa -mcpu=gfx1032 --show-encoding
```

Version/hash/output and linked-library records are retained. LLVM is not
invoked by Kyanite. `host-checks` copies the final complete exit-zero run from
`results/host-checks-A1ZWkp`; its doctor names the parent commit because the
tested changes had not yet been committed. The earlier full host run is also
retained under `results/host-checks-YZPvFl`. No host log measures inference TPS.

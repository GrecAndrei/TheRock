# Reduction gate provenance

Original artifacts are retained under `results/kyanite-reduction-NCqDIM`.
This directory contains compact raw logs, exits, journals and hashes; binaries
and source archives remain in ignored results storage.

Build from the outer workspace:

```sh
CARGO_TARGET_DIR=/home/alex/TheRock-qwen38/engine-workspace/build/kyanite-host \
CARGO_BUILD_JOBS=2 cargo build --locked --offline -p kyanite-bringup \
  --example hardware_dispatch --manifest-path engine-workspace/sources/kyanite/Cargo.toml
```

Queue, no-op, fill, vector-add and reduction each ran separately in that order
on the frozen executable, with this command shape:

```sh
timeout --signal=TERM --kill-after=5s 30s RESULTS/executable \
  /sys/class/kfd/kfd/topology /dev/kfd /dev/dri 4096 PHASE
```

Each process performs three repetitions. Both wrapper and tee logger exits
were captured from PIPESTATUS. Kernel journals were inspected after every
phase from `start.txt`; failure would stop progression. After committing
`083c4e28f834107c6450b06320359d85decf18d0`, the build retained the exact same
binary hash and reduction ran again; see `reduction-committed*`,
`committed-start.txt` and `kernel-after-committed.log`. All exits are zero.

`inputs.sha256` records the working-source snapshot taken before final docs;
`committed-inputs.sha256` records `git archive HEAD` after commit. Both original
archives remain available. No claim that their hashes are identical is made.

The external oracle command was:

```sh
printf 'v_xor_b32 v3, v4, v5\n' | \
  /home/alex/TheRock/build/dist/rocm/llvm/bin/llvm-mc \
    -triple=amdgcn-amd-amdhsa -mcpu=gfx1032 --show-encoding
```

Oracle version/hash and output are retained. LLVM was not invoked by the
hardware gate and does not appear in linked libraries.

`host-checks` comes from `results/host-checks-XaAKgQ`, the final full
`bash engine-workspace/scripts/check-host.sh` run, completed exit zero. Its
doctor metadata names the parent commit because tested edits were not committed
yet. No host log is treated as inference speed or broad GPU numerical evidence.
Interpretation: `../../KYANITE_REDUCTION_PROGRESS.md`.

# Raw direct-KFD checkpoint evidence

`committed` records Kyanite commit
`709a5f491b82e616b87dabbdb8e5f80410951889` on 2026-10-12.
The exact executable/source hashes, raw kernel bytes, completion/readback/
teardown messages, wrapper/logger exits and phase journals are retained.
The original executable and source archive remain in ignored
`results/kyanite-committed-dispatch-WB59P4`; these records do not contain binaries.

Build command, from the outer workspace:

```sh
CARGO_TARGET_DIR=/home/alex/TheRock-qwen38/engine-workspace/build/kyanite-host \
CARGO_BUILD_JOBS=2 cargo build --locked --offline -p kyanite-bringup \
  --example hardware_dispatch --manifest-path engine-workspace/sources/kyanite/Cargo.toml
```

The frozen executable was invoked separately for `queue`, `noop`, `fill` and
`vector-add`, in that order:

```sh
timeout --signal=TERM --kill-after=5s 30s RESULTS/executable \
  /sys/class/kfd/kfd/topology /dev/kfd /dev/dri 4096 PHASE
```

Output was piped through `tee`, with both `PIPESTATUS` values recorded. The
kernel journal was captured after each phase from the recorded start timestamp.
Failure of either command, or a relevant new fault/reset, stopped progression.
All four committed phases returned wrapper/logger zero.

`host-checks` is the full `bash engine-workspace/scripts/check-host.sh` run from
`results/host-checks-iDpdnq`, which completed exit zero. Its doctor observation
records the then-current parent commit; source changes had not yet been
committed. The tests covered the source subsequently committed as `709a5f4`.

Other directories retain incremental failed and successful silicon attempts.
The wrong-root vector-add failure is a boundary-input failure, not a GPU timeout;
both it and the corrected invocation are retained. See
`../../KYANITE_DISPATCH_PROGRESS.md` from this directory for interpretation.
No timing here is an inference TPS benchmark or 128K validation.

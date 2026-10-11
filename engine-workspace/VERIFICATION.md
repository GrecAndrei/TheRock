# Workspace setup verification, 2026-10-11

This is host correctness evidence. No model inference, hardware dispatch,
inference comparison or tokens/s measurement was performed.

- Kyanite formatting passed.
- Offline, locked, all-target/all-feature Clippy passed with warnings denied.
- Offline, locked workspace/all-feature Rust tests passed: **549 passed,
  0 failed, 0 ignored**, including three compile-fail documentation tests.
  Logs: `results/host-checks-XDFDn6/cargo-test.log` and `cargo-clippy.log`.
- All four native host test programs passed using Clang. The projection suite
  includes **285 bit-exact fixtures**. The remaining programs cover runtime
  flags, attention staging/causality and worker concurrency/lifetime behavior.
- Python lifecycle suite: **8 passed**. Native logs are under
  `results/native-checks-nyLj0v`.
- Copied native sources, llama.cpp reference files and both frozen fixtures
  matched every recorded SHA256 in the initial provenance manifest.
- Original source diffs and original repository status matched their setup
  records after the checks. No original pending edit was changed.
- `doctor.py --require-gpu` returned **2**, correctly reporting unavailable KFD
  and DRM render nodes. Model files exist; their full hashes were not recomputed
  during setup. No hardware identity or model performance certification follows.

The first Clippy attempt found a pre-existing path debug-formatting warning in
the copied `hardware_smoke.rs`. Only its display formatting was corrected in
the new Kyanite clone; the original and historical log remain intact. The
initial source manifest predates this one display-only change. The failed
Clippy output remains under `results/host-checks-SOKAhb`.

The first native build attempt used system GCC and stopped on Clang-specific
unroll pragmas and its return-path analysis. No native source was altered to
accommodate it. The workspace runner now defaults to Clang, matching the native
engine's build assumptions, and all native host checks then passed. The full
runner was not rerun after this compiler selection change: Rust gates passed
in the recorded workspace run and native gates passed in the separate targeted
run above. Shell syntax checks passed after the runner update.

Network fetches failed because the terminal could not resolve GitHub. The
llama.cpp checkout is the preserved local DFlash2 revision, not a fresh upstream
pull. Upstream refresh and real GPU execution remain outstanding.

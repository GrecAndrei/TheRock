# gfx1032 (RDNA2 / RX 6650 XT) hipBLASLt Support — PR Plan

## Summary

This document organizes the work that enables hipBLASLt GEMM on gfx1032
(AMD Radeon RX 6650 XT / Navi 23 / RDNA2) and related build fixes, into
reviewable pull requests with clear upstream targets.

**Result:** hipBLASLt-bench now loads the gfx1032 Tensile library and runs
real GEMM on the GPU — up to **28.9 TFLOPS FP16** (2048³) and **26.5 TFLOPS
BF16** (4096³), verified on an RX 6650 XT.

---

## Repository Structure

TheRock is a monorepo that vendors sources via git submodules:

| Submodule | Upstream repo | Fork |
|-----------|--------------|------|
| `rocm-libraries` | `ROCm/rocm-libraries` (branch `develop`) | `GrecAndrei/rocm-libraries` |
| `compiler/amd-llvm` | `ROCm/llvm-project` | (not forked — left unchanged) |
| Outer `TheRock` | `ROCm/TheRock` | `GrecAndrei/TheRock` |

**Where to PR:**
- **hipBLASLt / TensileLite / stinkytofu code changes** → `ROCm/rocm-libraries`
- **Build infrastructure / inference tools** → `ROCm/TheRock`

---

## Commits in `rocm-libraries` submodule (→ PR to ROCm/rocm-libraries)

### PR branches (pushed to `GrecAndrei/rocm-libraries` fork):

| PR | Branch | Upstream URL | Risk |
|----|--------|-------------|------|
| **PR-A** | `pr-a/fix-amdgpu-processor-enum` | `ROCm/rocm-libraries:develop` ← `GrecAndrei/rocm-libraries:pr-a/fix-amdgpu-processor-enum` | 🟢 Safe |
| **PR-B** | `pr-b/register-gfx1032-arch` | `ROCm/rocm-libraries:develop` ← `GrecAndrei/rocm-libraries:pr-b/register-gfx1032-arch` | 🟢 Safe |
| **PR-C** | `pr-c/cblas-fallback` | `ROCm/rocm-libraries:develop` ← `GrecAndrei/rocm-libraries:pr-c/cblas-fallback` | 🟡 Fixed |
| PR-D | `gfx103-wmma-experiments` | (stays in fork — experimental) | 🔴 High |

### PR-A: Critical bugfix (2 commits, +42 lines)
> stinkytofu `#include <cstdint>` + AMDGPU::Processor enum fix (gfx1031/1032/1034/1035).
>
> This is the root-cause fix that unblocks **all** RDNA2 consumer GPUs.
> Without it, `TensileLibrary_lazy_gfx103*.dat` silently fails to load with
> `Enum not found! gfx1032`.
>
> **Status: Ready to submit.** Clean, small, follows existing patterns.

### PR-B: Architecture registration (1 commit, +6 lines, depends on PR-A)
> Adds gfx1032 to Python `Architectures.py` + CMake `tensilelite_supported_architectures.cmake`.
>
> **Status: Ready to submit** (mark as depending on PR-A).

### PR-C: cblas fallback (1 commit, +24 lines, independent)
> Opt-in fallback (`-DHIPBLASLT_ENABLE_CBLAS_FALLBACK=ON`) for bench client
> when BLIS is unavailable. Uses pkg-config first, then find_library.
> No hardcoded paths. Emits WARNING if flag set but cblas not found.
>
> **Status: Ready to submit.** Fixed from original — now opt-in, no hardcoded paths.

### PR-D: QuickTuning + generated Tensile logic (experimental, stays in fork)
> QuickTuning runtime + generated gfx1032 Tensile YAML (301k lines).
> Now gated behind `HIPBLASLT_ENABLE_QUICKTUNING=OFF` by default —
> zero overhead when disabled, no code in the hot path.
>
> **Status: NOT ready for upstream.** Still needs:
> - Run the GSU/WGM sweep to populate `tuning_gfx1032.json` with real data
> - Move `test_quicktuning.cpp`/`bench_gsuwgm.cpp` to `clients/`
> - Consider using existing UserDrivenTuning mechanism instead of parallel system
> - Add generation provenance for the 301k YAML lines

---

## Commits in outer `TheRock` repo (→ PR to ROCm/TheRock)

Branch: `users/grecandrei/gguf-tune`

| # | Commit | Files | Lines |
|---|--------|-------|-------|
| 1 | `018f8cb` TheRock: build infrastructure (inference subdir, client OFF, elfutils) | 3 | +3 |
| 2 | `dc849a6` inference: enhance GGUF runner + Tensile config generator | 2 | +284/-58 |
| 3 | `dc5e0e3` rocm-libraries: bump submodule pointer | 1 | +1/-1 |

### Suggested PR for TheRock

**PR-E (TheRock monorepo integration):**
> All three outer commits — wires up the inference subproject, sets
> `HIPBLASLT_ENABLE_CLIENT=OFF` by default, fixes elfutils build, enhances
> the GGUF runner / Tensile config generator, and bumps the rocm-libraries
> submodule to include PR-A through PR-D.
>
> Note: depends on the rocm-libraries PRs being merged first (the submodule
> bump references the new submodule commits).
>
> Target: `ROCm/TheRock`

---

## Sub-byte precision testing results (FP4 / BF4 / FP8 / 2-bit)

Tested on RX 6650 XT (gfx1032) via hipBLASLt-bench:

| Format | Bench accepts? | Kernels for gfx1032? | Why |
|--------|---------------|---------------------|-----|
| **FP8** (f8_r E5M2, bf8_r E4M3) | ✅ Yes (compute_input_type) | ❌ No solution | FP8 matrix units are gfx940+ (MI300) only |
| **FP8_fnuz** (f8_fnuz_r, bf8_fnuz_r) | ✅ Yes | ❌ No solution | Same — needs gfx940+ |
| **FP4** (f4_r / E2M1 / mxfp4) | ✅ Yes (recognized) | ❌ Init kernel missing + no GEMM kernels | MX matrix instructions are gfx950 (MI350) only |
| **FP6** (f6_r) | ✅ Yes (recognized) | ❌ Same as FP4 | gfx950 only |
| **BF6** (bf6_r) | ✅ Yes (recognized) | ❌ Same | gfx950 only |
| **2-bit** (nf2 / q2) | ❌ Not a ROCm datatype | N/A | 2-bit quant is handled by inference frameworks (llama.cpp/GGUF), not BLAS |

**Conclusion:** gfx1032 (RDNA2 consumer) **cannot** run sub-byte GEMM in
hipBLASLt because it lacks the native MX/FP8 matrix-multiply instructions.
Those require CDNA datacenter GPUs:
- FP8 (E5M2/E4M3): gfx940 / gfx941 / gfx942 (MI300 series)
- FP4/FP6/BF6 + block-scaled MX (E2M1, E2M3, E3M2): gfx950 (MI350 series)

For 2-bit/4-bit quantized inference on RDNA2, use the GGUF/llama.cpp path
(dequantize → FP16 → WMMA GEMM), not native sub-byte BLAS.

---

## Best precision on RX 6650 XT

| Precision | Best measured | Hardware path |
|-----------|--------------|---------------|
| **FP16** | 28.9 TFLOPS (2048³) | WMMA matrix units |
| **BF16** | 26.5 TFLOPS (4096³) | WMMA matrix units |
| FP32 | ~0.7 TFLOPS | Scalar ALUs (dual-issue FMA) |
| FP8/FP4/2-bit | N/A (no native HW) | — |

**FP16 and BF16 are tied as best** — both use the RDNA2 WMMA engines at equal
throughput. Use BF16 for training (wider dynamic range), FP16 for inference.

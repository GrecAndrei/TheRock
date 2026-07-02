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

Branch: `gfx103-wmma-experiments`

| # | Commit | Files | Lines | Priority | Standalone? |
|---|--------|-------|-------|----------|-------------|
| 1 | `e9ec5ad` stinkytofu: add missing `#include <cstdint>` | 10 | +10 | High | ✅ Yes |
| 2 | `4fee212` tensilelite: add gfx1031/1032/1034/1035 to `AMDGPU::Processor` enum | 2 | +32 | **Critical** | ✅ Yes |
| 3 | `36b6491` hipblaslt: register gfx1032 in supported architectures | 2 | +6 | High | Depends on #2 |
| 4 | `613f346` hipblaslt: fall back to system cblas when BLIS unavailable | 1 | +10 | Medium | ✅ Yes |
| 5 | `5b1a9a9` hipblaslt: add QuickTuning infrastructure | 10 | +953/-42 | Medium | Experimental |
| 6 | `4b45bf6` hipblaslt: add generated gfx1032 Tensile logic YAML | 46 | +301k | Low | Mechanical |

### Suggested PR split for rocm-libraries

**PR-A (Critical bugfix — highest priority, smallest, most reviewable):**
> Commits #1 + #2 — stinkytofu `#include <cstdint>` + AMDGPU::Processor enum fix.
>
> This is the one-line-root-cause fix that unblocks **all** RDNA2 consumer GPUs
> (gfx1031 RX 6700 XT, gfx1032 RX 6600/6650 XT, gfx1034, gfx1035). Without it,
> `TensileLibrary_lazy_gfx103*.dat` silently fails to load with
> `Enum not found! gfx1032`.
>
> Target: `ROCm/rocm-libraries` → `projects/hipblaslt/tensilelite/`

**PR-B (Architecture registration):**
> Commit #3 — adds gfx1032 to the Python + CMake arch lists so Tensile generates
> kernels for it. Logically depends on PR-A but is a separate concern.
>
> Target: `ROCm/rocm-libraries` → `projects/hipblaslt/`

**PR-C (Bench client build fix):**
> Commit #4 — lets the bench client link against system libcblas when BLIS/AOCL
> is not installed. Useful for distros that ship reference BLAS but not AOCL.
>
> Target: `ROCm/rocm-libraries` → `projects/hipblaslt/clients/`

**PR-D (QuickTuning + generated Tensile logic — experimental):**
> Commits #5 + #6 — the QuickTuning runtime feature and the generated gfx1032
> Tensile YAML logic files. This is the larger, experimental work.
>
> Target: `ROCm/rocm-libraries` → `projects/hipblaslt/library/`

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

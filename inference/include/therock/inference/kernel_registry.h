// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Kernel registry: maps (op, hardware, exec_mode) → kernel implementation.
// The registry owns the dispatch decision between prefill and decode paths.

#pragma once

#include "therock/inference/hardware.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Execution mode — the single most important dispatch axis.
// Prefill: compute-bound, large M, use GEMM.
// Decode:  bandwidth-bound, M=1..few, use GEMV or fused dequant-GEMV.
typedef enum TheRockExecMode {
  THEROCK_EXEC_PREFILL = 0,
  THEROCK_EXEC_DECODE  = 1,
} TheRockExecMode;

// Data layout for weight tensors.
typedef enum TheRockWeightLayout {
  THEROCK_LAYOUT_ROW_MAJOR  = 0,  // standard row-major fp16
  THEROCK_LAYOUT_Q4_0       = 1,  // llama.cpp Q4_0: 32-element blocks, fp16 scale
  THEROCK_LAYOUT_Q4_R       = 2,  // our format: WMMA thread-native, WUSH-transformed
} TheRockWeightLayout;

// Opaque kernel handle — implementation is backend-specific.
typedef struct TheRockKernel TheRockKernel;

// Operation descriptor passed to the registry for lookup.
typedef struct TheRockOpDesc {
  // What operation
  enum { THEROCK_OP_GEMM, THEROCK_OP_GEMV, THEROCK_OP_ATTN } op;

  // Problem dimensions
  uint32_t M, N, K;
  uint32_t batch;

  // Data types
  TheRockWeightLayout weight_layout;

  // Execution context
  TheRockExecMode     exec_mode;
} TheRockOpDesc;

// Kernel function pointer types.
// All kernels take (stream, desc, A, B, C, D) where:
//   A = activations (fp16, [M x K])
//   B = weights     (layout per weight_layout, [N x K] row-major logical)
//   C = bias/residual or NULL
//   D = output      (fp16, [M x N])
typedef void (*TheRockGemmFn)(void *stream, const TheRockOpDesc *desc,
                               const void *A, const void *B,
                               const void *C, void *D);

typedef void (*TheRockGemvFn)(void *stream, const TheRockOpDesc *desc,
                               const void *x, const void *W,
                               const void *bias, void *y);

typedef union TheRockKernelFn {
  TheRockGemmFn gemm;
  TheRockGemvFn gemv;
} TheRockKernelFn;

// Registry handle.
typedef struct TheRockKernelRegistry TheRockKernelRegistry;

TheRockKernelRegistry *therock_registry_create(const TheRockHardwareDesc *hw);
void                   therock_registry_destroy(TheRockKernelRegistry *reg);

// Look up the best kernel for an operation. Returns NULL if no kernel found.
// The returned fn is valid for the lifetime of the registry.
TheRockKernelFn therock_registry_lookup(const TheRockKernelRegistry *reg,
                                         const TheRockOpDesc *desc);

// Threshold: M <= this → dispatch GEMV instead of GEMM.
// Exposed so the executor can make the same decision without calling lookup.
uint32_t therock_decode_threshold(const TheRockHardwareDesc *hw);

#ifdef __cplusplus
}
#endif

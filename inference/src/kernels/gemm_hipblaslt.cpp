// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Prefill GEMM via hipBLASLt — dispatches to TensileLite-tuned kernels.
// The tuned kernel selection for gfx1032 Gemma 4 shapes lives in the
// hipBLASLt library logic built by TensileLite benchmarking.

#include "therock/inference/kernel_registry.h"

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>
#include <cassert>
#include <cstdio>

// One hipBLASLt handle per process — created lazily, never destroyed.
// Thread safety: handle creation is protected by HIP's internal serialization.
static hipblasLtHandle_t s_handle = nullptr;

static hipblasLtHandle_t get_handle() {
  if (!s_handle)
    hipblasLtCreate(&s_handle);
  return s_handle;
}

extern "C" void therock_gemm_hipblaslt(hipStream_t stream,
                                        const TheRockOpDesc *desc,
                                        const void *A, const void *B,
                                        const void *C, void *D) {
  hipblasLtHandle_t handle = get_handle();

  hipblasLtMatmulDesc_t   matmul_desc;
  hipblasLtMatrixLayout_t layout_A, layout_B, layout_C, layout_D;
  hipblasLtMatmulPreference_t pref;

  int M = (int)desc->M, N = (int)desc->N, K = (int)desc->K;

  // A: [M x K] fp16, row-major (activations, not transposed)
  // B: [K x N] fp16, col-major = [N x K] row-major transposed
  // D: [M x N] fp16, row-major
  hipblasOperation_t transa = HIPBLAS_OP_N;
  hipblasOperation_t transb = HIPBLAS_OP_T;

  hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
  hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                   &transa, sizeof(transa));
  hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                   &transb, sizeof(transb));

  hipblasLtMatrixLayoutCreate(&layout_A, HIP_R_16F, M, K, K);
  hipblasLtMatrixLayoutCreate(&layout_B, HIP_R_16F, N, K, K);
  hipblasLtMatrixLayoutCreate(&layout_C, HIP_R_16F, M, N, N);
  hipblasLtMatrixLayoutCreate(&layout_D, HIP_R_16F, M, N, N);

  hipblasLtMatmulPreferenceCreate(&pref);
  size_t workspace_size = 32 * 1024 * 1024;  // 32MB scratch
  hipblasLtMatmulPreferenceSetAttribute(pref,
    HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
    &workspace_size, sizeof(workspace_size));

  hipblasLtMatmulHeuristicResult_t heuristic;
  int returned = 0;
  hipblasLtMatmulAlgoGetHeuristic(handle, matmul_desc,
                                   layout_A, layout_B, layout_C, layout_D,
                                   pref, 1, &heuristic, &returned);

  const float alpha = 1.0f, beta = (C != nullptr) ? 1.0f : 0.0f;

  // Workspace — allocate once per unique size via a simple bump allocator.
  // TODO: wire into the memory subsystem properly.
  static void *s_workspace = nullptr;
  static size_t s_workspace_size = 0;
  if (s_workspace_size < workspace_size) {
    if (s_workspace) hipFree(s_workspace);
    hipMalloc(&s_workspace, workspace_size);
    s_workspace_size = workspace_size;
  }

  hipblasLtMatmul(handle, matmul_desc,
                  &alpha,
                  A, layout_A,
                  B, layout_B,
                  &beta,
                  C ? C : D, layout_C,
                  D, layout_D,
                  returned ? &heuristic.algo : nullptr,
                  s_workspace, workspace_size,
                  stream);

  hipblasLtMatmulPreferenceDestroy(pref);
  hipblasLtMatrixLayoutDestroy(layout_A);
  hipblasLtMatrixLayoutDestroy(layout_B);
  hipblasLtMatrixLayoutDestroy(layout_C);
  hipblasLtMatrixLayoutDestroy(layout_D);
  hipblasLtMatmulDescDestroy(matmul_desc);
}

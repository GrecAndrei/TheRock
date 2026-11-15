// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rxinfer/arena.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace rxinfer::ops {

// RMSNorm: y[m, d] = x[m, d] * inv_rms(x[m]) * (plus_one ? (1.0f + w[d]) : w[d])
void rmsnorm(const __half* x, const __half* weight, __half* y,
             int M, int D, float eps, bool plus_one,
             hipStream_t stream = nullptr);

// Single-pass Residual Add + RMSNorm (Mega-mHC fusion principle):
//   residual[m, d] += x[m, d]
//   y[m, d] = residual[m, d] * inv_rms(residual[m]) * (plus_one ? (1.0f + w[d]) : w[d])
void fused_residual_rmsnorm(const __half* x, __half* residual, const __half* weight, __half* y,
                            int M, int D, float eps, bool plus_one,
                            hipStream_t stream = nullptr);

// Fused SwiGLU: out[m, i] = silu(gate[m, i]) * up[m, i]
void fused_silu_mul(const __half* gate, const __half* up, __half* out,
                    int M, int I, hipStream_t stream = nullptr);

// Fused SwiGLU for vertically packed [gate_proj; up_proj] -> gate_up[M, 2*I]:
//   out[m, i] = silu(gate_up[m, i]) * gate_up[m, I + i]
void fused_silu_mul_packed(const __half* gate_up, __half* out,
                           int M, int I, hipStream_t stream = nullptr);

// Embedding lookup (FP16 table [V, H])
void embed_lookup_f16_step(const __half* table, const StepParams* d_step, __half* out,
                           int V, int H, hipStream_t stream = nullptr);
void embed_lookup_f16_batch(const __half* table, const int32_t* d_token_ids, __half* out,
                            int M, int V, int H, hipStream_t stream = nullptr);

// Embedding lookup directly from R4_GFX12_G64 compressed table (saves ~1.8 GB VRAM on 248k vocab!)
void embed_lookup_r4_step(const void* W_qs, const __half* W_scales, const StepParams* d_step,
                          __half* out, int V, int H, hipStream_t stream = nullptr);
void embed_lookup_r4_batch(const void* W_qs, const __half* W_scales, const int32_t* d_token_ids,
                           __half* out, int M, int V, int H, hipStream_t stream = nullptr);

// Small/dense FP16 Linear: C[M, N] = A[M, K] * W[N, K]^T (e.g., GDN in_proj_ba [96, 5120] or F16 fallback)
void gemm_f16(const __half* A, const __half* W, __half* C,
              int M, int N, int K, hipStream_t stream = nullptr);

// Strided 2D FP16 copy: dst[r, c] = src[r * src_stride + c] for r in [0..rows), c in [0..cols).
void copy_strided_f16(const __half* src, int src_stride,
                      __half* dst, int dst_stride,
                      int rows, int cols, hipStream_t stream = nullptr);

// Device-side greedy argmax over logits[V] or batch logits[M, V]
size_t argmax_tmp_bytes(int max_M = 1);
void argmax_sample(const __half* logits, int V, void* tmp_buf,
                   int32_t* d_out_token, StepParams* d_step_optional = nullptr,
                   hipStream_t stream = nullptr);
void argmax_sample_batch(const __half* logits, int M, int V, void* tmp_buf,
                         int32_t* d_out_tokens, hipStream_t stream = nullptr);

} // namespace rxinfer::ops

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstddef>

namespace rxinfer::quant {

// Standard GGUF Q4_0 block (18 bytes per 32 weights -> 4.5 bpw)
struct __attribute__((packed)) BlockQ4_0 {
    uint16_t d;      // IEEE FP16 scale
    uint8_t qs[16];  // low nibbles: 0..15, high nibbles: 16..31 (stored as q + 8)
};
static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be 18 bytes");

// Host packers (W is row-major [N, K])
void pack_gguf_q4_0(const float* src_f32, void* dst_blocks, int N, int K);

// Custom gfx1201-native R4_GFX12_G64 (4.25 bpw, 256B-aligned tiles):
//   qs:     [N/16, K/64, 32] uint4 (512 bytes per 16x64 tile)
//   scales: [N/16, K/64, 16] __half (32 bytes per 16x64 tile)
size_t r4_g64_qs_bytes(int N, int K);
size_t r4_g64_scales_bytes(int N, int K);
size_t r4_g64_tmp_bytes(int max_M, int max_N);
void pack_r4_gfx12_g64(const float* src_f32, void* dst_qs, void* dst_scales, int N, int K);

// GEMM / GEMV launchers: C[M, N] = A[M, K] * W[N, K]^T (A in FP16, C in FP16)
void gemm_gguf_q4_0(const __half* A, const void* W_blocks, __half* C, float* tmp_f32,
                    int M, int N, int K, hipStream_t stream = nullptr);

void gemm_r4_gfx12_w4a16(const __half* A, const void* W_qs, const __half* W_scales,
                         __half* C, float* tmp_f32,
                         int M, int N, int K, hipStream_t stream = nullptr);

// Fused small-batch (1 <= M <= 16) Gate+Up+SwiGLU: W is packed [2*I, K] (gate rows [0..I), up rows [I..2I)), C is [M, I].
// Requires (I & 15) == 0 and (K & 63) == 0.
void gemv_r4_gfx12_w4a16_swiglu(const __half* A, const void* W_qs, const __half* W_scales,
                                __half* C, int I, int K, hipStream_t stream = nullptr, int M = 1);

void gemm_r4_gfx12_fp8(const __half* A, const void* W_qs, const __half* W_scales,
                       __half* C, float* tmp_f32,
                       uint8_t* tmp_a_fp8, float* tmp_a_sc,
                       int M, int N, int K, hipStream_t stream = nullptr);

} // namespace rxinfer::quant

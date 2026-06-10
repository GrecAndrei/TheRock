// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// GEMV kernels for wave32 decode path (gfx103x, gfx11xx).
//
// Design principles:
//   - One wavefront (32 threads) per output row
//   - Each thread loads a contiguous 8-element chunk → coalesced 128-bit loads
//   - Reduction over K with __builtin_amdgcn_wave_reduce_add (single instruction)
//   - Q4_0 variant dequants on the fly — weights never land in VRAM as fp16
//
// This is a starting implementation. WUSH-transformed Q4_R will slot in here.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "therock/inference/kernel_registry.h"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int WAVE_SIZE    = 32;
static constexpr int VEC_WIDTH    = 8;   // fp16 elements per thread per step
static constexpr int Q4_BLOCK     = 32;  // Q4_0 block size (elements)

// ---------------------------------------------------------------------------
// fp16 GEMV: y[n] = sum_k W[n,k] * x[k]
// Grid:  (N,)   — one block per output element
// Block: (32,)  — one wavefront
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(WAVE_SIZE)
kernel_gemv_fp16_wave32(int N, int K,
                         const __half * __restrict__ x,
                         const __half * __restrict__ W,
                         const __half * __restrict__ bias,
                         __half       * __restrict__ y) {
  int row = blockIdx.x;
  if (row >= N) return;

  const __half *w_row = W + (long)row * K;

  float acc = 0.0f;
  int tid = threadIdx.x;

  // Each thread handles every WAVE_SIZE-th group of VEC_WIDTH elements
  for (int k = tid * VEC_WIDTH; k < K; k += WAVE_SIZE * VEC_WIDTH) {
    // 128-bit load: 8 fp16 at once
    float4 wv = *reinterpret_cast<const float4 *>(w_row + k);
    float4 xv = *reinterpret_cast<const float4 *>(x     + k);

    const __half *wp = reinterpret_cast<const __half *>(&wv);
    const __half *xp = reinterpret_cast<const __half *>(&xv);

    for (int i = 0; i < VEC_WIDTH; i++)
      acc += __half2float(wp[i]) * __half2float(xp[i]);
  }

  // Wave reduction: sum across all 32 lanes
  acc += __shfl_xor(acc, 16);
  acc += __shfl_xor(acc, 8);
  acc += __shfl_xor(acc, 4);
  acc += __shfl_xor(acc, 2);
  acc += __shfl_xor(acc, 1);

  if (tid == 0) {
    float result = acc;
    if (bias) result += __half2float(bias[row]);
    y[row] = __float2half(result);
  }
}

// ---------------------------------------------------------------------------
// Q4_0 fused dequant-GEMV
//
// Q4_0 memory layout per block (32 elements):
//   [fp16 scale (2 bytes)] [16 bytes: 32 x 4-bit values packed]
// Total: 18 bytes per 32 elements = 4.5 bits/weight
//
// Each thread handles one Q4_0 block per iteration.
// 32 threads × 1 block = 32 elements per iteration step.
// ---------------------------------------------------------------------------

struct Q4_0Block {
  __half scale;
  uint8_t qs[16];  // 32 nibbles packed into 16 bytes
};

__global__ void __launch_bounds__(WAVE_SIZE)
kernel_gemv_q4_0_wave32(int N, int K,
                          const __half      * __restrict__ x,
                          const Q4_0Block   * __restrict__ W,  // [N * K/32] blocks
                          const __half      * __restrict__ bias,
                          __half            * __restrict__ y) {
  int row = blockIdx.x;
  if (row >= N) return;

  int num_blocks = K / Q4_BLOCK;
  const Q4_0Block *w_row = W + (long)row * num_blocks;

  float acc = 0.0f;
  int tid = threadIdx.x;

  // Each thread handles one block per iteration
  for (int blk = tid; blk < num_blocks; blk += WAVE_SIZE) {
    const Q4_0Block &b = w_row[blk];
    float scale = __half2float(b.scale);

    // Load 32 activations for this block
    const __half *xblk = x + blk * Q4_BLOCK;

    // Unpack 16 bytes → 32 nibbles, dequant, dot
    float block_acc = 0.0f;
    for (int i = 0; i < 16; i++) {
      uint8_t packed = b.qs[i];
      // Q4_0: values are offset by 8 (range -8..7)
      float w0 = scale * ((float)(packed & 0xF) - 8.0f);
      float w1 = scale * ((float)(packed >> 4)  - 8.0f);
      block_acc += w0 * __half2float(xblk[i * 2    ]);
      block_acc += w1 * __half2float(xblk[i * 2 + 1]);
    }
    acc += block_acc;
  }

  // Wave reduction
  acc += __shfl_xor(acc, 16);
  acc += __shfl_xor(acc, 8);
  acc += __shfl_xor(acc, 4);
  acc += __shfl_xor(acc, 2);
  acc += __shfl_xor(acc, 1);

  if (tid == 0) {
    float result = acc;
    if (bias) result += __half2float(bias[row]);
    y[row] = __float2half(result);
  }
}

// ---------------------------------------------------------------------------
// C wrappers (called from kernel_registry.cpp)
// ---------------------------------------------------------------------------

extern "C" void therock_gemv_fp16_wave32(hipStream_t stream,
                                          const TheRockOpDesc *desc,
                                          const void *x, const void *W,
                                          const void *bias, void *y) {
  dim3 grid(desc->N);
  dim3 block(WAVE_SIZE);
  hipLaunchKernelGGL(kernel_gemv_fp16_wave32, grid, block, 0, stream,
                     (int)desc->N, (int)desc->K,
                     (const __half *)x, (const __half *)W,
                     (const __half *)bias, (__half *)y);
}

extern "C" void therock_gemv_q4_0_wave32(hipStream_t stream,
                                          const TheRockOpDesc *desc,
                                          const void *x, const void *W,
                                          const void *bias, void *y) {
  dim3 grid(desc->N);
  dim3 block(WAVE_SIZE);
  hipLaunchKernelGGL(kernel_gemv_q4_0_wave32, grid, block, 0, stream,
                     (int)desc->N, (int)desc->K,
                     (const __half *)x, (const Q4_0Block *)W,
                     (const __half *)bias, (__half *)y);
}

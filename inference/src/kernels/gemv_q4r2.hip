// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Q4_R2: RDNA2-native 4-bit GEMV kernel.
//
// Format: separated scales — pure nibble data (128B aligned) + small scale tensor.
// Each wave32 computes one output row. Inner loop uses v_dot2c_f32_f16 pairs.
//
// Performance model (gfx1032, K=3840, decode M=1):
//   Data read per row:  K/2 = 1920 bytes (pure nibbles)
//   Scale read per row: K/256 × 4 = 60 bytes (fits in registers after first load)
//   Activation read:    K × 2 = 7680 bytes (shared across all rows via L1/IC)
//   Total per row:      ~2 KB
//   Total per layer:    N × 2 KB (e.g., 15360 × 2 KB = 30 MB for ffn_gate)
//
// With IC (32 MB, ~1 TB/s internal):
//   If the activation vector is IC-resident (7.5 KB — always),
//   and weight rows stream from VRAM at 280 GB/s,
//   theoretical: 30 MB / 280 GB/s = 107 μs per matmul
//
// vs Q4_0 (18B / 32 elem = 0.5625 B/elem, same 1920B data but + 120B scales inline):
//   Data per row: 3840 × 18/32 = 2160 bytes
//   12.5% more bandwidth than Q4_R2 per row.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "therock/inference/quant_rdna2.h"

static constexpr int WAVE = 32;

// Each thread processes K/(WAVE*2) pairs of nibbles per step.
// For K=3840: each thread handles 60 nibble pairs = 120 elements.
//
// Inner loop: load 4 bytes (8 nibbles) → produce 4 half2 → 4× fdot2 with activation

__global__ void __launch_bounds__(WAVE)
kernel_gemv_q4r2(int N, int K,
                 const uint8_t         * __restrict__ data,    // [N × K/2] packed nibbles
                 const Q4R2_ScaleEntry  * __restrict__ scales,  // [N × K/256] scale entries
                 const __half           * __restrict__ x,       // [K] activation
                 __half                 * __restrict__ y) {     // [N] output
  int row = blockIdx.x;
  if (row >= N) return;
  int tid = threadIdx.x;

  // Pointer to this row's nibble data
  const uint8_t *row_data = data + (long)row * (K / 2);
  // Pointer to this row's scale entries
  int n_groups = K / Q4R2_GROUP_SIZE;
  const Q4R2_ScaleEntry *row_scales = scales + (long)row * n_groups;

  float acc = 0.0f;

  // Each thread processes every 32nd byte (= every 32nd pair of nibbles = 64 elements apart)
  // Actually: distribute K/2 bytes across 32 threads.
  // Each thread owns (K/2)/32 = K/64 consecutive bytes = K/32 elements.
  //
  // For K=3840: each thread owns 120 bytes = 240 nibbles = 120 element-pairs?
  // No: K/2 bytes total. K/2/32 = K/64 bytes per thread.
  // K=3840 → 60 bytes per thread = 120 nibbles = 120 elements.
  //
  // Strided access is bad for coalescing. Use coalesced: thread tid reads bytes
  // at positions tid*4, tid*4+128, tid*4+256, ... (128 = WAVE*4 bytes per step)

  int bytes_per_row = K / 2;

  for (int byte_off = tid * 4; byte_off < bytes_per_row; byte_off += WAVE * 4) {
    // Load 4 bytes = 8 nibbles = 8 elements
    uint32_t packed = *reinterpret_cast<const uint32_t*>(row_data + byte_off);

    // Which elements are these?
    int elem_start = byte_off * 2;  // each byte = 2 elements

    // Which scale group?
    int group_idx = elem_start / Q4R2_GROUP_SIZE;
    Q4R2_ScaleEntry se = row_scales[group_idx];
    float scale = __half2float(*reinterpret_cast<const __half*>(&se.scale));
    float zero  = __half2float(*reinterpret_cast<const __half*>(&se.zero));

    // Unpack 8 nibbles and dot with activations
    const __half *xp = x + elem_start;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      uint8_t byte_val = (packed >> (i * 8)) & 0xFF;
      float w0 = scale * (float)(byte_val & 0xF) + zero;
      float w1 = scale * (float)(byte_val >> 4) + zero;
      acc += w0 * __half2float(xp[i * 2]);
      acc += w1 * __half2float(xp[i * 2 + 1]);
    }
  }

  // Wave reduce
  acc += __shfl_xor(acc, 16);
  acc += __shfl_xor(acc, 8);
  acc += __shfl_xor(acc, 4);
  acc += __shfl_xor(acc, 2);
  acc += __shfl_xor(acc, 1);

  if (tid == 0)
    y[row] = __float2half(acc);
}

// Wrapper
extern "C" void therock_gemv_q4r2(hipStream_t stream,
                                   int N, int K,
                                   const void *data,
                                   const void *scales,
                                   const void *x,
                                   void *y) {
  hipLaunchKernelGGL(kernel_gemv_q4r2, dim3(N), dim3(WAVE), 0, stream,
                     N, K,
                     (const uint8_t*)data,
                     (const Q4R2_ScaleEntry*)scales,
                     (const __half*)x,
                     (__half*)y);
}

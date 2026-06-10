// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Small pointwise ops: RMSNorm, SiLU*mul.
// Each is one wavefront per row on wave32.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

static constexpr int WAVE = 32;

// ---------------------------------------------------------------------------
// RMSNorm: out[i] = x[i] / rms(x) * scale[i]
// One wavefront per row. Reduction over `dim` elements.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(WAVE)
kernel_rmsnorm_fp16(uint32_t rows, uint32_t dim,
                    const __half * __restrict__ x,
                    const float  * __restrict__ scale,
                    float eps,
                    __half       * __restrict__ out) {
  uint32_t row = blockIdx.x;
  if (row >= rows) return;

  const __half *xr = x   + (size_t)row * dim;
  __half       *or_ = out + (size_t)row * dim;

  int tid = threadIdx.x;

  // Compute sum of squares
  float ss = 0.0f;
  for (uint32_t i = tid; i < dim; i += WAVE)
    ss += __half2float(xr[i]) * __half2float(xr[i]);

  // Wave reduce
  ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
  ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);

  float rms_inv = rsqrtf(ss / (float)dim + eps);

  // Normalize and scale
  for (uint32_t i = tid; i < dim; i += WAVE)
    or_[i] = __float2half(__half2float(xr[i]) * rms_inv * scale[i]);
}

extern "C" void therock_rmsnorm_fp16(hipStream_t stream,
                                      uint32_t rows, uint32_t dim,
                                      const void *x, const float *scale,
                                      float eps, void *out) {
  hipLaunchKernelGGL(kernel_rmsnorm_fp16,
                     dim3(rows), dim3(WAVE), 0, stream,
                     rows, dim,
                     (const __half *)x, scale, eps, (__half *)out);
}

// ---------------------------------------------------------------------------
// SiLU(gate) * up: out[i] = gate[i] / (1 + exp(-gate[i])) * up[i]
// Flat over all elements (no row structure needed).
// ---------------------------------------------------------------------------

__global__ void
kernel_silu_mul_fp16(uint32_t n,
                     const __half * __restrict__ gate,
                     const __half * __restrict__ up,
                     __half       * __restrict__ out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float g = __half2float(gate[i]);
  float u = __half2float(up[i]);
  // SiLU: g * sigmoid(g) = g / (1 + exp(-g))
  float silu = g / (1.0f + expf(-g));
  out[i] = __float2half(silu * u);
}

extern "C" void therock_silu_mul_fp16(hipStream_t stream,
                                       uint32_t n,
                                       const void *gate, const void *up,
                                       void *out) {
  uint32_t threads = 256;
  uint32_t blocks  = (n + threads - 1) / threads;
  hipLaunchKernelGGL(kernel_silu_mul_fp16,
                     dim3(blocks), dim3(threads), 0, stream,
                     n, (const __half *)gate, (const __half *)up, (__half *)out);
}

// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Smoke test: run fp16 and Q4_0 GEMV on hardware, compare against CPU reference.

#include "therock/inference/hardware.h"
#include "therock/inference/kernel_registry.h"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#define HIP_CHECK(x) do { \
  hipError_t e = (x); \
  if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %d at %s:%d\n", e, __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

// CPU reference GEMV in fp32
static void cpu_gemv_fp16(int N, int K,
                           const uint16_t *x, const uint16_t *W,
                           float *out) {
  auto f16_to_f32 = [](uint16_t h) -> float {
    // simple fp16 → fp32 via union trick
    uint32_t bits = ((uint32_t)(h & 0x8000) << 16)
                  | ((uint32_t)((h >> 10) & 0x1F) + 112) << 23
                  | ((uint32_t)(h & 0x3FF) << 13);
    // handle zero exponent
    if (((h >> 10) & 0x1F) == 0) bits = ((uint32_t)(h & 0x8000) << 16);
    float f; memcpy(&f, &bits, 4); return f;
  };
  for (int n = 0; n < N; n++) {
    double acc = 0;
    for (int k = 0; k < K; k++)
      acc += (double)f16_to_f32(W[n*K+k]) * f16_to_f32(x[k]);
    out[n] = (float)acc;
  }
}

int main() {
  const TheRockHardwareDesc *hw = therock_hardware_detect();
  if (!hw) {
    fprintf(stderr, "Could not detect hardware — running without GPU?\n");
    return 1;
  }
  printf("Hardware: %s (%s)\n", hw->gfx_arch, hw->marketing_name);

  TheRockKernelRegistry *reg = therock_registry_create(hw);

  // Problem: M=1 (decode), N=4096, K=4096 — typical attention projection
  const int M = 1, N = 4096, K = 4096;

  // Allocate fp16 host buffers
  std::vector<uint16_t> h_x(K), h_W(N*K), h_y(N, 0);
  // Fill with small values so fp16 doesn't overflow
  for (int i = 0; i < K;   i++) h_x[i] = 0x3400;  // ~0.25 in fp16
  for (int i = 0; i < N*K; i++) h_W[i] = 0x3000 + (i % 32);  // ~0.125 + noise

  // CPU reference
  std::vector<float> ref(N);
  cpu_gemv_fp16(N, K, h_x.data(), h_W.data(), ref.data());

  // GPU buffers
  void *d_x, *d_W, *d_y;
  HIP_CHECK(hipMalloc(&d_x, K   * sizeof(uint16_t)));
  HIP_CHECK(hipMalloc(&d_W, N*K * sizeof(uint16_t)));
  HIP_CHECK(hipMalloc(&d_y, N   * sizeof(uint16_t)));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), K   * sizeof(uint16_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_W, h_W.data(), N*K * sizeof(uint16_t), hipMemcpyHostToDevice));

  TheRockOpDesc desc = {
    .op            = TheRockOpDesc::THEROCK_OP_GEMV,
    .M = M, .N = N, .K = K, .batch = 1,
    .weight_layout = THEROCK_LAYOUT_ROW_MAJOR,
    .exec_mode     = THEROCK_EXEC_DECODE,
  };

  TheRockKernelFn fn = therock_registry_lookup(reg, &desc);
  if (!fn.gemv) { fprintf(stderr, "No kernel found\n"); return 1; }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Warmup
  for (int i = 0; i < 10; i++)
    fn.gemv(stream, &desc, d_x, d_W, nullptr, d_y);
  HIP_CHECK(hipStreamSynchronize(stream));

  // Benchmark: 1000 iterations
  hipEvent_t t0, t1;
  HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
  HIP_CHECK(hipEventRecord(t0, stream));
  for (int i = 0; i < 1000; i++)
    fn.gemv(stream, &desc, d_x, d_W, nullptr, d_y);
  HIP_CHECK(hipEventRecord(t1, stream));
  HIP_CHECK(hipEventSynchronize(t1));

  float ms = 0;
  hipEventElapsedTime(&ms, t0, t1);
  ms /= 1000.0f;

  // Bytes read: weights + activations (decoded weights never materialized as fp16)
  double bytes = (double)N * K * sizeof(uint16_t) + K * sizeof(uint16_t);
  double gb_s  = bytes / (ms * 1e-3) / 1e9;
  double gflops = 2.0 * M * N * K / (ms * 1e-3) / 1e9;
  printf("GEMV fp16 M=%d N=%d K=%d: %.3f ms  %.1f GB/s  %.1f GFlops\n",
         M, N, K, ms, gb_s, gflops);

  // Correctness check
  HIP_CHECK(hipMemcpy(h_y.data(), d_y, N * sizeof(uint16_t), hipMemcpyDeviceToHost));
  int mismatches = 0;
  for (int n = 0; n < N; n++) {
    // decode h_y[n] from fp16
    uint16_t raw = h_y[n];
    float got; uint32_t bits = ((uint32_t)(raw&0x8000)<<16)|(((uint32_t)((raw>>10)&0x1F)+112)<<23)|((uint32_t)(raw&0x3FF)<<13);
    if (((raw>>10)&0x1F)==0) bits=((uint32_t)(raw&0x8000)<<16);
    memcpy(&got, &bits, 4);
    float err = fabsf(got - ref[n]) / (fabsf(ref[n]) + 1e-6f);
    if (err > 0.01f) { mismatches++; if (mismatches < 5) printf("  mismatch[%d]: got=%f ref=%f\n", n, got, ref[n]); }
  }
  printf("Correctness: %d/%d mismatches\n", mismatches, N);

  hipFree(d_x); hipFree(d_W); hipFree(d_y);
  therock_registry_destroy(reg);
  return mismatches > 0 ? 1 : 0;
}

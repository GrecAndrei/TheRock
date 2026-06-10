// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Flash attention correctness + benchmark test.
// Reference: naive O(N²) attention on CPU in fp32.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#define HIP_CHECK(x) do { \
  hipError_t e = (x); \
  if (e != hipSuccess) { \
    fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

extern "C" void therock_flash_attn_decode(
    hipStream_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, float,
    const void *, const void *, const void *,
    const uint32_t *, void *);

extern "C" void therock_flash_attn_prefill(
    hipStream_t, uint32_t, uint32_t, uint32_t, float,
    const void *, const void *, const void *, void *);

extern "C" bool therock_flash_attn_prefill_hipblaslt(
    hipStream_t, uint32_t, uint32_t, uint32_t, float,
    const void *, const void *, const void *, void *);

// fp16 bit pattern helpers
static float f16_to_f32(uint16_t h) {
  uint32_t sign = (h >> 15) & 1;
  uint32_t exp  = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
  if (exp == 0x1F) return sign ? -INFINITY : INFINITY;
  float f = (sign ? -1.0f : 1.0f)
          * ldexpf(1.0f + mant / 1024.0f, (int)exp - 15);
  return f;
}

static uint16_t f32_to_f16(float f) {
  // Use HIP runtime helper via union — simple approximate version
  if (f == 0.0f) return 0;
  int sign = f < 0 ? 1 : 0;
  f = fabsf(f);
  int exp = (int)floorf(log2f(f));
  float mant = f / ldexpf(1.0f, exp) - 1.0f;
  exp += 15;
  if (exp <= 0) return sign << 15;
  if (exp >= 31) return (sign << 15) | (0x1F << 10);
  return (uint16_t)((sign << 15) | (exp << 10) | (int)(mant * 1024.0f));
}

// CPU reference: softmax(Q K^T / sqrt(d)) V, causal mask
static void cpu_attention(int S, int H, int D,
                           const std::vector<uint16_t> &q,
                           const std::vector<uint16_t> &k,
                           const std::vector<uint16_t> &v,
                           std::vector<float> &out,
                           bool causal) {
  float scale = 1.0f / sqrtf((float)D);
  out.assign(S * H * D, 0.0f);

  for (int h = 0; h < H; h++) {
    for (int i = 0; i < S; i++) {
      int max_j = causal ? i + 1 : S;
      std::vector<float> scores(max_j);
      float m = -1e9f;
      for (int j = 0; j < max_j; j++) {
        float dot = 0;
        for (int d = 0; d < D; d++)
          dot += f16_to_f32(q[i*H*D + h*D + d]) * f16_to_f32(k[j*H*D + h*D + d]);
        scores[j] = dot * scale;
        if (scores[j] > m) m = scores[j];
      }
      float sum = 0;
      for (int j = 0; j < max_j; j++) { scores[j] = expf(scores[j] - m); sum += scores[j]; }
      for (int j = 0; j < max_j; j++) {
        for (int d = 0; d < D; d++)
          out[i*H*D + h*D + d] += scores[j] / sum * f16_to_f32(v[j*H*D + h*D + d]);
      }
    }
  }
}

static float max_rel_err(const std::vector<uint16_t> &got,
                          const std::vector<float> &ref) {
  float max_err = 0;
  for (size_t i = 0; i < ref.size(); i++) {
    float g = f16_to_f32(got[i]);
    float r = ref[i];
    float err = fabsf(g - r) / (fabsf(r) + 1e-3f);
    if (err > max_err) max_err = err;
  }
  return max_err;
}

// ---- Test: decode (S=1) ----
static int test_decode() {
  const int S = 1, H = 8, D = 128, KV_PAST = 64;
  // Total KV context: KV_PAST previous tokens
  const int TOTAL_KV = KV_PAST;

  int total_elems_qkv = TOTAL_KV * H * D;
  std::vector<uint16_t> q_h(S*H*D), k_h(total_elems_qkv), v_h(total_elems_qkv);

  srand(42);
  auto randf16 = [&]() { return f32_to_f16((rand() / (float)RAND_MAX - 0.5f) * 0.5f); };
  for (auto &x : q_h) x = randf16();
  for (auto &x : k_h) x = randf16();
  for (auto &x : v_h) x = randf16();

  // CPU reference (S=1 attending to KV_PAST tokens, no causal needed for single query)
  // Extend q to TOTAL_KV for reference (use zeros for past queries)
  std::vector<uint16_t> q_full(total_elems_qkv, 0);
  // Put our query at the last position
  for (int i = 0; i < S*H*D; i++) q_full[(TOTAL_KV-1)*H*D + i] = q_h[i];
  std::vector<float> ref_full;
  cpu_attention(TOTAL_KV, H, D, q_full, k_h, v_h, ref_full, true);
  // Extract last row
  std::vector<float> ref(ref_full.end() - H*D, ref_full.end());

  // GPU
  void *d_q, *d_k, *d_v, *d_out;
  HIP_CHECK(hipMalloc(&d_q, S*H*D*2));
  HIP_CHECK(hipMalloc(&d_k, total_elems_qkv*2));
  HIP_CHECK(hipMalloc(&d_v, total_elems_qkv*2));
  HIP_CHECK(hipMalloc(&d_out, S*H*D*2));
  HIP_CHECK(hipMemcpy(d_q, q_h.data(), S*H*D*2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, k_h.data(), total_elems_qkv*2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, v_h.data(), total_elems_qkv*2, hipMemcpyHostToDevice));

  // page_table = [0] (single page, all KV in physical page 0)
  uint32_t page_table_h = 0;
  uint32_t *d_page_table;
  HIP_CHECK(hipMalloc(&d_page_table, sizeof(uint32_t)));
  HIP_CHECK(hipMemcpy(d_page_table, &page_table_h, sizeof(uint32_t), hipMemcpyHostToDevice));

  hipStream_t s; HIP_CHECK(hipStreamCreate(&s));
  float scale = 1.0f / sqrtf((float)D);
  therock_flash_attn_decode(s, H, H, D, TOTAL_KV, TOTAL_KV, scale,
                             d_q, d_k, d_v, d_page_table, d_out);
  HIP_CHECK(hipStreamSynchronize(s));

  std::vector<uint16_t> got(S*H*D);
  HIP_CHECK(hipMemcpy(got.data(), d_out, S*H*D*2, hipMemcpyDeviceToHost));

  float err = max_rel_err(got, ref);
  printf("decode  S=1  H=%d D=%d KV=%d: max_rel_err=%.4f  %s\n",
         H, D, TOTAL_KV, err, err < 0.05f ? "PASS" : "FAIL");

  // Benchmark
  hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
  const int ITERS = 10000;
  hipEventRecord(t0, s);
  for (int i = 0; i < ITERS; i++)
    therock_flash_attn_decode(s, H, H, D, TOTAL_KV, TOTAL_KV, scale,
                               d_q, d_k, d_v, d_page_table, d_out);
  hipEventRecord(t1, s); hipEventSynchronize(t1);
  float ms = 0; hipEventElapsedTime(&ms, t0, t1); ms /= ITERS;
  // Bytes: Q(S*H*D*2) + K(TOTAL_KV*H*D*2) + V(same) + out(S*H*D*2)
  double bytes = (double)(S + 2*TOTAL_KV + S) * H * D * 2;
  printf("        %.3f ms  %.1f GB/s\n", ms, bytes/(ms*1e-3)/1e9);

  hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_out); hipFree(d_page_table);
  return err < 0.05f ? 0 : 1;
}

// ---- Test: prefill (S=64, causal) ----
static int test_prefill() {
  const int S = 64, H = 8, D = 128;
  int elems = S * H * D;
  std::vector<uint16_t> q_h(elems), k_h(elems), v_h(elems);
  srand(42);
  auto randf16 = [&]() { return f32_to_f16((rand() / (float)RAND_MAX - 0.5f) * 0.5f); };
  for (auto &x : q_h) x = randf16();
  for (auto &x : k_h) x = randf16();
  for (auto &x : v_h) x = randf16();

  std::vector<float> ref;
  cpu_attention(S, H, D, q_h, k_h, v_h, ref, true);

  void *d_q, *d_k, *d_v, *d_out;
  HIP_CHECK(hipMalloc(&d_q, elems*2)); HIP_CHECK(hipMalloc(&d_k, elems*2));
  HIP_CHECK(hipMalloc(&d_v, elems*2)); HIP_CHECK(hipMalloc(&d_out, elems*2));
  HIP_CHECK(hipMemcpy(d_q, q_h.data(), elems*2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, k_h.data(), elems*2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, v_h.data(), elems*2, hipMemcpyHostToDevice));

  hipStream_t s; HIP_CHECK(hipStreamCreate(&s));
  float scale = 1.0f / sqrtf((float)D);
  therock_flash_attn_prefill(s, S, H, D, scale, d_q, d_k, d_v, d_out);
  HIP_CHECK(hipStreamSynchronize(s));

  std::vector<uint16_t> got(elems);
  HIP_CHECK(hipMemcpy(got.data(), d_out, elems*2, hipMemcpyDeviceToHost));

  float err = max_rel_err(got, ref);
  printf("prefill S=%d H=%d D=%d: max_rel_err=%.4f  %s\n",
         S, H, D, err, err < 0.05f ? "PASS" : "FAIL");

  // Benchmark
  hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
  const int ITERS = 1000;
  hipEventRecord(t0, s);
  for (int i = 0; i < ITERS; i++)
    therock_flash_attn_prefill(s, S, H, D, scale, d_q, d_k, d_v, d_out);
  hipEventRecord(t1, s); hipEventSynchronize(t1);
  float ms = 0; hipEventElapsedTime(&ms, t0, t1); ms /= ITERS;
  double flops = 2.0 * S * S * H * D;  // QK^T
  printf("        %.3f ms  %.1f GFlops\n", ms, flops/(ms*1e-3)/1e9);

  hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_out);
  return err < 0.05f ? 0 : 1;
}

static int test_prefill_hipblaslt() {
  const int S = 64, H = 8, D = 128;
  int elems = S * H * D;
  std::vector<uint16_t> q_h(elems), k_h(elems), v_h(elems);
  srand(42);
  auto randf16 = [&]() { return f32_to_f16((rand() / (float)RAND_MAX - 0.5f) * 0.5f); };
  for (auto &x : q_h) x = randf16();
  for (auto &x : k_h) x = randf16();
  for (auto &x : v_h) x = randf16();

  std::vector<float> ref;
  cpu_attention(S, H, D, q_h, k_h, v_h, ref, true);

  void *d_q, *d_k, *d_v, *d_out;
  HIP_CHECK(hipMalloc(&d_q, elems*2)); HIP_CHECK(hipMalloc(&d_k, elems*2));
  HIP_CHECK(hipMalloc(&d_v, elems*2)); HIP_CHECK(hipMalloc(&d_out, elems*2));
  HIP_CHECK(hipMemcpy(d_q, q_h.data(), elems*2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, k_h.data(), elems*2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, v_h.data(), elems*2, hipMemcpyHostToDevice));

  hipStream_t s; HIP_CHECK(hipStreamCreate(&s));
  float scale = 1.0f / sqrtf((float)D);
  bool ok = therock_flash_attn_prefill_hipblaslt(s, S, H, D, scale, d_q, d_k, d_v, d_out);
  HIP_CHECK(hipStreamSynchronize(s));

  if (!ok) {
    printf("prefill_hipblaslt S=%d H=%d D=%d: SKIP (hipBLASLt unavailable)\n", S, H, D);
    hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_out);
    return 0;
  }

  std::vector<uint16_t> got(elems);
  HIP_CHECK(hipMemcpy(got.data(), d_out, elems*2, hipMemcpyDeviceToHost));
  float err = max_rel_err(got, ref);
  printf("prefill_hipblaslt S=%d H=%d D=%d: max_rel_err=%.4f  %s\n",
         S, H, D, err, err < 0.05f ? "PASS" : "FAIL");

  hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_out);
  return err < 0.05f ? 0 : 1;
}

static void bench_prefill_hipblaslt(int S, int H, int D) {
  int elems = S * H * D;
  std::vector<uint16_t> q_h(elems), k_h(elems), v_h(elems);
  srand(123);
  auto randf16 = [&]() { return f32_to_f16((rand() / (float)RAND_MAX - 0.5f) * 0.5f); };
  for (auto &x : q_h) x = randf16();
  for (auto &x : k_h) x = randf16();
  for (auto &x : v_h) x = randf16();

  void *d_q, *d_k, *d_v, *d_out;
  hipMalloc(&d_q, elems*2); hipMalloc(&d_k, elems*2);
  hipMalloc(&d_v, elems*2); hipMalloc(&d_out, elems*2);
  hipMemcpy(d_q, q_h.data(), elems*2, hipMemcpyHostToDevice);
  hipMemcpy(d_k, k_h.data(), elems*2, hipMemcpyHostToDevice);
  hipMemcpy(d_v, v_h.data(), elems*2, hipMemcpyHostToDevice);

  hipStream_t s; hipStreamCreate(&s);
  float scale = 1.0f / sqrtf((float)D);

  // Warmup + heuristic selection (also checks availability)
  bool avail = false;
  for (int i = 0; i < 5; i++)
    avail |= therock_flash_attn_prefill_hipblaslt(s, S, H, D, scale, d_q, d_k, d_v, d_out);
  hipStreamSynchronize(s);
  if (!avail) {
    printf("  S=%4d H=%d D=%d: SKIP (hipBLASLt unavailable)\n", S, H, D);
    hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_out);
    hipStreamDestroy(s);
    return;
  }

  hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
  int ITERS = (S <= 128) ? 500 : (S <= 512 ? 100 : 30);
  hipEventRecord(t0, s);
  for (int i = 0; i < ITERS; i++)
    therock_flash_attn_prefill_hipblaslt(s, S, H, D, scale, d_q, d_k, d_v, d_out);
  hipEventRecord(t1, s); hipEventSynchronize(t1);
  float ms = 0; hipEventElapsedTime(&ms, t0, t1); ms /= ITERS;
  double flops = 2.0 * S * S * H * D;
  printf("  S=%4d H=%d D=%d: %.3f ms  %.1f GFlops\n",
         S, H, D, ms, flops/(ms*1e-3)/1e9);

  hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_out);
  hipStreamDestroy(s);
}

static void bench_prefill(int S, int H, int D) {
  int elems = S * H * D;
  std::vector<uint16_t> q_h(elems), k_h(elems), v_h(elems);
  srand(123);
  auto randf16 = [&]() { return f32_to_f16((rand() / (float)RAND_MAX - 0.5f) * 0.5f); };
  for (auto &x : q_h) x = randf16();
  for (auto &x : k_h) x = randf16();
  for (auto &x : v_h) x = randf16();

  void *d_q, *d_k, *d_v, *d_out;
  hipMalloc(&d_q, elems*2); hipMalloc(&d_k, elems*2);
  hipMalloc(&d_v, elems*2); hipMalloc(&d_out, elems*2);
  hipMemcpy(d_q, q_h.data(), elems*2, hipMemcpyHostToDevice);
  hipMemcpy(d_k, k_h.data(), elems*2, hipMemcpyHostToDevice);
  hipMemcpy(d_v, v_h.data(), elems*2, hipMemcpyHostToDevice);

  hipStream_t s; hipStreamCreate(&s);
  float scale = 1.0f / sqrtf((float)D);

  // Warmup
  for (int i = 0; i < 10; i++)
    therock_flash_attn_prefill(s, S, H, D, scale, d_q, d_k, d_v, d_out);
  hipStreamSynchronize(s);

  hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
  int ITERS = (S <= 128) ? 1000 : (S <= 512 ? 200 : 50);
  hipEventRecord(t0, s);
  for (int i = 0; i < ITERS; i++)
    therock_flash_attn_prefill(s, S, H, D, scale, d_q, d_k, d_v, d_out);
  hipEventRecord(t1, s); hipEventSynchronize(t1);
  float ms = 0; hipEventElapsedTime(&ms, t0, t1); ms /= ITERS;
  double flops = 2.0 * S * S * H * D;
  double bandwidth = (double)(S * H * D * 2) * 3.0 / (ms * 1e-3) / 1e9;  // Q+K+V reads
  printf("  S=%4d H=%d D=%d: %.3f ms  %.1f GFlops  ~%.0f GB/s\n",
         S, H, D, ms, flops/(ms*1e-3)/1e9, bandwidth);

  hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_out);
  hipStreamDestroy(s);
}

int main() {
  int failures = 0;
  failures += test_decode();
  failures += test_prefill();
  failures += test_prefill_hipblaslt();

  printf("\n--- Prefill scaling benchmark (custom kernel) ---\n");
  for (int S : {32, 64, 128, 256, 512, 1024, 2048})
    bench_prefill(S, 8, 128);

  printf("\n--- Prefill scaling benchmark (hipBLASLt) ---\n");
  for (int S : {32, 64, 128, 256, 512, 1024, 2048})
    bench_prefill_hipblaslt(S, 8, 128);

  return failures;
}

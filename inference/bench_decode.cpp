// Decode-only benchmark with multiple KV sizes to see 4-token unroll impact.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)

extern "C" void therock_flash_attn_decode(
    hipStream_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, float,
    const void *, const void *, const void *,
    const uint32_t *, void *);

int main() {
  const int D = 128, PAGE = 16;
  int kv_sizes[] = {64, 256, 1024, 4096, 16384, 32768, 65536, 131072};
  const int n = sizeof(kv_sizes)/sizeof(kv_sizes[0]);

  // Run for 2 configs: H_q=8 H_kv=8 (no GQA, baseline) and H_q=8 H_kv=1 (GQA, 8x share)
  for (int config = 0; config < 3; config++) {
    int H_Q, H_KV;
    if (config == 0)      { H_Q = 8;  H_KV = 8; printf("\n=== H_q=%d H_kv=%d (no GQA, baseline) ===\n", H_Q, H_KV); }
    else if (config == 1) { H_Q = 8;  H_KV = 1; printf("\n=== H_q=%d H_kv=%d (8x GQA, LDS-shared) ===\n", H_Q, H_KV); }
    else                  { H_Q = 32; H_KV = 8; printf("\n=== H_q=%d H_kv=%d (4x GQA, LLaMA-3 8B / Mistral 7B) ===\n", H_Q, H_KV); }

    for (int sidx = 0; sidx < n; sidx++) {
      int TOTAL_KV = kv_sizes[sidx];
      int npages = (TOTAL_KV + PAGE - 1) / PAGE;

      std::vector<__half> q_h(H_Q*D), k_h(TOTAL_KV*H_KV*D), v_h(TOTAL_KV*H_KV*D);
      srand(42);
      for (auto &x : q_h) x = __float2half((rand()/(float)RAND_MAX - 0.5f) * 0.5f);
      for (auto &x : k_h) x = __float2half((rand()/(float)RAND_MAX - 0.5f) * 0.5f);
      for (auto &x : v_h) x = __float2half((rand()/(float)RAND_MAX - 0.5f) * 0.5f);

      // Validate correctness
      std::vector<float> ref(H_Q*D);
      for (int h = 0; h < H_Q; h++) {
        float m = -1e9f, l = 0.0f;
        std::vector<float> o(D, 0.0f);
        for (int t = 0; t < TOTAL_KV; t++) {
          float dot = 0.0f;
          for (int d = 0; d < D; d++) {
            int kidx = (t * H_KV + (h * H_KV / H_Q)) * D + d;
            int qidx = h * D + d;
            dot += __half2float(q_h[qidx]) * __half2float(k_h[kidx]);
          }
          float s = dot / sqrtf((float)D);
          float mn = (s > m) ? s : m;
          float es = expf(s - mn), esh = expf(m - mn);
          l = esh * l + es;
          for (int d = 0; d < D; d++) {
            int vidx = (t * H_KV + (h * H_KV / H_Q)) * D + d;
            o[d] = esh * o[d] + es * __half2float(v_h[vidx]);
          }
          m = mn;
        }
        float li = l > 0 ? 1.0f/l : 0.0f;
        for (int d = 0; d < D; d++) ref[h * D + d] = o[d] * li;
      }

      void *d_q, *d_k, *d_v, *d_out;
      HIP_CHECK(hipMalloc(&d_q, H_Q*D*sizeof(__half)));
      HIP_CHECK(hipMalloc(&d_k, TOTAL_KV*H_KV*D*sizeof(__half)));
      HIP_CHECK(hipMalloc(&d_v, TOTAL_KV*H_KV*D*sizeof(__half)));
      HIP_CHECK(hipMalloc(&d_out, H_Q*D*sizeof(__half)));
      HIP_CHECK(hipMemcpy(d_q, q_h.data(), H_Q*D*sizeof(__half), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(d_k, k_h.data(), TOTAL_KV*H_KV*D*sizeof(__half), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(d_v, v_h.data(), TOTAL_KV*H_KV*D*sizeof(__half), hipMemcpyHostToDevice));

      std::vector<uint32_t> pt(npages);
      for (int i = 0; i < npages; i++) pt[i] = i;
      uint32_t *d_pt;
      HIP_CHECK(hipMalloc(&d_pt, npages*sizeof(uint32_t)));
      HIP_CHECK(hipMemcpy(d_pt, pt.data(), npages*sizeof(uint32_t), hipMemcpyHostToDevice));

      hipStream_t s; HIP_CHECK(hipStreamCreate(&s));
      float scale = 1.0f / sqrtf((float)D);
      for (int i = 0; i < 5; i++)
        therock_flash_attn_decode(s, H_Q, H_KV, D, TOTAL_KV, PAGE, scale,
                                   d_q, d_k, d_v, d_pt, d_out);
      HIP_CHECK(hipStreamSynchronize(s));

      // Validate
      std::vector<__half> got(H_Q*D);
      HIP_CHECK(hipMemcpy(got.data(), d_out, H_Q*D*sizeof(__half), hipMemcpyDeviceToHost));
      double max_err = 0;
      for (int i = 0; i < H_Q*D; i++) {
        double e = std::abs(__half2float(got[i]) - ref[i]);
        double r = std::abs(ref[i]);
        if (r > 1e-3 && e/r > max_err) max_err = e/r;
      }
      const char* tag = max_err < 0.05 ? "OK" : "FAIL";

      hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
      const int ITERS = 1000;
      hipEventRecord(t0, s);
      for (int i = 0; i < ITERS; i++)
        therock_flash_attn_decode(s, H_Q, H_KV, D, TOTAL_KV, PAGE, scale,
                                   d_q, d_k, d_v, d_pt, d_out);
      hipEventRecord(t1, s); hipEventSynchronize(t1);
      float ms = 0; hipEventElapsedTime(&ms, t0, t1); ms /= ITERS;

      double flops = 2.0 * 2.0 * H_Q * D * TOTAL_KV;
      double gflops = flops/1e9/(ms/1000.0);
      double bytes = (double)(H_Q*D + 2*TOTAL_KV*H_KV*D + H_Q*D) * 2;
      double gbs = bytes/(ms*1e-3)/1e9;

      printf("  KV=%5d  %.4f ms  %7.1f GFlops  %5.1f GB/s  err %.4f %s\n", TOTAL_KV, ms, gflops, gbs, max_err, tag);

      hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_out); hipFree(d_pt);
    }
  }
  return 0;
}

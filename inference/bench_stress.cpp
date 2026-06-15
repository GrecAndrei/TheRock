// Stress test: run the split-K kernel with many random seeds and KV sizes
// to make sure it's not just lucky at srand(42).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)

extern "C" void therock_flash_attn_decode(
    hipStream_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, float,
    const void *, const void *, const void *,
    const uint32_t *, void *);

int main() {
  const int D = 128, PAGE = 16;
  struct Cfg { int h_q, h_kv, kv; };
  Cfg cfgs[] = {
    {8, 8, 64}, {8, 8, 256}, {8, 8, 1024}, {8, 8, 4096}, {8, 8, 16384},
    {8, 1, 64}, {8, 1, 256}, {8, 1, 1024}, {8, 1, 4096}, {8, 1, 16384},
    {8, 4, 1024},  // GQA ratio 2
    {16, 8, 4096}, // larger H_q
  };
  const int n = sizeof(cfgs)/sizeof(cfgs[0]);
  int total_pass = 0, total_fail = 0;

  for (int c = 0; c < n; c++) {
    int H_Q = cfgs[c].h_q, H_KV = cfgs[c].h_kv, TOTAL_KV = cfgs[c].kv;
    int npages = (TOTAL_KV + PAGE - 1) / PAGE;

    std::vector<__half> q_h(H_Q*D), k_h(TOTAL_KV*H_KV*D), v_h(TOTAL_KV*H_KV*D);
    srand(7 + c);  // different seed per config
    for (auto &x : q_h) x = __float2half(((rand()%1000) - 500) / 1000.0f);
    for (auto &x : k_h) x = __float2half(((rand()%1000) - 500) / 1000.0f);
    for (auto &x : v_h) x = __float2half(((rand()%1000) - 500) / 1000.0f);

    // CPU reference
    std::vector<float> ref(H_Q*D);
    for (int h = 0; h < H_Q; h++) {
      float m = -1e9f, l = 0.0f;
      std::vector<float> o(D, 0.0f);
      for (int t = 0; t < TOTAL_KV; t++) {
        float dot = 0.0f;
        int kvh = h * H_KV / H_Q;
        for (int d = 0; d < D; d++) {
          int kidx = (t * H_KV + kvh) * D + d;
          int qidx = h * D + d;
          dot += __half2float(q_h[qidx]) * __half2float(k_h[kidx]);
        }
        float s = dot / sqrtf((float)D);
        float mn = (s > m) ? s : m;
        float es = expf(s - mn), esh = expf(m - mn);
        l = esh * l + es;
        for (int d = 0; d < D; d++) {
          int vidx = (t * H_KV + kvh) * D + d;
          o[d] = esh * o[d] + es * __half2float(v_h[vidx]);
        }
        m = mn;
      }
      float li = l > 0 ? 1.0f/l : 0.0f;
      for (int d = 0; d < D; d++) ref[h * D + d] = o[d] * li;
    }

    // GPU
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

    hipStream_t st; HIP_CHECK(hipStreamCreate(&st));
    float scale = 1.0f / sqrtf((float)D);
    therock_flash_attn_decode(st, H_Q, H_KV, D, TOTAL_KV, PAGE, scale,
                               d_q, d_k, d_v, d_pt, d_out);
    HIP_CHECK(hipStreamSynchronize(st));

    std::vector<__half> got(H_Q*D);
    HIP_CHECK(hipMemcpy(got.data(), d_out, H_Q*D*sizeof(__half), hipMemcpyDeviceToHost));

    double max_err = 0;
    for (int i = 0; i < H_Q*D; i++) {
      double e = std::abs(__half2float(got[i]) - ref[i]);
      double r = std::abs(ref[i]);
      if (r > 1e-3 && e/r > max_err) max_err = e/r;
    }
    const char* tag = max_err < 0.05 ? "PASS" : "FAIL";
    if (max_err < 0.05) total_pass++; else total_fail++;
    printf("H_q=%d H_kv=%d KV=%-6d seed=%-3d max_rel_err=%.6f %s\n",
           H_Q, H_KV, TOTAL_KV, 7+c, max_err, tag);

    hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_out); hipFree(d_pt);
  }

  printf("\n=== %d/%d passed, %d failed ===\n", total_pass, n, total_fail);
  return total_fail > 0 ? 1 : 0;
}

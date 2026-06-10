// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Flash attention prefill via hipBLASLt — TensileLite-tuned batched GEMM
// for Q@K^T and P@V, with a custom causal softmax kernel in between.
//
// Flow:
//   1. Pack:    [seq, heads, dim] → [heads, seq, dim]  (interleave layout)
//   2. GEMM1:   Q @ K^T  →  scores [heads, seq, seq]  fp32, alpha=scale
//   3. Softmax: causal mask + normalize  →  probs [heads, seq, seq] fp16
//   4. GEMM2:   probs @ V  →  o_packed [heads, seq, dim]  fp16
//   5. Unpack:  [heads, seq, dim] → [seq, heads, dim]

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <stdint.h>

static constexpr float FLASH_NEG_INF  = -1e9f;
static constexpr int   SOFTMAX_BLOCKS = 256;

// ---------------------------------------------------------------------------
// Pack: [seq_len, heads, head_pairs] → [heads, seq_len, head_pairs]  (half2)
// ---------------------------------------------------------------------------
__global__ void kernel_lt_pack(
    uint32_t seq_len, uint32_t heads, uint32_t head_pairs,
    const __half2 *qin, const __half2 *kin, const __half2 *vin,
    __half2 *qout, __half2 *kout, __half2 *vout) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)seq_len * heads * head_pairs;
  if (idx >= total) return;
  uint32_t pair = idx % head_pairs;
  uint32_t head = (idx / head_pairs) % heads;
  uint32_t row  = (idx / head_pairs) / heads;
  size_t dst = ((size_t)head * seq_len + row) * head_pairs + pair;
  qout[dst] = qin[idx];
  kout[dst] = kin[idx];
  vout[dst] = vin[idx];
}

// ---------------------------------------------------------------------------
// Unpack: [heads, seq_len, head_pairs] → [seq_len, heads, head_pairs]  (half2)
// ---------------------------------------------------------------------------
__global__ void kernel_lt_unpack(
    uint32_t seq_len, uint32_t heads, uint32_t head_pairs,
    const __half2 *src, __half2 *dst) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)seq_len * heads * head_pairs) return;
  uint32_t pair = idx % head_pairs;
  uint32_t head = (idx / head_pairs) % heads;
  uint32_t row  = (idx / head_pairs) / heads;
  size_t s = ((size_t)head * seq_len + row) * head_pairs + pair;
  dst[idx] = src[s];
}

// ---------------------------------------------------------------------------
// Causal softmax: scores fp32 [heads, seq, seq] → probs fp16 (causal masked)
// Scale is already baked into scores by the GEMM alpha.
// Grid: (seq_len, heads),  Block: SOFTMAX_BLOCKS
// ---------------------------------------------------------------------------
__global__ void kernel_lt_softmax(
    uint32_t seq_len, const float *scores, __half *probs) {
  uint32_t row  = blockIdx.x;
  uint32_t head = blockIdx.y;
  uint32_t tid  = threadIdx.x;
  size_t off    = ((size_t)head * seq_len + row) * seq_len;
  const float *sr = scores + off;
  __half      *pr = probs  + off;

  __shared__ float sh[SOFTMAX_BLOCKS];

  // Reduce max over causal window [0..row]
  float lmax = FLASH_NEG_INF;
  for (uint32_t c = tid; c <= row; c += blockDim.x)
    lmax = fmaxf(lmax, sr[c]);
  sh[tid] = lmax;
  __syncthreads();
  for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid + s]);
    __syncthreads();
  }
  float m = sh[0];

  // Reduce sum-exp
  float lsum = 0.0f;
  for (uint32_t c = tid; c <= row; c += blockDim.x)
    lsum += __expf(sr[c] - m);
  sh[tid] = lsum;
  __syncthreads();
  for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) sh[tid] += sh[tid + s];
    __syncthreads();
  }
  float inv = 1.0f / sh[0];

  for (uint32_t c = tid; c <= row; c += blockDim.x)
    pr[c] = __float2half(__expf(sr[c] - m) * inv);
  for (uint32_t c = row + 1 + tid; c < seq_len; c += blockDim.x)
    pr[c] = __float2half(0.0f);
}

// ---------------------------------------------------------------------------
// hipBLASLt state
// ---------------------------------------------------------------------------

static hipblasLtHandle_t s_lt = nullptr;
static void *s_ws = nullptr;
static size_t s_ws_sz = 0;
static constexpr size_t WS_BYTES = 32u << 20;  // 32 MB

struct LtPlan {
  bool        valid = false;
  uint32_t    M = 0, N = 0, K = 0, batch = 0;
  bool        transA = false, transB = false;
  bool        has_algo = false;
  hipblasLtMatmulAlgo_t algo;
};
static LtPlan s_qk_plan;
static LtPlan s_pv_plan;

static hipblasLtHandle_t get_lt() {
  if (!s_lt) hipblasLtCreate(&s_lt);
  return s_lt;
}

static bool ensure_ws() {
  if (s_ws_sz >= WS_BYTES) return true;
  if (s_ws) { hipFree(s_ws); s_ws = nullptr; s_ws_sz = 0; }
  if (hipMalloc(&s_ws, WS_BYTES) != hipSuccess) return false;
  s_ws_sz = WS_BYTES;
  return true;
}

// Run one batched-strided GEMM via hipBLASLt.
// Layouts: physical matrices described in row-major (ld = #cols per row).
// Strides in BYTES between consecutive batches.
// beta is always 0 — C pointer is aliased to D.
static bool run_lt_gemm(
    hipStream_t stream, LtPlan &plan,
    bool transA, bool transB,
    int M, int N, int K, int batch,
    float alpha,
    hipDataType typeA, const void *A, int ldA, int64_t strA_elems,
    hipDataType typeB, const void *B, int ldB, int64_t strB_elems,
    hipDataType typeD, void *D, int ldD, int64_t strD_elems) {

  hipblasLtHandle_t h = get_lt();
  if (!h || !ensure_ws()) return false;

  hipblasOperation_t opA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  hipblasOperation_t opB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

  hipblasLtMatmulDesc_t   md;
  hipblasLtMatrixLayout_t layA, layB, layC, layD;

  hipblasLtMatmulDescCreate(&md, HIPBLAS_COMPUTE_32F, HIP_R_32F);
  hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
  hipblasLtMatmulDescSetAttribute(md, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

  // Physical layout: transA=N → M×K, transA=T → K×M (stored, then transposed)
  int64_t rA = transA ? K : M,  cA = transA ? M : K;
  int64_t rB = transB ? N : K,  cB = transB ? K : N;
  int64_t b64 = batch;

  int32_t ibatch = batch;
  hipblasLtMatrixLayoutCreate(&layA, typeA, rA, cA, ldA);
  hipblasLtMatrixLayoutSetAttribute(layA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,          &ibatch,      sizeof(ibatch));
  hipblasLtMatrixLayoutSetAttribute(layA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strA_elems,  sizeof(strA_elems));

  hipblasLtMatrixLayoutCreate(&layB, typeB, rB, cB, ldB);
  hipblasLtMatrixLayoutSetAttribute(layB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,          &ibatch,      sizeof(ibatch));
  hipblasLtMatrixLayoutSetAttribute(layB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strB_elems,  sizeof(strB_elems));

  hipblasLtMatrixLayoutCreate(&layC, typeD, M, N, ldD);
  hipblasLtMatrixLayoutSetAttribute(layC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,          &ibatch,      sizeof(ibatch));
  hipblasLtMatrixLayoutSetAttribute(layC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strD_elems,  sizeof(strD_elems));

  hipblasLtMatrixLayoutCreate(&layD, typeD, M, N, ldD);
  hipblasLtMatrixLayoutSetAttribute(layD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,          &ibatch,      sizeof(ibatch));
  hipblasLtMatrixLayoutSetAttribute(layD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strD_elems,  sizeof(strD_elems));

  bool need_plan = !plan.valid || plan.M != (uint32_t)M || plan.N != (uint32_t)N ||
                   plan.K != (uint32_t)K || plan.batch != (uint32_t)batch ||
                   plan.transA != transA || plan.transB != transB;

  if (need_plan) {
    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    size_t ws = s_ws_sz;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    hipblasLtMatmulHeuristicResult_t res;
    int ret = 0;
    hipblasLtMatmulAlgoGetHeuristic(h, md, layA, layB, layC, layD, pref, 1, &res, &ret);
    plan.valid    = true;
    plan.M = M; plan.N = N; plan.K = K; plan.batch = batch;
    plan.transA   = transA; plan.transB = transB;
    plan.has_algo = ret > 0;
    if (plan.has_algo) plan.algo = res.algo;
    hipblasLtMatmulPreferenceDestroy(pref);
  }

  const float beta = 0.0f;
  bool ok = (hipblasLtMatmul(h, md,
                              &alpha, A, layA, B, layB,
                              &beta,  D, layC, D, layD,
                              plan.has_algo ? &plan.algo : nullptr,
                              s_ws, s_ws_sz, stream) == HIPBLAS_STATUS_SUCCESS);

  hipblasLtMatrixLayoutDestroy(layD);
  hipblasLtMatrixLayoutDestroy(layC);
  hipblasLtMatrixLayoutDestroy(layB);
  hipblasLtMatrixLayoutDestroy(layA);
  hipblasLtMatmulDescDestroy(md);
  return ok;
}

// ---------------------------------------------------------------------------
// Scratch buffers
// ---------------------------------------------------------------------------

struct LtBufs {
  __half2 *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
  float   *scores = nullptr;
  __half  *probs  = nullptr;
  size_t   qkv_cap = 0, sc_cap = 0;
};
static LtBufs s_bufs;

static bool ensure_bufs(uint32_t seq_len, uint32_t heads, uint32_t head_dim) {
  size_t qkv = (size_t)seq_len * heads * (head_dim / 2);  // __half2 elements
  size_t sc  = (size_t)seq_len * seq_len * heads;

  if (s_bufs.qkv_cap < qkv) {
    hipFree(s_bufs.q); hipFree(s_bufs.k); hipFree(s_bufs.v); hipFree(s_bufs.o);
    s_bufs.q = s_bufs.k = s_bufs.v = s_bufs.o = nullptr;
    if (hipMalloc(&s_bufs.q, qkv * sizeof(__half2)) != hipSuccess) return false;
    if (hipMalloc(&s_bufs.k, qkv * sizeof(__half2)) != hipSuccess) return false;
    if (hipMalloc(&s_bufs.v, qkv * sizeof(__half2)) != hipSuccess) return false;
    if (hipMalloc(&s_bufs.o, qkv * sizeof(__half2)) != hipSuccess) return false;
    s_bufs.qkv_cap = qkv;
  }

  if (s_bufs.sc_cap < sc) {
    hipFree(s_bufs.scores); hipFree(s_bufs.probs);
    s_bufs.scores = nullptr; s_bufs.probs = nullptr;
    if (hipMalloc(&s_bufs.scores, sc * sizeof(float))  != hipSuccess) return false;
    if (hipMalloc(&s_bufs.probs,  sc * sizeof(__half)) != hipSuccess) return false;
    s_bufs.sc_cap = sc;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

extern "C" bool therock_flash_attn_prefill_hipblaslt(
    hipStream_t stream,
    uint32_t seq_len, uint32_t num_heads, uint32_t head_dim, float scale,
    const void *q, const void *k, const void *v, void *out) {

  if (head_dim & 1) return false;
  uint32_t head_pairs = head_dim / 2;
  if (!ensure_bufs(seq_len, num_heads, head_dim)) return false;

  // 1. Pack
  size_t total_pairs = (size_t)seq_len * num_heads * head_pairs;
  dim3 blk(256), grd((total_pairs + 255) / 256);
  hipLaunchKernelGGL(kernel_lt_pack, grd, blk, 0, stream,
    seq_len, num_heads, head_pairs,
    (const __half2 *)q, (const __half2 *)k, (const __half2 *)v,
    s_bufs.q, s_bufs.k, s_bufs.v);

  // 2. QK^T: Q[h,sq,d] @ K[h,sk,d]^T → scores[h,sq,sk] fp32
  // hipBLASLt is column-major. For row-major inputs:
  //   scores_rowmaj = Q_rowmaj @ K_rowmaj^T
  //   ≡ scores_colmaj^T = K_colmaj^T @ Q_colmaj
  //   i.e. C=K^T*Q in colmaj → M=sk=seq, N=sq=seq, K=d
  //   A=K (no transpose, colmaj [d,sk] = rowmaj [sk,d]), ldA=seq
  //   B=Q (transpose,    colmaj [d,sq] = rowmaj [sq,d]), ldB=seq  transB=T
  // But we store as [sq,d] row-major → ldA=ldB=head_dim, stride=sq*d
  // Simplest: pass K as A (transA=T meaning K^T[d,sk]), Q as B (transB=N, [sq,d])
  // M=sk=seq N=sq=seq K=d → C[sk,sq] = K^T[sk,d]*Q[d,sq] ✓
  {
    int64_t ldK = head_dim, ldQ = head_dim, ldS = seq_len;
    int64_t strK = (int64_t)seq_len * head_dim;
    int64_t strQ = (int64_t)seq_len * head_dim;
    int64_t strS = (int64_t)seq_len * seq_len;
    if (!run_lt_gemm(stream, s_qk_plan, true, false,
                     seq_len, seq_len, head_dim, num_heads, scale,
                     HIP_R_16F, s_bufs.k, ldK, strK,
                     HIP_R_16F, s_bufs.q, ldQ, strQ,
                     HIP_R_32F, s_bufs.scores, ldS, strS))
      return false;
  }

  // 3. Causal softmax: scores fp32 → probs fp16
  hipLaunchKernelGGL(kernel_lt_softmax,
    dim3(seq_len, num_heads), dim3(SOFTMAX_BLOCKS), 0, stream,
    seq_len, s_bufs.scores, s_bufs.probs);

  // 4. P@V: probs[h,s,s] fp16 @ V_packed[h,s,d] fp16 → o_packed[h,s,d] fp16
  //    M=seq N=dim K=seq  transA=N transB=N  alpha=1
  {
    int64_t ldP = seq_len, ldV = head_dim, ldO = head_dim;
    int64_t strP = (int64_t)seq_len * seq_len;   // elements
    int64_t strV = (int64_t)seq_len * head_dim;
    int64_t strO = (int64_t)seq_len * head_dim;
    if (!run_lt_gemm(stream, s_pv_plan, false, false,
                     seq_len, head_dim, seq_len, num_heads, 1.0f,
                     HIP_R_16F, s_bufs.probs, ldP, strP,
                     HIP_R_16F, s_bufs.v,     ldV, strV,
                     HIP_R_16F, s_bufs.o,     ldO, strO))
      return false;
  }

  // 5. Unpack
  hipLaunchKernelGGL(kernel_lt_unpack, grd, blk, 0, stream,
    seq_len, num_heads, head_pairs, s_bufs.o, (__half2 *)out);

  return hipGetLastError() == hipSuccess;
}

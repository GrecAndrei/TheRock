// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Transformer forward pass executor.
// Implements: embed → N × (RMSNorm → Attention → RMSNorm → MLP) → RMSNorm → lm_head

#include "therock/inference/executor.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#define HIP_CHECK(x) do { \
  hipError_t _e = (x); \
  if (_e != hipSuccess) { \
    fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
            _e, hipGetErrorString(_e), __FILE__, __LINE__); \
    return -1; \
  } \
} while(0)

// ---------------------------------------------------------------------------
// RMSNorm kernel (one wavefront per row)
// ---------------------------------------------------------------------------

// Declared in a separate .hip file; forward-declare here.
extern "C" void therock_rmsnorm_fp16(hipStream_t stream,
                                      uint32_t rows, uint32_t dim,
                                      const void *x, const float *scale,
                                      float eps, void *out);

// SiLU elementwise + multiply (fused gate*up for MLP)
extern "C" void therock_silu_mul_fp16(hipStream_t stream,
                                       uint32_t n,
                                       const void *gate, const void *up,
                                       void *out);

// Flash attention — decode (paged KV) and prefill (causal)
extern "C" void therock_flash_attn_decode(
    hipStream_t stream,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim,
    uint32_t total_kv_tokens, uint32_t page_tokens, float scale,
    const void *q, const void *k_pages, const void *v_pages,
    const uint32_t *page_table, void *out);

extern "C" void therock_flash_attn_prefill(
    hipStream_t stream,
    uint32_t seq_len, uint32_t num_heads, uint32_t head_dim, float scale,
    const void *q, const void *k, const void *v, void *out);

// ---------------------------------------------------------------------------
// Executor struct
// ---------------------------------------------------------------------------

struct TheRockExecutor {
  const TheRockHardwareDesc  *hw       = nullptr;
  const TheRockModelDesc     *model    = nullptr;
  TheRockKernelRegistry      *registry = nullptr;
  TheRockKVCache             *kv_cache = nullptr;
  TheRockPrefetcher          *prefetcher = nullptr;

  hipStream_t compute_stream = nullptr;

  // Scratch buffers (reused across layers)
  TheRockActivationPool *act_pool  = nullptr;
  void *scratch_attn  = nullptr;  // [max_tokens x hidden_dim] fp16
  void *scratch_mlp   = nullptr;  // [max_tokens x intermediate_dim] fp16 (gate)
  void *scratch_mlp2  = nullptr;  // [max_tokens x intermediate_dim] fp16 (up)
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TheRockExecutor *therock_executor_create(const TheRockHardwareDesc  *hw,
                                          const TheRockModelDesc     *model,
                                          TheRockKernelRegistry      *registry,
                                          TheRockKVCache             *kv_cache) {
  auto *ex = new TheRockExecutor{};
  ex->hw       = hw;
  ex->model    = model;
  ex->registry = registry;
  ex->kv_cache = kv_cache;

  hipStreamCreate(&ex->compute_stream);

  // Activation pool: double-buffer for hidden states
  // Max 2048 tokens for prefill; 1 for decode
  const uint32_t MAX_TOKENS = 2048;
  ex->act_pool = therock_actpool_create(hw, MAX_TOKENS, model->hidden_dim);

  // Scratch: attention output + MLP intermediates
  size_t attn_bytes  = (size_t)MAX_TOKENS * model->hidden_dim     * 2;
  size_t mlp_bytes   = (size_t)MAX_TOKENS * model->intermediate_dim * 2;
  hipMalloc(&ex->scratch_attn,  attn_bytes);
  hipMalloc(&ex->scratch_mlp,   mlp_bytes);
  hipMalloc(&ex->scratch_mlp2,  mlp_bytes);

  // Prefetcher
  if (model->layer_weights) {
    ex->prefetcher = therock_prefetcher_create(hw, model->num_layers,
                                                model->layer_weights);
  }

  return ex;
}

void therock_executor_destroy(TheRockExecutor *ex) {
  if (ex->prefetcher)     therock_prefetcher_destroy(ex->prefetcher);
  if (ex->act_pool)       therock_actpool_destroy(ex->act_pool);
  if (ex->scratch_attn)   hipFree(ex->scratch_attn);
  if (ex->scratch_mlp)    hipFree(ex->scratch_mlp);
  if (ex->scratch_mlp2)   hipFree(ex->scratch_mlp2);
  if (ex->compute_stream) hipStreamDestroy(ex->compute_stream);
  delete ex;
}

// ---------------------------------------------------------------------------
// Dispatch helper: choose GEMM vs GEMV based on M, dispatch via registry
// ---------------------------------------------------------------------------

static void dispatch_matmul(TheRockExecutor *ex, hipStream_t s,
                             uint32_t M, uint32_t N, uint32_t K,
                             const void *A, const void *W,
                             void *out) {
  TheRockOpDesc desc = {};
  desc.M             = M;
  desc.N             = N;
  desc.K             = K;
  desc.batch         = 1;
  desc.weight_layout = ex->model->weight_layout;
  desc.exec_mode     = (M <= therock_decode_threshold(ex->hw))
                         ? THEROCK_EXEC_DECODE
                         : THEROCK_EXEC_PREFILL;

  TheRockKernelFn fn = therock_registry_lookup(ex->registry, &desc);

  if (desc.exec_mode == THEROCK_EXEC_DECODE && fn.gemv)
    fn.gemv(s, &desc, A, W, nullptr, out);
  else if (fn.gemm)
    fn.gemm(s, &desc, A, W, nullptr, out);
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

int therock_executor_forward(TheRockExecutor      *ex,
                              const TheRockRequest *req,
                              void                 *logits_out) {
  const TheRockModelDesc *m = ex->model;
  hipStream_t s = ex->compute_stream;
  uint32_t T = req->num_tokens;

  // --- Token embedding lookup ---
  // TODO: custom gather kernel; for now copy via CPU path (bootstrap only)
  // embed: [T x hidden_dim] fp16 into act_pool write buffer
  // (placeholder: actual embed gather kernel goes here)

  // Kick off prefetch for first `lookahead` layers
  if (ex->prefetcher)
    therock_prefetcher_advance(ex->prefetcher, UINT32_MAX, s);

  // --- Transformer layers ---
  for (uint32_t l = 0; l < m->num_layers; l++) {
    void *h      = therock_actpool_read_buf(ex->act_pool);  // input hidden state
    void *h_out  = therock_actpool_write_buf(ex->act_pool); // output

    // Ensure this layer's weights are in cache
    if (ex->prefetcher)
      therock_prefetcher_ensure_ready(ex->prefetcher, l, s);

    // Prefetch next layers while we compute
    if (ex->prefetcher)
      therock_prefetcher_advance(ex->prefetcher, l, s);

    // --- Attention pre-norm ---
    therock_rmsnorm_fp16(s, T, m->hidden_dim, h, m->norm_attn[l],
                          1e-6f, ex->scratch_attn);

    // --- QKV projections ---
    uint32_t kv_dim = m->num_kv_heads * m->head_dim;

    // scratch_mlp  = Q   [T x hidden_dim]
    // scratch_mlp2 = K   [T x kv_dim]
    // scratch_attn = V   [T x kv_dim]  (reuse after norm, done with it)
    // h_out        = attn output [T x hidden_dim]
    dispatch_matmul(ex, s, T, m->hidden_dim, m->hidden_dim,
                    ex->scratch_attn, m->attn_q[l], ex->scratch_mlp);
    dispatch_matmul(ex, s, T, kv_dim,        m->hidden_dim,
                    ex->scratch_attn, m->attn_k[l], ex->scratch_mlp2);
    // V goes back into scratch_attn (we're done reading the normed input)
    dispatch_matmul(ex, s, T, kv_dim,        m->hidden_dim,
                    ex->scratch_attn, m->attn_v[l], ex->scratch_attn);

    // --- Flash attention ---
    float attn_scale = 1.0f / sqrtf((float)m->head_dim);
    if (T == 1) {
      // Decode: paged KV flash attention
      // For now use K/V directly from scratch (fresh projection, not cached yet)
      // TODO: integrate with KV cache pages + page_table from request
      therock_flash_attn_decode(
          s,
          m->num_heads, m->num_kv_heads, m->head_dim,
          T,           // total_kv_tokens (just the current token — bootstrap)
          T,           // page_tokens
          attn_scale,
          ex->scratch_mlp,   // Q
          ex->scratch_mlp2,  // K
          ex->scratch_attn,  // V
          nullptr,           // page_table (direct, no paging for bootstrap)
          h_out);
    } else {
      // Prefill: causal flash attention
      // K and V are the freshly projected tensors (no KV cache lookup yet)
      therock_flash_attn_prefill(
          s,
          T, m->num_heads, m->head_dim, attn_scale,
          ex->scratch_mlp,   // Q
          ex->scratch_mlp2,  // K
          ex->scratch_attn,  // V
          h_out);
    }

    // --- Output projection ---
    // Temporarily stash attn output in scratch_attn for the matmul
    // (h_out → scratch_attn → h_out via output proj)
    dispatch_matmul(ex, s, T, m->hidden_dim, m->hidden_dim,
                    h_out, m->attn_o[l], ex->scratch_attn);
    // TODO: fused residual add; for now h_out = output proj result
    // swap scratch_attn → h_out role (just update pointer in actpool — TODO)

    // Residual add: h_out += h  (TODO: fused into output projection)

    // --- MLP pre-norm ---
    therock_rmsnorm_fp16(s, T, m->hidden_dim, h_out, m->norm_mlp[l],
                          1e-6f, ex->scratch_attn);

    // --- MLP: gate and up projections ---
    dispatch_matmul(ex, s, T, m->intermediate_dim, m->hidden_dim,
                    ex->scratch_attn, m->mlp_gate[l], ex->scratch_mlp);
    dispatch_matmul(ex, s, T, m->intermediate_dim, m->hidden_dim,
                    ex->scratch_attn, m->mlp_up[l],   ex->scratch_mlp2);

    // --- SiLU(gate) * up → then down projection ---
    therock_silu_mul_fp16(s, T * m->intermediate_dim,
                           ex->scratch_mlp, ex->scratch_mlp2,
                           ex->scratch_mlp);

    dispatch_matmul(ex, s, T, m->hidden_dim, m->intermediate_dim,
                    ex->scratch_mlp, m->mlp_down[l], h_out);

    // Residual: h_out += h_pre_mlp  (TODO)

    therock_actpool_swap(ex->act_pool);
  }

  // --- Final norm + lm_head ---
  void *final_h = therock_actpool_read_buf(ex->act_pool);
  // Apply norm to last token only for decode efficiency
  uint32_t last = T - 1;
  // Use logits_out as scratch for normed final hidden state
  therock_rmsnorm_fp16(s, 1, m->hidden_dim,
                        static_cast<char*>(final_h) + last * m->hidden_dim * 2,
                        m->norm_final, 1e-6f, logits_out);

  // lm_head: [1 x hidden_dim] × [hidden_dim x vocab_size]
  dispatch_matmul(ex, s, 1, m->vocab_size, m->hidden_dim,
                  logits_out, m->embed_tokens,  // weight-tied in Gemma
                  logits_out);

  HIP_CHECK(hipStreamSynchronize(s));
  return 0;
}

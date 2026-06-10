// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Flash attention for wave32 (gfx103x, gfx11xx).
//
// Two kernels:
//
//   flash_attn_decode — M=1, paged KV cache, one wavefront per query head.
//     Online softmax over all KV pages. No full attention matrix materialized.
//     Memory: reads Q once, streams K/V pages sequentially → Infinity Cache
//     keeps hot pages resident across heads of the same layer.
//
//   flash_attn_prefill — M>1, causal mask, tiled QK^T in SRAM (LDS).
//     Tile size tuned to gfx1032 LDS (64KB per CU).
//     One workgroup per (query_tile, head).
//
// Both kernels implement the online softmax from Dao et al. 2022:
//   m_new = max(m_old, max(scores))
//   l_new = exp(m_old - m_new) * l_old + sum(exp(scores - m_new))
//   o_new = (exp(m_old - m_new) * o_old + sum(exp(s_i - m_new) * v_i)) / l_new

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

extern "C" bool therock_flash_attn_prefill_hipblaslt(
    hipStream_t stream,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t head_dim,
    float scale,
    const void *q,
    const void *k,
    const void *v,
    void *out);

static constexpr int WAVE       = 32;
static constexpr int HEAD_DIM   = 128;  // Gemma 4 head dim; parameterized below
static constexpr float NEG_INF  = -1e9f;

// ---------------------------------------------------------------------------
// Decode flash attention
//
// Grid:  (num_query_heads, batch=1)
// Block: (WAVE,)
// Each wavefront handles one query head, iterates over all KV pages.
//
// KV layout per page per layer: [page_tokens][num_kv_heads][head_dim] fp16
// (contiguous in the TheRockKVCache layout)
//
// GQA support: query_head → kv_head = query_head / (num_q_heads / num_kv_heads)
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(WAVE)
kernel_flash_attn_decode(
    uint32_t      num_q_heads,
    uint32_t      num_kv_heads,
    uint32_t      head_dim,
    uint32_t      total_kv_tokens,    // number of past tokens in KV cache
    uint32_t      page_tokens,        // tokens per KV page
    float         scale,              // 1/sqrt(head_dim)
    const __half * __restrict__ q,    // [num_q_heads x head_dim]
    const __half * __restrict__ k_pages,  // [num_pages x page_tokens x num_kv_heads x head_dim]
    const __half * __restrict__ v_pages,  // same layout
    const uint32_t * __restrict__ page_table,  // [ceil(total_kv_tokens/page_tokens)]
    __half       * __restrict__ out   // [num_q_heads x head_dim]
) {
  uint32_t qh  = blockIdx.x;
  uint32_t kvh = qh / (num_q_heads / num_kv_heads);  // GQA mapping
  int      tid = threadIdx.x;

  // Load query for this head into registers (HEAD_DIM / WAVE elements per thread)
  // Each thread owns head_dim/WAVE contiguous elements
  // For head_dim=128, WAVE=32: each thread owns 4 elements
  const int ELEMS_PER_THREAD = HEAD_DIM / WAVE;  // = 4 for standard config
  float q_reg[ELEMS_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; i++)
    q_reg[i] = __half2float(q[qh * head_dim + tid * ELEMS_PER_THREAD + i]);

  // Online softmax state
  float m = NEG_INF;   // running max
  float l = 0.0f;      // running sum(exp)
  float o[ELEMS_PER_THREAD] = {0.0f};  // running output accumulator

  uint32_t num_pages = (total_kv_tokens + page_tokens - 1) / page_tokens;

  for (uint32_t pg = 0; pg < num_pages; pg++) {
    uint32_t physical_page = page_table[pg];
    uint32_t tokens_this_page = min(page_tokens,
                                    total_kv_tokens - pg * page_tokens);

    // Base pointers for K and V in this page for our KV head
    // Layout: [page_tokens][num_kv_heads][head_dim]
    const __half *k_page = k_pages
      + (size_t)physical_page * page_tokens * num_kv_heads * head_dim
      + kvh * head_dim;
    const __half *v_page = v_pages
      + (size_t)physical_page * page_tokens * num_kv_heads * head_dim
      + kvh * head_dim;

    for (uint32_t t = 0; t < tokens_this_page; t++) {
      const __half *kt = k_page + t * num_kv_heads * head_dim;
      const __half *vt = v_page + t * num_kv_heads * head_dim;

      // Dot product q·k for this token: each thread computes partial sum
      float dot = 0.0f;
      #pragma unroll
      for (int i = 0; i < ELEMS_PER_THREAD; i++)
        dot += q_reg[i] * __half2float(kt[tid * ELEMS_PER_THREAD + i]);

      // Wave reduce dot product
      dot += __shfl_xor(dot, 16); dot += __shfl_xor(dot, 8);
      dot += __shfl_xor(dot, 4);  dot += __shfl_xor(dot, 2); dot += __shfl_xor(dot, 1);
      float s = dot * scale;  // scaled attention score

      // Online softmax update
      float m_new = (s > m) ? s : m;
      float exp_shift = expf(m - m_new);
      float exp_s     = expf(s - m_new);

      l = exp_shift * l + exp_s;

      // Update output accumulator
      #pragma unroll
      for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        float vi = __half2float(vt[tid * ELEMS_PER_THREAD + i]);
        o[i] = exp_shift * o[i] + exp_s * vi;
      }

      m = m_new;
    }
  }

  // Normalize by l and write output
  float l_inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
  #pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; i++)
    out[qh * head_dim + tid * ELEMS_PER_THREAD + i] = __float2half(o[i] * l_inv);
}

// ---------------------------------------------------------------------------
// Prefill flash attention (causal) — tiled LDS with uniform cooperative load
//
// Grid:  (ceil(seq_len / Q_TILE), num_heads)
// Block: (WAVE * Q_TILE,) — one wave per query row in tile
//
// Strategy: Q_TILE waves cooperatively load a KV tile into LDS. All waves use
// the SAME kv_len (the maximum needed by any row in the tile — the last row's
// causal bound). Individual rows apply the causal mask in the score computation
// via NEG_INF predication. This keeps K/V tile reuse across 16 query rows while
// replacing the per-lane 4x fp16 multiply chain with 2x v_dot2_f32_f16.
//
// LDS budget (gfx1032: 64KB per CU, target 2 blocks/CU = 32KB each):
//   K tile: KV_TILE * HEAD_DIM * 2 = 64*128*2 = 16KB
//   V tile: same = 16KB
//   Total: 32KB ✓ (allows 2 blocks per CU)
// ---------------------------------------------------------------------------

static constexpr uint32_t Q_TILE  = 16;   // query rows per block
static constexpr uint32_t KV_TILE = 64;   // KV tokens loaded per iteration
static constexpr uint32_t BLK_SZ  = WAVE * Q_TILE;  // 512 threads
static constexpr uint32_t HEAD_PAIRS = HEAD_DIM / 2;
static constexpr uint32_t PAIRS_PER_LANE = HEAD_DIM / WAVE / 2;

__global__ void __launch_bounds__(BLK_SZ)
kernel_flash_attn_prefill(
    uint32_t      seq_len,
    uint32_t      num_heads,
    uint32_t      head_dim,
    float         scale,
    const __half * __restrict__ q,   // [seq_len x num_heads x head_dim]
    const __half * __restrict__ k,   // [seq_len x num_heads x head_dim]
    const __half * __restrict__ v,   // [seq_len x num_heads x head_dim]
    __half       * __restrict__ out  // [seq_len x num_heads x head_dim]
) {
  uint32_t tile_start = blockIdx.x * Q_TILE;
  uint32_t head       = blockIdx.y;
  uint32_t wave_id    = threadIdx.x / WAVE;
  uint32_t lane       = threadIdx.x % WAVE;

  uint32_t q_row = tile_start + wave_id;

  const int EPT = HEAD_DIM / WAVE;
  const size_t row_stride = (size_t)num_heads * head_dim;
  const size_t row_stride2 = row_stride / 2;
  const __half2 *q2 = reinterpret_cast<const __half2 *>(q);
  const __half2 *k2 = reinterpret_cast<const __half2 *>(k);
  const __half2 *v2 = reinterpret_cast<const __half2 *>(v);

  __shared__ __half2 lds_k2[KV_TILE * HEAD_PAIRS];
  __shared__ __half2 lds_v2[KV_TILE * HEAD_PAIRS];

  __half2 q_reg0;
  __half2 q_reg1;
  if (q_row < seq_len) {
    size_t q_base = (size_t)q_row * row_stride2 + head * HEAD_PAIRS + lane * PAIRS_PER_LANE;
    q_reg0 = q2[q_base + 0];
    q_reg1 = q2[q_base + 1];
  } else {
    q_reg0 = __float2half2_rn(0.0f);
    q_reg1 = __float2half2_rn(0.0f);
  }

  float m = NEG_INF;
  float l = 0.0f;
  float o0 = 0.0f;
  float o1 = 0.0f;
  float o2 = 0.0f;
  float o3 = 0.0f;

  // Causal bound for entire tile: last row's bound (uniform across block)
  uint32_t tile_last_row = min(tile_start + Q_TILE - 1, seq_len - 1);
  uint32_t kv_total = tile_last_row + 1;

  for (uint32_t kv_start = 0; kv_start < kv_total; kv_start += KV_TILE) {
    uint32_t kv_len = min(KV_TILE, kv_total - kv_start);

    for (uint32_t e = threadIdx.x; e < KV_TILE * HEAD_PAIRS; e += BLK_SZ) {
      uint32_t tok = e / HEAD_PAIRS;
      uint32_t pair = e % HEAD_PAIRS;
      if (tok < kv_len) {
        size_t src = (size_t)(kv_start + tok) * row_stride2 + head * HEAD_PAIRS + pair;
        lds_k2[e] = k2[src];
        lds_v2[e] = v2[src];
      } else {
        lds_k2[e] = __float2half2_rn(0.0f);
        lds_v2[e] = __float2half2_rn(0.0f);
      }
    }
    __syncthreads();

    if (q_row < seq_len) {
      const uint32_t pair_base = lane * PAIRS_PER_LANE;
      uint32_t t = 0;
      for (; t + 4 <= kv_len; t += 4) {
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

        const __half2 *k0 = lds_k2 + (t + 0) * HEAD_PAIRS + pair_base;
        const __half2 *k1 = lds_k2 + (t + 1) * HEAD_PAIRS + pair_base;
        const __half2 *k2_row = lds_k2 + (t + 2) * HEAD_PAIRS + pair_base;
        const __half2 *k3 = lds_k2 + (t + 3) * HEAD_PAIRS + pair_base;

        d0 = __builtin_amdgcn_fdot2(q_reg0, k0[0], d0, false);
        d0 = __builtin_amdgcn_fdot2(q_reg1, k0[1], d0, false);
        d1 = __builtin_amdgcn_fdot2(q_reg0, k1[0], d1, false);
        d1 = __builtin_amdgcn_fdot2(q_reg1, k1[1], d1, false);
        d2 = __builtin_amdgcn_fdot2(q_reg0, k2_row[0], d2, false);
        d2 = __builtin_amdgcn_fdot2(q_reg1, k2_row[1], d2, false);
        d3 = __builtin_amdgcn_fdot2(q_reg0, k3[0], d3, false);
        d3 = __builtin_amdgcn_fdot2(q_reg1, k3[1], d3, false);

        d0 += __shfl_xor(d0, 16); d1 += __shfl_xor(d1, 16);
        d2 += __shfl_xor(d2, 16); d3 += __shfl_xor(d3, 16);
        d0 += __shfl_xor(d0, 8);  d1 += __shfl_xor(d1, 8);
        d2 += __shfl_xor(d2, 8);  d3 += __shfl_xor(d3, 8);
        d0 += __shfl_xor(d0, 4);  d1 += __shfl_xor(d1, 4);
        d2 += __shfl_xor(d2, 4);  d3 += __shfl_xor(d3, 4);
        d0 += __shfl_xor(d0, 2);  d1 += __shfl_xor(d1, 2);
        d2 += __shfl_xor(d2, 2);  d3 += __shfl_xor(d3, 2);
        d0 += __shfl_xor(d0, 1);  d1 += __shfl_xor(d1, 1);
        d2 += __shfl_xor(d2, 1);  d3 += __shfl_xor(d3, 1);

        float scores[4] = {d0, d1, d2, d3};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          float s_t = (kv_start + t + j <= q_row) ? scores[j] * scale : NEG_INF;
          float m_new = (s_t > m) ? s_t : m;
          float exp_shft = __expf(m - m_new);
          float exp_s = __expf(s_t - m_new);
          l = exp_shft * l + exp_s;

          const __half2 v01_h = lds_v2[(t + j) * HEAD_PAIRS + pair_base + 0];
          const __half2 v23_h = lds_v2[(t + j) * HEAD_PAIRS + pair_base + 1];
          float2 v01 = __half22float2(v01_h);
          float2 v23 = __half22float2(v23_h);
          o0 = exp_shft * o0 + exp_s * v01.x;
          o1 = exp_shft * o1 + exp_s * v01.y;
          o2 = exp_shft * o2 + exp_s * v23.x;
          o3 = exp_shft * o3 + exp_s * v23.y;
          m = m_new;
        }
      }

      for (; t < kv_len; t++) {
        const __half2 *k_row2 = lds_k2 + t * HEAD_PAIRS + pair_base;
        float dot = 0.0f;
        dot = __builtin_amdgcn_fdot2(q_reg0, k_row2[0], dot, false);
        dot = __builtin_amdgcn_fdot2(q_reg1, k_row2[1], dot, false);
        dot += __shfl_xor(dot, 16);
        dot += __shfl_xor(dot, 8);
        dot += __shfl_xor(dot, 4);
        dot += __shfl_xor(dot, 2);
        dot += __shfl_xor(dot, 1);

        float s_t = (kv_start + t <= q_row) ? dot * scale : NEG_INF;
        float m_new = (s_t > m) ? s_t : m;
        float exp_shft = __expf(m - m_new);
        float exp_s = __expf(s_t - m_new);
        l = exp_shft * l + exp_s;

        const __half2 v01_h = lds_v2[t * HEAD_PAIRS + pair_base + 0];
        const __half2 v23_h = lds_v2[t * HEAD_PAIRS + pair_base + 1];
        float2 v01 = __half22float2(v01_h);
        float2 v23 = __half22float2(v23_h);
        o0 = exp_shft * o0 + exp_s * v01.x;
        o1 = exp_shft * o1 + exp_s * v01.y;
        o2 = exp_shft * o2 + exp_s * v23.x;
        o3 = exp_shft * o3 + exp_s * v23.y;
        m = m_new;
      }
    }
    __syncthreads();
  }

  // Write output
  if (q_row < seq_len) {
    float l_inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    __half *orow = out + q_row * row_stride + head * head_dim;
    orow[lane * EPT + 0] = __float2half(o0 * l_inv);
    orow[lane * EPT + 1] = __float2half(o1 * l_inv);
    orow[lane * EPT + 2] = __float2half(o2 * l_inv);
    orow[lane * EPT + 3] = __float2half(o3 * l_inv);
  }
}

// ---------------------------------------------------------------------------
// C wrappers
// ---------------------------------------------------------------------------

extern "C" void therock_flash_attn_decode(
    hipStream_t stream,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim,
    uint32_t total_kv_tokens, uint32_t page_tokens, float scale,
    const void *q, const void *k_pages, const void *v_pages,
    const uint32_t *page_table, void *out)
{
  dim3 grid(num_q_heads);
  dim3 block(WAVE);
  hipLaunchKernelGGL(kernel_flash_attn_decode, grid, block, 0, stream,
    num_q_heads, num_kv_heads, head_dim,
    total_kv_tokens, page_tokens, scale,
    (const __half *)q, (const __half *)k_pages, (const __half *)v_pages,
    page_table, (__half *)out);
}

extern "C" void therock_flash_attn_prefill(
    hipStream_t stream,
    uint32_t seq_len, uint32_t num_heads, uint32_t head_dim, float scale,
    const void *q, const void *k, const void *v, void *out)
{
  if (seq_len >= 512 && therock_flash_attn_prefill_hipblaslt(
          stream, seq_len, num_heads, head_dim, scale, q, k, v, out))
    return;

  dim3 grid((seq_len + Q_TILE - 1) / Q_TILE, num_heads);
  dim3 block(BLK_SZ);
  hipLaunchKernelGGL(kernel_flash_attn_prefill, grid, block, 0, stream,
    seq_len, num_heads, head_dim, scale,
    (const __half *)q, (const __half *)k, (const __half *)v, (__half *)out);
}

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

    // 8-token unroll: process 8 tokens per iteration to maximize the dot
    // chain's ILP, hiding both the 5-cycle shuffle-reduce chain and the
    // V-load latency behind independent dot products. 5 shfl_xor's cover
    // all 8 reductions in 5 cycles (vs 5*8=40 serially).
    uint32_t t = 0;
    for (; t + 8 <= tokens_this_page; t += 8) {
      const __half *kt[8] = {
        k_page + (t + 0) * num_kv_heads * head_dim,
        k_page + (t + 1) * num_kv_heads * head_dim,
        k_page + (t + 2) * num_kv_heads * head_dim,
        k_page + (t + 3) * num_kv_heads * head_dim,
        k_page + (t + 4) * num_kv_heads * head_dim,
        k_page + (t + 5) * num_kv_heads * head_dim,
        k_page + (t + 6) * num_kv_heads * head_dim,
        k_page + (t + 7) * num_kv_heads * head_dim,
      };
      const __half *vt[8] = {
        v_page + (t + 0) * num_kv_heads * head_dim,
        v_page + (t + 1) * num_kv_heads * head_dim,
        v_page + (t + 2) * num_kv_heads * head_dim,
        v_page + (t + 3) * num_kv_heads * head_dim,
        v_page + (t + 4) * num_kv_heads * head_dim,
        v_page + (t + 5) * num_kv_heads * head_dim,
        v_page + (t + 6) * num_kv_heads * head_dim,
        v_page + (t + 7) * num_kv_heads * head_dim,
      };

      // Issue all 16 loads (8 K + 8 V) up front so the V loads can hide
      // behind the K dot+reduce work. Without this hoisting, the compiler
      // emits V loads inside the o-update loop, serialized after the K
      // reduce chain.
      const int load_off = tid * ELEMS_PER_THREAD;
      __half k0_0 = kt[0][load_off + 0]; __half k0_1 = kt[0][load_off + 1];
      __half k0_2 = kt[0][load_off + 2]; __half k0_3 = kt[0][load_off + 3];
      __half k1_0 = kt[1][load_off + 0]; __half k1_1 = kt[1][load_off + 1];
      __half k1_2 = kt[1][load_off + 2]; __half k1_3 = kt[1][load_off + 3];
      __half k2_0 = kt[2][load_off + 0]; __half k2_1 = kt[2][load_off + 1];
      __half k2_2 = kt[2][load_off + 2]; __half k2_3 = kt[2][load_off + 3];
      __half k3_0 = kt[3][load_off + 0]; __half k3_1 = kt[3][load_off + 1];
      __half k3_2 = kt[3][load_off + 2]; __half k3_3 = kt[3][load_off + 3];
      __half k4_0 = kt[4][load_off + 0]; __half k4_1 = kt[4][load_off + 1];
      __half k4_2 = kt[4][load_off + 2]; __half k4_3 = kt[4][load_off + 3];
      __half k5_0 = kt[5][load_off + 0]; __half k5_1 = kt[5][load_off + 1];
      __half k5_2 = kt[5][load_off + 2]; __half k5_3 = kt[5][load_off + 3];
      __half k6_0 = kt[6][load_off + 0]; __half k6_1 = kt[6][load_off + 1];
      __half k6_2 = kt[6][load_off + 2]; __half k6_3 = kt[6][load_off + 3];
      __half k7_0 = kt[7][load_off + 0]; __half k7_1 = kt[7][load_off + 1];
      __half k7_2 = kt[7][load_off + 2]; __half k7_3 = kt[7][load_off + 3];

      __half v0_0 = vt[0][load_off + 0]; __half v0_1 = vt[0][load_off + 1];
      __half v0_2 = vt[0][load_off + 2]; __half v0_3 = vt[0][load_off + 3];
      __half v1_0 = vt[1][load_off + 0]; __half v1_1 = vt[1][load_off + 1];
      __half v1_2 = vt[1][load_off + 2]; __half v1_3 = vt[1][load_off + 3];
      __half v2_0 = vt[2][load_off + 0]; __half v2_1 = vt[2][load_off + 1];
      __half v2_2 = vt[2][load_off + 2]; __half v2_3 = vt[2][load_off + 3];
      __half v3_0 = vt[3][load_off + 0]; __half v3_1 = vt[3][load_off + 1];
      __half v3_2 = vt[3][load_off + 2]; __half v3_3 = vt[3][load_off + 3];
      __half v4_0 = vt[4][load_off + 0]; __half v4_1 = vt[4][load_off + 1];
      __half v4_2 = vt[4][load_off + 2]; __half v4_3 = vt[4][load_off + 3];
      __half v5_0 = vt[5][load_off + 0]; __half v5_1 = vt[5][load_off + 1];
      __half v5_2 = vt[5][load_off + 2]; __half v5_3 = vt[5][load_off + 3];
      __half v6_0 = vt[6][load_off + 0]; __half v6_1 = vt[6][load_off + 1];
      __half v6_2 = vt[6][load_off + 2]; __half v6_3 = vt[6][load_off + 3];
      __half v7_0 = vt[7][load_off + 0]; __half v7_1 = vt[7][load_off + 1];
      __half v7_2 = vt[7][load_off + 2]; __half v7_3 = vt[7][load_off + 3];

      // 8 parallel dot products (independent FMA chains, no dependency)
      float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
      float d4 = 0.0f, d5 = 0.0f, d6 = 0.0f, d7 = 0.0f;
      d0 += q_reg[0] * __half2float(k0_0); d0 += q_reg[1] * __half2float(k0_1);
      d0 += q_reg[2] * __half2float(k0_2); d0 += q_reg[3] * __half2float(k0_3);
      d1 += q_reg[0] * __half2float(k1_0); d1 += q_reg[1] * __half2float(k1_1);
      d1 += q_reg[2] * __half2float(k1_2); d1 += q_reg[3] * __half2float(k1_3);
      d2 += q_reg[0] * __half2float(k2_0); d2 += q_reg[1] * __half2float(k2_1);
      d2 += q_reg[2] * __half2float(k2_2); d2 += q_reg[3] * __half2float(k2_3);
      d3 += q_reg[0] * __half2float(k3_0); d3 += q_reg[1] * __half2float(k3_1);
      d3 += q_reg[2] * __half2float(k3_2); d3 += q_reg[3] * __half2float(k3_3);
      d4 += q_reg[0] * __half2float(k4_0); d4 += q_reg[1] * __half2float(k4_1);
      d4 += q_reg[2] * __half2float(k4_2); d4 += q_reg[3] * __half2float(k4_3);
      d5 += q_reg[0] * __half2float(k5_0); d5 += q_reg[1] * __half2float(k5_1);
      d5 += q_reg[2] * __half2float(k5_2); d5 += q_reg[3] * __half2float(k5_3);
      d6 += q_reg[0] * __half2float(k6_0); d6 += q_reg[1] * __half2float(k6_1);
      d6 += q_reg[2] * __half2float(k6_2); d6 += q_reg[3] * __half2float(k6_3);
      d7 += q_reg[0] * __half2float(k7_0); d7 += q_reg[1] * __half2float(k7_1);
      d7 += q_reg[2] * __half2float(k7_2); d7 += q_reg[3] * __half2float(k7_3);

      // Interleaved wave-reduce: 8 dots, 5 xor offsets, 5 cycles total
      d0 += __shfl_xor(d0, 16); d1 += __shfl_xor(d1, 16);
      d2 += __shfl_xor(d2, 16); d3 += __shfl_xor(d3, 16);
      d4 += __shfl_xor(d4, 16); d5 += __shfl_xor(d5, 16);
      d6 += __shfl_xor(d6, 16); d7 += __shfl_xor(d7, 16);
      d0 += __shfl_xor(d0, 8);  d1 += __shfl_xor(d1, 8);
      d2 += __shfl_xor(d2, 8);  d3 += __shfl_xor(d3, 8);
      d4 += __shfl_xor(d4, 8);  d5 += __shfl_xor(d5, 8);
      d6 += __shfl_xor(d6, 8);  d7 += __shfl_xor(d7, 8);
      d0 += __shfl_xor(d0, 4);  d1 += __shfl_xor(d1, 4);
      d2 += __shfl_xor(d2, 4);  d3 += __shfl_xor(d3, 4);
      d4 += __shfl_xor(d4, 4);  d5 += __shfl_xor(d5, 4);
      d6 += __shfl_xor(d6, 4);  d7 += __shfl_xor(d7, 4);
      d0 += __shfl_xor(d0, 2);  d1 += __shfl_xor(d1, 2);
      d2 += __shfl_xor(d2, 2);  d3 += __shfl_xor(d3, 2);
      d4 += __shfl_xor(d4, 2);  d5 += __shfl_xor(d5, 2);
      d6 += __shfl_xor(d6, 2);  d7 += __shfl_xor(d7, 2);
      d0 += __shfl_xor(d0, 1);  d1 += __shfl_xor(d1, 1);
      d2 += __shfl_xor(d2, 1);  d3 += __shfl_xor(d3, 1);
      d4 += __shfl_xor(d4, 1);  d5 += __shfl_xor(d5, 1);
      d6 += __shfl_xor(d6, 1);  d7 += __shfl_xor(d7, 1);

      float s0 = d0 * scale, s1 = d1 * scale, s2 = d2 * scale, s3 = d3 * scale;
      float s4 = d4 * scale, s5 = d5 * scale, s6 = d6 * scale, s7 = d7 * scale;

      float s[8] = {s0, s1, s2, s3, s4, s5, s6, s7};

      // Online softmax: one max update for all 8 tokens
      float m_new = m;
      #pragma unroll
      for (int k = 0; k < 8; k++) m_new = (s[k] > m_new) ? s[k] : m_new;

      float exp_shift = __expf(m - m_new);
      float e[8];
      #pragma unroll
      for (int k = 0; k < 8; k++) e[k] = __expf(s[k] - m_new);

      float e_sum = 0.0f;
      #pragma unroll
      for (int k = 0; k < 8; k++) e_sum += e[k];
      l = exp_shift * l + e_sum;

      // 8 parallel output updates (V already loaded)
      o[0] = exp_shift * o[0] + (e[0] * __half2float(v0_0) + e[1] * __half2float(v1_0) +
                                  e[2] * __half2float(v2_0) + e[3] * __half2float(v3_0) +
                                  e[4] * __half2float(v4_0) + e[5] * __half2float(v5_0) +
                                  e[6] * __half2float(v6_0) + e[7] * __half2float(v7_0));
      o[1] = exp_shift * o[1] + (e[0] * __half2float(v0_1) + e[1] * __half2float(v1_1) +
                                  e[2] * __half2float(v2_1) + e[3] * __half2float(v3_1) +
                                  e[4] * __half2float(v4_1) + e[5] * __half2float(v5_1) +
                                  e[6] * __half2float(v6_1) + e[7] * __half2float(v7_1));
      o[2] = exp_shift * o[2] + (e[0] * __half2float(v0_2) + e[1] * __half2float(v1_2) +
                                  e[2] * __half2float(v2_2) + e[3] * __half2float(v3_2) +
                                  e[4] * __half2float(v4_2) + e[5] * __half2float(v5_2) +
                                  e[6] * __half2float(v6_2) + e[7] * __half2float(v7_2));
      o[3] = exp_shift * o[3] + (e[0] * __half2float(v0_3) + e[1] * __half2float(v1_3) +
                                  e[2] * __half2float(v2_3) + e[3] * __half2float(v3_3) +
                                  e[4] * __half2float(v4_3) + e[5] * __half2float(v5_3) +
                                  e[6] * __half2float(v6_3) + e[7] * __half2float(v7_3));

      m = m_new;
    }

    // Tail: <8 leftover tokens, fall back to the serialized path
    for (; t < tokens_this_page; t++) {
      const __half *kt = k_page + t * num_kv_heads * head_dim;
      const __half *vt = v_page + t * num_kv_heads * head_dim;

      float dot = 0.0f;
      #pragma unroll
      for (int i = 0; i < ELEMS_PER_THREAD; i++)
        dot += q_reg[i] * __half2float(kt[tid * ELEMS_PER_THREAD + i]);

      dot += __shfl_xor(dot, 16); dot += __shfl_xor(dot, 8);
      dot += __shfl_xor(dot, 4);  dot += __shfl_xor(dot, 2); dot += __shfl_xor(dot, 1);
      float s = dot * scale;

      float m_new = (s > m) ? s : m;
      float exp_shift = __expf(m - m_new);
      float exp_s     = __expf(s - m_new);

      l = exp_shift * l + exp_s;

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

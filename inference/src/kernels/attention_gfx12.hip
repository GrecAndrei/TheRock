// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/attention.hpp"
#include "rxinfer/common.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace rxinfer::attn {
namespace {

using h2_t = __half2;
using v2f16 = _Float16 __attribute__((ext_vector_type(2)));
using v2f32 = float __attribute__((ext_vector_type(2)));
using f16x8 = _Float16 __attribute__((ext_vector_type(8)));
using f32x8 = float __attribute__((ext_vector_type(8)));

static constexpr float LOG2E = 1.4426950408889634f;
static constexpr double TWO_PI = 6.283185307179586476925286766559;

__device__ __forceinline__ float sigmoid_f32(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float fdot2_h2(h2_t a, h2_t b, float acc) {
    v2f16 va, vb;
    __builtin_memcpy(&va, &a, 4);
    __builtin_memcpy(&vb, &b, 4);
    return __builtin_amdgcn_fdot2(va, vb, acc, false);
}

// Convert positive OCP FP8 E4M3 scale byte -> float using native gfx12 cvt_pk_f32_fp8.
__device__ __forceinline__ float fp8_e4m3_to_f32(uint8_t b) {
    v2f32 v = __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<unsigned int>(b), 0);
    return v[0];
}

// Quantize float x -> 4-bit E2M1 nibble (0..15) assuming x has already been divided by scale (>0).
// E2M1 levels: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (midpoint thresholds: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0).
__device__ __forceinline__ uint32_t quant_f32_to_e2m1_nibble(float x) {
    const uint32_t sign = (x < 0.0f) ? 8u : 0u;
    const float ax = fabsf(x);
    uint32_t mag = 0;
    if (ax >= 5.0f) mag = 7;
    else if (ax >= 3.5f) mag = 6;
    else if (ax >= 2.5f) mag = 5;
    else if (ax >= 1.75f) mag = 4;
    else if (ax >= 1.25f) mag = 3;
    else if (ax >= 0.75f) mag = 2;
    else if (ax >= 0.25f) mag = 1;
    else mag = 0;
    return sign | mag;
}

// Dequantize 2 E2M1 nibbles packed in one byte (lo nibble = even elem, hi nibble = odd elem) -> half2!
// IEEE FP16 representations of E2M1 magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6} ALL have low byte 0x00,
// and high bytes {0x00, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46}.
__device__ __forceinline__ h2_t dequant_e2m1_byte_to_h2(uint8_t byte_val) {
    const uint32_t u0 = byte_val & 0x0Fu;
    const uint32_t u1 = (byte_val >> 4) & 0x0Fu;
    const uint32_t pair_idx = u0 | (u1 << 8);
    const uint32_t mag_hi = __builtin_amdgcn_perm(0x46444240u, 0x3E3C3800u, pair_idx & 0x0707u);
    const uint32_t sign_hi = (pair_idx & 0x0808u) << 4;
    const uint32_t hi_bytes = mag_hi | sign_hi;
    const uint32_t packed_h2 = __builtin_amdgcn_perm(hi_bytes, 0u, 0x05000400u);
    h2_t res;
    __builtin_memcpy(&res, &packed_h2, 4);
    return res;
}

__device__ __forceinline__ int32_t select_mrope_pos(int pair_idx,
                                                    int32_t pos_default,
                                                    const int32_t* mrope_pos,
                                                    const int32_t* mrope_sec,
                                                    bool interleaved) {
    if (!mrope_pos || (mrope_sec[0] + mrope_sec[1] + mrope_sec[2]) == 0) {
        return pos_default;
    }
    if (interleaved) {
        const int mod3 = pair_idx % 3;
        if (mod3 == 1 && pair_idx < mrope_sec[1] * 3) return mrope_pos[1];
        if (mod3 == 2 && pair_idx < mrope_sec[2] * 3) return mrope_pos[2];
        return mrope_pos[0];
    } else {
        if (pair_idx < mrope_sec[0]) return mrope_pos[0];
        if (pair_idx < mrope_sec[0] + mrope_sec[1]) return mrope_pos[1];
        return mrope_pos[2];
    }
}

// =============================================================================
// Fused QK-Norm + Double-Precision RoPE/mRoPE + Paged Head-Major KV Write
//   1 Wave32 (32 threads) per (token m, head_slot)
// =============================================================================
__global__ __launch_bounds__(32)
void fused_qknorm_rope_kv_write_kernel(const __half* __restrict__ q_in,
                                       const __half* __restrict__ k_in,
                                       const __half* __restrict__ v_in,
                                       const __half* __restrict__ q_norm_w,
                                       const __half* __restrict__ k_norm_w,
                                       __half* __restrict__ q_out,
                                       KVCacheView kv,
                                       AttentionParams p,
                                       int M,
                                       int32_t pos_offset,
                                       const StepParams* __restrict__ d_step,
                                       int kv_in_stride) {
    __shared__ float s_head[256];

    const int lane = threadIdx.x & 31;
    const int slot = blockIdx.x;
    const int m = blockIdx.y;
    if (m >= M) return;

    const int H_q = p.num_q_heads;
    const int H_kv = p.num_kv_heads;
    const int D = p.head_dim;
    const int kv_row_elems = (kv_in_stride > 0) ? kv_in_stride : (H_kv * D);

    const int32_t pos = d_step ? (d_step->pos + m) : (pos_offset + m);
    int32_t mrope_p[3];
    if (d_step) {
        mrope_p[0] = d_step->mrope_pos[0] + m;
        mrope_p[1] = d_step->mrope_pos[1] + m;
        mrope_p[2] = d_step->mrope_pos[2] + m;
    } else {
        mrope_p[0] = pos;
        mrope_p[1] = pos;
        mrope_p[2] = pos;
    }

    const bool is_q = (slot < H_q);
    const bool is_k = (slot >= H_q && slot < H_q + H_kv);
    const bool is_v = (slot >= H_q + H_kv);

    const __half* src_ptr = nullptr;
    const __half* norm_w = nullptr;
    int head_idx = 0;

    if (is_q) {
        head_idx = slot;
        const size_t q_stride_m = static_cast<size_t>(H_q) * (p.attn_output_gate ? 2 * D : D);
        const size_t q_stride_h = static_cast<size_t>(p.attn_output_gate ? 2 * D : D);
        src_ptr = q_in + static_cast<size_t>(m) * q_stride_m + static_cast<size_t>(head_idx) * q_stride_h;
        norm_w = q_norm_w;
    } else if (is_k) {
        head_idx = slot - H_q;
        src_ptr = k_in + static_cast<size_t>(m) * kv_row_elems + static_cast<size_t>(head_idx) * D;
        norm_w = k_norm_w;
    } else {
        head_idx = slot - H_q - H_kv;
        src_ptr = v_in + static_cast<size_t>(m) * kv_row_elems + static_cast<size_t>(head_idx) * D;
    }

    // 1. Load vector into s_head[0..D-1] and compute sum_sq for Q/K norm if needed
    float sum_sq = 0.0f;
    for (int d = lane; d < D; d += 32) {
        const float val = __half2float(src_ptr[d]);
        s_head[d] = val;
        sum_sq += val * val;
    }

    // 2. QK-Norm
    if (!is_v && p.qk_norm && norm_w != nullptr) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_sq += __shfl_xor(sum_sq, off);
        }
        const float inv_rms = rsqrtf(sum_sq / static_cast<float>(D) + p.rms_norm_eps);
        for (int d = lane; d < D; d += 32) {
            const float w = __half2float(norm_w[d]);
            const float scale = p.qk_norm_plus_one ? (1.0f + w) : w;
            s_head[d] *= inv_rms * scale;
        }
    }
    __syncthreads();

    // 3. RoPE with double-precision inv_freq & 2*pi range reduction (preserves full precision at 200k+ ctx)
    if (!is_v && p.rope_kind != RopeKind::None && p.rotary_dim > 0) {
        const int half_rot = p.rotary_dim >> 1;
        for (int i = lane; i < half_rot; i += 32) {
            const int32_t tok_pos = select_mrope_pos(i, pos, mrope_p, p.mrope_section, p.mrope_interleaved);
            const double inv_freq = pow(static_cast<double>(p.rope_theta),
                                        -2.0 * static_cast<double>(i) / static_cast<double>(p.rotary_dim));
            const double prod = static_cast<double>(tok_pos) * inv_freq;
            const double angle_red = prod - TWO_PI * floor(prod / TWO_PI);
            float s_ang, c_ang;
            sincosf(static_cast<float>(angle_red), &s_ang, &c_ang);

            const int i0 = (p.rope_kind == RopeKind::NeoX) ? i : (2 * i);
            const int i1 = (p.rope_kind == RopeKind::NeoX) ? (i + half_rot) : (2 * i + 1);
            const float x0 = s_head[i0];
            const float x1 = s_head[i1];
            s_head[i0] = x0 * c_ang - x1 * s_ang;
            s_head[i1] = x1 * c_ang + x0 * s_ang;
        }
    }
    __syncthreads();

    // 4. Store Q output or quantize & write K/V into paged head-major cache at position `pos`
    if (is_q) {
        __half* dst = q_out + (static_cast<size_t>(m) * H_q + head_idx) * D;
        for (int d = lane; d < D; d += 32) {
            dst[d] = __float2half(s_head[d]);
        }
        return;
    }

    const int32_t pg = pos / KV_PAGE_TOKENS;
    const int32_t tok = pos % KV_PAGE_TOKENS;
    uint8_t* pg_base = kv.page_ptrs[pg];

    const int32_t row_data_bytes =
        (p.kv_dtype == DType::F16) ? (D * 2) : ((p.kv_dtype == DType::FP8_E4M3) ? D : (D >> 1));
    const int32_t row_sc_bytes =
        (p.kv_dtype == DType::FP8_E4M3) ? 2 : ((p.kv_dtype == DType::FP4_E2M1_G16) ? (D >> 4) : 0);

    const size_t data_off = (is_k ? 0 : kv.data_sec_bytes) +
                            (static_cast<size_t>(head_idx) * KV_PAGE_TOKENS + tok) * row_data_bytes;
    uint8_t* dst_data = pg_base + data_off;

    if (p.kv_dtype == DType::F16) {
        __half* dst = reinterpret_cast<__half*>(dst_data);
        for (int d = lane; d < D; d += 32) {
            dst[d] = __float2half(s_head[d]);
        }
    } else if (p.kv_dtype == DType::FP8_E4M3) {
        const size_t sc_off = 2 * kv.data_sec_bytes + (is_k ? 0 : kv.sc_sec_bytes) +
                              (static_cast<size_t>(head_idx) * KV_PAGE_TOKENS + tok) * row_sc_bytes;
        __half* dst_sc = reinterpret_cast<__half*>(pg_base + sc_off);

        float amax = 0.0f;
        for (int d = lane; d < D; d += 32) {
            amax = fmaxf(amax, fabsf(s_head[d]));
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor(amax, off));
        }
        const float sc = (amax > 1e-12f) ? (amax / 448.0f) : 1.0f;
        const float inv_sc = (amax > 1e-12f) ? (448.0f / amax) : 0.0f;
        if (lane == 0) {
            *dst_sc = __float2half(sc);
        }
        for (int d = lane * 2; d < D; d += 64) {
            const int pk = __builtin_amdgcn_cvt_pk_fp8_f32(s_head[d] * inv_sc, s_head[d + 1] * inv_sc, 0, false);
            *reinterpret_cast<uint16_t*>(dst_data + d) = static_cast<uint16_t>(pk & 0xFFFF);
        }
    } else if (p.kv_dtype == DType::FP4_E2M1_G16) {
        const int num_groups = D >> 4;
        const size_t sc_off = 2 * kv.data_sec_bytes + (is_k ? 0 : kv.sc_sec_bytes) +
                              (static_cast<size_t>(head_idx) * KV_PAGE_TOKENS + tok) * row_sc_bytes;
        uint8_t* dst_s = pg_base + sc_off;

        for (int g0 = 0; g0 < num_groups; g0 += 2) {
            const int g = g0 + (lane >> 4);
            const int c = lane & 15;
            const float x = (g < num_groups) ? s_head[g * 16 + c] : 0.0f;
            float amax = fabsf(x);
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                amax = fmaxf(amax, __shfl_xor(amax, off));
            }
            float raw_sc = amax / 6.0f;
            if (raw_sc > 448.0f) raw_sc = 448.0f;
            const int pk_sc = __builtin_amdgcn_cvt_pk_fp8_f32(raw_sc, 0.0f, 0, false);
            uint8_t sc_u8 = (amax > 1e-12f) ? static_cast<uint8_t>(pk_sc & 0x7F) : 0u;
            if (amax > 1e-12f && sc_u8 == 0u) sc_u8 = 1u;
            const float sc_recon = (sc_u8 != 0) ? fp8_e4m3_to_f32(sc_u8) : 1.0f;
            const float inv_sc = (sc_u8 != 0) ? (1.0f / sc_recon) : 0.0f;

            const uint32_t nib = quant_f32_to_e2m1_nibble(x * inv_sc);
            const uint32_t nib_odd = __shfl_xor(nib, 1);
            if (g < num_groups) {
                if (c == 0) dst_s[g] = sc_u8;
                if ((c & 1) == 0) {
                    dst_data[g * 8 + (c >> 1)] = static_cast<uint8_t>(nib | (nib_odd << 4));
                }
            }
        }
    }
}

// =============================================================================
// GQA-Grouped Paged Split-KV Decode Attention (M <= 16, Wave32 online softmax)
//   Each block handles up to G_CHUNK Q-heads sharing 1 KV head!
//   Loads K and V once from paged head-major cache across all G_CHUNK Q-heads.
// =============================================================================
template <int D, DType KV_DT, int G_CHUNK, int N_WARPS = 4>
__global__ __launch_bounds__(N_WARPS * 32)
void gqa_decode_paged_kernel(const __half* __restrict__ q_in,
                             KVCacheView kv,
                             float* __restrict__ scratch,
                             AttentionParams p,
                             int M,
                             int32_t fixed_kv_len,
                             const StepParams* __restrict__ d_step,
                             int splits) {
    constexpr int ELEMS = D / 32; // 2, 4, or 8
    constexpr int PAIRS = ELEMS / 2;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int H_q = p.num_q_heads;
    const int H_kv = p.num_kv_heads;
    const int gqa_ratio = H_q / H_kv;
    const int g_chunks = (gqa_ratio + G_CHUNK - 1) / G_CHUNK;

    const int h_kv = blockIdx.x / g_chunks;
    const int g_blk = (blockIdx.x % g_chunks) * G_CHUNK;
    const int split_id = blockIdx.y;
    const int m = blockIdx.z;
    if (m >= M) return;

    const int32_t kv_len = d_step ? d_step->kv_seq_len : fixed_kv_len;
    const int32_t total_kv = kv_len - (M - 1 - m);
    const int32_t t_start = (total_kv > 0) ? static_cast<int32_t>((static_cast<int64_t>(total_kv) * split_id) / splits) : 0;
    const int32_t t_end = (total_kv > 0) ? static_cast<int32_t>((static_cast<int64_t>(total_kv) * (split_id + 1)) / splits) : 0;

    const float scale = (p.attn_scale > 0.0f) ? p.attn_scale : rsqrtf(static_cast<float>(D));
    const float scale_log2 = scale * LOG2E;

    const int d0 = lane * ELEMS;
    h2_t q_h2[G_CHUNK][PAIRS];
    float o_acc[G_CHUNK][ELEMS];
    float m_max[G_CHUNK];
    float l_sum[G_CHUNK];

#pragma unroll
    for (int g = 0; g < G_CHUNK; ++g) {
        m_max[g] = -FLT_MAX;
        l_sum[g] = 0.0f;
#pragma unroll
        for (int i = 0; i < ELEMS; ++i) o_acc[g][i] = 0.0f;
        const int h_in_group = g_blk + g;
        if (h_in_group < gqa_ratio) {
            const int hq = h_kv * gqa_ratio + h_in_group;
            const __half* qp = q_in + (static_cast<size_t>(m) * H_q + hq) * D + d0;
#pragma unroll
            for (int i = 0; i < PAIRS; ++i) q_h2[g][i] = *reinterpret_cast<const h2_t*>(qp + i * 2);
        }
    }

    constexpr int32_t ROW_DATA_BYTES =
        (KV_DT == DType::F16) ? (D * 2) : ((KV_DT == DType::FP8_E4M3) ? D : (D >> 1));
    constexpr int32_t ROW_SC_BYTES =
        (KV_DT == DType::FP8_E4M3) ? 2 : ((KV_DT == DType::FP4_E2M1_G16) ? (D >> 4) : 0);

    const size_t h_data_off = static_cast<size_t>(h_kv) * KV_PAGE_TOKENS * ROW_DATA_BYTES;
    const size_t h_sc_off = static_cast<size_t>(h_kv) * KV_PAGE_TOKENS * ROW_SC_BYTES;
    const int grp = d0 >> 4;

    if (t_start < t_end) {
        const int pg_start = t_start / KV_PAGE_TOKENS;
        const int pg_end = (t_end - 1) / KV_PAGE_TOKENS;

        for (int pg = pg_start; pg <= pg_end; ++pg) {
            const uint8_t* pg_base = kv.page_ptrs[pg];
            const int tok_lo = (pg == pg_start) ? (t_start % KV_PAGE_TOKENS) : 0;
            const int tok_hi = (pg == pg_end) ? (t_end - pg * KV_PAGE_TOKENS) : KV_PAGE_TOKENS;

            const uint8_t* hk_d = pg_base + h_data_off;
            const uint8_t* hv_d = pg_base + kv.data_sec_bytes + h_data_off;
            const uint8_t* hk_s = pg_base + 2 * kv.data_sec_bytes + h_sc_off;
            const uint8_t* hv_s = pg_base + 2 * kv.data_sec_bytes + kv.sc_sec_bytes + h_sc_off;

            for (int tok = tok_lo + warp; tok < tok_hi; tok += N_WARPS) {
                h2_t k_h2[PAIRS];
                h2_t v_h2[PAIRS];
                float k_sc = 1.0f;
                float v_sc = 1.0f;

                if constexpr (KV_DT == DType::F16) {
                    const __half* kp = reinterpret_cast<const __half*>(hk_d + static_cast<size_t>(tok) * ROW_DATA_BYTES) + d0;
                    const __half* vp = reinterpret_cast<const __half*>(hv_d + static_cast<size_t>(tok) * ROW_DATA_BYTES) + d0;
#pragma unroll
                    for (int i = 0; i < PAIRS; ++i) {
                        k_h2[i] = *reinterpret_cast<const h2_t*>(kp + i * 2);
                        v_h2[i] = *reinterpret_cast<const h2_t*>(vp + i * 2);
                    }
                } else if constexpr (KV_DT == DType::FP8_E4M3) {
                    const uint8_t* kp = hk_d + static_cast<size_t>(tok) * ROW_DATA_BYTES + d0;
                    const uint8_t* vp = hv_d + static_cast<size_t>(tok) * ROW_DATA_BYTES + d0;
                    k_sc = __half2float(reinterpret_cast<const __half*>(hk_s)[tok]);
                    v_sc = __half2float(reinterpret_cast<const __half*>(hv_s)[tok]);
#pragma unroll
                    for (int i = 0; i < PAIRS; ++i) {
                        const uint32_t kb2 = *reinterpret_cast<const uint16_t*>(kp + i * 2);
                        const uint32_t vb2 = *reinterpret_cast<const uint16_t*>(vp + i * 2);
                        const v2f32 kf2 = __builtin_amdgcn_cvt_pk_f32_fp8(kb2, 0);
                        const v2f32 vf2 = __builtin_amdgcn_cvt_pk_f32_fp8(vb2, 0);
                        k_h2[i] = __floats2half2_rn(kf2[0], kf2[1]);
                        v_h2[i] = __floats2half2_rn(vf2[0], vf2[1]);
                    }
                } else if constexpr (KV_DT == DType::FP4_E2M1_G16) {
                    const uint8_t* kp = hk_d + static_cast<size_t>(tok) * ROW_DATA_BYTES + (d0 >> 1);
                    const uint8_t* vp = hv_d + static_cast<size_t>(tok) * ROW_DATA_BYTES + (d0 >> 1);
                    k_sc = fp8_e4m3_to_f32(hk_s[static_cast<size_t>(tok) * ROW_SC_BYTES + grp]);
                    v_sc = fp8_e4m3_to_f32(hv_s[static_cast<size_t>(tok) * ROW_SC_BYTES + grp]);
#pragma unroll
                    for (int i = 0; i < PAIRS; ++i) {
                        k_h2[i] = dequant_e2m1_byte_to_h2(kp[i]);
                        v_h2[i] = dequant_e2m1_byte_to_h2(vp[i]);
                    }
                }

                float dots[G_CHUNK];
#pragma unroll
                for (int g = 0; g < G_CHUNK; ++g) {
                    float d = 0.0f;
#pragma unroll
                    for (int i = 0; i < PAIRS; ++i) d = fdot2_h2(q_h2[g][i], k_h2[i], d);
                    dots[g] = d * k_sc;
                }
#pragma unroll
                for (int g = 0; g < G_CHUNK; ++g) {
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) dots[g] += __shfl_xor(dots[g], off);
                }

                float2 vf[PAIRS];
#pragma unroll
                for (int i = 0; i < PAIRS; ++i) vf[i] = __half22float2(v_h2[i]);

#pragma unroll
                for (int g = 0; g < G_CHUNK; ++g) {
                    if (g_blk + g < gqa_ratio) {
                        const float score = dots[g] * scale_log2;
                        const float m_new = fmaxf(m_max[g], score);
                        const float alpha = (m_max[g] > -FLT_MAX * 0.5f) ? exp2f(m_max[g] - m_new) : 0.0f;
                        const float w = exp2f(score - m_new);
                        l_sum[g] = l_sum[g] * alpha + w;
                        m_max[g] = m_new;
                        const float wv = w * v_sc;
#pragma unroll
                        for (int i = 0; i < PAIRS; ++i) {
                            o_acc[g][i * 2 + 0] = o_acc[g][i * 2 + 0] * alpha + wv * vf[i].x;
                            o_acc[g][i * 2 + 1] = o_acc[g][i * 2 + 1] * alpha + wv * vf[i].y;
                        }
                    }
                }
            }
        }
    }

    __shared__ float s_m[N_WARPS];
    __shared__ float s_l[N_WARPS];
    __shared__ float s_o[N_WARPS][D];

    for (int g = 0; g < G_CHUNK; ++g) {
        const int h_in_group = g_blk + g;
        if (h_in_group >= gqa_ratio) break;

        if (lane == 0) {
            s_m[warp] = m_max[g];
            s_l[warp] = l_sum[g];
        }
#pragma unroll
        for (int i = 0; i < ELEMS; ++i) s_o[warp][d0 + i] = o_acc[g][i];
        __syncthreads();

        const int hq = h_kv * gqa_ratio + h_in_group;
        const size_t slot_idx = (static_cast<size_t>(m) * H_q + hq) * splits + split_id;
        float* dst_slot = scratch + slot_idx * (2 + D);

        float gm = s_m[0];
#pragma unroll
        for (int w = 1; w < N_WARPS; ++w) gm = fmaxf(gm, s_m[w]);

        float gl = 0.0f;
        if (gm > -FLT_MAX * 0.5f) {
#pragma unroll
            for (int w = 0; w < N_WARPS; ++w) {
                if (s_l[w] > 0.0f) gl += s_l[w] * exp2f(s_m[w] - gm);
            }
        }
        if (tid == 0) {
            dst_slot[0] = gm;
            dst_slot[1] = gl;
        }
        for (int d = tid; d < D; d += N_WARPS * 32) {
            float go = 0.0f;
            if (gm > -FLT_MAX * 0.5f) {
#pragma unroll
                for (int w = 0; w < N_WARPS; ++w) {
                    if (s_l[w] > 0.0f) go += s_o[w][d] * exp2f(s_m[w] - gm);
                }
            }
            dst_slot[2 + d] = go;
        }
        __syncthreads();
    }
}

template <int D>
__global__ __launch_bounds__(256)
void gqa_decode_reduce_kernel(const float* __restrict__ scratch,
                              const __half* __restrict__ gate_src,
                              __half* __restrict__ out,
                              AttentionParams p,
                              int M,
                              int splits) {
    const int d = threadIdx.x;
    const int hq = blockIdx.x;
    const int m = blockIdx.y;
    if (m >= M || d >= D) return;

    const int H_q = p.num_q_heads;
    const size_t base_slot = (static_cast<size_t>(m) * H_q + hq) * splits;

    float gm = -FLT_MAX;
    for (int s = 0; s < splits; ++s) {
        gm = fmaxf(gm, scratch[(base_slot + s) * (2 + D) + 0]);
    }

    float gl = 0.0f;
    float go = 0.0f;
    if (gm > -FLT_MAX * 0.5f) {
        for (int s = 0; s < splits; ++s) {
            const float* slot = scratch + (base_slot + s) * (2 + D);
            const float sl = slot[1];
            if (sl > 0.0f) {
                const float sc = exp2f(slot[0] - gm);
                gl += sl * sc;
                go += slot[2 + d] * sc;
            }
        }
    }

    float val = (gl > 0.0f) ? (go / gl) : 0.0f;
    if (p.attn_output_gate && gate_src != nullptr) {
        const size_t gate_idx = (static_cast<size_t>(m) * H_q + hq) * (2 * D) + D + d;
        val *= sigmoid_f32(__half2float(gate_src[gate_idx]));
    }
    out[(static_cast<size_t>(m) * H_q + hq) * D + d] = __float2half(val);
}

// =============================================================================
// Split-KV WMMA FlashAttention Prefill + VRAM Staging for Tiered GTT Pages
// =============================================================================
__global__ __launch_bounds__(256)
void stage_kv_pages_kernel(uint8_t* const* __restrict__ page_ptrs,
                           int32_t start_pg,
                           uint8_t* __restrict__ dst_stage,
                           size_t page_bytes) {
    const int32_t pg = start_pg + static_cast<int32_t>(blockIdx.x);
    const uint4* src = reinterpret_cast<const uint4*>(page_ptrs[pg]);
    uint4* dst = reinterpret_cast<uint4*>(dst_stage + static_cast<size_t>(blockIdx.x) * page_bytes);
    const size_t n_u4 = page_bytes >> 4;
    for (size_t i = threadIdx.x; i < n_u4; i += blockDim.x) {
        dst[i] = src[i];
    }
}

template <int D, int BKV, DType KV_DT>
__device__ __forceinline__ void load_kv_tile_to_lds(const uint8_t* __restrict__ h_data,
                                                    const uint8_t* __restrict__ h_scales,
                                                    _Float16 lds_kv[][D + 8],
                                                    int tid,
                                                    int n_threads,
                                                    int kv0,
                                                    int kv_end,
                                                    int tok_in_pg_base) {
    constexpr int GROUPS_PER_TOK = D >> 4;
    constexpr int TOTAL_GROUPS = BKV * GROUPS_PER_TOK;
    constexpr int32_t ROW_DATA_BYTES =
        (KV_DT == DType::F16) ? (D * 2) : ((KV_DT == DType::FP8_E4M3) ? D : (D >> 1));

    for (int idx = tid; idx < TOTAL_GROUPS; idx += n_threads) {
        const int j = idx / GROUPS_PER_TOK;
        const int g = idx % GROUPS_PER_TOK;
        const int kv_pos = kv0 + j;
        const int d = g << 4;
        if (kv_pos < kv_end) {
            const int tok = tok_in_pg_base + j;
            if constexpr (KV_DT == DType::F16) {
                const uint4* src_u4 =
                    reinterpret_cast<const uint4*>(h_data + static_cast<size_t>(tok) * ROW_DATA_BYTES + d * 2);
                *reinterpret_cast<uint4*>(&lds_kv[j][d + 0]) = src_u4[0];
                *reinterpret_cast<uint4*>(&lds_kv[j][d + 8]) = src_u4[1];
            } else if constexpr (KV_DT == DType::FP8_E4M3) {
                const uint4 u =
                    *reinterpret_cast<const uint4*>(h_data + static_cast<size_t>(tok) * ROW_DATA_BYTES + d);
                const float sc = __half2float(reinterpret_cast<const __half*>(h_scales)[tok]);
                h2_t out0[4], out1[4];
                const uint32_t words[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int w = 0; w < 2; ++w) {
                    const uint32_t wx = words[w];
                    const v2f32 f0 = __builtin_amdgcn_cvt_pk_f32_fp8(wx & 0xFFFFu, 0);
                    const v2f32 f1 = __builtin_amdgcn_cvt_pk_f32_fp8((wx >> 16) & 0xFFFFu, 0);
                    out0[w * 2 + 0] = __floats2half2_rn(f0[0] * sc, f0[1] * sc);
                    out0[w * 2 + 1] = __floats2half2_rn(f1[0] * sc, f1[1] * sc);
                }
#pragma unroll
                for (int w = 0; w < 2; ++w) {
                    const uint32_t wx = words[2 + w];
                    const v2f32 f0 = __builtin_amdgcn_cvt_pk_f32_fp8(wx & 0xFFFFu, 0);
                    const v2f32 f1 = __builtin_amdgcn_cvt_pk_f32_fp8((wx >> 16) & 0xFFFFu, 0);
                    out1[w * 2 + 0] = __floats2half2_rn(f0[0] * sc, f0[1] * sc);
                    out1[w * 2 + 1] = __floats2half2_rn(f1[0] * sc, f1[1] * sc);
                }
                *reinterpret_cast<uint4*>(&lds_kv[j][d + 0]) = *reinterpret_cast<const uint4*>(out0);
                *reinterpret_cast<uint4*>(&lds_kv[j][d + 8]) = *reinterpret_cast<const uint4*>(out1);
            } else if constexpr (KV_DT == DType::FP4_E2M1_G16) {
                const uint2 u =
                    *reinterpret_cast<const uint2*>(h_data + static_cast<size_t>(tok) * ROW_DATA_BYTES + (d >> 1));
                const float sc = fp8_e4m3_to_f32(h_scales[static_cast<size_t>(tok) * GROUPS_PER_TOK + g]);
                const h2_t sc2 = __float2half2_rn(sc);
                h2_t out0[4], out1[4];
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    out0[b] = __hmul2(dequant_e2m1_byte_to_h2(static_cast<uint8_t>((u.x >> (b * 8)) & 0xFFu)), sc2);
                    out1[b] = __hmul2(dequant_e2m1_byte_to_h2(static_cast<uint8_t>((u.y >> (b * 8)) & 0xFFu)), sc2);
                }
                *reinterpret_cast<uint4*>(&lds_kv[j][d + 0]) = *reinterpret_cast<const uint4*>(out0);
                *reinterpret_cast<uint4*>(&lds_kv[j][d + 8]) = *reinterpret_cast<const uint4*>(out1);
            }
        } else {
            *reinterpret_cast<uint4*>(&lds_kv[j][d + 0]) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&lds_kv[j][d + 8]) = make_uint4(0, 0, 0, 0);
        }
    }
}

template <int D, int Q_TILES, int G_HEADS, int BKV, DType KV_DT>
__global__ __launch_bounds__(Q_TILES * G_HEADS * 32)
void fa_wmma_paged_kernel(const __half* __restrict__ q_in,
                          const __half* __restrict__ gate_src,
                          KVCacheView kv,
                          const uint8_t* __restrict__ contiguous_stage_base,
                          float* __restrict__ split_scratch,
                          __half* __restrict__ out,
                          AttentionParams p,
                          int q_row_base,
                          int M_sub,
                          int M_tot,
                          int kv_offset,
                          int kv_range_start,
                          int kv_range_end,
                          int split_kv,
                          bool write_direct_out) {
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int N_WARPS = Q_TILES * G_HEADS;
    constexpr int D_TILES = D / 16;
    constexpr int KV_TILES = BKV / 16;
    constexpr int K_STRIDE = D + 8;
    constexpr int P_STRIDE = BKV + 8;

    __shared__ _Float16 lds_kv[BKV][K_STRIDE];
    __shared__ _Float16 lds_p[N_WARPS][16][P_STRIDE];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const int k_half8 = (lane >= 16) ? 8 : 0;
    const int m_half = (lane >= 16) ? 8 : 0;

    const int H_q = p.num_q_heads;
    const int H_kv = p.num_kv_heads;
    const int gqa_ratio = H_q / H_kv;
    const int g_chunks = (gqa_ratio + G_HEADS - 1) / G_HEADS;
    const int h_kv = blockIdx.x / g_chunks;
    const int g_blk = (blockIdx.x % g_chunks) * G_HEADS;
    const int qt = warp / G_HEADS;
    const int gh = warp % G_HEADS;
    const int h_in_group = g_blk + gh;
    const int hq = h_kv * gqa_ratio + h_in_group;
    const bool valid_head = (h_in_group < gqa_ratio) && (hq < H_q);

    const int q_sub_top = blockIdx.y * (Q_TILES * 16);
    const int q_sub_block = q_sub_top + qt * 16;
    const int split_id = blockIdx.z;

    const int q_global_top = q_row_base + q_sub_top;
    const int causal_max_kv = min(kv_offset + min(q_global_top + Q_TILES * 16, M_tot), kv_range_end);
    const int valid_kv_len = max(0, causal_max_kv - kv_range_start);
    const int total_tiles = (valid_kv_len + BKV - 1) / BKV;
    const int t0 = (total_tiles * split_id) / split_kv;
    const int t1 = (total_tiles * (split_id + 1)) / split_kv;
    const int kv_start = kv_range_start + t0 * BKV;
    const int kv_end = min(kv_range_start + t1 * BKV, causal_max_kv);

    const int q_row_sub = q_sub_block + lane16;
    const int q_row_global = q_row_base + q_row_sub;
    const __half* qp = (valid_head && q_row_sub < M_sub && q_row_global < M_tot)
                           ? (q_in + (static_cast<size_t>(q_row_global) * H_q + hq) * D + k_half8)
                           : nullptr;

    f32x8 o_acc[D_TILES];
#pragma unroll
    for (int dt = 0; dt < D_TILES; ++dt) {
        for (int e = 0; e < 8; ++e) o_acc[dt][e] = 0.0f;
    }
    float m_max[8], l_sum[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        m_max[e] = -FLT_MAX;
        l_sum[e] = 0.0f;
    }

    constexpr int32_t ROW_DATA_BYTES =
        (KV_DT == DType::F16) ? (D * 2) : ((KV_DT == DType::FP8_E4M3) ? D : (D >> 1));
    constexpr int32_t ROW_SC_BYTES =
        (KV_DT == DType::FP8_E4M3) ? 2 : ((KV_DT == DType::FP4_E2M1_G16) ? (D >> 4) : 0);

    const size_t h_data_off = static_cast<size_t>(h_kv) * KV_PAGE_TOKENS * ROW_DATA_BYTES;
    const size_t h_sc_off = static_cast<size_t>(h_kv) * KV_PAGE_TOKENS * ROW_SC_BYTES;

    const float scale = (p.attn_scale > 0.0f) ? p.attn_scale : rsqrtf(static_cast<float>(D));
    const float scale_log2 = scale * LOG2E;
    const int start_pg_base = kv_range_start / KV_PAGE_TOKENS;

    for (int kv0 = kv_start; kv0 < kv_end; kv0 += BKV) {
        const int pg = kv0 / KV_PAGE_TOKENS;
        const uint8_t* pg_base = contiguous_stage_base
                                     ? (contiguous_stage_base + static_cast<size_t>(pg - start_pg_base) * kv.page_bytes)
                                     : kv.page_ptrs[pg];
        const int tok_in_pg_base = kv0 % KV_PAGE_TOKENS;

        const uint8_t* hk_d = pg_base + h_data_off;
        const uint8_t* hk_s = pg_base + 2 * kv.data_sec_bytes + h_sc_off;

        load_kv_tile_to_lds<D, BKV, KV_DT>(hk_d, hk_s, lds_kv, tid, N_WARPS * 32, kv0, kv_end, tok_in_pg_base);
        __syncthreads();

        f32x8 s_acc[KV_TILES];
#pragma unroll
        for (int kt = 0; kt < KV_TILES; ++kt) {
            for (int e = 0; e < 8; ++e) s_acc[kt][e] = 0.0f;
        }
#pragma unroll
        for (int dt = 0; dt < D_TILES; ++dt) {
            f16x8 q_frag{};
            if (qp) q_frag = *reinterpret_cast<const f16x8*>(qp + dt * 16);
#pragma unroll
            for (int kt = 0; kt < KV_TILES; ++kt) {
                const int j_col = kt * 16 + lane16;
                const f16x8 k_frag = *reinterpret_cast<const f16x8*>(&lds_kv[j_col][dt * 16 + k_half8]);
                s_acc[kt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(q_frag, k_frag, s_acc[kt]);
            }
        }
        __syncthreads();

        const uint8_t* hv_d = pg_base + kv.data_sec_bytes + h_data_off;
        const uint8_t* hv_s = pg_base + 2 * kv.data_sec_bytes + kv.sc_sec_bytes + h_sc_off;

        load_kv_tile_to_lds<D, BKV, KV_DT>(hv_d, hv_s, lds_kv, tid, N_WARPS * 32, kv0, kv_end, tok_in_pg_base);

        float m_tile[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int row_sub = q_sub_block + m_half + e;
            const int row_glob = q_row_base + row_sub;
            const int lim = kv_offset + row_glob;
            float row_mx = -FLT_MAX;
#pragma unroll
            for (int kt = 0; kt < KV_TILES; ++kt) {
                const int kv_pos = kv0 + kt * 16 + lane16;
                const float sc = (row_sub < M_sub && row_glob < M_tot && kv_pos < kv_end && kv_pos <= lim)
                                     ? (s_acc[kt][e] * scale_log2)
                                     : -FLT_MAX;
                s_acc[kt][e] = sc;
                row_mx = fmaxf(row_mx, sc);
            }
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) row_mx = fmaxf(row_mx, __shfl_xor(row_mx, off));
            m_tile[e] = row_mx;
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float m_new = fmaxf(m_max[e], m_tile[e]);
            const float alpha = (m_max[e] > -FLT_MAX * 0.5f && m_new > -FLT_MAX * 0.5f) ? exp2f(m_max[e] - m_new) : 0.0f;
            if (m_new > -FLT_MAX * 0.5f) {
                l_sum[e] *= alpha;
                m_max[e] = m_new;
            }
            m_tile[e] = alpha;
        }
#pragma unroll
        for (int dt = 0; dt < D_TILES; ++dt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) o_acc[dt][e] *= m_tile[e];
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float mn = m_max[e];
            float p_sum = 0.0f;
#pragma unroll
            for (int kt = 0; kt < KV_TILES; ++kt) {
                const float sc = s_acc[kt][e];
                const float prob = (sc > -FLT_MAX * 0.5f && mn > -FLT_MAX * 0.5f) ? exp2f(sc - mn) : 0.0f;
                lds_p[warp][m_half + e][kt * 16 + lane16] = static_cast<_Float16>(prob);
                p_sum += prob;
            }
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) p_sum += __shfl_xor(p_sum, off);
            l_sum[e] += p_sum;
        }
        __syncthreads();

        if (valid_head) {
#pragma unroll
            for (int kt = 0; kt < KV_TILES; ++kt) {
                const f16x8 p_frag = *reinterpret_cast<const f16x8*>(&lds_p[warp][lane16][kt * 16 + k_half8]);
                const int k0 = kt * 16 + k_half8;
#pragma unroll
                for (int dt = 0; dt < D_TILES; ++dt) {
                    const int d_col = dt * 16 + lane16;
                    f16x8 v_frag;
#pragma unroll
                    for (int e = 0; e < 8; ++e) v_frag[e] = lds_kv[k0 + e][d_col];
                    o_acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(p_frag, v_frag, o_acc[dt]);
                }
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        if (write_direct_out) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row_sub = q_sub_block + m_half + e;
                const int row_glob = q_row_base + row_sub;
                if (row_sub < M_sub && row_glob < M_tot) {
                    const float inv_l = (l_sum[e] > 0.0f) ? (1.0f / l_sum[e]) : 0.0f;
                    __half* dst = out + (static_cast<size_t>(row_glob) * H_q + hq) * D;
                    const __half* gp = (p.attn_output_gate && gate_src != nullptr)
                                           ? (gate_src + (static_cast<size_t>(row_glob) * H_q + hq) * (2 * D) + D)
                                           : nullptr;
#pragma unroll
                    for (int dt = 0; dt < D_TILES; ++dt) {
                        const int d = dt * 16 + lane16;
                        float val = o_acc[dt][e] * inv_l;
                        if (gp != nullptr) val *= sigmoid_f32(__half2float(gp[d]));
                        dst[d] = __float2half(val);
                    }
                }
            }
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row_sub = q_sub_block + m_half + e;
                const int row_glob = q_row_base + row_sub;
                if (row_sub < M_sub && row_glob < M_tot) {
                    const size_t slot = ((static_cast<size_t>(split_id) * M_sub + row_sub) * H_q + hq) * (2 + D);
                    if (lane16 == 0) {
                        split_scratch[slot + 0] = m_max[e];
                        split_scratch[slot + 1] = l_sum[e];
                    }
#pragma unroll
                    for (int dt = 0; dt < D_TILES; ++dt) {
                        split_scratch[slot + 2 + dt * 16 + lane16] = o_acc[dt][e];
                    }
                }
            }
        }
    }
#endif
}

template <int D>
__global__ __launch_bounds__(256)
void fa_prefill_merge_reduce(const float* __restrict__ split_scratch,
                             float* __restrict__ acc_buf,
                             const __half* __restrict__ gate_src,
                             __half* __restrict__ out,
                             AttentionParams p,
                             int q_row_base,
                             int M_sub,
                             int M_tot,
                             int split_kv,
                             bool init_acc,
                             bool finalize) {
    const int d = threadIdx.x;
    const int hq = blockIdx.x;
    const int m_sub = blockIdx.y;
    const int m_glob = q_row_base + m_sub;
    if (m_sub >= M_sub || m_glob >= M_tot || d >= D) return;

    const int H_q = p.num_q_heads;
    const size_t split_m_stride = static_cast<size_t>(M_sub) * H_q * (2 + D);
    const size_t split_row_off = (static_cast<size_t>(m_sub) * H_q + hq) * (2 + D);
    const size_t acc_row_off = (static_cast<size_t>(m_glob) * H_q + hq) * (2 + D);

    float gm = init_acc ? -FLT_MAX : acc_buf[acc_row_off + 0];
    for (int s = 0; s < split_kv; ++s) {
        gm = fmaxf(gm, split_scratch[s * split_m_stride + split_row_off + 0]);
    }

    float gl = 0.0f;
    float go = 0.0f;
    if (gm > -FLT_MAX * 0.5f) {
        if (!init_acc && acc_buf[acc_row_off + 1] > 0.0f) {
            const float w = exp2f(acc_buf[acc_row_off + 0] - gm);
            gl += acc_buf[acc_row_off + 1] * w;
            go += acc_buf[acc_row_off + 2 + d] * w;
        }
        for (int s = 0; s < split_kv; ++s) {
            const float* sl = split_scratch + s * split_m_stride + split_row_off;
            if (sl[1] > 0.0f) {
                const float w = exp2f(sl[0] - gm);
                gl += sl[1] * w;
                go += sl[2 + d] * w;
            }
        }
    }

    if (finalize) {
        float val = (gl > 0.0f) ? (go / gl) : 0.0f;
        if (p.attn_output_gate && gate_src != nullptr) {
            const size_t gate_idx = (static_cast<size_t>(m_glob) * H_q + hq) * (2 * D) + D + d;
            val *= sigmoid_f32(__half2float(gate_src[gate_idx]));
        }
        out[(static_cast<size_t>(m_glob) * H_q + hq) * D + d] = __float2half(val);
    } else {
        if (d == 0) {
            acc_buf[acc_row_off + 0] = gm;
            acc_buf[acc_row_off + 1] = gl;
        }
        acc_buf[acc_row_off + 2 + d] = go;
    }
}

// =============================================================================
// Host Dispatchers
// =============================================================================
template <int D, int G_CHUNK>
void launch_decode_gqa_dt(const __half* q_in,
                          const __half* gate_src,
                          KVCacheView kv,
                          __half* out,
                          float* fscratch,
                          const AttentionParams& p,
                          int M,
                          int32_t kv_seq_len,
                          const StepParams* d_step,
                          int splits,
                          hipStream_t stream) {
    constexpr int N_WARPS = 4;
    const int gqa_ratio = p.num_q_heads / p.num_kv_heads;
    const int g_chunks = (gqa_ratio + G_CHUNK - 1) / G_CHUNK;
    dim3 grid1(p.num_kv_heads * g_chunks, splits, M);
    dim3 block1(N_WARPS * 32);

    if (p.kv_dtype == DType::F16) {
        hipLaunchKernelGGL((gqa_decode_paged_kernel<D, DType::F16, G_CHUNK, N_WARPS>), grid1, block1, 0, stream,
                           q_in, kv, fscratch, p, M, kv_seq_len, d_step, splits);
    } else if (p.kv_dtype == DType::FP8_E4M3) {
        hipLaunchKernelGGL((gqa_decode_paged_kernel<D, DType::FP8_E4M3, G_CHUNK, N_WARPS>), grid1, block1, 0, stream,
                           q_in, kv, fscratch, p, M, kv_seq_len, d_step, splits);
    } else {
        hipLaunchKernelGGL((gqa_decode_paged_kernel<D, DType::FP4_E2M1_G16, G_CHUNK, N_WARPS>), grid1, block1, 0, stream,
                           q_in, kv, fscratch, p, M, kv_seq_len, d_step, splits);
    }

    dim3 grid2(p.num_q_heads, M);
    hipLaunchKernelGGL((gqa_decode_reduce_kernel<D>), grid2, dim3(D), 0, stream,
                       fscratch, gate_src, out, p, M, splits);
}

template <int D>
void launch_decode_dispatch(const __half* q_in,
                            const __half* gate_src,
                            KVCacheView kv,
                            __half* out,
                            void* split_scratch,
                            const AttentionParams& p,
                            int M,
                            int32_t kv_seq_len,
                            const StepParams* d_step,
                            int splits,
                            hipStream_t stream) {
    auto* fscratch = static_cast<float*>(split_scratch);
    const int gqa_ratio = p.num_q_heads / p.num_kv_heads;
    if (gqa_ratio == 1) {
        launch_decode_gqa_dt<D, 1>(q_in, gate_src, kv, out, fscratch, p, M, kv_seq_len, d_step, splits, stream);
    } else if (gqa_ratio == 2) {
        launch_decode_gqa_dt<D, 2>(q_in, gate_src, kv, out, fscratch, p, M, kv_seq_len, d_step, splits, stream);
    } else if (gqa_ratio % 6 == 0 || gqa_ratio == 5 || gqa_ratio == 3) {
        launch_decode_gqa_dt<D, 6>(q_in, gate_src, kv, out, fscratch, p, M, kv_seq_len, d_step, splits, stream);
    } else {
        launch_decode_gqa_dt<D, 4>(q_in, gate_src, kv, out, fscratch, p, M, kv_seq_len, d_step, splits, stream);
    }
}

template <int D, int Q_TILES, int G_HEADS, int BKV>
void launch_wmma_fa_call(const __half* q_in,
                         const __half* gate_src,
                         KVCacheView kv,
                         const uint8_t* contiguous_stage_base,
                         float* split_scratch,
                         __half* out,
                         const AttentionParams& p,
                         int q_row_base,
                         int m_sub,
                         int M_tot,
                         int kv_offset,
                         int r_start,
                         int r_end,
                         int split_kv,
                         bool write_direct_out,
                         hipStream_t stream) {
    const int gqa_ratio = p.num_q_heads / p.num_kv_heads;
    const int g_chunks = (gqa_ratio + G_HEADS - 1) / G_HEADS;
    dim3 grid(p.num_kv_heads * g_chunks, div_ceil(m_sub, Q_TILES * 16), split_kv);
    dim3 block(Q_TILES * G_HEADS * 32);

    if (p.kv_dtype == DType::F16) {
        hipLaunchKernelGGL((fa_wmma_paged_kernel<D, Q_TILES, G_HEADS, BKV, DType::F16>), grid, block, 0, stream,
                           q_in, gate_src, kv, contiguous_stage_base, split_scratch, out,
                           p, q_row_base, m_sub, M_tot, kv_offset, r_start, r_end, split_kv, write_direct_out);
    } else if (p.kv_dtype == DType::FP8_E4M3) {
        hipLaunchKernelGGL((fa_wmma_paged_kernel<D, Q_TILES, G_HEADS, BKV, DType::FP8_E4M3>), grid, block, 0, stream,
                           q_in, gate_src, kv, contiguous_stage_base, split_scratch, out,
                           p, q_row_base, m_sub, M_tot, kv_offset, r_start, r_end, split_kv, write_direct_out);
    } else {
        hipLaunchKernelGGL((fa_wmma_paged_kernel<D, Q_TILES, G_HEADS, BKV, DType::FP4_E2M1_G16>), grid, block, 0, stream,
                           q_in, gate_src, kv, contiguous_stage_base, split_scratch, out,
                           p, q_row_base, m_sub, M_tot, kv_offset, r_start, r_end, split_kv, write_direct_out);
    }
}

template <int D, int Q_TILES, int G_HEADS, int BKV>
void launch_prefill_impl(const __half* q_in,
                         const __half* gate_src,
                         KVCacheView kv,
                         __half* out,
                         const AttentionParams& p,
                         int M,
                         int32_t kv_offset,
                         const PrefillScratch* scratch,
                         hipStream_t stream) {
    const int32_t max_kv = kv_offset + M;
    const int32_t vram_tok_lim = std::min(max_kv, kv.num_vram_pages * KV_PAGE_TOKENS);
    const bool has_gtt = (max_kv > vram_tok_lim);
    const bool has_scratch = (scratch != nullptr && scratch->split_scratch != nullptr &&
                              scratch->max_M >= M && scratch->sub_m > 0 && scratch->split_kv > 0);

    // Fast single-pass path when all KV is in VRAM and either short context or no scratch provided.
    if ((!has_gtt && (!has_scratch || max_kv <= 2048)) || !has_scratch) {
        launch_wmma_fa_call<D, Q_TILES, G_HEADS, BKV>(
            q_in, gate_src, kv, nullptr, nullptr, out, p, 0, M, M, kv_offset, 0, max_kv, 1, true, stream);
        return;
    }

    const int sub_m = scratch->sub_m;
    const int split_kv = scratch->split_kv;
    const size_t acc_floats = align_up(static_cast<size_t>(scratch->max_M) * p.num_q_heads * (2 + D), 64);
    float* acc_buf = scratch->split_scratch;
    float* split_buf = scratch->split_scratch + acc_floats;

    bool acc_initialized = false;

    if (vram_tok_lim > 0) {
        for (int q_base = 0; q_base < M; q_base += sub_m) {
            const int m_sub = std::min(sub_m, M - q_base);
            launch_wmma_fa_call<D, Q_TILES, G_HEADS, BKV>(
                q_in, gate_src, kv, nullptr, split_buf, out, p,
                q_base, m_sub, M, kv_offset, 0, vram_tok_lim, split_kv, false, stream);
            hipLaunchKernelGGL((fa_prefill_merge_reduce<D>), dim3(p.num_q_heads, m_sub), dim3(D), 0, stream,
                               split_buf, acc_buf, gate_src, out, p,
                               q_base, m_sub, M, split_kv, true, !has_gtt);
        }
        acc_initialized = true;
    }

    if (has_gtt) {
        const int gtt_start_pg = kv.num_vram_pages;
        const int gtt_end_pg = static_cast<int>(div_ceil(max_kv, KV_PAGE_TOKENS));
        if (scratch->stage_buf != nullptr && scratch->stage_pages > 0) {
            for (int pg0 = gtt_start_pg; pg0 < gtt_end_pg; pg0 += scratch->stage_pages) {
                const int n_pg = std::min(scratch->stage_pages, gtt_end_pg - pg0);
                hipLaunchKernelGGL(stage_kv_pages_kernel, dim3(n_pg), dim3(256), 0, stream,
                                   kv.page_ptrs, pg0, scratch->stage_buf, kv.page_bytes);
                const int r_start = pg0 * KV_PAGE_TOKENS;
                const int r_end = std::min((pg0 + n_pg) * KV_PAGE_TOKENS, max_kv);
                const bool is_last = (pg0 + n_pg >= gtt_end_pg);
                for (int q_base = 0; q_base < M; q_base += sub_m) {
                    const int m_sub = std::min(sub_m, M - q_base);
                    launch_wmma_fa_call<D, Q_TILES, G_HEADS, BKV>(
                        q_in, gate_src, kv, scratch->stage_buf, split_buf, out, p,
                        q_base, m_sub, M, kv_offset, r_start, r_end, split_kv, false, stream);
                    hipLaunchKernelGGL((fa_prefill_merge_reduce<D>), dim3(p.num_q_heads, m_sub), dim3(D), 0, stream,
                                       split_buf, acc_buf, gate_src, out, p,
                                       q_base, m_sub, M, split_kv, !acc_initialized, is_last);
                }
                acc_initialized = true;
            }
        } else {
            for (int q_base = 0; q_base < M; q_base += sub_m) {
                const int m_sub = std::min(sub_m, M - q_base);
                launch_wmma_fa_call<D, Q_TILES, G_HEADS, BKV>(
                    q_in, gate_src, kv, nullptr, split_buf, out, p,
                    q_base, m_sub, M, kv_offset, vram_tok_lim, max_kv, split_kv, false, stream);
                hipLaunchKernelGGL((fa_prefill_merge_reduce<D>), dim3(p.num_q_heads, m_sub), dim3(D), 0, stream,
                                   split_buf, acc_buf, gate_src, out, p,
                                   q_base, m_sub, M, split_kv, !acc_initialized, true);
            }
        }
    }
}

template <int D>
void launch_prefill_dispatch(const __half* q_in,
                             const __half* gate_src,
                             KVCacheView kv,
                             __half* out,
                             const AttentionParams& p,
                             int M,
                             int32_t kv_offset,
                             const PrefillScratch* scratch,
                             hipStream_t stream) {
    const int gqa_ratio = p.num_q_heads / p.num_kv_heads;
    if (gqa_ratio == 1) {
        launch_prefill_impl<D, 4, 1, 32>(q_in, gate_src, kv, out, p, M, kv_offset, scratch, stream);
    } else if (gqa_ratio % 6 == 0) {
        launch_prefill_impl<D, 2, 6, 32>(q_in, gate_src, kv, out, p, M, kv_offset, scratch, stream);
    } else if (gqa_ratio % 4 == 0) {
        launch_prefill_impl<D, 2, 4, 32>(q_in, gate_src, kv, out, p, M, kv_offset, scratch, stream);
    } else if (gqa_ratio % 3 == 0) {
        launch_prefill_impl<D, 2, 3, 32>(q_in, gate_src, kv, out, p, M, kv_offset, scratch, stream);
    } else {
        launch_prefill_impl<D, 2, 2, 32>(q_in, gate_src, kv, out, p, M, kv_offset, scratch, stream);
    }
}

} // namespace

KVPageLayout compute_kv_page_layout(const AttentionParams& p) {
    KVPageLayout lay{};
    if (p.kv_dtype == DType::F16) {
        lay.row_data_bytes = p.head_dim * 2;
        lay.row_sc_bytes = 0;
    } else if (p.kv_dtype == DType::FP8_E4M3) {
        lay.row_data_bytes = p.head_dim;
        lay.row_sc_bytes = 2;
    } else if (p.kv_dtype == DType::FP4_E2M1_G16) {
        lay.row_data_bytes = p.head_dim / 2;
        lay.row_sc_bytes = p.head_dim / 16;
    }
    lay.data_per_head = static_cast<size_t>(KV_PAGE_TOKENS) * lay.row_data_bytes;
    lay.sc_per_head = static_cast<size_t>(KV_PAGE_TOKENS) * lay.row_sc_bytes;
    lay.data_sec_bytes = align_up(static_cast<size_t>(p.num_kv_heads) * lay.data_per_head, 256);
    lay.sc_sec_bytes = (lay.row_sc_bytes > 0) ? align_up(static_cast<size_t>(p.num_kv_heads) * lay.sc_per_head, 256) : 0;
    lay.page_bytes = 2 * lay.data_sec_bytes + 2 * lay.sc_sec_bytes;
    return lay;
}

size_t decode_splitk_scratch_bytes(const AttentionParams& p, int max_M, int splits) {
    return static_cast<size_t>(max_M) * p.num_q_heads * splits * (2 + p.head_dim) * sizeof(float);
}

size_t prefill_splitk_scratch_bytes(const AttentionParams& p, int max_M, int sub_m, int split_kv) {
    const size_t acc_floats = align_up(static_cast<size_t>(max_M) * p.num_q_heads * (2 + p.head_dim), 64);
    const size_t split_floats = static_cast<size_t>(split_kv) * sub_m * p.num_q_heads * (2 + p.head_dim);
    return (acc_floats + split_floats) * sizeof(float);
}

void fused_qknorm_rope_kv_write(const __half* q_in,
                                const __half* k_in,
                                const __half* v_in,
                                const __half* q_norm_w,
                                const __half* k_norm_w,
                                __half* q_out,
                                KVCacheView kv_cache,
                                const AttentionParams& p,
                                int M,
                                int32_t pos_offset,
                                const StepParams* d_step,
                                hipStream_t stream,
                                int kv_in_stride) {
    dim3 grid(p.num_q_heads + 2 * p.num_kv_heads, M);
    hipLaunchKernelGGL(fused_qknorm_rope_kv_write_kernel, grid, dim3(32), 0, stream,
                       q_in, k_in, v_in, q_norm_w, k_norm_w, q_out, kv_cache,
                       p, M, pos_offset, d_step, kv_in_stride);
}

void gqa_decode_attention(const __half* q_in,
                          const __half* gate_src,
                          KVCacheView kv_cache,
                          __half* out,
                          void* split_scratch,
                          const AttentionParams& p,
                          int M,
                          int32_t kv_seq_len,
                          const StepParams* d_step,
                          int splits,
                          hipStream_t stream) {
    if (p.num_kv_heads <= 0 || p.num_q_heads % p.num_kv_heads != 0) {
        std::fprintf(stderr, "Invalid H_q=%d H_kv=%d in gqa_decode_attention\n", p.num_q_heads, p.num_kv_heads);
        std::abort();
    }
    if (p.head_dim == 64) {
        launch_decode_dispatch<64>(q_in, gate_src, kv_cache, out, split_scratch, p, M, kv_seq_len, d_step, splits, stream);
    } else if (p.head_dim == 128) {
        launch_decode_dispatch<128>(q_in, gate_src, kv_cache, out, split_scratch, p, M, kv_seq_len, d_step, splits, stream);
    } else if (p.head_dim == 256) {
        launch_decode_dispatch<256>(q_in, gate_src, kv_cache, out, split_scratch, p, M, kv_seq_len, d_step, splits, stream);
    } else {
        std::fprintf(stderr, "Unsupported head_dim %d in gqa_decode_attention\n", p.head_dim);
        std::abort();
    }
}

void flash_attention_prefill(const __half* q_in,
                             const __half* gate_src,
                             KVCacheView kv_cache,
                             __half* out,
                             const AttentionParams& p,
                             int M,
                             int32_t kv_offset,
                             const PrefillScratch* scratch,
                             hipStream_t stream) {
    if (p.num_kv_heads <= 0 || p.num_q_heads % p.num_kv_heads != 0) {
        std::fprintf(stderr, "Invalid H_q=%d H_kv=%d in flash_attention_prefill\n", p.num_q_heads, p.num_kv_heads);
        std::abort();
    }
    if (p.head_dim == 64) {
        launch_prefill_dispatch<64>(q_in, gate_src, kv_cache, out, p, M, kv_offset, scratch, stream);
    } else if (p.head_dim == 128) {
        launch_prefill_dispatch<128>(q_in, gate_src, kv_cache, out, p, M, kv_offset, scratch, stream);
    } else if (p.head_dim == 256) {
        launch_prefill_dispatch<256>(q_in, gate_src, kv_cache, out, p, M, kv_offset, scratch, stream);
    } else {
        std::fprintf(stderr, "Unsupported head_dim %d in flash_attention_prefill\n", p.head_dim);
        std::abort();
    }
}

} // namespace rxinfer::attn

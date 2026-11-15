// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/attention.hpp"
#include "rxinfer/common.hpp"

#include <cfloat>
#include <cmath>

namespace rxinfer::attn {
namespace {

using h2_t = __half2;
using v2f16 = _Float16 __attribute__((ext_vector_type(2)));
using v2f32 = float __attribute__((ext_vector_type(2)));

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
// and high bytes {0x00, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46}!
__device__ __forceinline__ h2_t dequant_e2m1_byte_to_h2(uint8_t byte_val) {
    const uint32_t u0 = byte_val & 0x0Fu;
    const uint32_t u1 = (byte_val >> 4) & 0x0Fu;
    const uint32_t pair_idx = u0 | (u1 << 8);
    const uint32_t mag_hi = __builtin_amdgcn_perm(0x46444240u, 0x3E3C3800u, pair_idx & 0x0707u);
    const uint32_t sign_hi = (pair_idx & 0x0808u) << 4;
    const uint32_t hi_bytes = mag_hi | sign_hi;
    // Assemble half2: [0x00, hi_byte0, 0x00, hi_byte1]
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
        // Interleaved mRoPE: H at 1,4,7... (< 3*sec[1]); W at 2,5,8... (< 3*sec[2]); else T.
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
// Fused QK-Norm + Partial/Full RoPE/mRoPE + Post-RoPE KV Write
//   1 Wave32 (32 threads) per (token m, head_slot)
// =============================================================================
__global__ __launch_bounds__(32)
void fused_qknorm_rope_kv_write_kernel(const __half* __restrict__ q_in,
                                       const __half* __restrict__ k_in,
                                       const __half* __restrict__ v_in,
                                       const __half* __restrict__ q_norm_w,
                                       const __half* __restrict__ k_norm_w,
                                       __half* __restrict__ q_out,
                                       void* __restrict__ k_data,
                                       void* __restrict__ v_data,
                                       void* __restrict__ k_scales,
                                       void* __restrict__ v_scales,
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

    // 3. RoPE (post-norm, pre-KV-quantization as in DeepSeek-V4.1)
    if (!is_v && p.rope_kind != RopeKind::None && p.rotary_dim > 0) {
        const int half_rot = p.rotary_dim >> 1;
        for (int i = lane; i < half_rot; i += 32) {
            const int32_t tok_pos = select_mrope_pos(i, pos, mrope_p, p.mrope_section, p.mrope_interleaved);
            const float inv_freq = __powf(p.rope_theta, -2.0f * static_cast<float>(i) / static_cast<float>(p.rotary_dim));
            const float angle = static_cast<float>(tok_pos) * inv_freq;
            float s_ang, c_ang;
            __sincosf(angle, &s_ang, &c_ang);

            const int i0 = (p.rope_kind == RopeKind::NeoX) ? i : (2 * i);
            const int i1 = (p.rope_kind == RopeKind::NeoX) ? (i + half_rot) : (2 * i + 1);
            const float x0 = s_head[i0];
            const float x1 = s_head[i1];
            s_head[i0] = x0 * c_ang - x1 * s_ang;
            s_head[i1] = x1 * c_ang + x0 * s_ang;
        }
    }
    __syncthreads();

    // 4. Store Q output or quantize & write K/V into cache at position `pos`
    if (is_q) {
        __half* dst = q_out + (static_cast<size_t>(m) * H_q + head_idx) * D;
        for (int d = lane; d < D; d += 32) {
            dst[d] = __float2half(s_head[d]);
        }
        return;
    }

    const size_t kv_slot = static_cast<size_t>(pos) * H_kv + head_idx;
    void* dst_data = is_k ? k_data : v_data;
    void* dst_scales = is_k ? k_scales : v_scales;

    if (p.kv_dtype == DType::F16) {
        __half* dst = static_cast<__half*>(dst_data) + kv_slot * D;
        for (int d = lane; d < D; d += 32) {
            dst[d] = __float2half(s_head[d]);
        }
    } else if (p.kv_dtype == DType::FP8_E4M3) {
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
            static_cast<__half*>(dst_scales)[kv_slot] = __float2half(sc);
        }
        uint8_t* dst = static_cast<uint8_t*>(dst_data) + kv_slot * D;
        for (int d = lane * 2; d < D; d += 64) {
            const int pk = __builtin_amdgcn_cvt_pk_fp8_f32(s_head[d] * inv_sc, s_head[d + 1] * inv_sc, 0, false);
            *reinterpret_cast<uint16_t*>(dst + d) = static_cast<uint16_t>(pk & 0xFFFF);
        }
    } else if (p.kv_dtype == DType::FP4_E2M1_G16) {
        // DeepSeek-V4.1 post-RoPE FP4 KV: 16 channels per group, 1 OCP E4M3 scale per group!
        const int num_groups = D >> 4;
        uint8_t* dst_q = static_cast<uint8_t*>(dst_data) + kv_slot * (D >> 1);
        uint8_t* dst_s = static_cast<uint8_t*>(dst_scales) + kv_slot * num_groups;

        // Process 2 groups of 16 channels per step across 32 lanes (lanes 0..15 -> g, lanes 16..31 -> g+1)
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
            // Pack pair (even c, odd c+1) using __shfl_xor(..., 1)
            const uint32_t nib_odd = __shfl_xor(nib, 1);
            if (g < num_groups) {
                if (c == 0) dst_s[g] = sc_u8;
                if ((c & 1) == 0) {
                    dst_q[g * 8 + (c >> 1)] = static_cast<uint8_t>(nib | (nib_odd << 4));
                }
            }
        }
    }
}

// =============================================================================
// Split-KV Decode Attention (M <= 16, head_dim <= 256, Wave32 online softmax)
//   Stage 1: 4 wave32 warps (128 threads) per (hq, split_id, m).
//   Each lane holds ELEMS = D / 32 channels (2 for D=64, 4 for D=128, 8 for D=256).
// =============================================================================
template <int D, DType KV_DT>
__global__ __launch_bounds__(128)
void gqa_decode_splitk_kernel(const __half* __restrict__ q_in,
                              const void* __restrict__ k_data,
                              const void* __restrict__ v_data,
                              const void* __restrict__ k_scales,
                              const void* __restrict__ v_scales,
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

    const int hq = blockIdx.x;
    const int split_id = blockIdx.y;
    const int m = blockIdx.z;
    if (m >= M) return;

    const int32_t total_kv = d_step ? (d_step->kv_seq_len - (M - 1 - m)) : (fixed_kv_len - (M - 1 - m));
    const int32_t t_start = (total_kv > 0) ? ((total_kv * split_id) / splits) : 0;
    const int32_t t_end = (total_kv > 0) ? ((total_kv * (split_id + 1)) / splits) : 0;

    const int H_q = p.num_q_heads;
    const int H_kv = p.num_kv_heads;
    const int gqa_ratio = H_q / H_kv;
    const int h_kv = hq / gqa_ratio;
    const float scale = (p.attn_scale > 0.0f) ? p.attn_scale : rsqrtf(static_cast<float>(D));

    // Load Q slice for this lane (ELEMS consecutive channels: d0 .. d0 + ELEMS - 1)
    const int d0 = lane * ELEMS;
    const __half* q_ptr = q_in + (static_cast<size_t>(m) * H_q + hq) * D + d0;
    h2_t q_h2[PAIRS];
    float o_acc[ELEMS];
#pragma unroll
    for (int i = 0; i < PAIRS; ++i) {
        q_h2[i] = *reinterpret_cast<const h2_t*>(q_ptr + i * 2);
    }
#pragma unroll
    for (int i = 0; i < ELEMS; ++i) o_acc[i] = 0.0f;

    float m_max = -FLT_MAX;
    float l_sum = 0.0f;

    for (int32_t t = t_start + warp; t < t_end; t += 4) {
        const size_t kv_slot = static_cast<size_t>(t) * H_kv + h_kv;
        float dot = 0.0f;
        h2_t v_h2[PAIRS];
        float v_sc = 1.0f;

        if constexpr (KV_DT == DType::F16) {
            const __half* kp = static_cast<const __half*>(k_data) + kv_slot * D + d0;
            const __half* vp = static_cast<const __half*>(v_data) + kv_slot * D + d0;
#pragma unroll
            for (int i = 0; i < PAIRS; ++i) {
                const h2_t kh = *reinterpret_cast<const h2_t*>(kp + i * 2);
                dot = fdot2_h2(q_h2[i], kh, dot);
                v_h2[i] = *reinterpret_cast<const h2_t*>(vp + i * 2);
            }
        } else if constexpr (KV_DT == DType::FP8_E4M3) {
            const uint8_t* kp = static_cast<const uint8_t*>(k_data) + kv_slot * D + d0;
            const uint8_t* vp = static_cast<const uint8_t*>(v_data) + kv_slot * D + d0;
            const float k_sc = __half2float(static_cast<const __half*>(k_scales)[kv_slot]);
            v_sc = __half2float(static_cast<const __half*>(v_scales)[kv_slot]);
#pragma unroll
            for (int i = 0; i < PAIRS; ++i) {
                const uint32_t kb2 = *reinterpret_cast<const uint16_t*>(kp + i * 2);
                const uint32_t vb2 = *reinterpret_cast<const uint16_t*>(vp + i * 2);
                const v2f32 kf2 = __builtin_amdgcn_cvt_pk_f32_fp8(kb2, 0);
                const v2f32 vf2 = __builtin_amdgcn_cvt_pk_f32_fp8(vb2, 0);
                const h2_t kh = __floats2half2_rn(kf2[0], kf2[1]);
                dot = fdot2_h2(q_h2[i], kh, dot);
                v_h2[i] = __floats2half2_rn(vf2[0], vf2[1]);
            }
            dot *= k_sc;
        } else if constexpr (KV_DT == DType::FP4_E2M1_G16) {
            const int g = d0 >> 4; // All ELEMS channels of this lane lie inside group g!
            const int num_groups = D >> 4;
            const uint8_t* kp = static_cast<const uint8_t*>(k_data) + kv_slot * (D >> 1) + (d0 >> 1);
            const uint8_t* vp = static_cast<const uint8_t*>(v_data) + kv_slot * (D >> 1) + (d0 >> 1);
            const float k_sc = fp8_e4m3_to_f32(static_cast<const uint8_t*>(k_scales)[kv_slot * num_groups + g]);
            v_sc = fp8_e4m3_to_f32(static_cast<const uint8_t*>(v_scales)[kv_slot * num_groups + g]);
#pragma unroll
            for (int i = 0; i < PAIRS; ++i) {
                const h2_t kh = dequant_e2m1_byte_to_h2(kp[i]);
                dot = fdot2_h2(q_h2[i], kh, dot);
                v_h2[i] = dequant_e2m1_byte_to_h2(vp[i]);
            }
            dot *= k_sc;
        }

#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            dot += __shfl_xor(dot, off);
        }
        const float score = dot * scale;
        const float m_new = fmaxf(m_max, score);
        const float alpha = (m_max > -FLT_MAX * 0.5f) ? __expf(m_max - m_new) : 0.0f;
        const float w = __expf(score - m_new);
        l_sum = l_sum * alpha + w;
        m_max = m_new;

        const float wv_sc = w * v_sc;
#pragma unroll
        for (int i = 0; i < PAIRS; ++i) {
            const float2 vf = __half22float2(v_h2[i]);
            o_acc[i * 2 + 0] = o_acc[i * 2 + 0] * alpha + wv_sc * vf.x;
            o_acc[i * 2 + 1] = o_acc[i * 2 + 1] * alpha + wv_sc * vf.y;
        }
    }

    // Merge 4 warps in LDS
    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_o[4][D];

    if (lane == 0) {
        s_m[warp] = m_max;
        s_l[warp] = l_sum;
    }
#pragma unroll
    for (int i = 0; i < ELEMS; ++i) {
        s_o[warp][d0 + i] = o_acc[i];
    }
    __syncthreads();

    // Scratch layout per (m, hq, split_id): (2 + D) floats: [m_max, l_sum, o[0..D-1]]
    const size_t slot_idx = ((static_cast<size_t>(m) * H_q + hq) * splits + split_id);
    float* dst_slot = scratch + slot_idx * (2 + D);

    // 128 threads merge all D channels cleanly across the 4 warps!
    const float gm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
    float gl = 0.0f;
    if (gm > -FLT_MAX * 0.5f) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (s_l[w] > 0.0f) gl += s_l[w] * __expf(s_m[w] - gm);
        }
    }
    if (tid == 0) {
        dst_slot[0] = gm;
        dst_slot[1] = gl;
    }
    for (int d = tid; d < D; d += 128) {
        float go = 0.0f;
        if (gm > -FLT_MAX * 0.5f) {
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (s_l[w] > 0.0f) {
                    go += s_o[w][d] * __expf(s_m[w] - gm);
                }
            }
        }
        dst_slot[2 + d] = go;
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
        const float sm = scratch[(base_slot + s) * (2 + D) + 0];
        gm = fmaxf(gm, sm);
    }

    float gl = 0.0f;
    float go = 0.0f;
    if (gm > -FLT_MAX * 0.5f) {
        for (int s = 0; s < splits; ++s) {
            const float* slot = scratch + (base_slot + s) * (2 + D);
            const float sl = slot[1];
            if (sl > 0.0f) {
                const float sc = __expf(slot[0] - gm);
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
// Tiled Causal FlashAttention Prefill (M >= 1, head_dim <= 256, F16/FP8/FP4 KV)
//   Block = 128 threads (4 wave32 warps), BQ = 16 queries (4 queries per warp!), BKV = 16 keys/values.
//   K and V tiles dequantized once into LDS[16][D] (16 KB LDS for D=256).
// =============================================================================
template <int D, DType KV_DT>
__global__ __launch_bounds__(128)
void flash_attention_prefill_kernel(const __half* __restrict__ q_in,
                                    const __half* __restrict__ gate_src,
                                    const void* __restrict__ k_data,
                                    const void* __restrict__ v_data,
                                    const void* __restrict__ k_scales,
                                    const void* __restrict__ v_scales,
                                    __half* __restrict__ out,
                                    AttentionParams p,
                                    int M,
                                    int32_t kv_offset) {
    constexpr int BQ = 16;
    constexpr int BKV = 16;
    constexpr int ELEMS = D / 32;
    constexpr int PAIRS = ELEMS / 2;

    __shared__ _Float16 lds_K[BKV][D];
    __shared__ _Float16 lds_V[BKV][D];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int hq = blockIdx.x;
    const int q_block = blockIdx.y * BQ;
    const int H_q = p.num_q_heads;
    const int H_kv = p.num_kv_heads;
    const int h_kv = hq / (H_q / H_kv);
    const float scale = (p.attn_scale > 0.0f) ? p.attn_scale : rsqrtf(static_cast<float>(D));

    const int d0 = lane * ELEMS;

    // Each warp handles 4 query rows: qm = 0..3 -> row = q_block + warp * 4 + qm
    h2_t q_reg[4][PAIRS];
    float o_acc[4][ELEMS];
    float m_max[4];
    float l_sum[4];

#pragma unroll
    for (int qm = 0; qm < 4; ++qm) {
        const int row = q_block + warp * 4 + qm;
        m_max[qm] = -FLT_MAX;
        l_sum[qm] = 0.0f;
#pragma unroll
        for (int i = 0; i < ELEMS; ++i) o_acc[qm][i] = 0.0f;
        if (row < M) {
            const __half* qp = q_in + (static_cast<size_t>(row) * H_q + hq) * D + d0;
#pragma unroll
            for (int i = 0; i < PAIRS; ++i) {
                q_reg[qm][i] = *reinterpret_cast<const h2_t*>(qp + i * 2);
            }
        }
    }

    const int max_kv_for_block = min(kv_offset + min(q_block + BQ, M), kv_offset + M);

    for (int kv0 = 0; kv0 < max_kv_for_block; kv0 += BKV) {
        // Load 16 KV tokens into LDS (4 warps load 4 tokens each at a time -> 4 passes)
#pragma unroll
        for (int pass = 0; pass < 4; ++pass) {
            const int j = pass * 4 + warp; // 0..15
            const int kv_pos = kv0 + j;
            if (kv_pos < max_kv_for_block) {
                const size_t kv_slot = static_cast<size_t>(kv_pos) * H_kv + h_kv;
                if constexpr (KV_DT == DType::F16) {
                    const __half* kp = static_cast<const __half*>(k_data) + kv_slot * D + d0;
                    const __half* vp = static_cast<const __half*>(v_data) + kv_slot * D + d0;
                    for (int i = 0; i < ELEMS; ++i) {
                        lds_K[j][d0 + i] = static_cast<_Float16>(kp[i]);
                        lds_V[j][d0 + i] = static_cast<_Float16>(vp[i]);
                    }
                } else if constexpr (KV_DT == DType::FP8_E4M3) {
                    const uint8_t* kp = static_cast<const uint8_t*>(k_data) + kv_slot * D + d0;
                    const uint8_t* vp = static_cast<const uint8_t*>(v_data) + kv_slot * D + d0;
                    const float k_sc = __half2float(static_cast<const __half*>(k_scales)[kv_slot]);
                    const float v_sc = __half2float(static_cast<const __half*>(v_scales)[kv_slot]);
#pragma unroll
                    for (int i = 0; i < PAIRS; ++i) {
                        const uint32_t kb2 = *reinterpret_cast<const uint16_t*>(kp + i * 2);
                        const uint32_t vb2 = *reinterpret_cast<const uint16_t*>(vp + i * 2);
                        const v2f32 kf2 = __builtin_amdgcn_cvt_pk_f32_fp8(kb2, 0);
                        const v2f32 vf2 = __builtin_amdgcn_cvt_pk_f32_fp8(vb2, 0);
                        lds_K[j][d0 + i * 2 + 0] = static_cast<_Float16>(kf2[0] * k_sc);
                        lds_K[j][d0 + i * 2 + 1] = static_cast<_Float16>(kf2[1] * k_sc);
                        lds_V[j][d0 + i * 2 + 0] = static_cast<_Float16>(vf2[0] * v_sc);
                        lds_V[j][d0 + i * 2 + 1] = static_cast<_Float16>(vf2[1] * v_sc);
                    }
                } else if constexpr (KV_DT == DType::FP4_E2M1_G16) {
                    const int g = d0 >> 4;
                    const int num_groups = D >> 4;
                    const uint8_t* kp = static_cast<const uint8_t*>(k_data) + kv_slot * (D >> 1) + (d0 >> 1);
                    const uint8_t* vp = static_cast<const uint8_t*>(v_data) + kv_slot * (D >> 1) + (d0 >> 1);
                    const float k_sc = fp8_e4m3_to_f32(static_cast<const uint8_t*>(k_scales)[kv_slot * num_groups + g]);
                    const float v_sc = fp8_e4m3_to_f32(static_cast<const uint8_t*>(v_scales)[kv_slot * num_groups + g]);
#pragma unroll
                    for (int i = 0; i < PAIRS; ++i) {
                        const float2 kf2 = __half22float2(dequant_e2m1_byte_to_h2(kp[i]));
                        const float2 vf2 = __half22float2(dequant_e2m1_byte_to_h2(vp[i]));
                        lds_K[j][d0 + i * 2 + 0] = static_cast<_Float16>(kf2.x * k_sc);
                        lds_K[j][d0 + i * 2 + 1] = static_cast<_Float16>(kf2.y * k_sc);
                        lds_V[j][d0 + i * 2 + 0] = static_cast<_Float16>(vf2.x * v_sc);
                        lds_V[j][d0 + i * 2 + 1] = static_cast<_Float16>(vf2.y * v_sc);
                    }
                }
            }
        }
        __syncthreads();

        for (int j = 0; j < BKV; ++j) {
            const int kv_pos = kv0 + j;
            if (kv_pos >= max_kv_for_block) break;

            h2_t k_h2[PAIRS];
            h2_t v_h2[PAIRS];
#pragma unroll
            for (int i = 0; i < PAIRS; ++i) {
                k_h2[i] = *reinterpret_cast<const h2_t*>(&lds_K[j][d0 + i * 2]);
                v_h2[i] = *reinterpret_cast<const h2_t*>(&lds_V[j][d0 + i * 2]);
            }

#pragma unroll
            for (int qm = 0; qm < 4; ++qm) {
                const int row = q_block + warp * 4 + qm;
                if (row < M && kv_pos <= kv_offset + row) {
                    float dot = 0.0f;
#pragma unroll
                    for (int i = 0; i < PAIRS; ++i) {
                        dot = fdot2_h2(q_reg[qm][i], k_h2[i], dot);
                    }
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        dot += __shfl_xor(dot, off);
                    }
                    const float score = dot * scale;
                    const float m_new = fmaxf(m_max[qm], score);
                    const float alpha = (m_max[qm] > -FLT_MAX * 0.5f) ? __expf(m_max[qm] - m_new) : 0.0f;
                    const float w = __expf(score - m_new);
                    l_sum[qm] = l_sum[qm] * alpha + w;
                    m_max[qm] = m_new;

#pragma unroll
                    for (int i = 0; i < PAIRS; ++i) {
                        const float2 vf = __half22float2(v_h2[i]);
                        o_acc[qm][i * 2 + 0] = o_acc[qm][i * 2 + 0] * alpha + w * vf.x;
                        o_acc[qm][i * 2 + 1] = o_acc[qm][i * 2 + 1] * alpha + w * vf.y;
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int qm = 0; qm < 4; ++qm) {
        const int row = q_block + warp * 4 + qm;
        if (row < M) {
            const float inv_l = (l_sum[qm] > 0.0f) ? (1.0f / l_sum[qm]) : 0.0f;
            __half* dst = out + (static_cast<size_t>(row) * H_q + hq) * D + d0;
            const __half* gp = (p.attn_output_gate && gate_src != nullptr)
                                   ? (gate_src + (static_cast<size_t>(row) * H_q + hq) * (2 * D) + D + d0)
                                   : nullptr;
#pragma unroll
            for (int i = 0; i < ELEMS; ++i) {
                float val = o_acc[qm][i] * inv_l;
                if (gp != nullptr) {
                    val *= sigmoid_f32(__half2float(gp[i]));
                }
                dst[i] = __float2half(val);
            }
        }
    }
}

template <int D>
void launch_decode_dispatch(const __half* q_in, const __half* gate_src, KVCacheView kv, __half* out,
                            void* split_scratch, const AttentionParams& p, int M,
                            int32_t kv_seq_len, const StepParams* d_step, int splits, hipStream_t stream) {
    dim3 grid1(p.num_q_heads, splits, M);
    dim3 block1(128);
    auto* fscratch = static_cast<float*>(split_scratch);
    if (p.kv_dtype == DType::F16) {
        hipLaunchKernelGGL((gqa_decode_splitk_kernel<D, DType::F16>), grid1, block1, 0, stream,
                           q_in, kv.k_data, kv.v_data, kv.k_scales, kv.v_scales,
                           fscratch, p, M, kv_seq_len, d_step, splits);
    } else if (p.kv_dtype == DType::FP8_E4M3) {
        hipLaunchKernelGGL((gqa_decode_splitk_kernel<D, DType::FP8_E4M3>), grid1, block1, 0, stream,
                           q_in, kv.k_data, kv.v_data, kv.k_scales, kv.v_scales,
                           fscratch, p, M, kv_seq_len, d_step, splits);
    } else {
        hipLaunchKernelGGL((gqa_decode_splitk_kernel<D, DType::FP4_E2M1_G16>), grid1, block1, 0, stream,
                           q_in, kv.k_data, kv.v_data, kv.k_scales, kv.v_scales,
                           fscratch, p, M, kv_seq_len, d_step, splits);
    }

    dim3 grid2(p.num_q_heads, M);
    hipLaunchKernelGGL((gqa_decode_reduce_kernel<D>), grid2, dim3(256), 0, stream,
                       fscratch, gate_src, out, p, M, splits);
}

template <int D>
void launch_prefill_dispatch(const __half* q_in, const __half* gate_src, KVCacheView kv, __half* out,
                             const AttentionParams& p, int M, int32_t kv_offset, hipStream_t stream) {
    dim3 grid(p.num_q_heads, div_ceil(M, 16));
    dim3 block(128);
    if (p.kv_dtype == DType::F16) {
        hipLaunchKernelGGL((flash_attention_prefill_kernel<D, DType::F16>), grid, block, 0, stream,
                           q_in, gate_src, kv.k_data, kv.v_data, kv.k_scales, kv.v_scales, out, p, M, kv_offset);
    } else if (p.kv_dtype == DType::FP8_E4M3) {
        hipLaunchKernelGGL((flash_attention_prefill_kernel<D, DType::FP8_E4M3>), grid, block, 0, stream,
                           q_in, gate_src, kv.k_data, kv.v_data, kv.k_scales, kv.v_scales, out, p, M, kv_offset);
    } else {
        hipLaunchKernelGGL((flash_attention_prefill_kernel<D, DType::FP4_E2M1_G16>), grid, block, 0, stream,
                           q_in, gate_src, kv.k_data, kv.v_data, kv.k_scales, kv.v_scales, out, p, M, kv_offset);
    }
}

} // namespace

size_t kv_cache_data_bytes(const AttentionParams& p) {
    const size_t slots = static_cast<size_t>(p.max_seq_len) * p.num_kv_heads;
    if (p.kv_dtype == DType::F16) return slots * p.head_dim * sizeof(__half);
    if (p.kv_dtype == DType::FP8_E4M3) return slots * p.head_dim;
    if (p.kv_dtype == DType::FP4_E2M1_G16) return slots * (p.head_dim / 2);
    return 0;
}

size_t kv_cache_scales_bytes(const AttentionParams& p) {
    const size_t slots = static_cast<size_t>(p.max_seq_len) * p.num_kv_heads;
    if (p.kv_dtype == DType::FP8_E4M3) return slots * sizeof(__half);
    if (p.kv_dtype == DType::FP4_E2M1_G16) return slots * (p.head_dim / 16);
    return 0;
}

size_t decode_splitk_scratch_bytes(const AttentionParams& p, int max_M, int splits) {
    return static_cast<size_t>(max_M) * p.num_q_heads * splits * (2 + p.head_dim) * sizeof(float);
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
                       q_in, k_in, v_in, q_norm_w, k_norm_w, q_out,
                       kv_cache.k_data, kv_cache.v_data, kv_cache.k_scales, kv_cache.v_scales,
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
                             hipStream_t stream) {
    if (p.head_dim == 64) {
        launch_prefill_dispatch<64>(q_in, gate_src, kv_cache, out, p, M, kv_offset, stream);
    } else if (p.head_dim == 128) {
        launch_prefill_dispatch<128>(q_in, gate_src, kv_cache, out, p, M, kv_offset, stream);
    } else if (p.head_dim == 256) {
        launch_prefill_dispatch<256>(q_in, gate_src, kv_cache, out, p, M, kv_offset, stream);
    } else {
        std::fprintf(stderr, "Unsupported head_dim %d in flash_attention_prefill\n", p.head_dim);
        std::abort();
    }
}

} // namespace rxinfer::attn

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rxinfer/arena.hpp"
#include "rxinfer/config.hpp"
#include "rxinfer/tensor.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace rxinfer::attn {

struct AttentionParams {
    int32_t num_q_heads = 0;
    int32_t num_kv_heads = 0;
    int32_t head_dim = 0;           // 64, 128, or 256
    int32_t rotary_dim = 0;         // <= head_dim
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-6f;
    float attn_scale = 0.0f;        // 0 -> 1.0f / sqrtf(head_dim)
    RopeKind rope_kind = RopeKind::NeoX;
    bool qk_norm = false;
    bool qk_norm_plus_one = false;
    bool attn_output_gate = false;  // If true, q_in has layout [M, H_q, 2, D] (Q, Gate interleaved per head)
    DType kv_dtype = DType::F16;    // F16, FP8_E4M3, or FP4_E2M1_G16
    int32_t max_seq_len = 0;
    int32_t mrope_section[3] = {0, 0, 0};
    bool mrope_interleaved = false;
};

struct KVCacheView {
    void* k_data = nullptr;
    void* v_data = nullptr;
    void* k_scales = nullptr; // nullptr for F16; __half[S, H_kv] for FP8_E4M3; uint8_t[S, H_kv, D/16] for FP4_E2M1_G16
    void* v_scales = nullptr;
};

size_t kv_cache_data_bytes(const AttentionParams& p);
size_t kv_cache_scales_bytes(const AttentionParams& p);
size_t decode_splitk_scratch_bytes(const AttentionParams& p, int max_M, int splits = 16);

// Fused QK-Norm (optional) + Partial/Full RoPE/mRoPE + Post-RoPE KV Cache Write (F16/FP8/FP4):
//   - q_in:  [M, H_q, D] (or [M, H_q, 2, D] when attn_output_gate=true)
//   - k_in:  [M, H_kv, D]
//   - v_in:  [M, H_kv, D]
//   - q_out: [M, H_q, D] (normalized & rotated Q ready for attention)
//   - If d_step != nullptr (decode M=1 inside hipGraph), reads pos & mrope_pos from *d_step.
//     Otherwise uses pos_offset + m for m in [0..M-1].
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
                                const StepParams* d_step = nullptr,
                                hipStream_t stream = nullptr,
                                int kv_in_stride = 0);

// Split-KV Decode Attention (M <= 16, head_dim <= 256, static hipGraph grid via d_step):
//   - q_in:     [M, H_q, D] (from fused_qknorm_rope_kv_write)
//   - gate_src: original q_proj output [M, H_q, 2, D] if attn_output_gate=true, else nullptr
//   - out:      [M, H_q, D]
//   - If d_step != nullptr, kv_seq_len = d_step->kv_seq_len (static hipGraph!).
void gqa_decode_attention(const __half* q_in,
                          const __half* gate_src,
                          KVCacheView kv_cache,
                          __half* out,
                          void* split_scratch,
                          const AttentionParams& p,
                          int M,
                          int32_t kv_seq_len,
                          const StepParams* d_step = nullptr,
                          int splits = 16,
                          hipStream_t stream = nullptr);

// Tiled Causal FlashAttention Prefill (M >= 1, head_dim <= 256, F16/FP8/FP4 KV cache):
//   - Query m attends to KV positions [0 .. kv_offset + m] (causal!).
void flash_attention_prefill(const __half* q_in,
                             const __half* gate_src,
                             KVCacheView kv_cache,
                             __half* out,
                             const AttentionParams& p,
                             int M,
                             int32_t kv_offset,
                             hipStream_t stream = nullptr);

} // namespace rxinfer::attn

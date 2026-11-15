// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rxinfer/arena.hpp"
#include "rxinfer/config.hpp"

#include <hip/hip_fp16.h>

#include <cstdint>
#include <vector>

namespace rxinfer {

struct R4Matrix {
    void* qs = nullptr;          // [N/16, K/64, 32] uint4
    __half* scales = nullptr;    // [N/16, K/64, 16] __half
    int32_t N = 0;
    int32_t K = 0;

    bool valid() const { return qs != nullptr && scales != nullptr && N > 0 && K > 0; }
};

struct LayerWeights {
    LayerKind kind = LayerKind::FullAttention;

    const __half* input_norm = nullptr;     // [H]
    const __half* post_attn_norm = nullptr; // [H]

    // FullAttention weights
    R4Matrix q_proj{};                      // [H_q * D * (attn_output_gate ? 2 : 1), H]
    R4Matrix kv_proj{};                     // [2 * H_kv * D, H] (K rows followed by V rows)
    R4Matrix o_proj{};                      // [H, H_q * D]
    const __half* q_norm = nullptr;         // [D]
    const __half* k_norm = nullptr;         // [D]
    const __half* q_bias = nullptr;         // Optional [H_q * D] (Qwen2)
    const __half* kv_bias = nullptr;        // Optional [2 * H_kv * D]

    // LinearAttention (GatedDeltaNet) weights
    R4Matrix in_proj_qkvz{};                // [conv_dim + value_dim, H]
    const __half* in_proj_ba = nullptr;     // [2 * num_v_heads, H] FP16 (b rows then a rows)
    const __half* conv_w = nullptr;         // [conv_k, conv_dim] FP16 (coalesced)
    const float* A_log = nullptr;           // [num_v_heads]
    const float* dt_bias = nullptr;         // [num_v_heads]
    const __half* gdn_norm_w = nullptr;     // [head_v_dim]
    R4Matrix out_proj{};                    // [H, value_dim]

    // MLP (SwiGLU) weights
    R4Matrix gate_up_proj{};                // [2 * I, H] (gate rows followed by up rows)
    R4Matrix down_proj{};                   // [H, I]
};

struct ModelWeights {
    ModelConfig config;
    DeviceArena weight_arena;

    R4Matrix embed_tokens{};                // [V, H]
    const __half* final_norm = nullptr;     // [H]
    R4Matrix lm_head{};                     // [V, H] (aliases embed_tokens if tie_word_embeddings)

    std::vector<LayerWeights> layers;
};

} // namespace rxinfer

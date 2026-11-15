// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rxinfer/tensor.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace rxinfer {

enum class LayerKind : uint8_t {
    FullAttention = 0,
    LinearAttentionGDN = 1,
};

enum class NormKind : uint8_t {
    RMSNorm = 0,          // y = x * inv_rms * w
    RMSNormPlusOne = 1,   // y = x * inv_rms * (1 + w) (Qwen3.5/Qwen3.8 style)
    LayerNorm = 2,
};

enum class RopeKind : uint8_t {
    None = 0,
    NeoX = 1,        // split-half rotate
    Interleaved = 2, // adjacent pair rotate (e.g. mRoPE interleaved)
};

struct ModelConfig {
    std::string model_type;   // e.g. "qwen3_5", "qwen3_5_text", "llama", "qwen3", "qwen2"
    std::string arch_name;    // HF architectures[0]

    // Core transformer dimensions
    int32_t vocab_size = 0;
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t max_position_embeddings = 32768;
    bool tie_word_embeddings = false;

    // Normalization
    NormKind norm_kind = NormKind::RMSNorm;
    float rms_norm_eps = 1e-6f;

    // Full Attention parameters
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t head_dim = 0;
    bool attention_bias = false;
    bool qk_norm = false;
    bool attn_output_gate = false; // e.g. Qwen3.5/3.8 full attention swish output gate

    // RoPE parameters
    RopeKind rope_kind = RopeKind::NeoX;
    float rope_theta = 10000.0f;
    float partial_rotary_factor = 1.0f;
    std::vector<int32_t> mrope_section;
    bool mrope_interleaved = false;

    // Linear Attention / Gated DeltaNet (GDN) parameters
    int32_t linear_num_key_heads = 0;
    int32_t linear_key_head_dim = 0;
    int32_t linear_num_value_heads = 0;
    int32_t linear_value_head_dim = 0;
    int32_t linear_conv_kernel_dim = 4;

    // Layer schedule (empty -> all FullAttention)
    std::vector<LayerKind> layer_types;

    // Optional Speculative / MTP & special token IDs
    int32_t mtp_num_hidden_layers = 0;
    int32_t bos_token_id = -1;
    std::vector<int32_t> eos_token_ids;

    // Default quantization target for linear weights
    DType weight_dtype = DType::R4_GFX12_G64;
    DType act_dtype = DType::F16;
    DType kv_dtype = DType::F16;

    int32_t rotary_dim() const {
        return static_cast<int32_t>(head_dim * partial_rotary_factor);
    }

    int32_t num_full_attention_layers() const;
    int32_t num_linear_attention_layers() const;

    static ModelConfig from_hf_json_file(const std::string& path);
    static ModelConfig from_hf_json_string(const std::string& json_text);

    // Memory estimation helpers (bytes)
    uint64_t estimate_kv_cache_bytes_per_token() const;
    uint64_t estimate_recurrent_state_bytes() const;
};

} // namespace rxinfer

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rxinfer/common.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace rxinfer::gdn {

struct GDNParams {
    int32_t num_k_heads = 16;
    int32_t head_k_dim = 128;
    int32_t num_v_heads = 48;
    int32_t head_v_dim = 128;
    int32_t conv_kernel_size = 4;
    float rms_norm_eps = 1e-6f;

    __host__ __device__ int32_t key_dim() const { return num_k_heads * head_k_dim; }
    __host__ __device__ int32_t value_dim() const { return num_v_heads * head_v_dim; }
    __host__ __device__ int32_t conv_dim() const { return 2 * key_dim() + value_dim(); }
};

struct GDNStateView {
    __half* conv_state = nullptr;       // [conv_kernel_size - 1, conv_dim]
    float* recurrent_state = nullptr;   // [num_v_heads, head_k_dim, head_v_dim]
};

size_t gdn_conv_state_bytes(const GDNParams& p);
size_t gdn_recurrent_state_bytes(const GDNParams& p);

// Causal Depthwise Conv1D + SiLU over mixed QKV:
//   - qkv_in:   [M, in_stride] (first conv_dim cols are mixed_qkv; allows packed [qkvz] projection!)
//   - conv_w:   [conv_kernel_size, conv_dim] (coalesced layout, transposed on load from HF [conv_dim, 1, K])
//   - conv_bias:[conv_dim] or nullptr
//   - qkv_out:  [M, conv_dim]
void conv1d_silu(const __half* qkv_in,
                 int in_stride,
                 const __half* conv_w,
                 const __half* conv_bias,
                 __half* conv_state,
                 __half* qkv_out,
                 const GDNParams& p,
                 int M,
                 hipStream_t stream = nullptr);

// Fused GatedDeltaNet Step + Gated RMSNorm (register-resident S across M tokens!):
//   - qkv:      [M, conv_dim] (output of conv1d_silu)
//   - z:        [M, z_stride] (output gate; z_stride = conv_dim + value_dim if packed, else value_dim)
//   - b_in:     [M, ba_stride] (beta gate logits)
//   - a_in:     [M, ba_stride] (decay gate logits)
//   - A_log:    [num_v_heads] (float)
//   - dt_bias:  [num_v_heads] (float)
//   - norm_w:   [head_v_dim] (__half, standard RMSNorm weight, NOT +1)
//   - S:        [num_v_heads, head_k_dim, head_v_dim] (float, updated in-place)
//   - out:      [M, value_dim] (__half)
void fused_gdn_gated_norm(const __half* qkv,
                          const __half* z,
                          int z_stride,
                          const __half* b_in,
                          const __half* a_in,
                          int ba_stride,
                          const float* A_log,
                          const float* dt_bias,
                          const __half* norm_w,
                          float* recurrent_state,
                          __half* out,
                          const GDNParams& p,
                          int M,
                          hipStream_t stream = nullptr);

} // namespace rxinfer::gdn

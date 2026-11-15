// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/gdn.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace rxinfer::gdn {
namespace {

using h2_t = __half2;

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float softplus_f32(float x) {
    return (x > 20.0f) ? x : log1pf(__expf(x));
}

// Fast decode Conv1D + SiLU for M == 1 (vectorized by half2 across conv_dim).
template <int CONV_K>
__global__ __launch_bounds__(256)
void conv1d_silu_m1_kernel(const __half* __restrict__ qkv_in,
                           const __half* __restrict__ conv_w,
                           const __half* __restrict__ conv_bias,
                           __half* __restrict__ conv_state,
                           __half* __restrict__ qkv_out,
                           int conv_dim) {
    const int vc = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_dim = conv_dim >> 1;
    if (vc >= vec_dim) return;

    const h2_t x_new = reinterpret_cast<const h2_t*>(qkv_in)[vc];
    h2_t state_h2[CONV_K - 1];
#pragma unroll
    for (int k = 0; k < CONV_K - 1; ++k) {
        state_h2[k] = reinterpret_cast<const h2_t*>(conv_state + static_cast<size_t>(k) * conv_dim)[vc];
    }

    float2 acc = make_float2(0.0f, 0.0f);
    if (conv_bias != nullptr) {
        acc = __half22float2(reinterpret_cast<const h2_t*>(conv_bias)[vc]);
    }
#pragma unroll
    for (int k = 0; k < CONV_K - 1; ++k) {
        const float2 sf = __half22float2(state_h2[k]);
        const float2 wf = __half22float2(reinterpret_cast<const h2_t*>(conv_w + static_cast<size_t>(k) * conv_dim)[vc]);
        acc.x += sf.x * wf.x;
        acc.y += sf.y * wf.y;
    }
    {
        const float2 xf = __half22float2(x_new);
        const float2 wf = __half22float2(reinterpret_cast<const h2_t*>(conv_w + static_cast<size_t>(CONV_K - 1) * conv_dim)[vc]);
        acc.x += xf.x * wf.x;
        acc.y += xf.y * wf.y;
    }

    // Shift state in-place: [0] <- [1], ..., [CONV_K - 2] <- x_new
#pragma unroll
    for (int k = 0; k < CONV_K - 2; ++k) {
        reinterpret_cast<h2_t*>(conv_state + static_cast<size_t>(k) * conv_dim)[vc] = state_h2[k + 1];
    }
    reinterpret_cast<h2_t*>(conv_state + static_cast<size_t>(CONV_K - 2) * conv_dim)[vc] = x_new;

    reinterpret_cast<h2_t*>(qkv_out)[vc] = __floats2half2_rn(silu_f32(acc.x), silu_f32(acc.y));
}

// Multi-token prefill Conv1D + SiLU (keeps sliding window in registers across M tokens).
template <int CONV_K>
__global__ __launch_bounds__(256)
void conv1d_silu_batch_kernel(const __half* __restrict__ qkv_in,
                              int in_stride,
                              const __half* __restrict__ conv_w,
                              const __half* __restrict__ conv_bias,
                              __half* __restrict__ conv_state,
                              __half* __restrict__ qkv_out,
                              int conv_dim,
                              int M) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;

    float win[CONV_K - 1];
#pragma unroll
    for (int k = 0; k < CONV_K - 1; ++k) {
        win[k] = __half2float(conv_state[static_cast<size_t>(k) * conv_dim + c]);
    }
    float w[CONV_K];
#pragma unroll
    for (int k = 0; k < CONV_K; ++k) {
        w[k] = __half2float(conv_w[static_cast<size_t>(k) * conv_dim + c]);
    }
    const float bias = (conv_bias != nullptr) ? __half2float(conv_bias[c]) : 0.0f;

    for (int m = 0; m < M; ++m) {
        const float x_new = __half2float(qkv_in[static_cast<size_t>(m) * in_stride + c]);
        float acc = bias + x_new * w[CONV_K - 1];
#pragma unroll
        for (int k = 0; k < CONV_K - 1; ++k) {
            acc += win[k] * w[k];
        }
#pragma unroll
        for (int k = 0; k < CONV_K - 2; ++k) {
            win[k] = win[k + 1];
        }
        win[CONV_K - 2] = x_new;
        qkv_out[static_cast<size_t>(m) * conv_dim + c] = __float2half(silu_f32(acc));
    }

#pragma unroll
    for (int k = 0; k < CONV_K - 1; ++k) {
        conv_state[static_cast<size_t>(k) * conv_dim + c] = __float2half(win[k]);
    }
}

// =============================================================================
// Fused GatedDeltaNet Step + Gated RMSNorm (DK = 128, DV = 128)
//   512 threads (16 wave32 warps) per value head h:
//     wg_i = warp >> 2 (0..3 -> 32 rows of K each)
//     wg_j = warp & 3  (0..3 -> 32 cols of V each, j = wg_j * 32 + lane)
//   Reads/writes S[h, DK, DV] ONCE across ALL M tokens (register-resident S!).
// =============================================================================
__global__ __launch_bounds__(512)
void gdn_fused_128x128_kernel(const __half* __restrict__ qkv,
                              const __half* __restrict__ z,
                              int z_stride,
                              const __half* __restrict__ b_in,
                              const __half* __restrict__ a_in,
                              int ba_stride,
                              const float* __restrict__ A_log,
                              const float* __restrict__ dt_bias,
                              const __half* __restrict__ norm_w,
                              float* __restrict__ recurrent_state,
                              __half* __restrict__ out,
                              GDNParams p,
                              int M) {
    constexpr int DK = 128;
    constexpr int DV = 128;
    constexpr int ROWS_PER_WG = 32;

    __shared__ float s_q[DK];
    __shared__ float s_k[DK];
    __shared__ float s_v[DV];
    __shared__ float s_z[DV];
    __shared__ float s_delta[DV];
    __shared__ float s_red[4][DV];
    __shared__ float s_q_red[4];
    __shared__ float s_k_red[4];
    __shared__ float s_norm_red[4];
    __shared__ float s_decay;
    __shared__ float s_beta;
    __shared__ float s_norm_w[DV];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;   // 0..15
    const int lane = tid & 31;   // 0..31
    const int wg_i = warp >> 2;  // 0..3
    const int wg_j = warp & 3;   // 0..3
    const int i_base = wg_i * ROWS_PER_WG;
    const int j = wg_j * 32 + lane;

    const int h = blockIdx.x;
    const int gqa_ratio = p.num_v_heads / p.num_k_heads;
    const int hk = p.tiled_v_heads ? (h % p.num_k_heads) : (h / gqa_ratio);
    const int key_dim = p.key_dim();
    const int value_dim = p.value_dim();
    const int conv_dim = p.conv_dim();

    if (tid < DV) {
        s_norm_w[tid] = __half2float(norm_w[tid]);
    }

    const float a_log_h = A_log[h];
    const float dt_bias_h = dt_bias[h];
    const float neg_exp_a = -__expf(a_log_h);
    const float q_scale = rsqrtf(static_cast<float>(DK));

    // Load recurrent state slice S[h, i_base + 0..31, j] once into 32 registers!
    float* S_head = recurrent_state + static_cast<size_t>(h) * (DK * DV);
    float s_reg[ROWS_PER_WG];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WG; ++r) {
        s_reg[r] = S_head[static_cast<size_t>(i_base + r) * DV + j];
    }

    for (int m = 0; m < M; ++m) {
        const __half* qkv_m = qkv + static_cast<size_t>(m) * conv_dim;
        const __half* z_m = z + static_cast<size_t>(m) * z_stride + h * DV;

        // 1. Load Q, K, V, Z, and gates; compute L2-norm sums for Q and K
        float q_raw = 0.0f;
        float k_raw = 0.0f;
        if (tid < DK) {
            q_raw = __half2float(qkv_m[hk * DK + tid]);
            k_raw = __half2float(qkv_m[key_dim + hk * DK + tid]);
            s_v[tid] = __half2float(qkv_m[2 * key_dim + h * DV + tid]);
            s_z[tid] = __half2float(z_m[tid]);

            float q2 = q_raw * q_raw;
            float k2 = k_raw * k_raw;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                q2 += __shfl_xor(q2, off);
                k2 += __shfl_xor(k2, off);
            }
            if (lane == 0) {
                s_q_red[warp] = q2;
                s_k_red[warp] = k2;
            }
        }
        if (tid == 0) {
            const float a_val = __half2float(a_in[static_cast<size_t>(m) * ba_stride + h]);
            const float b_val = __half2float(b_in[static_cast<size_t>(m) * ba_stride + h]);
            s_decay = __expf(neg_exp_a * softplus_f32(a_val + dt_bias_h));
            s_beta = 1.0f / (1.0f + __expf(-b_val));
        }
        __syncthreads();

        if (tid < DK) {
            const float sum_q2 = s_q_red[0] + s_q_red[1] + s_q_red[2] + s_q_red[3];
            const float sum_k2 = s_k_red[0] + s_k_red[1] + s_k_red[2] + s_k_red[3];
            s_q[tid] = q_raw * rsqrtf(sum_q2 + 1e-6f) * q_scale;
            s_k[tid] = k_raw * rsqrtf(sum_k2 + 1e-6f);
        }
        __syncthreads();

        // 2. Decay S and compute partial kv_mem[j] = sum_r (S * decay * k[r])
        const float decay = s_decay;
        float kv_part = 0.0f;
#pragma unroll
        for (int r = 0; r < ROWS_PER_WG; ++r) {
            const float sr = s_reg[r] * decay;
            s_reg[r] = sr;
            kv_part += sr * s_k[i_base + r];
        }
        s_red[wg_i][j] = kv_part;
        __syncthreads();

        // 3. Reduce kv_mem and compute delta[j] = (v[j] - kv_mem[j]) * beta
        if (wg_i == 0) {
            const float kv_mem = s_red[0][j] + s_red[1][j] + s_red[2][j] + s_red[3][j];
            s_delta[j] = (s_v[j] - kv_mem) * s_beta;
        }
        __syncthreads();

        // 4. Rank-1 update S += k * delta[j] and compute partial y[j] = sum_r (S_new * q[r])
        const float dj = s_delta[j];
        float y_part = 0.0f;
#pragma unroll
        for (int r = 0; r < ROWS_PER_WG; ++r) {
            const float s_new = s_reg[r] + s_k[i_base + r] * dj;
            s_reg[r] = s_new;
            y_part += s_new * s_q[i_base + r];
        }
        s_red[wg_i][j] = y_part;
        __syncthreads();

        // 5. Reduce y[j] and compute RMSNorm(y) * silu(z)
        float y_val = 0.0f;
        if (wg_i == 0) {
            y_val = s_red[0][j] + s_red[1][j] + s_red[2][j] + s_red[3][j];
            float y2 = y_val * y_val;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                y2 += __shfl_xor(y2, off);
            }
            if (lane == 0) s_norm_red[warp] = y2;
        }
        __syncthreads();

        if (wg_i == 0) {
            const float sum_y2 = s_norm_red[0] + s_norm_red[1] + s_norm_red[2] + s_norm_red[3];
            const float inv_rms = rsqrtf(sum_y2 / static_cast<float>(DV) + p.rms_norm_eps);
            const float gated = y_val * inv_rms * s_norm_w[j] * silu_f32(s_z[j]);
            out[static_cast<size_t>(m) * value_dim + h * DV + j] = __float2half(gated);
        }
        if (m + 1 < M) {
            __syncthreads();
        }
    }

    // Store updated recurrent state S[h, DK, DV] back to VRAM once!
#pragma unroll
    for (int r = 0; r < ROWS_PER_WG; ++r) {
        S_head[static_cast<size_t>(i_base + r) * DV + j] = s_reg[r];
    }
}

} // namespace

size_t gdn_conv_state_bytes(const GDNParams& p) {
    return static_cast<size_t>(p.conv_kernel_size - 1) * p.conv_dim() * sizeof(__half);
}

size_t gdn_recurrent_state_bytes(const GDNParams& p) {
    return static_cast<size_t>(p.num_v_heads) * p.head_k_dim * p.head_v_dim * sizeof(float);
}

void conv1d_silu(const __half* qkv_in,
                 int in_stride,
                 const __half* conv_w,
                 const __half* conv_bias,
                 __half* conv_state,
                 __half* qkv_out,
                 const GDNParams& p,
                 int M,
                 hipStream_t stream) {
    const int conv_dim = p.conv_dim();
    if (p.conv_kernel_size != 4) {
        std::fprintf(stderr, "Unsupported GDN conv_kernel_size=%d (expected 4)\n", p.conv_kernel_size);
        std::abort();
    }
    if (M == 1 && (conv_dim & 1) == 0) {
        const int vecs = conv_dim >> 1;
        hipLaunchKernelGGL((conv1d_silu_m1_kernel<4>), dim3(div_ceil(vecs, 256)), dim3(256), 0, stream,
                           qkv_in, conv_w, conv_bias, conv_state, qkv_out, conv_dim);
    } else {
        hipLaunchKernelGGL((conv1d_silu_batch_kernel<4>), dim3(div_ceil(conv_dim, 256)), dim3(256), 0, stream,
                           qkv_in, in_stride, conv_w, conv_bias, conv_state, qkv_out, conv_dim, M);
    }
}

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
                          hipStream_t stream) {
    if (p.head_k_dim == 128 && p.head_v_dim == 128) {
        hipLaunchKernelGGL(gdn_fused_128x128_kernel, dim3(p.num_v_heads), dim3(512), 0, stream,
                           qkv, z, z_stride, b_in, a_in, ba_stride,
                           A_log, dt_bias, norm_w, recurrent_state, out, p, M);
    } else {
        std::fprintf(stderr, "Unsupported GDN head dims (%d, %d)\n", p.head_k_dim, p.head_v_dim);
        std::abort();
    }
}

} // namespace rxinfer::gdn

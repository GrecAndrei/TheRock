// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/ops.hpp"
#include "rxinfer/common.hpp"

#include <cfloat>
#include <cmath>

namespace rxinfer::ops {
namespace {

using f16x8 = _Float16 __attribute__((ext_vector_type(8)));
using f32x8 = float __attribute__((ext_vector_type(8)));
using h2_t = __half2;

__device__ __forceinline__ float block_reduce_sum_256(float val) {
    __shared__ float smem[8]; // 256 threads = 8 wave32 warps
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        val += __shfl_xor(val, off);
    }
    if (lane == 0) smem[warp] = val;
    __syncthreads();

    if (warp == 0) {
        float wv = (lane < 8) ? smem[lane] : 0.0f;
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            wv += __shfl_xor(wv, off);
        }
        if (lane == 0) smem[0] = wv;
    }
    __syncthreads();
    return smem[0];
}

// Single-pass fused ResidualAdd + RMSNorm (or standalone RMSNorm if FUSED_RESIDUAL == false).
// Keeps up to MAX_CHUNKS uint4 (8x FP16 each) in registers when D <= 256 * 8 * MAX_CHUNKS = 8192,
// eliminating the second global memory read of x/residual.
template <bool FUSED_RESIDUAL, bool PLUS_ONE, int MAX_CHUNKS>
__global__ __launch_bounds__(256)
void rmsnorm_cached_kernel(const __half* __restrict__ x,
                           __half* __restrict__ residual,
                           const __half* __restrict__ weight,
                           __half* __restrict__ y,
                           int D, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t base = static_cast<size_t>(row) * D;

    uint4 r_vec[MAX_CHUNKS];
    float sum_sq = 0.0f;

    const int vec_D = D >> 3; // number of uint4 (8 halfs) per row
#pragma unroll
    for (int c = 0; c < MAX_CHUNKS; ++c) {
        const int idx = tid + c * 256;
        if (idx < vec_D) {
            const uint4 xv = reinterpret_cast<const uint4*>(x + base)[idx];
            uint4 rv;
            if constexpr (FUSED_RESIDUAL) {
                const uint4 resv = reinterpret_cast<const uint4*>(residual + base)[idx];
                const h2_t* xh = reinterpret_cast<const h2_t*>(&xv);
                const h2_t* rh = reinterpret_cast<const h2_t*>(&resv);
                h2_t* out_rh = reinterpret_cast<h2_t*>(&rv);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float2 fx = __half22float2(xh[k]);
                    const float2 fr = __half22float2(rh[k]);
                    const float s0 = fx.x + fr.x;
                    const float s1 = fx.y + fr.y;
                    sum_sq += s0 * s0 + s1 * s1;
                    out_rh[k] = __floats2half2_rn(s0, s1);
                }
                reinterpret_cast<uint4*>(residual + base)[idx] = rv;
            } else {
                rv = xv;
                const h2_t* xh = reinterpret_cast<const h2_t*>(&rv);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float2 fx = __half22float2(xh[k]);
                    sum_sq += fx.x * fx.x + fx.y * fx.y;
                }
            }
            r_vec[c] = rv;
        }
    }

    const float total_sq = block_reduce_sum_256(sum_sq);
    const float inv_rms = rsqrtf(total_sq / static_cast<float>(D) + eps);

#pragma unroll
    for (int c = 0; c < MAX_CHUNKS; ++c) {
        const int idx = tid + c * 256;
        if (idx < vec_D) {
            const uint4 wv = reinterpret_cast<const uint4*>(weight)[idx];
            const h2_t* rh = reinterpret_cast<const h2_t*>(&r_vec[c]);
            const h2_t* wh = reinterpret_cast<const h2_t*>(&wv);
            uint4 yv;
            h2_t* yh = reinterpret_cast<h2_t*>(&yv);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 fr = __half22float2(rh[k]);
                const float2 fw = __half22float2(wh[k]);
                const float w0 = PLUS_ONE ? (1.0f + fw.x) : fw.x;
                const float w1 = PLUS_ONE ? (1.0f + fw.y) : fw.y;
                yh[k] = __floats2half2_rn(fr.x * inv_rms * w0, fr.y * inv_rms * w1);
            }
            reinterpret_cast<uint4*>(y + base)[idx] = yv;
        }
    }
}

// Generic fallback RMSNorm for arbitrary D (supports non-multiple-of-8 or D > 8192).
template <bool FUSED_RESIDUAL, bool PLUS_ONE>
__global__ __launch_bounds__(256)
void rmsnorm_generic_kernel(const __half* __restrict__ x,
                            __half* __restrict__ residual,
                            const __half* __restrict__ weight,
                            __half* __restrict__ y,
                            int D, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t base = static_cast<size_t>(row) * D;

    float sum_sq = 0.0f;
    for (int d = tid; d < D; d += 256) {
        float v = __half2float(x[base + d]);
        if constexpr (FUSED_RESIDUAL) {
            v += __half2float(residual[base + d]);
            residual[base + d] = __float2half(v);
        }
        sum_sq += v * v;
    }

    const float total_sq = block_reduce_sum_256(sum_sq);
    const float inv_rms = rsqrtf(total_sq / static_cast<float>(D) + eps);

    for (int d = tid; d < D; d += 256) {
        const float r = __half2float(FUSED_RESIDUAL ? residual[base + d] : x[base + d]);
        const float w = __half2float(weight[d]);
        const float scale = PLUS_ONE ? (1.0f + w) : w;
        y[base + d] = __float2half(r * inv_rms * scale);
    }
}

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ __launch_bounds__(256)
void silu_mul_vec_kernel(const __half* __restrict__ gate,
                         const __half* __restrict__ up,
                         __half* __restrict__ out,
                         size_t total_elems) {
    const size_t vec_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t idx = vec_idx << 3;
    if (idx + 7 < total_elems) {
        const uint4 gv = *reinterpret_cast<const uint4*>(gate + idx);
        const uint4 uv = *reinterpret_cast<const uint4*>(up + idx);
        const h2_t* gh = reinterpret_cast<const h2_t*>(&gv);
        const h2_t* uh = reinterpret_cast<const h2_t*>(&uv);
        uint4 ov;
        h2_t* oh = reinterpret_cast<h2_t*>(&ov);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 fg = __half22float2(gh[k]);
            const float2 fu = __half22float2(uh[k]);
            oh[k] = __floats2half2_rn(silu_f32(fg.x) * fu.x, silu_f32(fg.y) * fu.y);
        }
        *reinterpret_cast<uint4*>(out + idx) = ov;
    } else if (idx < total_elems) {
        for (size_t i = idx; i < total_elems; ++i) {
            const float g = __half2float(gate[i]);
            const float u = __half2float(up[i]);
            out[i] = __float2half(silu_f32(g) * u);
        }
    }
}

__global__ __launch_bounds__(256)
void silu_mul_packed_vec_kernel(const __half* __restrict__ gate_up,
                                __half* __restrict__ out,
                                int M, int I) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_I = I >> 3;
    if ((I & 7) == 0) {
        const int total_vecs = M * vec_I;
        if (tid >= total_vecs) return;
        const int m = tid / vec_I;
        const int vi = tid % vec_I;
        const size_t row_in = static_cast<size_t>(m) * (2 * I);
        const size_t row_out = static_cast<size_t>(m) * I;

        const uint4 gv = reinterpret_cast<const uint4*>(gate_up + row_in)[vi];
        const uint4 uv = reinterpret_cast<const uint4*>(gate_up + row_in + I)[vi];
        const h2_t* gh = reinterpret_cast<const h2_t*>(&gv);
        const h2_t* uh = reinterpret_cast<const h2_t*>(&uv);
        uint4 ov;
        h2_t* oh = reinterpret_cast<h2_t*>(&ov);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 fg = __half22float2(gh[k]);
            const float2 fu = __half22float2(uh[k]);
            oh[k] = __floats2half2_rn(silu_f32(fg.x) * fu.x, silu_f32(fg.y) * fu.y);
        }
        reinterpret_cast<uint4*>(out + row_out)[vi] = ov;
    } else {
        const size_t total = static_cast<size_t>(M) * I;
        if (static_cast<size_t>(tid) >= total) return;
        const int m = tid / I;
        const int i = tid % I;
        const size_t row_in = static_cast<size_t>(m) * (2 * I);
        const float g = __half2float(gate_up[row_in + i]);
        const float u = __half2float(gate_up[row_in + I + i]);
        out[tid] = __float2half(silu_f32(g) * u);
    }
}

__global__ __launch_bounds__(256)
void embed_lookup_f16_kernel(const __half* __restrict__ table,
                             const StepParams* __restrict__ step_ptr,
                             const int32_t* __restrict__ token_ids,
                             __half* __restrict__ out,
                             int M, int V, int H) {
    const int m = blockIdx.y;
    if (m >= M) return;
    int32_t tok = token_ids ? token_ids[m] : step_ptr->token_id;
    if (tok < 0 || tok >= V) tok = 0;

    const int h = (blockIdx.x * blockDim.x + threadIdx.x) << 3;
    if (h + 7 < H) {
        *reinterpret_cast<uint4*>(out + static_cast<size_t>(m) * H + h) =
            *reinterpret_cast<const uint4*>(table + static_cast<size_t>(tok) * H + h);
    } else if (h < H) {
        for (int i = h; i < H; ++i) {
            out[static_cast<size_t>(m) * H + i] = table[static_cast<size_t>(tok) * H + i];
        }
    }
}

// Same nibble unpacker as quant_r4_gfx12.hip, yielding 4x half2:
//   h01 = [w0, w1], h23 = [w2, w3], h45 = [w4, w5], h67 = [w6, w7]
__device__ __forceinline__ void unpack_r4_u32_to_4xh2_ops(
    uint32_t p, h2_t& h01, h2_t& h23, h2_t& h45, h2_t& h67) {
    const uint32_t mask = 0x000F000Fu;
    const uint32_t bias = 0x64006400u;
    const h2_t neg1032 = __halves2half2(__float2half(-1032.0f), __float2half(-1032.0f));

    uint32_t b0 = (p & mask) | bias;
    uint32_t b1 = ((p >> 4) & mask) | bias;
    uint32_t b2 = ((p >> 8) & mask) | bias;
    uint32_t b3 = ((p >> 12) & mask) | bias;

    h2_t t0, t1, t2, t3;
    __builtin_memcpy(&t0, &b0, 4);
    __builtin_memcpy(&t1, &b1, 4);
    __builtin_memcpy(&t2, &b2, 4);
    __builtin_memcpy(&t3, &b3, 4);

    h01 = __hadd2(t0, neg1032);
    h23 = __hadd2(t1, neg1032);
    h45 = __hadd2(t2, neg1032);
    h67 = __hadd2(t3, neg1032);
}

// Direct single-row R4_GFX12_G64 embedding dequantization:
// 1 thread per k_tile=64 (dequantizes both k_half=0 and k_half=8 across s=0..3).
__global__ __launch_bounds__(256)
void embed_lookup_r4_kernel(const uint4* __restrict__ W_qs,
                            const __half* __restrict__ W_scales,
                            const StepParams* __restrict__ step_ptr,
                            const int32_t* __restrict__ token_ids,
                            __half* __restrict__ out,
                            int M, int V, int H) {
    const int m = blockIdx.y;
    if (m >= M) return;
    int32_t tok = token_ids ? token_ids[m] : step_ptr->token_id;
    if (tok < 0 || tok >= V) tok = 0;

    const int kt = blockIdx.x * blockDim.x + threadIdx.x;
    const int k_tiles = H >> 6;
    if (kt >= k_tiles) return;

    const int nt = tok >> 4;
    const int nl = tok & 15;
    const size_t tile_idx = static_cast<size_t>(nt) * k_tiles + kt;

    const __half sc_h = W_scales[tile_idx * 16 + nl];
    const h2_t sc2 = __half2half2(sc_h);

    const uint4 w_lo = W_qs[tile_idx * 32 + nl];
    const uint4 w_hi = W_qs[tile_idx * 32 + nl + 16];
    const uint32_t* p_lo = reinterpret_cast<const uint32_t*>(&w_lo);
    const uint32_t* p_hi = reinterpret_cast<const uint32_t*>(&w_hi);

    h2_t* dst = reinterpret_cast<h2_t*>(out + static_cast<size_t>(m) * H + (kt << 6));
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        h2_t h01, h23, h45, h67;
        unpack_r4_u32_to_4xh2_ops(p_lo[s], h01, h23, h45, h67);
        dst[s * 8 + 0] = __hmul2(h01, sc2);
        dst[s * 8 + 1] = __hmul2(h23, sc2);
        dst[s * 8 + 2] = __hmul2(h45, sc2);
        dst[s * 8 + 3] = __hmul2(h67, sc2);

        unpack_r4_u32_to_4xh2_ops(p_hi[s], h01, h23, h45, h67);
        dst[s * 8 + 4] = __hmul2(h01, sc2);
        dst[s * 8 + 5] = __hmul2(h23, sc2);
        dst[s * 8 + 6] = __hmul2(h45, sc2);
        dst[s * 8 + 7] = __hmul2(h67, sc2);
    }
}

// Fast F16 WG-SplitK GEMV for M <= 16 and small N (e.g. GDN in_proj_ba [96, 5120]):
// 1 workgroup of WG_WARPS (8 warps = 256 threads) per output row, coalesced uint4 + fdot2.
template <int WG_WARPS, int M_MAX>
__global__ __launch_bounds__(WG_WARPS * 32)
void gemv_f16_wg_kernel(const __half* __restrict__ A,
                        const __half* __restrict__ W,
                        __half* __restrict__ C,
                        int M, int N, int K) {
    __shared__ float s_acc[WG_WARPS][M_MAX];
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const __half* w_row = W + static_cast<size_t>(row) * K;
    const int vec_K = K >> 3;

    float acc[M_MAX];
#pragma unroll
    for (int m = 0; m < M_MAX; ++m) acc[m] = 0.0f;

    for (int vk = tid; vk < vec_K; vk += WG_WARPS * 32) {
        const uint4 wv = reinterpret_cast<const uint4*>(w_row)[vk];
        const h2_t* wh = reinterpret_cast<const h2_t*>(&wv);
#pragma unroll
        for (int m = 0; m < M_MAX; ++m) {
            if (m < M) {
                const uint4 av = reinterpret_cast<const uint4*>(A + static_cast<size_t>(m) * K)[vk];
                const h2_t* ah = reinterpret_cast<const h2_t*>(&av);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    using v2f16 = _Float16 __attribute__((ext_vector_type(2)));
                    v2f16 va, vw;
                    __builtin_memcpy(&va, &ah[k], 4);
                    __builtin_memcpy(&vw, &wh[k], 4);
                    acc[m] = __builtin_amdgcn_fdot2(va, vw, acc[m], false);
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M_MAX; ++m) {
        if (m < M) {
            float v = acc[m];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
            if (lane == 0) s_acc[warp][m] = v;
        }
    }
    __syncthreads();
    if (warp == 0) {
        for (int m = 0; m < M; ++m) {
            float sum = (lane < WG_WARPS) ? s_acc[lane][m] : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor(sum, off);
            if (lane == 0) C[static_cast<size_t>(m) * N + row] = __float2half(sum);
        }
    }
}

// Fast FP16 GEMV for small M <= 16 and small N (e.g., GDN in_proj_ba [96, 5120]).
// 1 wave32 per output row n, vectorized uint4 (8 FP16s) per lane -> 256 FP16s per iteration.
template <int M_MAX>
__global__ __launch_bounds__(128)
void gemv_f16_kernel(const __half* __restrict__ A,
                     const __half* __restrict__ W,
                     __half* __restrict__ C,
                     int M, int N, int K) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 4 + warp;
    if (row >= N) return;

    float acc[M_MAX];
#pragma unroll
    for (int m = 0; m < M_MAX; ++m) acc[m] = 0.0f;

    const __half* w_row = W + static_cast<size_t>(row) * K;
    const int vec_K = K >> 3;
    for (int vk = lane; vk < vec_K; vk += 32) {
        const uint4 wv = reinterpret_cast<const uint4*>(w_row)[vk];
        const h2_t* wh = reinterpret_cast<const h2_t*>(&wv);
#pragma unroll
        for (int m = 0; m < M_MAX; ++m) {
            if (m < M) {
                const uint4 av = reinterpret_cast<const uint4*>(A + static_cast<size_t>(m) * K)[vk];
                const h2_t* ah = reinterpret_cast<const h2_t*>(&av);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    using v2f16 = _Float16 __attribute__((ext_vector_type(2)));
                    v2f16 va, vw;
                    __builtin_memcpy(&va, &ah[k], 4);
                    __builtin_memcpy(&vw, &wh[k], 4);
                    acc[m] = __builtin_amdgcn_fdot2(va, vw, acc[m], false);
                }
            }
        }
    }
    // Tail if K not divisible by 256 (still aligned or scalar remainder)
    const int tail_start = vec_K << 3;
    for (int k = tail_start + lane; k < K; k += 32) {
        const float wv = __half2float(w_row[k]);
#pragma unroll
        for (int m = 0; m < M_MAX; ++m) {
            if (m < M) {
                acc[m] += __half2float(A[static_cast<size_t>(m) * K + k]) * wv;
            }
        }
    }

#pragma unroll
    for (int m = 0; m < M_MAX; ++m) {
        if (m < M) {
            float v = acc[m];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_xor(v, off);
            }
            if (lane == 0) {
                C[static_cast<size_t>(m) * N + row] = __float2half(v);
            }
        }
    }
}

// Tiled Wave32 WMMA FP16 GEMM (M > 16): C[M, N] = A[M, K] * W[N, K]^T
__global__ __launch_bounds__(128)
void gemm_f16_wmma_kernel(const __half* __restrict__ A,
                          const __half* __restrict__ W,
                          __half* __restrict__ C,
                          int M, int N, int K) {
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int STRIDE = BK + 8;

    __shared__ _Float16 lds_A[BM][STRIDE];
    __shared__ _Float16 lds_B[BN][STRIDE];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;

    const int m_block = blockIdx.y * BM;
    const int n_block = blockIdx.x * BN;
    const int warp_m = (warp >> 1) * 32;
    const int warp_n = (warp & 1) * 32;

    f32x8 acc[2][2]{};
#pragma unroll
    for (int wm = 0; wm < 2; ++wm)
        for (int wn = 0; wn < 2; ++wn)
            for (int e = 0; e < 8; ++e) acc[wm][wn][e] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // 128 threads load 64x32 halfs (256 uint4 -> 2 passes!)
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int chunk = tid + pass * 128; // 0..255
            const int r = chunk >> 2;           // 0..63
            const int c = (chunk & 3) << 3;     // 0, 8, 16, 24

            if (m_block + r < M && k0 + c + 7 < K) {
                *reinterpret_cast<uint4*>(&lds_A[r][c]) =
                    *reinterpret_cast<const uint4*>(A + static_cast<size_t>(m_block + r) * K + k0 + c);
            } else {
                for (int i = 0; i < 8; ++i) {
                    lds_A[r][c + i] = (m_block + r < M && k0 + c + i < K)
                                          ? static_cast<_Float16>(A[static_cast<size_t>(m_block + r) * K + k0 + c + i])
                                          : static_cast<_Float16>(0.0f);
                }
            }

            if (n_block + r < N && k0 + c + 7 < K) {
                *reinterpret_cast<uint4*>(&lds_B[r][c]) =
                    *reinterpret_cast<const uint4*>(W + static_cast<size_t>(n_block + r) * K + k0 + c);
            } else {
                for (int i = 0; i < 8; ++i) {
                    lds_B[r][c + i] = (n_block + r < N && k0 + c + i < K)
                                          ? static_cast<_Float16>(W[static_cast<size_t>(n_block + r) * K + k0 + c + i])
                                          : static_cast<_Float16>(0.0f);
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const int k_sub = ks * 16 + (lane >= 16 ? 8 : 0);
            f16x8 a_frag[2], b_frag[2];
#pragma unroll
            for (int wm = 0; wm < 2; ++wm) {
                a_frag[wm] = *reinterpret_cast<const f16x8*>(&lds_A[warp_m + wm * 16 + lane16][k_sub]);
            }
#pragma unroll
            for (int wn = 0; wn < 2; ++wn) {
                b_frag[wn] = *reinterpret_cast<const f16x8*>(&lds_B[warp_n + wn * 16 + lane16][k_sub]);
            }
#pragma unroll
            for (int wm = 0; wm < 2; ++wm) {
#pragma unroll
                for (int wn = 0; wn < 2; ++wn) {
                    acc[wm][wn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        a_frag[wm], b_frag[wn], acc[wm][wn]);
                }
            }
        }
        __syncthreads();
    }

    const int m_half = (lane >= 16) ? 8 : 0;
#pragma unroll
    for (int wm = 0; wm < 2; ++wm) {
#pragma unroll
        for (int wn = 0; wn < 2; ++wn) {
            const int col = n_block + warp_n + wn * 16 + lane16;
            if (col < N) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int row = m_block + warp_m + wm * 16 + m_half + e;
                    if (row < M) {
                        C[static_cast<size_t>(row) * N + col] = __float2half(acc[wm][wn][e]);
                    }
                }
            }
        }
    }
#endif
}

constexpr int ARGMAX_BLOCKS = 512;

__global__ __launch_bounds__(256)
void argmax_stage1_kernel(const __half* __restrict__ logits, int V,
                          float* __restrict__ part_val, int32_t* __restrict__ part_idx) {
    __shared__ float s_val[8];
    __shared__ int32_t s_idx[8];

    const int m = blockIdx.y;
    const __half* row_logits = logits + static_cast<size_t>(m) * V;
    float* row_val = part_val + static_cast<size_t>(m) * ARGMAX_BLOCKS;
    int32_t* row_idx = part_idx + static_cast<size_t>(m) * ARGMAX_BLOCKS;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int stride = gridDim.x * blockDim.x;

    float best_v = -FLT_MAX;
    int32_t best_i = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < V; i += stride) {
        const float v = __half2float(row_logits[i]);
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor(best_v, off);
        const int32_t oi = __shfl_xor(best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }
    if (lane == 0) {
        s_val[warp] = best_v;
        s_idx[warp] = best_i;
    }
    __syncthreads();

    if (warp == 0) {
        float wv = (lane < 8) ? s_val[lane] : -FLT_MAX;
        int32_t wi = (lane < 8) ? s_idx[lane] : 0;
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            const float ov = __shfl_xor(wv, off);
            const int32_t oi = __shfl_xor(wi, off);
            if (ov > wv || (ov == wv && oi < wi)) {
                wv = ov;
                wi = oi;
            }
        }
        if (lane == 0) {
            row_val[blockIdx.x] = wv;
            row_idx[blockIdx.x] = wi;
        }
    }
}

__global__ __launch_bounds__(256)
void argmax_stage2_kernel(const float* __restrict__ part_val, const int32_t* __restrict__ part_idx,
                          int count, int32_t* __restrict__ out_token, StepParams* __restrict__ d_step) {
    __shared__ float s_val[8];
    __shared__ int32_t s_idx[8];

    const int m = blockIdx.y;
    const float* row_val = part_val + static_cast<size_t>(m) * ARGMAX_BLOCKS;
    const int32_t* row_idx = part_idx + static_cast<size_t>(m) * ARGMAX_BLOCKS;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float best_v = -FLT_MAX;
    int32_t best_i = 0;
    for (int i = tid; i < count; i += 256) {
        const float v = row_val[i];
        const int32_t idx = row_idx[i];
        if (v > best_v || (v == best_v && idx < best_i)) {
            best_v = v;
            best_i = idx;
        }
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor(best_v, off);
        const int32_t oi = __shfl_xor(best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }
    if (lane == 0) {
        s_val[warp] = best_v;
        s_idx[warp] = best_i;
    }
    __syncthreads();

    if (warp == 0) {
        float wv = (lane < 8) ? s_val[lane] : -FLT_MAX;
        int32_t wi = (lane < 8) ? s_idx[lane] : 0;
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            const float ov = __shfl_xor(wv, off);
            const int32_t oi = __shfl_xor(wi, off);
            if (ov > wv || (ov == wv && oi < wi)) {
                wv = ov;
                wi = oi;
            }
        }
        if (lane == 0) {
            if (out_token) out_token[m] = wi;
            if (d_step && m == 0) d_step->token_id = wi;
        }
    }
}

} // namespace

void rmsnorm(const __half* x, const __half* weight, __half* y,
             int M, int D, float eps, bool plus_one, hipStream_t stream) {
    if ((D & 7) == 0 && D <= 8192) {
        if (plus_one) {
            hipLaunchKernelGGL((rmsnorm_cached_kernel<false, true, 4>), dim3(M), dim3(256), 0, stream,
                               x, nullptr, weight, y, D, eps);
        } else {
            hipLaunchKernelGGL((rmsnorm_cached_kernel<false, false, 4>), dim3(M), dim3(256), 0, stream,
                               x, nullptr, weight, y, D, eps);
        }
    } else {
        if (plus_one) {
            hipLaunchKernelGGL((rmsnorm_generic_kernel<false, true>), dim3(M), dim3(256), 0, stream,
                               x, nullptr, weight, y, D, eps);
        } else {
            hipLaunchKernelGGL((rmsnorm_generic_kernel<false, false>), dim3(M), dim3(256), 0, stream,
                               x, nullptr, weight, y, D, eps);
        }
    }
}

void fused_residual_rmsnorm(const __half* x, __half* residual, const __half* weight, __half* y,
                            int M, int D, float eps, bool plus_one, hipStream_t stream) {
    if ((D & 7) == 0 && D <= 8192) {
        if (plus_one) {
            hipLaunchKernelGGL((rmsnorm_cached_kernel<true, true, 4>), dim3(M), dim3(256), 0, stream,
                               x, residual, weight, y, D, eps);
        } else {
            hipLaunchKernelGGL((rmsnorm_cached_kernel<true, false, 4>), dim3(M), dim3(256), 0, stream,
                               x, residual, weight, y, D, eps);
        }
    } else {
        if (plus_one) {
            hipLaunchKernelGGL((rmsnorm_generic_kernel<true, true>), dim3(M), dim3(256), 0, stream,
                               x, residual, weight, y, D, eps);
        } else {
            hipLaunchKernelGGL((rmsnorm_generic_kernel<true, false>), dim3(M), dim3(256), 0, stream,
                               x, residual, weight, y, D, eps);
        }
    }
}

void fused_silu_mul(const __half* gate, const __half* up, __half* out,
                    int M, int I, hipStream_t stream) {
    const size_t total = static_cast<size_t>(M) * I;
    const size_t vecs = div_ceil(total, 8);
    hipLaunchKernelGGL(silu_mul_vec_kernel, dim3(div_ceil(vecs, 256)), dim3(256), 0, stream,
                       gate, up, out, total);
}

void fused_silu_mul_packed(const __half* gate_up, __half* out,
                           int M, int I, hipStream_t stream) {
    const size_t work = ((I & 7) == 0) ? (static_cast<size_t>(M) * (I >> 3)) : (static_cast<size_t>(M) * I);
    hipLaunchKernelGGL(silu_mul_packed_vec_kernel, dim3(div_ceil(work, 256)), dim3(256), 0, stream,
                       gate_up, out, M, I);
}

void embed_lookup_f16_step(const __half* table, const StepParams* d_step, __half* out,
                           int V, int H, hipStream_t stream) {
    dim3 grid(div_ceil(div_ceil(H, 8), 256), 1);
    hipLaunchKernelGGL(embed_lookup_f16_kernel, grid, dim3(256), 0, stream,
                       table, d_step, nullptr, out, 1, V, H);
}

void embed_lookup_f16_batch(const __half* table, const int32_t* d_token_ids, __half* out,
                            int M, int V, int H, hipStream_t stream) {
    dim3 grid(div_ceil(div_ceil(H, 8), 256), M);
    hipLaunchKernelGGL(embed_lookup_f16_kernel, grid, dim3(256), 0, stream,
                       table, nullptr, d_token_ids, out, M, V, H);
}

void embed_lookup_r4_step(const void* W_qs, const __half* W_scales, const StepParams* d_step,
                          __half* out, int V, int H, hipStream_t stream) {
    const int k_tiles = H >> 6;
    dim3 grid(div_ceil(k_tiles, 256), 1);
    hipLaunchKernelGGL(embed_lookup_r4_kernel, grid, dim3(256), 0, stream,
                       static_cast<const uint4*>(W_qs), W_scales, d_step, nullptr, out, 1, V, H);
}

void embed_lookup_r4_batch(const void* W_qs, const __half* W_scales, const int32_t* d_token_ids,
                           __half* out, int M, int V, int H, hipStream_t stream) {
    const int k_tiles = H >> 6;
    dim3 grid(div_ceil(k_tiles, 256), M);
    hipLaunchKernelGGL(embed_lookup_r4_kernel, grid, dim3(256), 0, stream,
                       static_cast<const uint4*>(W_qs), W_scales, nullptr, d_token_ids, out, M, V, H);
}

void gemm_f16(const __half* A, const __half* W, __half* C,
              int M, int N, int K, hipStream_t stream) {
    if (M <= 16 && N <= 512 && K >= 256 && (K & 7) == 0) {
        if (M == 1) {
            hipLaunchKernelGGL((gemv_f16_wg_kernel<8, 1>), dim3(N), dim3(256), 0, stream, A, W, C, M, N, K);
        } else if (M <= 4) {
            hipLaunchKernelGGL((gemv_f16_wg_kernel<8, 4>), dim3(N), dim3(256), 0, stream, A, W, C, M, N, K);
        } else if (M <= 8) {
            hipLaunchKernelGGL((gemv_f16_wg_kernel<8, 8>), dim3(N), dim3(256), 0, stream, A, W, C, M, N, K);
        } else {
            hipLaunchKernelGGL((gemv_f16_wg_kernel<8, 16>), dim3(N), dim3(256), 0, stream, A, W, C, M, N, K);
        }
        return;
    }
    if (M <= 16) {
        dim3 grid(div_ceil(N, 4));
        dim3 block(128);
        if (M == 1) {
            hipLaunchKernelGGL(gemv_f16_kernel<1>, grid, block, 0, stream, A, W, C, M, N, K);
        } else if (M <= 4) {
            hipLaunchKernelGGL(gemv_f16_kernel<4>, grid, block, 0, stream, A, W, C, M, N, K);
        } else if (M <= 8) {
            hipLaunchKernelGGL(gemv_f16_kernel<8>, grid, block, 0, stream, A, W, C, M, N, K);
        } else {
            hipLaunchKernelGGL(gemv_f16_kernel<16>, grid, block, 0, stream, A, W, C, M, N, K);
        }
    } else {
        dim3 grid(div_ceil(N, 64), div_ceil(M, 64));
        hipLaunchKernelGGL(gemm_f16_wmma_kernel, grid, dim3(128), 0, stream, A, W, C, M, N, K);
    }
}

__global__ __launch_bounds__(256)
void copy_strided_f16_kernel(const __half* __restrict__ src, int src_stride,
                             __half* __restrict__ dst, int dst_stride,
                             int rows, int cols) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((cols & 7) == 0 && (src_stride & 7) == 0 && (dst_stride & 7) == 0) {
        const int vec_cols = cols >> 3;
        const int total = rows * vec_cols;
        if (tid >= total) return;
        const int r = tid / vec_cols;
        const int vc = tid % vec_cols;
        reinterpret_cast<uint4*>(dst + static_cast<size_t>(r) * dst_stride)[vc] =
            reinterpret_cast<const uint4*>(src + static_cast<size_t>(r) * src_stride)[vc];
    } else {
        const int total = rows * cols;
        if (tid >= total) return;
        const int r = tid / cols;
        const int c = tid % cols;
        dst[static_cast<size_t>(r) * dst_stride + c] = src[static_cast<size_t>(r) * src_stride + c];
    }
}

void copy_strided_f16(const __half* src, int src_stride,
                      __half* dst, int dst_stride,
                      int rows, int cols, hipStream_t stream) {
    const int work = ((cols & 7) == 0 && (src_stride & 7) == 0 && (dst_stride & 7) == 0)
                         ? (rows * (cols >> 3))
                         : (rows * cols);
    if (work <= 0) return;
    hipLaunchKernelGGL(copy_strided_f16_kernel, dim3(div_ceil(work, 256)), dim3(256), 0, stream,
                       src, src_stride, dst, dst_stride, rows, cols);
}

size_t argmax_tmp_bytes(int max_M) {
    return align_up(static_cast<size_t>(max_M) * ARGMAX_BLOCKS * (sizeof(float) + sizeof(int32_t)), 256);
}

void argmax_sample(const __half* logits, int V, void* tmp_buf,
                   int32_t* d_out_token, StepParams* d_step_optional, hipStream_t stream) {
    auto* part_val = static_cast<float*>(tmp_buf);
    auto* part_idx = reinterpret_cast<int32_t*>(part_val + ARGMAX_BLOCKS);
    const int blocks = std::min(ARGMAX_BLOCKS, static_cast<int>(div_ceil(V, 256)));
    hipLaunchKernelGGL(argmax_stage1_kernel, dim3(blocks, 1), dim3(256), 0, stream,
                       logits, V, part_val, part_idx);
    hipLaunchKernelGGL(argmax_stage2_kernel, dim3(1, 1), dim3(256), 0, stream,
                       part_val, part_idx, blocks, d_out_token, d_step_optional);
}

void argmax_sample_batch(const __half* logits, int M, int V, void* tmp_buf,
                         int32_t* d_out_tokens, hipStream_t stream) {
    auto* part_val = static_cast<float*>(tmp_buf);
    auto* part_idx = reinterpret_cast<int32_t*>(part_val + static_cast<size_t>(M) * ARGMAX_BLOCKS);
    const int blocks = std::min(ARGMAX_BLOCKS, static_cast<int>(div_ceil(V, 256)));
    hipLaunchKernelGGL(argmax_stage1_kernel, dim3(blocks, M), dim3(256), 0, stream,
                       logits, V, part_val, part_idx);
    hipLaunchKernelGGL(argmax_stage2_kernel, dim3(1, M), dim3(256), 0, stream,
                       part_val, part_idx, blocks, d_out_tokens, nullptr);
}

} // namespace rxinfer::ops

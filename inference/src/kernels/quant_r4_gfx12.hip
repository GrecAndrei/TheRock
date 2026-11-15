// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/quant.hpp"
#include "rxinfer/common.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace rxinfer::quant {
namespace {

using f16x8 = _Float16 __attribute__((ext_vector_type(8)));
using f32x8 = float __attribute__((ext_vector_type(8)));
using i32x2 = int32_t __attribute__((ext_vector_type(2)));
using h2_t = __half2;

inline __device__ float fdot2_h2(h2_t a, h2_t b, float acc) {
    using v2f16 = _Float16 __attribute__((ext_vector_type(2)));
    v2f16 va, vb;
    __builtin_memcpy(&va, &a, sizeof(va));
    __builtin_memcpy(&vb, &b, sizeof(vb));
    return __builtin_amdgcn_fdot2(va, vb, acc, false);
}

// Weight nibble layout in uint32_t p (two's complement biased +8, u in 0..15 -> q = u - 8 in -8..+7):
//   bits  0..3 : u0,   bits 16..19: u1
//   bits  4..7 : u2,   bits 20..23: u3
//   bits  8..11: u4,   bits 24..27: u5
//   bits 12..15: u6,   bits 28..31: u7
// Unpacks 8 weights -> 4x half2 in ONLY 8 packed instructions (zero v_cvt, zero shuffles).
__device__ __forceinline__ void unpack_r4_u32_to_4xh2(
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

// Fast 16-level [-8..+7] -> OCP FP8 E4M3 via dual v_perm_b32 + carry-free byte mask.
// Input p has [u0, u2, u4, u6 | u1, u3, u5, u7].
// First gather [u0, u1, u2, u3] and [u4, u5, u6, u7] into two byte-packed u32s,
// then map each 4-bit u in 0..15 -> exact OCP FP8 E4M3!
__device__ __forceinline__ uint32_t u4_bytes_to_fp8(uint32_t b) {
    // u in 0..7 -> -8..-1: 0xD0, 0xCE, 0xCC, 0xCA, 0xC8, 0xC4, 0xC0, 0xB8
    // u in 8..15 -> 0..+7: 0x00, 0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E
    const uint32_t sel3 = b & 0x07070707u;
    const uint32_t neg_fp8 = __builtin_amdgcn_perm(0xB8C0C4C8u, 0xCACCCED0u, sel3);
    const uint32_t pos_fp8 = __builtin_amdgcn_perm(0x4E4C4A48u, 0x44403800u, sel3);

    // Build byte-wise mask 0xFF where bit 3 is 1, 0x00 where bit 3 is 0 (no borrow across bytes!)
    const uint32_t s = b & 0x08080808u;
    const uint32_t mask = ((s << 4) - (s >> 3)) | (s << 4);
    return (pos_fp8 & mask) | (neg_fp8 & ~mask);
}

__device__ __forceinline__ i32x2 unpack_r4_u32_to_fp8x8(uint32_t p) {
    // p = [u0(0..3), u2(4..7), u4(8..11), u6(12..15) | u1(16..19), u3(20..23), u5(24..27), u7(28..31)]
    // Pack b_lo = [u0, u1, u2, u3] into bytes 0..3:
    //   u0: p & 0xF (byte 0)
    //   u1: (p >> 8) & 0xF00 (byte 1)
    //   u2: (p << 12) & 0xF0000 (byte 2)
    //   u3: (p << 4) & 0xF000000 (byte 3)
    const uint32_t b_lo = (p & 0x0000000Fu) | ((p >> 8) & 0x00000F00u) |
                          ((p << 12) & 0x000F0000u) | ((p << 4) & 0x0F000000u);
    const uint32_t b_hi = ((p >> 8) & 0x0000000Fu) | ((p >> 16) & 0x00000F00u) |
                          ((p << 4) & 0x000F0000u) | ((p >> 4) & 0x0F000000u);

    i32x2 res;
    res[0] = static_cast<int32_t>(u4_bytes_to_fp8(b_lo));
    res[1] = static_cast<int32_t>(u4_bytes_to_fp8(b_hi));
    return res;
}

// =============================================================================
// Custom R4_GFX12_G64 Decode Kernel (M <= 16) — ZERO LDS barriers!
//   Supports SPLIT_K = 1 (single-kernel direct __half output, zero memset/atomics!)
//   and SPLIT_K > 1 (atomicAdd into float buffer + convert).
// =============================================================================
template <int M_MAX, int BLOCK_WARPS, int SPLIT_K>
__global__ __launch_bounds__(BLOCK_WARPS * 32)
void r4_gfx12_decode_fast_kernel(const __half* __restrict__ A,
                                 const uint4* __restrict__ W_qs,
                                 const __half* __restrict__ W_scales,
                                 __half* __restrict__ C_f16,
                                 float* __restrict__ C_f32,
                                 int M, int N, int K) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int n_tiles = (N + 15) >> 4;
    const int k_tiles = K >> 6;

    const int nt = blockIdx.x * BLOCK_WARPS + warp;
    if (nt >= n_tiles) return;

    const int split_id = blockIdx.y;
    const int kt_start = (k_tiles * split_id) / SPLIT_K;
    const int kt_end = (k_tiles * (split_id + 1)) / SPLIT_K;

    const int lane16 = lane & 15;
    const int k_half_off = (lane >= 16) ? 8 : 0;

    float acc[M_MAX];
#pragma unroll
    for (int m = 0; m < M_MAX; ++m) acc[m] = 0.0f;

    for (int kt = kt_start; kt < kt_end; ++kt) {
        const size_t tile_idx = static_cast<size_t>(nt) * k_tiles + kt;
        const uint4 w_pack = W_qs[tile_idx * 32 + lane];
        const float sc = __half2float(W_scales[tile_idx * 16 + lane16]);

        h2_t w_h2[16];
        unpack_r4_u32_to_4xh2(w_pack.x, w_h2[0], w_h2[1], w_h2[2], w_h2[3]);
        unpack_r4_u32_to_4xh2(w_pack.y, w_h2[4], w_h2[5], w_h2[6], w_h2[7]);
        unpack_r4_u32_to_4xh2(w_pack.z, w_h2[8], w_h2[9], w_h2[10], w_h2[11]);
        unpack_r4_u32_to_4xh2(w_pack.w, w_h2[12], w_h2[13], w_h2[14], w_h2[15]);

        const int k_base = (kt << 6) + k_half_off;
#pragma unroll
        for (int m = 0; m < M_MAX; ++m) {
            if (m < M) {
                const h2_t* a_h2 = reinterpret_cast<const h2_t*>(A + static_cast<size_t>(m) * K + k_base);
                float g_acc = 0.0f;
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const h2_t* as = a_h2 + s * 8;
                    g_acc = fdot2_h2(w_h2[s * 4 + 0], as[0], g_acc);
                    g_acc = fdot2_h2(w_h2[s * 4 + 1], as[1], g_acc);
                    g_acc = fdot2_h2(w_h2[s * 4 + 2], as[2], g_acc);
                    g_acc = fdot2_h2(w_h2[s * 4 + 3], as[3], g_acc);
                }
                acc[m] += g_acc * sc;
            }
        }
    }

    const int row = (nt << 4) + lane16;
    if (row < N) {
#pragma unroll
        for (int m = 0; m < M_MAX; ++m) {
            if (m < M) {
                float v = acc[m] + __shfl_xor(acc[m], 16);
                if (lane < 16) {
                    if constexpr (SPLIT_K == 1) {
                        C_f16[static_cast<size_t>(m) * N + row] = __float2half(v);
                    } else {
                        const size_t stride_mn = static_cast<size_t>(M) * N;
                        C_f32[static_cast<size_t>(split_id) * stride_mn + static_cast<size_t>(m) * N + row] = v;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Custom R4_GFX12_G64 Single-Kernel WG-SplitK GEMV (M = 1):
//   - All WG_K_WARPS warps in a workgroup collaborate on 1 N-tile (16 rows) by
//     striping K tiles (`kt = warp; kt < k_tiles; kt += WG_K_WARPS`), maximizing
//     VRAM burst coalescing and eliminating tmp_f32 + reduce_splitk kernel!
//   - Deterministic reduction order across warps (0..WG_K_WARPS-1) in LDS.
// =============================================================================
template <int WG_K_WARPS>
__global__ __launch_bounds__(WG_K_WARPS * 32)
void r4_gfx12_decode_wgsplitk_kernel(const __half* __restrict__ A,
                                     const uint4* __restrict__ W_qs,
                                     const __half* __restrict__ W_scales,
                                     __half* __restrict__ C_f16,
                                     int N, int K) {
    __shared__ float s_acc[WG_K_WARPS][16];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int lane16 = lane & 15;
    const int k_half_off = (lane >= 16) ? 8 : 0;

    const int nt = blockIdx.x;
    const int n_tiles = (N + 15) >> 4;
    if (nt >= n_tiles) return;
    const int k_tiles = K >> 6;

    float acc = 0.0f;
    for (int kt = warp; kt < k_tiles; kt += WG_K_WARPS) {
        const int k_base = (kt << 6) + k_half_off;
        const h2_t* a_h2 = reinterpret_cast<const h2_t*>(A + k_base);
        uint4 av[4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            av[s] = *reinterpret_cast<const uint4*>(a_h2 + s * 8);
        }

        const size_t tile_idx = static_cast<size_t>(nt) * k_tiles + kt;
        const uint4 w_pack = W_qs[tile_idx * 32 + lane];
        const float sc = __half2float(W_scales[tile_idx * 16 + lane16]);

        h2_t w_h2[16];
        unpack_r4_u32_to_4xh2(w_pack.x, w_h2[0], w_h2[1], w_h2[2], w_h2[3]);
        unpack_r4_u32_to_4xh2(w_pack.y, w_h2[4], w_h2[5], w_h2[6], w_h2[7]);
        unpack_r4_u32_to_4xh2(w_pack.z, w_h2[8], w_h2[9], w_h2[10], w_h2[11]);
        unpack_r4_u32_to_4xh2(w_pack.w, w_h2[12], w_h2[13], w_h2[14], w_h2[15]);

        float g_acc = 0.0f;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const h2_t* as = reinterpret_cast<const h2_t*>(&av[s]);
            g_acc = fdot2_h2(w_h2[s * 4 + 0], as[0], g_acc);
            g_acc = fdot2_h2(w_h2[s * 4 + 1], as[1], g_acc);
            g_acc = fdot2_h2(w_h2[s * 4 + 2], as[2], g_acc);
            g_acc = fdot2_h2(w_h2[s * 4 + 3], as[3], g_acc);
        }
        acc += g_acc * sc;
    }

    const float v = acc + __shfl_xor(acc, 16);
    if (lane < 16) s_acc[warp][lane] = v;
    __syncthreads();

    if (warp == 0 && lane < 16) {
        const int row = (nt << 4) + lane;
        if (row < N) {
            float sum = s_acc[0][lane];
#pragma unroll
            for (int w = 1; w < WG_K_WARPS; ++w) sum += s_acc[w][lane];
            C_f16[row] = __float2half(sum);
        }
    }
}

// Fused M=1 Gate+Up+SwiGLU WG-SplitK kernel: shares A loads across gate tile `nt`
// and up tile `nt + i_tiles`, reducing in LDS and fusing silu(gate) * up!
template <int WG_K_WARPS>
__global__ __launch_bounds__(WG_K_WARPS * 32)
void r4_gfx12_decode_swiglu_wgsplitk_kernel(const __half* __restrict__ A,
                                            const uint4* __restrict__ W_qs,
                                            const __half* __restrict__ W_scales,
                                            __half* __restrict__ C_f16,
                                            int I, int K) {
    __shared__ float s_g[WG_K_WARPS][16];
    __shared__ float s_u[WG_K_WARPS][16];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int lane16 = lane & 15;
    const int k_half_off = (lane >= 16) ? 8 : 0;

    const int i_tiles = I >> 4;
    const int k_tiles = K >> 6;
    const int nt = blockIdx.x;
    if (nt >= i_tiles) return;
    const int nt_up = nt + i_tiles;

    float acc_g = 0.0f, acc_u = 0.0f;
    for (int kt = warp; kt < k_tiles; kt += WG_K_WARPS) {
        const int k_base = (kt << 6) + k_half_off;
        const h2_t* a_h2 = reinterpret_cast<const h2_t*>(A + k_base);
        uint4 av[4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            av[s] = *reinterpret_cast<const uint4*>(a_h2 + s * 8);
        }

        const size_t ti_g = static_cast<size_t>(nt) * k_tiles + kt;
        const size_t ti_u = static_cast<size_t>(nt_up) * k_tiles + kt;
        const uint4 wg = W_qs[ti_g * 32 + lane];
        const float scg = __half2float(W_scales[ti_g * 16 + lane16]);
        const uint4 wu = W_qs[ti_u * 32 + lane];
        const float scu = __half2float(W_scales[ti_u * 16 + lane16]);

        h2_t w_h2[16];
        unpack_r4_u32_to_4xh2(wg.x, w_h2[0], w_h2[1], w_h2[2], w_h2[3]);
        unpack_r4_u32_to_4xh2(wg.y, w_h2[4], w_h2[5], w_h2[6], w_h2[7]);
        unpack_r4_u32_to_4xh2(wg.z, w_h2[8], w_h2[9], w_h2[10], w_h2[11]);
        unpack_r4_u32_to_4xh2(wg.w, w_h2[12], w_h2[13], w_h2[14], w_h2[15]);
        float gg = 0.0f;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const h2_t* as = reinterpret_cast<const h2_t*>(&av[s]);
            gg = fdot2_h2(w_h2[s * 4 + 0], as[0], gg);
            gg = fdot2_h2(w_h2[s * 4 + 1], as[1], gg);
            gg = fdot2_h2(w_h2[s * 4 + 2], as[2], gg);
            gg = fdot2_h2(w_h2[s * 4 + 3], as[3], gg);
        }
        acc_g += gg * scg;

        unpack_r4_u32_to_4xh2(wu.x, w_h2[0], w_h2[1], w_h2[2], w_h2[3]);
        unpack_r4_u32_to_4xh2(wu.y, w_h2[4], w_h2[5], w_h2[6], w_h2[7]);
        unpack_r4_u32_to_4xh2(wu.z, w_h2[8], w_h2[9], w_h2[10], w_h2[11]);
        unpack_r4_u32_to_4xh2(wu.w, w_h2[12], w_h2[13], w_h2[14], w_h2[15]);
        float gu = 0.0f;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const h2_t* as = reinterpret_cast<const h2_t*>(&av[s]);
            gu = fdot2_h2(w_h2[s * 4 + 0], as[0], gu);
            gu = fdot2_h2(w_h2[s * 4 + 1], as[1], gu);
            gu = fdot2_h2(w_h2[s * 4 + 2], as[2], gu);
            gu = fdot2_h2(w_h2[s * 4 + 3], as[3], gu);
        }
        acc_u += gu * scu;
    }

    const float vg = acc_g + __shfl_xor(acc_g, 16);
    const float vu = acc_u + __shfl_xor(acc_u, 16);
    if (lane < 16) {
        s_g[warp][lane] = vg;
        s_u[warp][lane] = vu;
    }
    __syncthreads();

    if (warp == 0 && lane < 16) {
        const int row = (nt << 4) + lane;
        if (row < I) {
            float sg = s_g[0][lane];
            float su = s_u[0][lane];
#pragma unroll
            for (int w = 1; w < WG_K_WARPS; ++w) {
                sg += s_g[w][lane];
                su += s_u[w][lane];
            }
            const float silu_g = sg / (1.0f + __expf(-sg));
            C_f16[row] = __float2half(silu_g * su);
        }
    }
}

// =============================================================================
// Custom R4_GFX12_G64 Single-Kernel WG-SplitK WMMA Decode (2 <= M <= 16)
//   - All WG_K_WARPS (8) warps collaborate on 1 N-tile (16 cols) by striping kt.
//   - Loads A[lane16, k_sub] straight via coalesced uint4/f16x8 into WMMA regs
//     (zero LDS in K-loop, zero __syncthreads() in K-loop, zero tmp_f32!).
// =============================================================================
template <int WG_K_WARPS, bool M_LE_8>
__global__ __launch_bounds__(WG_K_WARPS * 32)
void r4_gfx12_decode_wmma_m16_wgsplitk_kernel(const __half* __restrict__ A,
                                                const uint4* __restrict__ W_qs,
                                                const __half* __restrict__ W_scales,
                                                __half* __restrict__ C_f16,
                                                int M, int N, int K) {
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int M_SLOTS = M_LE_8 ? 8 : 16;
    __shared__ float s_out[WG_K_WARPS][M_SLOTS][16];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int lane16 = lane & 15;
    const int k_half_off = (lane >= 16) ? 8 : 0;

    const int nt = blockIdx.x;
    const int n_tiles = (N + 15) >> 4;
    if (nt >= n_tiles) return;
    const int k_tiles = K >> 6;
    const bool valid_a_lane = (lane16 < M);

    f32x8 acc{};
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.0f;

    for (int kt = warp; kt < k_tiles; kt += WG_K_WARPS) {
        const int k_base = kt << 6;
        const size_t tile_idx = static_cast<size_t>(nt) * k_tiles + kt;
        const uint4 w_pack = W_qs[tile_idx * 32 + lane];
        const float sc = __half2float(W_scales[tile_idx * 16 + lane16]);
        const uint32_t* wp = reinterpret_cast<const uint32_t*>(&w_pack);

        f32x8 grp{};
#pragma unroll
        for (int e = 0; e < 8; ++e) grp[e] = 0.0f;

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            f16x8 a_frag{};
            if (valid_a_lane) {
                const int k_sub = k_base + (ks << 4) + k_half_off;
                a_frag = *reinterpret_cast<const f16x8*>(
                    A + static_cast<size_t>(lane16) * K + k_sub);
            }
            f16x8 b_frag;
            h2_t h01, h23, h45, h67;
            unpack_r4_u32_to_4xh2(wp[ks], h01, h23, h45, h67);
            h2_t* dst = reinterpret_cast<h2_t*>(&b_frag);
            dst[0] = h01; dst[1] = h23; dst[2] = h45; dst[3] = h67;

            grp = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, grp);
        }

#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] += grp[e] * sc;
    }

    const int m_half = (lane >= 16) ? 8 : 0;
    if (!M_LE_8 || m_half == 0) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int row = m_half + e;
            if (row < M) s_out[warp][row][lane16] = acc[e];
        }
    }
    __syncthreads();

    const int row_m = tid >> 4;
    const int col16 = tid & 15;
    const int col = (nt << 4) + col16;
    if (row_m < M && col < N) {
        float sum = s_out[0][row_m][col16];
#pragma unroll
        for (int w = 1; w < WG_K_WARPS; ++w) sum += s_out[w][row_m][col16];
        C_f16[static_cast<size_t>(row_m) * N + col] = __float2half(sum);
    }
#endif
}

template <int WG_K_WARPS, bool M_LE_8>
__global__ __launch_bounds__(WG_K_WARPS * 32)
void r4_gfx12_decode_wmma_swiglu_wgsplitk_kernel(const __half* __restrict__ A,
                                                   const uint4* __restrict__ W_qs,
                                                   const __half* __restrict__ W_scales,
                                                   __half* __restrict__ C_f16,
                                                   int M, int I, int K) {
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int M_SLOTS = M_LE_8 ? 8 : 16;
    __shared__ float s_g[WG_K_WARPS][M_SLOTS][16];
    __shared__ float s_u[WG_K_WARPS][M_SLOTS][16];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int lane16 = lane & 15;
    const int k_half_off = (lane >= 16) ? 8 : 0;

    const int i_tiles = I >> 4;
    const int k_tiles = K >> 6;
    const int nt = blockIdx.x;
    if (nt >= i_tiles) return;
    const int nt_up = nt + i_tiles;
    const bool valid_a_lane = (lane16 < M);

    f32x8 acc_g{}, acc_u{};
#pragma unroll
    for (int e = 0; e < 8; ++e) { acc_g[e] = 0.0f; acc_u[e] = 0.0f; }

    for (int kt = warp; kt < k_tiles; kt += WG_K_WARPS) {
        const int k_base = kt << 6;
        const size_t tile_g = static_cast<size_t>(nt) * k_tiles + kt;
        const size_t tile_u = static_cast<size_t>(nt_up) * k_tiles + kt;
        const uint4 w_g = W_qs[tile_g * 32 + lane];
        const uint4 w_u = W_qs[tile_u * 32 + lane];
        const float sc_g = __half2float(W_scales[tile_g * 16 + lane16]);
        const float sc_u = __half2float(W_scales[tile_u * 16 + lane16]);
        const uint32_t* wgp = reinterpret_cast<const uint32_t*>(&w_g);
        const uint32_t* wup = reinterpret_cast<const uint32_t*>(&w_u);

        f32x8 grp_g{}, grp_u{};
#pragma unroll
        for (int e = 0; e < 8; ++e) { grp_g[e] = 0.0f; grp_u[e] = 0.0f; }

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            f16x8 a_frag{};
            if (valid_a_lane) {
                const int k_sub = k_base + (ks << 4) + k_half_off;
                a_frag = *reinterpret_cast<const f16x8*>(
                    A + static_cast<size_t>(lane16) * K + k_sub);
            }
            f16x8 bg, bu;
            h2_t h01, h23, h45, h67;
            unpack_r4_u32_to_4xh2(wgp[ks], h01, h23, h45, h67);
            reinterpret_cast<h2_t*>(&bg)[0] = h01; reinterpret_cast<h2_t*>(&bg)[1] = h23;
            reinterpret_cast<h2_t*>(&bg)[2] = h45; reinterpret_cast<h2_t*>(&bg)[3] = h67;

            unpack_r4_u32_to_4xh2(wup[ks], h01, h23, h45, h67);
            reinterpret_cast<h2_t*>(&bu)[0] = h01; reinterpret_cast<h2_t*>(&bu)[1] = h23;
            reinterpret_cast<h2_t*>(&bu)[2] = h45; reinterpret_cast<h2_t*>(&bu)[3] = h67;

            grp_g = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, bg, grp_g);
            grp_u = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, bu, grp_u);
        }

#pragma unroll
        for (int e = 0; e < 8; ++e) {
            acc_g[e] += grp_g[e] * sc_g;
            acc_u[e] += grp_u[e] * sc_u;
        }
    }

    const int m_half = (lane >= 16) ? 8 : 0;
    if (!M_LE_8 || m_half == 0) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int row = m_half + e;
            if (row < M) {
                s_g[warp][row][lane16] = acc_g[e];
                s_u[warp][row][lane16] = acc_u[e];
            }
        }
    }
    __syncthreads();

    const int row_m = tid >> 4;
    const int col16 = tid & 15;
    const int col = (nt << 4) + col16;
    if (row_m < M && col < I) {
        float sg = s_g[0][row_m][col16];
        float su = s_u[0][row_m][col16];
#pragma unroll
        for (int w = 1; w < WG_K_WARPS; ++w) {
            sg += s_g[w][row_m][col16];
            su += s_u[w][row_m][col16];
        }
        const float silu_g = sg / (1.0f + __expf(-sg));
        C_f16[static_cast<size_t>(row_m) * I + col] = __float2half(silu_g * su);
    }
#endif
}

// =============================================================================
// Custom R4_GFX12_G64 Small-Batch WMMA Decode Kernel (2 <= M <= 16)
//   - Cooperative load A[M, 64] once per block into LDS (shared by 4 warps!)
//   - Unpacks W once per N=16 tile, runs 4 WMMAs covering ALL M<=16 rows!
//   - Zero atomics, zero hipMemsetAsync, deterministic split-K output!
// =============================================================================
template <int BLOCK_WARPS, int SPLIT_K>
__global__ __launch_bounds__(BLOCK_WARPS * 32)
void r4_gfx12_decode_wmma_m16_kernel(const __half* __restrict__ A,
                                     const uint4* __restrict__ W_qs,
                                     const __half* __restrict__ W_scales,
                                     __half* __restrict__ C_f16,
                                     float* __restrict__ C_f32,
                                     int M, int N, int K) {
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int STRIDE = 64 + 8;
    __shared__ _Float16 lds_A[16][STRIDE];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int lane16 = lane & 15;

    const int n_tiles = (N + 15) >> 4;
    const int k_tiles = K >> 6;
    const int nt = blockIdx.x * BLOCK_WARPS + warp;
    const int split_id = blockIdx.y;
    const int kt_start = (k_tiles * split_id) / SPLIT_K;
    const int kt_end = (k_tiles * (split_id + 1)) / SPLIT_K;

    // Zero unused rows in lds_A once before the loop (128 threads cover 16x64 = 1024 halfs).
    {
        const int r = tid >> 3;         // 0..15
        const int c_off = (tid & 7) << 3;
        if (r >= M) {
            *reinterpret_cast<uint4*>(&lds_A[r][c_off]) = make_uint4(0, 0, 0, 0);
        }
    }

    f32x8 acc{};
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.0f;

    const int load_r = tid >> 3;
    const int load_c = (tid & 7) << 3;
    const bool valid_a_row = (load_r < M);
    const bool valid_nt = (nt < n_tiles);

    for (int kt = kt_start; kt < kt_end; ++kt) {
        const int k_base = kt << 6;
        if (valid_a_row) {
            *reinterpret_cast<uint4*>(&lds_A[load_r][load_c]) =
                *reinterpret_cast<const uint4*>(A + static_cast<size_t>(load_r) * K + k_base + load_c);
        }

        uint4 w_pack = make_uint4(0, 0, 0, 0);
        float sc = 0.0f;
        if (valid_nt) {
            const size_t tile_idx = static_cast<size_t>(nt) * k_tiles + kt;
            w_pack = W_qs[tile_idx * 32 + lane];
            sc = __half2float(W_scales[tile_idx * 16 + lane16]);
        }

        __syncthreads();

        f32x8 grp{};
#pragma unroll
        for (int e = 0; e < 8; ++e) grp[e] = 0.0f;

        const uint32_t* wp = reinterpret_cast<const uint32_t*>(&w_pack);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int k_sub = ks * 16 + (lane >= 16 ? 8 : 0);
            const f16x8 a_frag = *reinterpret_cast<const f16x8*>(&lds_A[lane16][k_sub]);

            f16x8 b_frag;
            h2_t h01, h23, h45, h67;
            unpack_r4_u32_to_4xh2(wp[ks], h01, h23, h45, h67);
            h2_t* dst = reinterpret_cast<h2_t*>(&b_frag);
            dst[0] = h01; dst[1] = h23; dst[2] = h45; dst[3] = h67;

            grp = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, grp);
        }

#pragma unroll
        for (int e = 0; e < 8; ++e) {
            acc[e] += grp[e] * sc;
        }
        __syncthreads();
    }

    if (!valid_nt) return;
    const int col = (nt << 4) + lane16;
    if (col >= N) return;

    const int m_half = (lane >= 16) ? 8 : 0;
    if (M <= 8 && m_half != 0) return;

    const size_t stride_mn = static_cast<size_t>(M) * N;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int row = m_half + e;
        if (row < M) {
            if constexpr (SPLIT_K == 1) {
                C_f16[static_cast<size_t>(row) * N + col] = __float2half(acc[e]);
            } else {
                C_f32[static_cast<size_t>(split_id) * stride_mn + static_cast<size_t>(row) * N + col] = acc[e];
            }
        }
    }
#endif
}

// Deterministic fixed-order Split-K reduction + FP32 -> FP16 conversion (vectorized by 4).
// N is always a multiple of 16, so total_elems = M * N is always a multiple of 16 (and 4).
template <int SPLIT_K>
__global__ __launch_bounds__(256)
void reduce_splitk_f32_to_f16_kernel(const float* __restrict__ src, __half* __restrict__ dst, size_t total_elems) {
    const size_t vec_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t idx = vec_idx << 2;
    if (idx + 3 < total_elems) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int s = 0; s < SPLIT_K; ++s) {
            const float4 v = *reinterpret_cast<const float4*>(src + static_cast<size_t>(s) * total_elems + idx);
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }
        uint32_t out_pk[2];
        h2_t h01 = __floats2half2_rn(sum.x, sum.y);
        h2_t h23 = __floats2half2_rn(sum.z, sum.w);
        __builtin_memcpy(&out_pk[0], &h01, 4);
        __builtin_memcpy(&out_pk[1], &h23, 4);
        *reinterpret_cast<uint2*>(dst + idx) = make_uint2(out_pk[0], out_pk[1]);
    } else if (idx < total_elems) {
        for (size_t i = idx; i < total_elems; ++i) {
            float sum = 0.0f;
#pragma unroll
            for (int s = 0; s < SPLIT_K; ++s) {
                sum += src[static_cast<size_t>(s) * total_elems + i];
            }
            dst[i] = __float2half(sum);
        }
    }
}

// =============================================================================
// Custom R4_GFX12_G64 W4A16 Prefill Kernel (M > 16):
//   - BM=64, BN=64, BK=64 (1 full group = 4 WMMA steps!)
//   - ZERO LDS for W! Weights load straight via coalesced uint4 into WMMA registers!
// =============================================================================
__global__ __launch_bounds__(128)
void r4_gfx12_prefill_w4a16_wmma_kernel(const __half* __restrict__ A,
                                        const uint4* __restrict__ W_qs,
                                        const __half* __restrict__ W_scales,
                                        __half* __restrict__ C,
                                        int M, int N, int K) {
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int STRIDE = 64 + 8;

    __shared__ _Float16 lds_A[BM][STRIDE];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;

    const int m_block = blockIdx.y * BM;
    const int n_block = blockIdx.x * BN;
    const int warp_m = (warp >> 1) * 32;
    const int warp_n_tile = (n_block >> 4) + (warp & 1) * 2;

    f32x8 acc[2][2]{};
#pragma unroll
    for (int wm = 0; wm < 2; ++wm)
        for (int wn = 0; wn < 2; ++wn)
            for (int e = 0; e < 8; ++e) acc[wm][wn][e] = 0.0f;

    const int k_tiles = K >> 6;

    for (int kt = 0; kt < k_tiles; ++kt) {
        const int k_base = kt << 6;

#pragma unroll
        for (int pass = 0; pass < 4; ++pass) {
            const int chunk = tid + pass * 128;
            const int r = chunk >> 3;
            const int c_off = (chunk & 7) << 3;
            if (m_block + r < M) {
                *reinterpret_cast<uint4*>(&lds_A[r][c_off]) =
                    *reinterpret_cast<const uint4*>(A + static_cast<size_t>(m_block + r) * K + k_base + c_off);
            } else {
                *reinterpret_cast<uint4*>(&lds_A[r][c_off]) = make_uint4(0, 0, 0, 0);
            }
        }

        uint4 w_pack[2];
        float w_sc[2];
#pragma unroll
        for (int wn = 0; wn < 2; ++wn) {
            const int nt = warp_n_tile + wn;
            if ((nt << 4) < N) {
                const size_t tile_idx = static_cast<size_t>(nt) * k_tiles + kt;
                w_pack[wn] = W_qs[tile_idx * 32 + lane];
                w_sc[wn] = __half2float(W_scales[tile_idx * 16 + lane16]);
            } else {
                w_pack[wn] = make_uint4(0, 0, 0, 0);
                w_sc[wn] = 0.0f;
            }
        }

        __syncthreads();

        f32x8 grp[2][2]{};
#pragma unroll
        for (int wm = 0; wm < 2; ++wm)
            for (int wn = 0; wn < 2; ++wn)
                for (int e = 0; e < 8; ++e) grp[wm][wn][e] = 0.0f;

        const uint32_t* wp0 = reinterpret_cast<const uint32_t*>(&w_pack[0]);
        const uint32_t* wp1 = reinterpret_cast<const uint32_t*>(&w_pack[1]);

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int k_sub = ks * 16 + (lane >= 16 ? 8 : 0);

            f16x8 a_frag[2];
#pragma unroll
            for (int wm = 0; wm < 2; ++wm) {
                a_frag[wm] = *reinterpret_cast<const f16x8*>(&lds_A[warp_m + wm * 16 + lane16][k_sub]);
            }

            f16x8 b_frag[2];
            {
                h2_t h01, h23, h45, h67;
                unpack_r4_u32_to_4xh2(wp0[ks], h01, h23, h45, h67);
                h2_t* dst = reinterpret_cast<h2_t*>(&b_frag[0]);
                dst[0] = h01; dst[1] = h23; dst[2] = h45; dst[3] = h67;
            }
            {
                h2_t h01, h23, h45, h67;
                unpack_r4_u32_to_4xh2(wp1[ks], h01, h23, h45, h67);
                h2_t* dst = reinterpret_cast<h2_t*>(&b_frag[1]);
                dst[0] = h01; dst[1] = h23; dst[2] = h45; dst[3] = h67;
            }

#pragma unroll
            for (int wm = 0; wm < 2; ++wm) {
#pragma unroll
                for (int wn = 0; wn < 2; ++wn) {
                    grp[wm][wn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        a_frag[wm], b_frag[wn], grp[wm][wn]);
                }
            }
        }

#pragma unroll
        for (int wm = 0; wm < 2; ++wm) {
#pragma unroll
            for (int wn = 0; wn < 2; ++wn) {
                const float sc = w_sc[wn];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    acc[wm][wn][e] += grp[wm][wn][e] * sc;
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
            const int col = (warp_n_tile + wn) * 16 + lane16;
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

// =============================================================================
// Pre-quantize activation matrix A[M, K] FP16 -> OCP FP8 E4M3 + group-64 scales
//   One wave32 per 64-elem group (2 FP16/lane -> max-abs warp reduce -> cvt_pk_fp8_f32).
//   In the full engine, this can even be fused into RMSNorm / SwiGLU for zero extra pass!
// =============================================================================
__global__ __launch_bounds__(128)
void quantize_f16_to_fp8_g64_kernel(const __half* __restrict__ A,
                                    uint8_t* __restrict__ A_fp8,
                                    float* __restrict__ A_scales,
                                    int M, int K) {
    const int k_tiles = K >> 6;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int group_idx = blockIdx.x * 4 + warp;
    const int total_groups = M * k_tiles;
    if (group_idx >= total_groups) return;

    const int m = group_idx / k_tiles;
    const int kt = group_idx % k_tiles;
    const int k_off = (kt << 6) + (lane << 1);

    const h2_t pair = *reinterpret_cast<const h2_t*>(A + static_cast<size_t>(m) * K + k_off);
    const float2 fv = __half22float2(pair);
    float amax = fmaxf(fabsf(fv.x), fabsf(fv.y));

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, off));
    }

    const float sc = (amax > 1e-12f) ? (amax / 448.0f) : 1.0f;
    const float inv_sc = (amax > 1e-12f) ? (448.0f / amax) : 0.0f;
    if (lane == 0) {
        A_scales[group_idx] = sc;
    }

    int packed2 = __builtin_amdgcn_cvt_pk_fp8_f32(fv.x * inv_sc, fv.y * inv_sc, 0, false);
    *reinterpret_cast<uint16_t*>(A_fp8 + static_cast<size_t>(m) * K + k_off) =
        static_cast<uint16_t>(packed2 & 0xFFFF);
}

// =============================================================================
// Custom R4_GFX12_G64 Turbo W4A8-FP8 Prefill Kernel (M > 16):
//   - Reads pre-quantized A_fp8[M, K] directly via uint4 into LDS (zero activation quant in loop!)
//   - Unpacks 16-level R4 weights -> OCP FP8 E4M3 via dual v_perm_b32!
//   - Fires wmma_f32_16x16x16_fp8_fp8_w32_gfx12 (8 cycles = 2x FP16 WMMA!)
// =============================================================================
template <int BN>
__global__ __launch_bounds__(128)
void r4_gfx12_prefill_fp8_wmma_kernel(const uint8_t* __restrict__ A_fp8,
                                      const float* __restrict__ A_scales,
                                      const uint4* __restrict__ W_qs,
                                      const __half* __restrict__ W_scales,
                                      __half* __restrict__ C,
                                      int M, int N, int K) {
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BM = 64;
    constexpr int STRIDE_B = 64 + 16;
    constexpr int WN_TILES = BN / 32; // 2 for BN=64, 4 for BN=128

    __shared__ uint8_t lds_A_fp8[BM][STRIDE_B];
    __shared__ float lds_A_sc[BM];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int lane16 = lane & 15;

    const int m_block = blockIdx.y * BM;
    const int n_block = blockIdx.x * BN;
    const int warp_m = (warp >> 1) * 32;
    const int warp_n_tile = (n_block >> 4) + (warp & 1) * WN_TILES;

    f32x8 acc[2][WN_TILES]{};
#pragma unroll
    for (int wm = 0; wm < 2; ++wm)
        for (int wn = 0; wn < WN_TILES; ++wn)
            for (int e = 0; e < 8; ++e) acc[wm][wn][e] = 0.0f;

    const int k_tiles = K >> 6;

    for (int kt = 0; kt < k_tiles; ++kt) {
        const int k_base = kt << 6;

        // 128 threads load 64x64 FP8 bytes (256 uint4 = 2 uint4/thread!)
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int chunk = tid + pass * 128; // 0..255
            const int r = chunk >> 2;           // 0..63
            const int c_off = (chunk & 3) << 4; // 0, 16, 32, 48
            if (m_block + r < M) {
                *reinterpret_cast<uint4*>(&lds_A_fp8[r][c_off]) =
                    *reinterpret_cast<const uint4*>(A_fp8 + static_cast<size_t>(m_block + r) * K + k_base + c_off);
                if (c_off == 0) {
                    lds_A_sc[r] = A_scales[static_cast<size_t>(m_block + r) * k_tiles + kt];
                }
            } else {
                *reinterpret_cast<uint4*>(&lds_A_fp8[r][c_off]) = make_uint4(0, 0, 0, 0);
                if (c_off == 0) lds_A_sc[r] = 0.0f;
            }
        }

        uint4 w_pack[WN_TILES];
        float w_sc[WN_TILES];
#pragma unroll
        for (int wn = 0; wn < WN_TILES; ++wn) {
            const int nt = warp_n_tile + wn;
            if ((nt << 4) < N) {
                const size_t tile_idx = static_cast<size_t>(nt) * k_tiles + kt;
                w_pack[wn] = W_qs[tile_idx * 32 + lane];
                w_sc[wn] = __half2float(W_scales[tile_idx * 16 + lane16]);
            } else {
                w_pack[wn] = make_uint4(0, 0, 0, 0);
                w_sc[wn] = 0.0f;
            }
        }

        __syncthreads();

        f32x8 grp[2][WN_TILES]{};
#pragma unroll
        for (int wm = 0; wm < 2; ++wm)
            for (int wn = 0; wn < WN_TILES; ++wn)
                for (int e = 0; e < 8; ++e) grp[wm][wn][e] = 0.0f;

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int k_sub = ks * 16 + (lane >= 16 ? 8 : 0);

            i32x2 a_frag[2];
#pragma unroll
            for (int wm = 0; wm < 2; ++wm) {
                a_frag[wm] = *reinterpret_cast<const i32x2*>(&lds_A_fp8[warp_m + wm * 16 + lane16][k_sub]);
            }

            i32x2 b_frag[WN_TILES];
#pragma unroll
            for (int wn = 0; wn < WN_TILES; ++wn) {
                const uint32_t* wp = reinterpret_cast<const uint32_t*>(&w_pack[wn]);
                b_frag[wn] = unpack_r4_u32_to_fp8x8(wp[ks]);
            }

#pragma unroll
            for (int wm = 0; wm < 2; ++wm) {
#pragma unroll
                for (int wn = 0; wn < WN_TILES; ++wn) {
                    grp[wm][wn] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        a_frag[wm], b_frag[wn], grp[wm][wn]);
                }
            }
        }

        const int m_half = (lane >= 16) ? 8 : 0;
#pragma unroll
        for (int wm = 0; wm < 2; ++wm) {
            float a_sc_vec[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                a_sc_vec[e] = lds_A_sc[warp_m + wm * 16 + m_half + e];
            }
#pragma unroll
            for (int wn = 0; wn < WN_TILES; ++wn) {
                const float wsc = w_sc[wn];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    acc[wm][wn][e] += grp[wm][wn][e] * (wsc * a_sc_vec[e]);
                }
            }
        }
        __syncthreads();
    }

    const int m_half = (lane >= 16) ? 8 : 0;
#pragma unroll
    for (int wm = 0; wm < 2; ++wm) {
#pragma unroll
        for (int wn = 0; wn < WN_TILES; ++wn) {
            const int col = (warp_n_tile + wn) * 16 + lane16;
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

} // namespace

size_t r4_g64_qs_bytes(int N, int K) {
    const size_t n_tiles = div_ceil(N, 16);
    const size_t k_tiles = div_ceil(K, 64);
    return n_tiles * k_tiles * 32 * sizeof(uint4);
}

size_t r4_g64_scales_bytes(int N, int K) {
    const size_t n_tiles = div_ceil(N, 16);
    const size_t k_tiles = div_ceil(K, 64);
    return n_tiles * k_tiles * 16 * sizeof(__half);
}

size_t r4_g64_tmp_bytes(int max_M, int max_N) {
    return 4ull * static_cast<size_t>(max_M) * static_cast<size_t>(max_N) * sizeof(float);
}

void pack_r4_gfx12_g64(const float* src_f32, void* dst_qs, void* dst_scales, int N, int K) {
    auto* qs = static_cast<uint4*>(dst_qs);
    auto* scales = static_cast<__half*>(dst_scales);

    const int n_tiles = div_ceil(N, 16);
    const int k_tiles = div_ceil(K, 64);

    for (int nt = 0; nt < n_tiles; ++nt) {
        for (int kt = 0; kt < k_tiles; ++kt) {
            const size_t tile_idx = static_cast<size_t>(nt) * k_tiles + kt;
            const int n0 = nt * 16;
            const int k0 = kt * 64;

            float d_row[16];
            float id_row[16];
            for (int nl = 0; nl < 16; ++nl) {
                if (n0 + nl >= N) {
                    d_row[nl] = 0.0f;
                    id_row[nl] = 0.0f;
                    scales[tile_idx * 16 + nl] = __float2half(0.0f);
                    continue;
                }
                const float* x = src_f32 + static_cast<size_t>(n0 + nl) * K + k0;
                const int k_lim = std::min(64, K - k0);
                float amax = 0.0f;
                float vmax = 0.0f;
                for (int i = 0; i < k_lim; ++i) {
                    if (amax < std::fabs(x[i])) {
                        amax = std::fabs(x[i]);
                        vmax = x[i];
                    }
                }
                d_row[nl] = vmax / -8.0f;
                id_row[nl] = (d_row[nl] != 0.0f) ? (1.0f / d_row[nl]) : 0.0f;
                scales[tile_idx * 16 + nl] = __float2half(d_row[nl]);
            }

            for (int lane = 0; lane < 32; ++lane) {
                const int nl = lane & 15;
                const int k_half = (lane >= 16) ? 8 : 0;
                const float id = id_row[nl];
                const bool valid_row = (n0 + nl) < N;
                const float* x = valid_row ? (src_f32 + static_cast<size_t>(n0 + nl) * K + k0) : nullptr;

                uint32_t words[4];
                for (int s = 0; s < 4; ++s) {
                    const int kb = s * 16 + k_half;
                    uint32_t u[8];
                    for (int i = 0; i < 8; ++i) {
                        const float xv = (valid_row && (k0 + kb + i) < K) ? x[kb + i] : 0.0f;
                        u[i] = static_cast<uint32_t>(std::min(15, std::max(0,
                                static_cast<int>(std::trunc(xv * id + 8.5f)))));
                    }
                    words[s] = (u[0]) | (u[2] << 4) | (u[4] << 8) | (u[6] << 12) |
                               (u[1] << 16) | (u[3] << 20) | (u[5] << 24) | (u[7] << 28);
                }
                qs[tile_idx * 32 + lane] = make_uint4(words[0], words[1], words[2], words[3]);
            }
        }
    }
}

void gemm_r4_gfx12_w4a16(const __half* A, const void* W_qs, const __half* W_scales,
                         __half* C, float* tmp_f32,
                         int M, int N, int K, hipStream_t stream) {
    const auto* qs = static_cast<const uint4*>(W_qs);
    if (M <= 16) {
        constexpr int WARPS = 4;
        const int n_tiles = div_ceil(N, 16);
        const size_t out_elems = static_cast<size_t>(M) * N;
        const bool use_split1 = (N >= 16384) || (K < 256);

        if (M == 1) {
            if (N < 65536 && K >= 256 && (K & 63) == 0) {
                hipLaunchKernelGGL((r4_gfx12_decode_wgsplitk_kernel<8>),
                                   dim3(n_tiles), dim3(256), 0, stream,
                                   A, qs, W_scales, C, N, K);
            } else {
                dim3 grid(div_ceil(n_tiles, WARPS), 1);
                dim3 block(WARPS * 32);
                hipLaunchKernelGGL((r4_gfx12_decode_fast_kernel<1, WARPS, 1>), grid, block, 0, stream,
                                   A, qs, W_scales, C, tmp_f32, M, N, K);
            }
            return;
        }

        // 2 <= M <= 16: Wave32 WMMA small-batch decode
        if (N < 65536 && K >= 256 && (K & 63) == 0) {
            if (M <= 8) {
                hipLaunchKernelGGL((r4_gfx12_decode_wmma_m16_wgsplitk_kernel<8, true>),
                                   dim3(n_tiles), dim3(256), 0, stream,
                                   A, qs, W_scales, C, M, N, K);
            } else {
                hipLaunchKernelGGL((r4_gfx12_decode_wmma_m16_wgsplitk_kernel<8, false>),
                                   dim3(n_tiles), dim3(256), 0, stream,
                                   A, qs, W_scales, C, M, N, K);
            }
        } else if (use_split1) {
            dim3 grid(div_ceil(n_tiles, WARPS), 1);
            dim3 block(WARPS * 32);
            hipLaunchKernelGGL((r4_gfx12_decode_wmma_m16_kernel<WARPS, 1>), grid, block, 0, stream,
                               A, qs, W_scales, C, tmp_f32, M, N, K);
        } else {
            constexpr int SPLIT_K = 4;
            dim3 grid(div_ceil(n_tiles, WARPS), SPLIT_K);
            dim3 block(WARPS * 32);
            hipLaunchKernelGGL((r4_gfx12_decode_wmma_m16_kernel<WARPS, SPLIT_K>), grid, block, 0, stream,
                               A, qs, W_scales, C, tmp_f32, M, N, K);
            hipLaunchKernelGGL((reduce_splitk_f32_to_f16_kernel<SPLIT_K>),
                               dim3(div_ceil(div_ceil(out_elems, 4), 256)), dim3(256), 0, stream,
                               tmp_f32, C, out_elems);
        }
    } else {
        dim3 grid(div_ceil(N, 64), div_ceil(M, 64));
        dim3 block(128);
        hipLaunchKernelGGL(r4_gfx12_prefill_w4a16_wmma_kernel, grid, block, 0, stream,
                           A, qs, W_scales, C, M, N, K);
    }
}

void gemv_r4_gfx12_w4a16_swiglu(const __half* A, const void* W_qs, const __half* W_scales,
                                __half* C, int I, int K, hipStream_t stream, int M) {
    const auto* qs = static_cast<const uint4*>(W_qs);
    if (M == 1) {
        hipLaunchKernelGGL((r4_gfx12_decode_swiglu_wgsplitk_kernel<8>),
                           dim3(I >> 4), dim3(256), 0, stream,
                           A, qs, W_scales, C, I, K);
    } else if (M <= 8) {
        hipLaunchKernelGGL((r4_gfx12_decode_wmma_swiglu_wgsplitk_kernel<8, true>),
                           dim3(I >> 4), dim3(256), 0, stream,
                           A, qs, W_scales, C, M, I, K);
    } else {
        hipLaunchKernelGGL((r4_gfx12_decode_wmma_swiglu_wgsplitk_kernel<8, false>),
                           dim3(I >> 4), dim3(256), 0, stream,
                           A, qs, W_scales, C, M, I, K);
    }
}

void gemm_r4_gfx12_fp8(const __half* A, const void* W_qs, const __half* W_scales,
                       __half* C, float* tmp_f32,
                       uint8_t* tmp_a_fp8, float* tmp_a_sc,
                       int M, int N, int K, hipStream_t stream) {
    if (M <= 16) {
        gemm_r4_gfx12_w4a16(A, W_qs, W_scales, C, tmp_f32, M, N, K, stream);
        return;
    }
    const auto* qs = static_cast<const uint4*>(W_qs);
    const int k_tiles = K >> 6;
    const int total_groups = M * k_tiles;
    hipLaunchKernelGGL(quantize_f16_to_fp8_g64_kernel,
                       dim3(div_ceil(total_groups, 4)), dim3(128), 0, stream,
                       A, tmp_a_fp8, tmp_a_sc, M, K);

    if (M < 256) {
        dim3 grid(div_ceil(N, 64), div_ceil(M, 64));
        hipLaunchKernelGGL((r4_gfx12_prefill_fp8_wmma_kernel<64>), grid, dim3(128), 0, stream,
                           tmp_a_fp8, tmp_a_sc, qs, W_scales, C, M, N, K);
    } else {
        dim3 grid(div_ceil(N, 128), div_ceil(M, 64));
        hipLaunchKernelGGL((r4_gfx12_prefill_fp8_wmma_kernel<128>), grid, dim3(128), 0, stream,
                           tmp_a_fp8, tmp_a_sc, qs, W_scales, C, M, N, K);
    }
}

} // namespace rxinfer::quant

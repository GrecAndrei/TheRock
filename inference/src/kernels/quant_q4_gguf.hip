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
using h2_t = __half2;

inline uint16_t float_to_fp16_bits(float v) {
    __half h = __float2half(v);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

inline __device__ float fdot2_h2(h2_t a, h2_t b, float acc) {
    using v2f16 = _Float16 __attribute__((ext_vector_type(2)));
    v2f16 va, vb;
    __builtin_memcpy(&va, &a, sizeof(va));
    __builtin_memcpy(&vb, &b, sizeof(vb));
    return __builtin_amdgcn_fdot2(va, vb, acc, false);
}

// Unpack 16 weights from uint32 (low/high nibble pairs? No, in Q4_0 qs[0..15]:
// byte j has low nibble = elem j (0..15), high nibble = elem j+16 (16..31)).
// Load 4 bytes = qs[0..3] (elems 0..3 in low nibbles, elems 16..19 in high nibbles).
__device__ __forceinline__ void unpack_4bytes_q4_0(
    uint32_t q4_bytes, h2_t& lo01, h2_t& lo23, h2_t& hi01, h2_t& hi23) {
    // Low nibbles: bytes 0,1 -> (q4_bytes & 0x0F) and ((q4_bytes >> 8) & 0x0F)
    const uint32_t bias = 0x64006400u;
    const h2_t neg1032 = __halves2half2(__float2half(-1032.0f), __float2half(-1032.0f));

    uint32_t p_lo01 = ((q4_bytes & 0x0000000Fu) | ((q4_bytes & 0x00000F00u) << 8)) | bias;
    uint32_t p_lo23 = (((q4_bytes >> 16) & 0x0000000Fu) | ((q4_bytes >> 8) & 0x000F0000u)) | bias;
    uint32_t p_hi01 = (((q4_bytes >> 4) & 0x0000000Fu) | ((q4_bytes & 0x0000F000u) << 4)) | bias;
    uint32_t p_hi23 = (((q4_bytes >> 20) & 0x0000000Fu) | ((q4_bytes >> 12) & 0x000F0000u)) | bias;

    h2_t t0, t1, t2, t3;
    __builtin_memcpy(&t0, &p_lo01, 4);
    __builtin_memcpy(&t1, &p_lo23, 4);
    __builtin_memcpy(&t2, &p_hi01, 4);
    __builtin_memcpy(&t3, &p_hi23, 4);
    lo01 = __hadd2(t0, neg1032);
    lo23 = __hadd2(t1, neg1032);
    hi01 = __hadd2(t2, neg1032);
    hi23 = __hadd2(t3, neg1032);
}

// Optimized GGUF Q4_0 Decode Kernel (M <= 16):
// Each Wave32 handles 1 output row per k-block stride? Wait: better!
// 1 Wave32 = 32 lanes; each lane processes 1 Q4_0 block (32 weights) per step -> 1024 K-weights/step!
// With blockDim = 128 (4 waves -> 4 rows/block or split-K across waves).
// Split-K across gridDim.y lets small-N matrices saturate all 32 WGPs.
template <int M_MAX, int BLOCK_WARPS, int SPLIT_K>
__global__ __launch_bounds__(BLOCK_WARPS * 32)
void gguf_q4_0_decode_kernel(const __half* __restrict__ A,
                             const BlockQ4_0* __restrict__ W,
                             float* __restrict__ C_f32,
                             int M, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * BLOCK_WARPS + warp;
    if (row >= N) return;

    const int blocks_per_row = K >> 5; // K / 32
    const int split_id = blockIdx.y;
    const int blk_start = (blocks_per_row * split_id) / SPLIT_K;
    const int blk_end = (blocks_per_row * (split_id + 1)) / SPLIT_K;

    const BlockQ4_0* w_row = W + static_cast<size_t>(row) * blocks_per_row;
    float acc[M_MAX];
#pragma unroll
    for (int m = 0; m < M_MAX; ++m) acc[m] = 0.0f;

    for (int b = blk_start + lane; b < blk_end; b += 32) {
        const BlockQ4_0 blk = w_row[b];
        __half d_h;
        __builtin_memcpy(&d_h, &blk.d, sizeof(d_h));
        const float d = __half2float(d_h);

        uint32_t q[4];
        __builtin_memcpy(q, blk.qs, 16);

        const int k_base = b << 5;
        h2_t wl[8], wh[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unpack_4bytes_q4_0(q[j], wl[j * 2 + 0], wl[j * 2 + 1], wh[j * 2 + 0], wh[j * 2 + 1]);
        }

#pragma unroll
        for (int m = 0; m < M_MAX; ++m) {
            if (m < M) {
                const h2_t* a_ptr = reinterpret_cast<const h2_t*>(A + static_cast<size_t>(m) * K + k_base);
                float g_acc = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    g_acc = fdot2_h2(wl[j], a_ptr[j], g_acc);
                    g_acc = fdot2_h2(wh[j], a_ptr[j + 8], g_acc);
                }
                acc[m] += g_acc * d;
            }
        }
    }

#pragma unroll
    for (int m = 0; m < M_MAX; ++m) {
        if (m < M) {
            float v = acc[m];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down(v, off);
            }
            if (lane == 0) {
                if constexpr (SPLIT_K == 1) {
                    C_f32[static_cast<size_t>(m) * N + row] = v;
                } else {
                    atomicAdd(&C_f32[static_cast<size_t>(m) * N + row], v);
                }
            }
        }
    }
}

__global__ void f32_to_f16_kernel(const float* __restrict__ src, __half* __restrict__ dst, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

// Optimized GGUF Q4_0 Prefill Kernel (M > 16):
// Tile: BM = 64, BN = 64, BK = 32 (one Q4_0 block along K).
// 4 warps/block (2x2 warps over 32x32? Wait: 4 warps cover 2x2 tiles of 32x32 = 4 WMMAs per warp per step,
// or 4 warps = 64x64 with 2x2 WMMA tiles per warp -> 32x32 per warp!).
// LDS holds A_tile[64][36] FP16 and dequantized W_tile[64][36] FP16 (zero bank conflicts!).
__global__ __launch_bounds__(128)
void gguf_q4_0_prefill_wmma_kernel(const __half* __restrict__ A,
                                   const BlockQ4_0* __restrict__ W,
                                   __half* __restrict__ C,
                                   int M, int N, int K) {
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int STRIDE = 32 + 8; // 40 FP16 = 80 bytes -> 16B aligned, bank-conflict free!

    __shared__ _Float16 lds_A[BM][STRIDE];
    __shared__ _Float16 lds_W[BN][STRIDE];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int m_block = blockIdx.y * BM;
    const int n_block = blockIdx.x * BN;

    // 4 warps arranged 2x2, each warp owns a 32x32 output tile (2x2 of 16x16 WMMAs)
    const int warp_m = (warp >> 1) * 32;
    const int warp_n = (warp & 1) * 32;

    f32x8 acc[2][2]{};
#pragma unroll
    for (int wm = 0; wm < 2; ++wm)
        for (int wn = 0; wn < 2; ++wn)
            for (int e = 0; e < 8; ++e) acc[wm][wn][e] = 0.0f;

    const int blocks_per_row = K >> 5;
    const int num_k_tiles = K / BK;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k_base = kt * BK;

        // 1) Load A[BM=64, BK=32]: 128 threads -> 1 thread per half-row (16 FP16 = 32 bytes), or 2 threads/row!
        // 64 rows * 2 chunks of 16 FP16 = 128 chunks -> exact 1 chunk per thread!
        {
            const int r = tid >> 1;          // 0..63
            const int c_off = (tid & 1) * 16; // 0 or 16
            if (m_block + r < M) {
                const uint4* src = reinterpret_cast<const uint4*>(A + static_cast<size_t>(m_block + r) * K + k_base + c_off);
                uint4* dst = reinterpret_cast<uint4*>(&lds_A[r][c_off]);
                dst[0] = src[0];
                dst[1] = src[1];
            } else {
                uint4* dst = reinterpret_cast<uint4*>(&lds_A[r][c_off]);
                dst[0] = make_uint4(0, 0, 0, 0);
                dst[1] = make_uint4(0, 0, 0, 0);
            }
        }

        // 2) Load + dequantize W[BN=64, BK=32] (64 blocks of Q4_0): 2 threads per block -> 128 threads!
        {
            const int r = tid >> 1;       // 0..63
            const int half_blk = tid & 1; // 0: bytes 0..7 (elems 0..7, 16..23), 1: bytes 8..15 (elems 8..15, 24..31)
            if (n_block + r < N) {
                const BlockQ4_0* blk_ptr = W + static_cast<size_t>(n_block + r) * blocks_per_row + kt;
                __half d_h;
                __builtin_memcpy(&d_h, &blk_ptr->d, sizeof(d_h));
                const __half d_half = d_h;
                const h2_t d_h2 = __half2half2(d_half);

                uint32_t q[2];
                __builtin_memcpy(q, blk_ptr->qs + half_blk * 8, 8);

                h2_t wl0, wl1, wh0, wh1, wl2, wl3, wh2, wh3;
                unpack_4bytes_q4_0(q[0], wl0, wl1, wh0, wh1);
                unpack_4bytes_q4_0(q[1], wl2, wl3, wh2, wh3);

                wl0 = __hmul2(wl0, d_h2); wl1 = __hmul2(wl1, d_h2);
                wl2 = __hmul2(wl2, d_h2); wl3 = __hmul2(wl3, d_h2);
                wh0 = __hmul2(wh0, d_h2); wh1 = __hmul2(wh1, d_h2);
                wh2 = __hmul2(wh2, d_h2); wh3 = __hmul2(wh3, d_h2);

                h2_t* dst_lo = reinterpret_cast<h2_t*>(&lds_W[r][half_blk * 8]);
                h2_t* dst_hi = reinterpret_cast<h2_t*>(&lds_W[r][16 + half_blk * 8]);
                dst_lo[0] = wl0; dst_lo[1] = wl1; dst_lo[2] = wl2; dst_lo[3] = wl3;
                dst_hi[0] = wh0; dst_hi[1] = wh1; dst_hi[2] = wh2; dst_hi[3] = wh3;
            } else {
                uint4* dst = reinterpret_cast<uint4*>(&lds_W[r][half_blk * 16]);
                dst[0] = make_uint4(0, 0, 0, 0);
                dst[1] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // 3) 2 WMMA K-steps of 16 across BK=32
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const int k_sub = ks * 16 + (lane >= 16 ? 8 : 0);
            const int lane16 = lane & 15;

            f16x8 a_frag[2];
            f16x8 b_frag[2];
#pragma unroll
            for (int wm = 0; wm < 2; ++wm) {
                a_frag[wm] = *reinterpret_cast<const f16x8*>(&lds_A[warp_m + wm * 16 + lane16][k_sub]);
            }
#pragma unroll
            for (int wn = 0; wn < 2; ++wn) {
                b_frag[wn] = *reinterpret_cast<const f16x8*>(&lds_W[warp_n + wn * 16 + lane16][k_sub]);
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

    // Store C[M, N] using verified GFX12 Wave32 output layout:
    // m_local = (lane >= 16 ? 8 : 0) + elem, n_local = lane & 15
    const int n_local = lane & 15;
    const int m_half = (lane >= 16) ? 8 : 0;
#pragma unroll
    for (int wm = 0; wm < 2; ++wm) {
#pragma unroll
        for (int wn = 0; wn < 2; ++wn) {
            const int col = n_block + warp_n + wn * 16 + n_local;
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

void pack_gguf_q4_0(const float* src_f32, void* dst_blocks, int N, int K) {
    auto* blocks = static_cast<BlockQ4_0*>(dst_blocks);
    const int blocks_per_row = K / 32;
    for (int n = 0; n < N; ++n) {
        const float* row = src_f32 + static_cast<size_t>(n) * K;
        for (int b = 0; b < blocks_per_row; ++b) {
            const float* x = row + b * 32;
            float amax = 0.0f;
            float vmax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                const float v = x[i];
                if (amax < std::fabs(v)) {
                    amax = std::fabs(v);
                    vmax = v;
                }
            }
            const float d = vmax / -8.0f;
            const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

            BlockQ4_0& out = blocks[static_cast<size_t>(n) * blocks_per_row + b];
            out.d = float_to_fp16_bits(d);
            for (int i = 0; i < 16; ++i) {
                float x0 = x[i] * id;
                float x1 = x[i + 16] * id;
                int xi0 = std::min(15, std::max(0, static_cast<int>(std::trunc(x0 + 8.5f))));
                int xi1 = std::min(15, std::max(0, static_cast<int>(std::trunc(x1 + 8.5f))));
                out.qs[i] = static_cast<uint8_t>(xi0 | (xi1 << 4));
            }
        }
    }
}

void gemm_gguf_q4_0(const __half* A, const void* W_blocks, __half* C, float* tmp_f32,
                    int M, int N, int K, hipStream_t stream) {
    const auto* W = static_cast<const BlockQ4_0*>(W_blocks);
    if (M <= 16) {
        constexpr int WARPS = 4;
        constexpr int SPLIT_K = 4;
        const size_t out_elems = static_cast<size_t>(M) * N;
        RX_HIP_CHECK(hipMemsetAsync(tmp_f32, 0, out_elems * sizeof(float), stream));

        dim3 grid(div_ceil(N, WARPS), SPLIT_K);
        dim3 block(WARPS * 32);
        if (M == 1) {
            hipLaunchKernelGGL((gguf_q4_0_decode_kernel<1, WARPS, SPLIT_K>), grid, block, 0, stream,
                               A, W, tmp_f32, M, N, K);
        } else if (M <= 4) {
            hipLaunchKernelGGL((gguf_q4_0_decode_kernel<4, WARPS, SPLIT_K>), grid, block, 0, stream,
                               A, W, tmp_f32, M, N, K);
        } else if (M <= 8) {
            hipLaunchKernelGGL((gguf_q4_0_decode_kernel<8, WARPS, SPLIT_K>), grid, block, 0, stream,
                               A, W, tmp_f32, M, N, K);
        } else {
            hipLaunchKernelGGL((gguf_q4_0_decode_kernel<16, WARPS, SPLIT_K>), grid, block, 0, stream,
                               A, W, tmp_f32, M, N, K);
        }
        hipLaunchKernelGGL(f32_to_f16_kernel, dim3(div_ceil(out_elems, 256)), dim3(256), 0, stream,
                           tmp_f32, C, out_elems);
    } else {
        dim3 grid(div_ceil(N, 64), div_ceil(M, 64));
        dim3 block(128);
        hipLaunchKernelGGL(gguf_q4_0_prefill_wmma_kernel, grid, block, 0, stream,
                           A, W, C, M, N, K);
    }
}

} // namespace rxinfer::quant

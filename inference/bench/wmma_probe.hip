// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/common.hpp"
#include "rxinfer/config.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using f16x8 = _Float16 __attribute__((ext_vector_type(8)));
using f32x8 = float __attribute__((ext_vector_type(8)));
using i32x2 = int32_t __attribute__((ext_vector_type(2)));
using i32x8 = int32_t __attribute__((ext_vector_type(8)));

// Probe 1: Determine exact GFX12 Wave32 lane->(row, col) mapping for
// __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12.
// Method: set A to unique row tags (lane 0..31, elem 0..7) or pass an identity-like pattern,
// and run 3 probe variants:
//   mode 0: A has single 1.0 at (target_lane, target_elem), B = all 1.0  -> D shows which M-rows are touched!
//   mode 1: A = all 1.0, B has single 1.0 at (target_lane, target_elem)  -> D shows which N-cols are touched!
//   mode 2: A has lane*8+elem+1, B = identity? Let's directly probe all 32x8 positions cleanly in 1 launch!
__global__ void probe_wmma_f16_layout(float* out_row_map, float* out_col_map, float* out_c_map) {
#if defined(__gfx1201__) || defined(__gfx1200__)
    const int lane = threadIdx.x & 31;

    // 1) Probe A[lane, elem]: A = 1 only at (probe_lane, probe_elem), B = 1 everywhere
    // Store summed output per (C_lane, C_elem) isn't needed; instead set A[lane, elem] = 1 at one bit,
    // wait - run a loop over (p_lane, p_elem)? Easier: set A[lane, elem] = (lane == blockIdx.x && elem == blockIdx.y) ? 1.0 : 0.0
    // Even simpler in 1 kernel with 256 blocks! blockIdx.x in [0..31], blockIdx.y in [0..7].
    if (blockIdx.z == 0) {
        f16x8 a{};
        f16x8 b{};
        f32x8 c{};
        for (int i = 0; i < 8; ++i) {
            a[i] = (lane == static_cast<int>(blockIdx.x) && i == static_cast<int>(blockIdx.y))
                       ? static_cast<_Float16>(1.0f)
                       : static_cast<_Float16>(0.0f);
            b[i] = static_cast<_Float16>(1.0f);
            c[i] = 0.0f;
        }
        f32x8 d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
        // If A has 1.0 at (row_m, k_idx) and B is all 1.0, then D has 1.0 across entire row_m!
        //Wait: check WHICH C[lane, elem] got 1.0! Store first (lane, elem) or sum per lane.
        // Actually we want BOTH: which C[lane, elem] are nonzero for a given A(p_lane, p_elem),
        // AND for a given B(p_lane, p_elem).
        // Let's write C[lane, i] for p_lane=0, p_elem=0..7 into out_c_map!
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            for (int i = 0; i < 8; ++i) {
                out_row_map[lane * 8 + i] = d[i];
            }
        }
        // Encode unique signature in out_col_map: for each (p_lane, p_elem), record C-index mask or first nonzero C lane/elem
        int hits = 0;
        int first_c_lane = -1;
        for (int i = 0; i < 8; ++i) {
            if (d[i] != 0.0f) {
                hits++;
                if (first_c_lane < 0) first_c_lane = lane * 8 + i;
            }
        }
        // Reduce across wave to find total nonzero C elements and min/max C lane
        unsigned mask = __ballot(hits > 0);
        if (lane == 0) {
            out_col_map[blockIdx.x * 8 + blockIdx.y] = static_cast<float>(mask);
        }
        // Also record which C indices are hit for (lane=0, elem=0..7) and (lane=16, elem=0..7)
        if ((blockIdx.x == 0 || blockIdx.x == 16) && blockIdx.y < 8) {
            const int slot = (blockIdx.x == 0 ? 0 : 8) + static_cast<int>(blockIdx.y);
            for (int i = 0; i < 8; ++i) {
                out_c_map[slot * 256 + lane * 8 + i] = d[i];
            }
        }
    }
#endif
}

// Probe 2: Direct identity + rank-1 verification on A, B, C, and FP8 test.
__global__ void probe_wmma_patterns(float* out_d_identity, float* out_d_fp8_ocp, float* out_d_fp8_fnuz) {
#if defined(__gfx1201__) || defined(__gfx1200__)
    const int lane = threadIdx.x & 31;

    // Test 1: Check assumed GFX12 Wave32 layout:
    // Hypothesis for 16x16x16 Wave32 on GFX12:
    //   C/D: 16x16 = 256 elems across 32 lanes -> 8 elems/lane.
    //   Let's also test A, B, C with exact coordinate encoding:
    //   First, encode C(row, col) = row * 100 + col by running A = 0, B = 0, C[i] = lane * 8 + i, wait no that just returns C.
    //   To find (row, col) of A[lane, i], B[lane, i], C[lane, i]:
    //   Run B = all 1s -> D[row, col] = sum_k A[row, k].
    //   Wait: if we instead set B[lane, i] = (lane < 16 ? 1 : 0)? No, let's test the hypothesis directly AND log!
    // Hypothesis:
    //   On GFX12 Wave32:
    //   A (16x16, M x K): lane 0..15 hold rows 0..15, k = 0..7 in elem 0..7? OR lane 0..15 hold k=0..7 and lane 16..31 hold k=8..15?
    //   Let's check BOTH by testing:
    //   Case H1: set A[lane, i] = (lane < 16 ? 1.0 : 0.0), B = all 1.0 -> if lane 0..15 hold k=0..7 and lane 16..31 hold k=8..15, D will be 8.0 everywhere!
    //   If lane 0..15 hold rows 0..15 (full k=0..15?) impossible since 8 elems < 16. So A MUST split K=16 across 2 groups of 16 lanes (8 elems each) OR split M=16 across 2 groups!
    //   With this single test (A = (lane < 16) ? 1 : 0, B = all 1):
    //     - If D is 8.0 across all 256 entries -> lane 0..15 hold ONE K-half (8 cols of K) and lane 16..31 hold the OTHER K-half!
    //     - If D is 16.0 in half of rows and 0.0 in other half -> lanes split rows!
    f16x8 a{}, b{};
    f32x8 c{};
    for (int i = 0; i < 8; ++i) {
        a[i] = static_cast<_Float16>(lane < 16 ? 1.0f : 0.0f);
        b[i] = static_cast<_Float16>(1.0f);
        c[i] = 0.0f;
    }
    f32x8 d1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    for (int i = 0; i < 8; ++i) {
        out_d_identity[lane * 8 + i] = d1[i];
    }

    // Test 2: Check FP8 encoding on gfx1201!
    // Fill all A with 1.0 and all B with 2.0 in OCP E4M3 (1.0 = 0x38, 2.0 = 0x40) -> dot product over K=16 should be 32.0!
    // In FNUZ E4M3 (bias=8), 0x38 = 0.5, 0x40 = 1.0 -> dot product over K=16 would be 8.0!
    {
        i32x2 a_ocp, b_ocp;
        a_ocp[0] = 0x38383838; a_ocp[1] = 0x38383838; // 8 x OCP E4M3 1.0
        b_ocp[0] = 0x40404040; b_ocp[1] = 0x40404040; // 8 x OCP E4M3 2.0
        f32x8 c_zero{};
        f32x8 d_ocp = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_ocp, b_ocp, c_zero);
        for (int i = 0; i < 8; ++i) out_d_fp8_ocp[lane * 8 + i] = d_ocp[i];
    }
    // Also test negative value -3.0 in OCP E4M3: sign=1, exp=7+1=8 (1000), man=100 (1.5) -> 0b1_1000_100 = 0xC4
    {
        i32x2 a_neg, b_one;
        a_neg[0] = static_cast<int32_t>(0xC4C4C4C4u);
        a_neg[1] = static_cast<int32_t>(0xC4C4C4C4u); // -3.0
        b_one[0] = 0x38383838;
        b_one[1] = 0x38383838; // +1.0
        f32x8 c_zero{};
        f32x8 d_neg = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_neg, b_one, c_zero);
        for (int i = 0; i < 8; ++i) out_d_fp8_fnuz[lane * 8 + i] = d_neg[i];
    }
#endif
}

// Probe 3: Probe exact (row, k) for A and (k, col) for B and (row, col) for C!
// For row r in 0..15, we put A[r, k] = (k == 0 ? (r + 1) : 0) — wait we don't know A's mapping yet!
// Instead: put a unique 1-hot bit per (p_lane, p_elem) across A, and read out which C-entries change!
// Even simpler:
//   Kernel A_probe: blockIdx.x = target_A_lane (0..31). Set A[target_A_lane, i] = pow(2.0, i) for i=0..7, B = 1.0.
//   Since all elements of B are 1.0, D[m, n] = sum_k A[m, k] = bitmask of which A elems hit row m!
//   Kernel B_probe: blockIdx.x = target_B_lane (0..31). Set B[target_B_lane, i] = pow(2.0, i) for i=0..7, A = 1.0.
//   Since all elements of A are 1.0, D[m, n] = sum_k B[k, n] = bitmask of which B elems hit col n!
//   Kernel K_probe: set A[lane, i] = (i == 0 ? 1 : 0) etc.? Even better:
//   Once we know which rows of D correspond to A's (lane, i), we test K matching by setting
//   A[0, 0] = 1.0 and B[b_lane, b_elem] = 1.0 -> D is nonzero ONLY if A[0, 0] and B[b_lane, b_elem] share the SAME K!
__global__ void probe_ab_mapping(float* a_to_c_mask, float* b_to_c_mask, int* k_match_b_for_a0) {
#if defined(__gfx1201__) || defined(__gfx1200__)
    const int lane = threadIdx.x & 31;
    const int p_lane = blockIdx.x; // 0..31

    // 1) A[p_lane, i] = 2^i, B = all 1
    {
        f16x8 a{}, b{};
        f32x8 c{};
        for (int i = 0; i < 8; ++i) {
            a[i] = (lane == p_lane) ? static_cast<_Float16>(static_cast<float>(1 << i))
                                    : static_cast<_Float16>(0.0f);
            b[i] = static_cast<_Float16>(1.0f);
            c[i] = 0.0f;
        }
        f32x8 d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
        for (int i = 0; i < 8; ++i) {
            a_to_c_mask[p_lane * 256 + lane * 8 + i] = d[i];
        }
    }

    // 2) B[p_lane, i] = 2^i, A = all 1
    {
        f16x8 a{}, b{};
        f32x8 c{};
        for (int i = 0; i < 8; ++i) {
            a[i] = static_cast<_Float16>(1.0f);
            b[i] = (lane == p_lane) ? static_cast<_Float16>(static_cast<float>(1 << i))
                                    : static_cast<_Float16>(0.0f);
            c[i] = 0.0f;
        }
        f32x8 d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
        for (int i = 0; i < 8; ++i) {
            b_to_c_mask[p_lane * 256 + lane * 8 + i] = d[i];
        }
    }

    // 3) K-match: A[0, 0] = 1.0 only, B[p_lane, i] = 2^i -> nonzero bits tell WHICH B(p_lane, i) share K with A(0, 0)!
    {
        f16x8 a{}, b{};
        f32x8 c{};
        for (int i = 0; i < 8; ++i) {
            a[i] = (lane == 0 && i == 0) ? static_cast<_Float16>(1.0f) : static_cast<_Float16>(0.0f);
            b[i] = (lane == p_lane) ? static_cast<_Float16>(static_cast<float>(1 << i))
                                    : static_cast<_Float16>(0.0f);
            c[i] = 0.0f;
        }
        f32x8 d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
        int sum = 0;
        for (int i = 0; i < 8; ++i) sum += static_cast<int>(d[i]);
        // Reduce sum across wave
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_down(sum, off);
        if (lane == 0) k_match_b_for_a0[p_lane] = sum;
    }
#endif
}

using f16x8_vs = __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16;

// Probe 4: Verify __builtin_amdgcn_global_load_tr_b128_v8f16!
__global__ void probe_global_load_tr(const _Float16* src_16x16, _Float16* out_per_lane) {
#if defined(__gfx1201__) || defined(__gfx1200__)
    const int lane = threadIdx.x & 31;
    f16x8_vs* ptr = reinterpret_cast<f16x8_vs*>(reinterpret_cast<uintptr_t>(src_16x16 + lane * 8));
    f16x8_vs v = __builtin_amdgcn_global_load_tr_b128_v8f16(ptr);
    for (int i = 0; i < 8; ++i) {
        out_per_lane[lane * 8 + i] = v[i];
    }
#endif
}

} // namespace

int main() {
    auto info = rxinfer::query_device_info(0);
    std::printf("Device: %s (%s)\n", info.name.c_str(), info.gcn_arch_name.c_str());
    std::printf("VRAM: %.2f GiB | CUs: %d | WaveSize: %d | LDS/blk: %zu KB | L2: %zu KB | LargeBAR: %s\n",
                info.total_vram_bytes / (1024.0 * 1024.0 * 1024.0),
                info.multiprocessor_count, info.warp_size,
                info.shared_mem_per_block / 1024, info.l2_cache_size / 1024,
                info.large_bar ? "yes" : "no");

    // Verify ModelConfig parsing on a synthetic Qwen3.8-27B-like config
    const char* sample_qwen_cfg = R"({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 248320,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "attn_output_gate": true,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mtp_num_hidden_layers": 1,
            "rope_parameters": {
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10]
            },
            "layer_types": [
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention",
                "linear_attention","linear_attention","linear_attention","full_attention"
            ]
        }
    })";
    auto mc = rxinfer::ModelConfig::from_hf_json_string(sample_qwen_cfg);
    std::printf("Config test: arch=%s hidden=%d layers=%d (full=%d, gdn=%d) kv/tok=%.1f KB gdn_state=%.2f MB\n",
                mc.arch_name.c_str(), mc.hidden_size, mc.num_hidden_layers,
                mc.num_full_attention_layers(), mc.num_linear_attention_layers(),
                mc.estimate_kv_cache_bytes_per_token() / 1024.0,
                mc.estimate_recurrent_state_bytes() / (1024.0 * 1024.0));

    // GPU Probes
    float *d_id, *d_fp8_ocp, *d_fp8_neg;
    RX_HIP_CHECK(hipMalloc(&d_id, 256 * sizeof(float)));
    RX_HIP_CHECK(hipMalloc(&d_fp8_ocp, 256 * sizeof(float)));
    RX_HIP_CHECK(hipMalloc(&d_fp8_neg, 256 * sizeof(float)));

    hipLaunchKernelGGL(probe_wmma_patterns, dim3(1), dim3(32), 0, 0, d_id, d_fp8_ocp, d_fp8_neg);
    RX_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> h_id(256), h_fp8_ocp(256), h_fp8_neg(256);
    RX_HIP_CHECK(hipMemcpy(h_id.data(), d_id, 256 * sizeof(float), hipMemcpyDeviceToHost));
    RX_HIP_CHECK(hipMemcpy(h_fp8_ocp.data(), d_fp8_ocp, 256 * sizeof(float), hipMemcpyDeviceToHost));
    RX_HIP_CHECK(hipMemcpy(h_fp8_neg.data(), d_fp8_neg, 256 * sizeof(float), hipMemcpyDeviceToHost));

    std::printf("\n--- Probe 1: Lane-split test (A = (lane<16)?1:0, B = 1) ---\n");
    std::printf("D[lane=0, 0..3] = %.1f %.1f %.1f %.1f | D[lane=16, 0..3] = %.1f %.1f %.1f %.1f\n",
                h_id[0], h_id[1], h_id[2], h_id[3], h_id[16 * 8 + 0], h_id[16 * 8 + 1], h_id[16 * 8 + 2], h_id[16 * 8 + 3]);

    std::printf("\n--- Probe 2: FP8 Encoding on gfx1201 ---\n");
    std::printf("FP8 (0x38 * 0x40 * 16): expected OCP=32.0, FNUZ=8.0 -> got %.2f\n", h_fp8_ocp[0]);
    std::printf("FP8 (0xC4 * 0x38 * 16): expected OCP(-3*1*16)=-48.0 -> got %.2f\n", h_fp8_neg[0]);

    float *d_a_mask, *d_b_mask;
    int *d_k_match;
    RX_HIP_CHECK(hipMalloc(&d_a_mask, 32 * 256 * sizeof(float)));
    RX_HIP_CHECK(hipMalloc(&d_b_mask, 32 * 256 * sizeof(float)));
    RX_HIP_CHECK(hipMalloc(&d_k_match, 32 * sizeof(int)));
    hipLaunchKernelGGL(probe_ab_mapping, dim3(32), dim3(32), 0, 0, d_a_mask, d_b_mask, d_k_match);
    RX_HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> h_a(32 * 256), h_b(32 * 256);
    std::vector<int> h_k(32);
    RX_HIP_CHECK(hipMemcpy(h_a.data(), d_a_mask, h_a.size() * sizeof(float), hipMemcpyDeviceToHost));
    RX_HIP_CHECK(hipMemcpy(h_b.data(), d_b_mask, h_b.size() * sizeof(float), hipMemcpyDeviceToHost));
    RX_HIP_CHECK(hipMemcpy(h_k.data(), d_k_match, 32 * sizeof(int), hipMemcpyDeviceToHost));

    std::printf("\n--- Probe 3: Exact GFX12 Wave32 A/B/C Layout ---\n");
    for (int pl : {0, 1, 15, 16, 17, 31}) {
        int nonzero_c = 0;
        int first_cl = -1, first_ce = -1;
        float first_val = 0.0f;
        for (int cl = 0; cl < 32; ++cl) {
            for (int ce = 0; ce < 8; ++ce) {
                float v = h_a[pl * 256 + cl * 8 + ce];
                if (v != 0.0f) {
                    if (first_cl < 0) { first_cl = cl; first_ce = ce; first_val = v; }
                    nonzero_c++;
                }
            }
        }
        std::printf("A_lane=%2d hits %3d C-entries (first: C[%2d,%d]=%.0f) | ",
                    pl, nonzero_c, first_cl, first_ce, first_val);

        nonzero_c = 0; first_cl = -1; first_ce = -1; first_val = 0.0f;
        for (int cl = 0; cl < 32; ++cl) {
            for (int ce = 0; ce < 8; ++ce) {
                float v = h_b[pl * 256 + cl * 8 + ce];
                if (v != 0.0f) {
                    if (first_cl < 0) { first_cl = cl; first_ce = ce; first_val = v; }
                    nonzero_c++;
                }
            }
        }
        std::printf("B_lane=%2d hits %3d C-entries (first: C[%2d,%d]=%.0f) | Kmatch(A0,0)=%d\n",
                    pl, nonzero_c, first_cl, first_ce, first_val, h_k[pl]);
    }

    // Print full C[lane, elem] dependency on A(0) and B(0) for clarity
    std::printf("C entries hit by A_lane=0: ");
    for (int cl = 0; cl < 32; ++cl)
        for (int ce = 0; ce < 8; ++ce)
            if (h_a[0 * 256 + cl * 8 + ce] != 0.0f)
                std::printf("(%d,%d) ", cl, ce);
    std::printf("\nC entries hit by B_lane=0: ");
    for (int cl = 0; cl < 32; ++cl)
        for (int ce = 0; ce < 8; ++ce)
            if (h_b[0 * 256 + cl * 8 + ce] != 0.0f)
                std::printf("(%d,%d) ", cl, ce);
    std::printf("\n");

    // Probe 4: global_load_tr_b128_v8f16
    std::vector<_Float16> h_src(256), h_tr(256);
    for (int i = 0; i < 256; ++i) h_src[i] = static_cast<_Float16>(static_cast<float>(i));
    _Float16 *d_src, *d_tr;
    RX_HIP_CHECK(hipMalloc(&d_src, 256 * sizeof(_Float16)));
    RX_HIP_CHECK(hipMalloc(&d_tr, 256 * sizeof(_Float16)));
    RX_HIP_CHECK(hipMemcpy(d_src, h_src.data(), 256 * sizeof(_Float16), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(probe_global_load_tr, dim3(1), dim3(32), 0, 0, d_src, d_tr);
    RX_HIP_CHECK(hipDeviceSynchronize());
    RX_HIP_CHECK(hipMemcpy(h_tr.data(), d_tr, 256 * sizeof(_Float16), hipMemcpyDeviceToHost));
    std::printf("\n--- Probe 4: global_load_tr_b128_v8f16 (src[i]=i) ---\n");
    for (int l : {0, 1, 15, 16, 17, 31}) {
        std::printf("lane %2d:", l);
        for (int e = 0; e < 8; ++e) std::printf(" %3.0f", static_cast<float>(h_tr[l * 8 + e]));
        std::printf("\n");
    }

    RX_HIP_CHECK(hipFree(d_id));
    RX_HIP_CHECK(hipFree(d_fp8_ocp));
    RX_HIP_CHECK(hipFree(d_fp8_neg));
    RX_HIP_CHECK(hipFree(d_a_mask));
    RX_HIP_CHECK(hipFree(d_b_mask));
    RX_HIP_CHECK(hipFree(d_k_match));
    RX_HIP_CHECK(hipFree(d_src));
    RX_HIP_CHECK(hipFree(d_tr));
    return 0;
}

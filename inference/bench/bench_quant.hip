// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/common.hpp"
#include "rxinfer/quant.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

struct Shape {
    const char* name;
    int N;
    int K;
};

struct Accuracy {
    double cos_sim;
    double rel_rmse;
    double max_abs;
};

Accuracy eval_accuracy(const std::vector<float>& ref, const std::vector<__half>& got) {
    double dot = 0.0, norm_r = 0.0, norm_g = 0.0, err2 = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double r = static_cast<double>(ref[i]);
        double g = static_cast<double>(__half2float(got[i]));
        dot += r * g;
        norm_r += r * r;
        norm_g += g * g;
        double d = std::fabs(r - g);
        err2 += d * d;
        if (d > max_abs) max_abs = d;
    }
    double cos_sim = dot / (std::sqrt(norm_r) * std::sqrt(norm_g) + 1e-12);
    double rel_rmse = std::sqrt(err2 / (norm_r + 1e-12));
    return {cos_sim, rel_rmse, max_abs};
}

template <typename Fn>
float time_kernel_us(Fn&& fn, hipStream_t stream, int warmup = 10, int iters = 50) {
    for (int i = 0; i < warmup; ++i) fn();
    RX_HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start, stop;
    RX_HIP_CHECK(hipEventCreate(&start));
    RX_HIP_CHECK(hipEventCreate(&stop));
    RX_HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) fn();
    RX_HIP_CHECK(hipEventRecord(stop, stream));
    RX_HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    RX_HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    RX_HIP_CHECK(hipEventDestroy(start));
    RX_HIP_CHECK(hipEventDestroy(stop));
    return (ms * 1000.0f) / iters;
}

} // namespace

int main() {
    auto info = rxinfer::query_device_info(0);
    std::printf("GPU: %s (%s) | VRAM: %.2f GiB | WGPs: %d\n\n",
                info.name.c_str(), info.gcn_arch_name.c_str(),
                info.total_vram_bytes / (1024.0 * 1024.0 * 1024.0),
                info.multiprocessor_count);

    hipStream_t stream = nullptr;
    RX_HIP_CHECK(hipStreamCreate(&stream));

    // 1. Accuracy verification against exact FP32 reference on (M=1, 4, 64) x (N=512, K=512)
    {
        const int N = 512, K = 512;
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.05f);

        std::vector<float> h_W_f32(static_cast<size_t>(N) * K);
        for (float& v : h_W_f32) v = dist(rng);

        std::vector<rxinfer::quant::BlockQ4_0> h_q4(static_cast<size_t>(N) * (K / 32));
        rxinfer::quant::pack_gguf_q4_0(h_W_f32.data(), h_q4.data(), N, K);

        std::vector<uint8_t> h_r4_qs(rxinfer::quant::r4_g64_qs_bytes(N, K));
        std::vector<__half> h_r4_sc(rxinfer::quant::r4_g64_scales_bytes(N, K) / sizeof(__half));
        rxinfer::quant::pack_r4_gfx12_g64(h_W_f32.data(), h_r4_qs.data(), h_r4_sc.data(), N, K);

        void* d_q4 = nullptr;
        void* d_r4_qs = nullptr;
        __half* d_r4_sc = nullptr;
        RX_HIP_CHECK(hipMalloc(&d_q4, h_q4.size() * sizeof(rxinfer::quant::BlockQ4_0)));
        RX_HIP_CHECK(hipMalloc(&d_r4_qs, h_r4_qs.size()));
        RX_HIP_CHECK(hipMalloc(&d_r4_sc, h_r4_sc.size() * sizeof(__half)));
        RX_HIP_CHECK(hipMemcpy(d_q4, h_q4.data(), h_q4.size() * sizeof(rxinfer::quant::BlockQ4_0), hipMemcpyHostToDevice));
        RX_HIP_CHECK(hipMemcpy(d_r4_qs, h_r4_qs.data(), h_r4_qs.size(), hipMemcpyHostToDevice));
        RX_HIP_CHECK(hipMemcpy(d_r4_sc, h_r4_sc.data(), h_r4_sc.size() * sizeof(__half), hipMemcpyHostToDevice));

        std::printf("=== Accuracy Check vs FP32 Reference (N=%d, K=%d) ===\n", N, K);
        for (int M : {1, 4, 64}) {
            std::vector<float> h_A_f32(static_cast<size_t>(M) * K);
            std::vector<__half> h_A_f16(static_cast<size_t>(M) * K);
            for (size_t i = 0; i < h_A_f32.size(); ++i) {
                float v = dist(rng) * 10.0f;
                h_A_f16[i] = __float2half(v);
                h_A_f32[i] = __half2float(h_A_f16[i]);
            }
            std::vector<float> h_ref(static_cast<size_t>(M) * N, 0.0f);
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    double sum = 0.0;
                    for (int k = 0; k < K; ++k) {
                        sum += static_cast<double>(h_A_f32[m * K + k]) * h_W_f32[n * K + k];
                    }
                    h_ref[m * N + n] = static_cast<float>(sum);
                }
            }

            __half *d_A, *d_C;
            float *d_tmp, *d_a_sc;
            uint8_t* d_a_fp8;
            RX_HIP_CHECK(hipMalloc(&d_A, h_A_f16.size() * sizeof(__half)));
            RX_HIP_CHECK(hipMalloc(&d_C, static_cast<size_t>(M) * N * sizeof(__half)));
            RX_HIP_CHECK(hipMalloc(&d_tmp, rxinfer::quant::r4_g64_tmp_bytes(M, N)));
            RX_HIP_CHECK(hipMalloc(&d_a_fp8, static_cast<size_t>(M) * K));
            RX_HIP_CHECK(hipMalloc(&d_a_sc, static_cast<size_t>(M) * (K / 64) * sizeof(float)));
            RX_HIP_CHECK(hipMemcpy(d_A, h_A_f16.data(), h_A_f16.size() * sizeof(__half), hipMemcpyHostToDevice));

            std::vector<__half> h_got(static_cast<size_t>(M) * N);

            rxinfer::quant::gemm_gguf_q4_0(d_A, d_q4, d_C, d_tmp, M, N, K, stream);
            RX_HIP_CHECK(hipMemcpyAsync(h_got.data(), d_C, h_got.size() * sizeof(__half), hipMemcpyDeviceToHost, stream));
            RX_HIP_CHECK(hipStreamSynchronize(stream));
            auto acc_q4 = eval_accuracy(h_ref, h_got);

            rxinfer::quant::gemm_r4_gfx12_w4a16(d_A, d_r4_qs, d_r4_sc, d_C, d_tmp, M, N, K, stream);
            RX_HIP_CHECK(hipMemcpyAsync(h_got.data(), d_C, h_got.size() * sizeof(__half), hipMemcpyDeviceToHost, stream));
            RX_HIP_CHECK(hipStreamSynchronize(stream));
            auto acc_r4 = eval_accuracy(h_ref, h_got);

            std::printf("M=%2d | GGUF_Q4_0 cos=%.5f rel=%.4f | R4_W4A16 cos=%.5f rel=%.4f",
                        M, acc_q4.cos_sim, acc_q4.rel_rmse, acc_r4.cos_sim, acc_r4.rel_rmse);

            if (M > 16) {
                rxinfer::quant::gemm_r4_gfx12_fp8(d_A, d_r4_qs, d_r4_sc, d_C, d_tmp, d_a_fp8, d_a_sc, M, N, K, stream);
                RX_HIP_CHECK(hipMemcpyAsync(h_got.data(), d_C, h_got.size() * sizeof(__half), hipMemcpyDeviceToHost, stream));
                RX_HIP_CHECK(hipStreamSynchronize(stream));
                auto acc_fp8 = eval_accuracy(h_ref, h_got);
                std::printf(" | R4_FP8 cos=%.5f rel=%.4f", acc_fp8.cos_sim, acc_fp8.rel_rmse);
            }
            std::printf("\n");

            RX_HIP_CHECK(hipFree(d_A));
            RX_HIP_CHECK(hipFree(d_C));
            RX_HIP_CHECK(hipFree(d_tmp));
            RX_HIP_CHECK(hipFree(d_a_fp8));
            RX_HIP_CHECK(hipFree(d_a_sc));
        }

        RX_HIP_CHECK(hipFree(d_q4));
        RX_HIP_CHECK(hipFree(d_r4_qs));
        RX_HIP_CHECK(hipFree(d_r4_sc));
    }

    // 2. Head-to-head Performance Benchmark on Qwen3.8-27B matrix shapes
    const Shape shapes[] = {
        {"attn_proj", 5120, 5120},
        {"qkv_proj ", 6144, 5120},
        {"ffn_up   ", 17408, 5120},
        {"ffn_down ", 5120, 17408},
    };
    const int Ms[] = {1, 4, 8, 64, 512, 2048};

    std::printf("\n=== Head-to-Head RX 9070 XT Benchmark: GGUF Q4_0 (4.5 bpw) vs R4_GFX12_G64 (4.25 bpw) ===\n");
    std::printf("%-9s %5s %5s %4s | %9s %8s | %9s %8s | %9s %8s | %7s\n",
                "Shape", "N", "K", "M",
                "GGUF(us)", "GB/s|TF",
                "R4_16(us)", "GB/s|TF",
                "R4_F8(us)", "GB/s|TF",
                "Speedup");

    for (const auto& s : shapes) {
        const size_t q4_bytes = static_cast<size_t>(s.N) * (s.K / 32) * sizeof(rxinfer::quant::BlockQ4_0);
        const size_t r4_qs_bytes = rxinfer::quant::r4_g64_qs_bytes(s.N, s.K);
        const size_t r4_sc_bytes = rxinfer::quant::r4_g64_scales_bytes(s.N, s.K);
        const size_t r4_total_bytes = r4_qs_bytes + r4_sc_bytes;

        void *d_q4 = nullptr, *d_r4_qs = nullptr;
        __half* d_r4_sc = nullptr;
        RX_HIP_CHECK(hipMalloc(&d_q4, q4_bytes));
        RX_HIP_CHECK(hipMalloc(&d_r4_qs, r4_qs_bytes));
        RX_HIP_CHECK(hipMalloc(&d_r4_sc, r4_sc_bytes));
        RX_HIP_CHECK(hipMemset(d_q4, 0x3C, q4_bytes));
        RX_HIP_CHECK(hipMemset(d_r4_qs, 0x88, r4_qs_bytes));
        RX_HIP_CHECK(hipMemset(d_r4_sc, 0x3C, r4_sc_bytes));

        for (int M : Ms) {
            __half *d_A = nullptr, *d_C = nullptr;
            float *d_tmp = nullptr, *d_a_sc = nullptr;
            uint8_t* d_a_fp8 = nullptr;
            RX_HIP_CHECK(hipMalloc(&d_A, static_cast<size_t>(M) * s.K * sizeof(__half)));
            RX_HIP_CHECK(hipMalloc(&d_C, static_cast<size_t>(M) * s.N * sizeof(__half)));
            RX_HIP_CHECK(hipMalloc(&d_tmp, rxinfer::quant::r4_g64_tmp_bytes(M, s.N)));
            RX_HIP_CHECK(hipMalloc(&d_a_fp8, static_cast<size_t>(M) * s.K));
            RX_HIP_CHECK(hipMalloc(&d_a_sc, static_cast<size_t>(M) * (s.K / 64) * sizeof(float)));
            RX_HIP_CHECK(hipMemset(d_A, 0x3C, static_cast<size_t>(M) * s.K * sizeof(__half)));

            const int iters = (M >= 512) ? 30 : 80;
            const float us_q4 = time_kernel_us([&] {
                rxinfer::quant::gemm_gguf_q4_0(d_A, d_q4, d_C, d_tmp, M, s.N, s.K, stream);
            }, stream, 10, iters);

            const float us_r4_16 = time_kernel_us([&] {
                rxinfer::quant::gemm_r4_gfx12_w4a16(d_A, d_r4_qs, d_r4_sc, d_C, d_tmp, M, s.N, s.K, stream);
            }, stream, 10, iters);

            float us_r4_f8 = us_r4_16;
            if (M > 16) {
                us_r4_f8 = time_kernel_us([&] {
                    rxinfer::quant::gemm_r4_gfx12_fp8(d_A, d_r4_qs, d_r4_sc, d_C, d_tmp,
                                                       d_a_fp8, d_a_sc, M, s.N, s.K, stream);
                }, stream, 10, iters);
            }

            const double flops = 2.0 * M * s.N * s.K;
            if (M <= 16) {
                const double bw_q4 = (q4_bytes + static_cast<double>(M) * (s.K + s.N) * 2.0) / (us_q4 * 1e3);
                const double bw_r4 = (r4_total_bytes + static_cast<double>(M) * (s.K + s.N) * 2.0) / (us_r4_16 * 1e3);
                std::printf("%-9s %5d %5d %4d | %9.1f %6.0fGB | %9.1f %6.0fGB | %9s %8s | %6.2fx\n",
                            s.name, s.N, s.K, M,
                            us_q4, bw_q4,
                            us_r4_16, bw_r4,
                            "-", "-",
                            us_q4 / us_r4_16);
            } else {
                const double tf_q4 = flops / (us_q4 * 1e6);
                const double tf_r4_16 = flops / (us_r4_16 * 1e6);
                const double tf_r4_f8 = flops / (us_r4_f8 * 1e6);
                const float best_r4 = std::min(us_r4_16, us_r4_f8);
                std::printf("%-9s %5d %5d %4d | %9.1f %6.2fTF | %9.1f %6.2fTF | %9.1f %6.2fTF | %6.2fx\n",
                            s.name, s.N, s.K, M,
                            us_q4, tf_q4,
                            us_r4_16, tf_r4_16,
                            us_r4_f8, tf_r4_f8,
                            us_q4 / best_r4);
            }

            RX_HIP_CHECK(hipFree(d_A));
            RX_HIP_CHECK(hipFree(d_C));
            RX_HIP_CHECK(hipFree(d_tmp));
            RX_HIP_CHECK(hipFree(d_a_fp8));
            RX_HIP_CHECK(hipFree(d_a_sc));
        }

        RX_HIP_CHECK(hipFree(d_q4));
        RX_HIP_CHECK(hipFree(d_r4_qs));
        RX_HIP_CHECK(hipFree(d_r4_sc));
    }

    // 3. Cold-L3 (>256 MB rotating buffers) Decode M=1 Benchmark — True GDDR6 VRAM Speed!
    {
        const int N = 17408, K = 5120;
        const size_t q4_bytes = static_cast<size_t>(N) * (K / 32) * sizeof(rxinfer::quant::BlockQ4_0);
        const size_t r4_qs_bytes = rxinfer::quant::r4_g64_qs_bytes(N, K);
        const size_t r4_sc_bytes = rxinfer::quant::r4_g64_scales_bytes(N, K);
        const int BANKS = 6; // 6 * 46 MB = 276 MB >> 64 MB L3 Infinity Cache

        std::vector<void*> d_q4_banks(BANKS), d_r4_qs_banks(BANKS);
        std::vector<__half*> d_r4_sc_banks(BANKS);
        for (int b = 0; b < BANKS; ++b) {
            RX_HIP_CHECK(hipMalloc(&d_q4_banks[b], q4_bytes));
            RX_HIP_CHECK(hipMalloc(&d_r4_qs_banks[b], r4_qs_bytes));
            RX_HIP_CHECK(hipMalloc(&d_r4_sc_banks[b], r4_sc_bytes));
            RX_HIP_CHECK(hipMemset(d_q4_banks[b], 0x3C + b, q4_bytes));
            RX_HIP_CHECK(hipMemset(d_r4_qs_banks[b], 0x22 + b, r4_qs_bytes));
            RX_HIP_CHECK(hipMemset(d_r4_sc_banks[b], 0x3C, r4_sc_bytes));
        }

        __half *d_A = nullptr, *d_C = nullptr;
        float* d_tmp = nullptr;
        RX_HIP_CHECK(hipMalloc(&d_A, static_cast<size_t>(K) * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_C, static_cast<size_t>(N) * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_tmp, rxinfer::quant::r4_g64_tmp_bytes(1, N)));
        RX_HIP_CHECK(hipMemset(d_A, 0x3C, static_cast<size_t>(K) * sizeof(__half)));

        int idx_q4 = 0;
        const float us_q4_cold = time_kernel_us([&] {
            rxinfer::quant::gemm_gguf_q4_0(d_A, d_q4_banks[idx_q4++ % BANKS], d_C, d_tmp, 1, N, K, stream);
        }, stream, 6, 48);

        int idx_r4 = 0;
        const float us_r4_cold = time_kernel_us([&] {
            rxinfer::quant::gemm_r4_gfx12_w4a16(d_A, d_r4_qs_banks[idx_r4], d_r4_sc_banks[idx_r4],
                                                 d_C, d_tmp, 1, N, K, stream);
            ++idx_r4;
            if (idx_r4 == BANKS) idx_r4 = 0;
        }, stream, 6, 48);

        const double bw_q4 = (q4_bytes + (K + N) * 2.0) / (us_q4_cold * 1e3);
        const double bw_r4 = (r4_qs_bytes + r4_sc_bytes + (K + N) * 2.0) / (us_r4_cold * 1e3);
        std::printf("\n=== Cold-L3 VRAM Decode M=1 (276 MB rotating pool, N=%d, K=%d) ===\n", N, K);
        std::printf("GGUF Q4_0:     %.1f us | %.1f GB/s\n", us_q4_cold, bw_q4);
        std::printf("R4_GFX12_G64:  %.1f us | %.1f GB/s (%.2fx faster, 5.5%% smaller)\n",
                    us_r4_cold, bw_r4, us_q4_cold / us_r4_cold);

        RX_HIP_CHECK(hipFree(d_A));
        RX_HIP_CHECK(hipFree(d_C));
        RX_HIP_CHECK(hipFree(d_tmp));
        for (int b = 0; b < BANKS; ++b) {
            RX_HIP_CHECK(hipFree(d_q4_banks[b]));
            RX_HIP_CHECK(hipFree(d_r4_qs_banks[b]));
            RX_HIP_CHECK(hipFree(d_r4_sc_banks[b]));
        }
    }

    RX_HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/arena.hpp"
#include "rxinfer/attention.hpp"
#include "rxinfer/common.hpp"
#include "rxinfer/gdn.hpp"
#include "rxinfer/ops.hpp"
#include "rxinfer/quant.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

struct Metrics {
    double cos_sim;
    double rel_rmse;
    double max_abs;
};

Metrics eval_f16_vs_f32(const std::vector<float>& ref, const std::vector<__half>& got) {
    double dot = 0.0, nr = 0.0, ng = 0.0, err2 = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = ref[i];
        const double g = __half2float(got[i]);
        dot += r * g;
        nr += r * r;
        ng += g * g;
        const double d = std::fabs(r - g);
        err2 += d * d;
        if (d > max_abs) max_abs = d;
    }
    return {dot / (std::sqrt(nr) * std::sqrt(ng) + 1e-12), std::sqrt(err2 / (nr + 1e-12)), max_abs};
}

Metrics eval_f32_vs_f32(const std::vector<float>& ref, const std::vector<float>& got) {
    double dot = 0.0, nr = 0.0, ng = 0.0, err2 = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = ref[i];
        const double g = got[i];
        dot += r * g;
        nr += r * r;
        ng += g * g;
        const double d = std::fabs(r - g);
        err2 += d * d;
        if (d > max_abs) max_abs = d;
    }
    return {dot / (std::sqrt(nr) * std::sqrt(ng) + 1e-12), std::sqrt(err2 / (nr + 1e-12)), max_abs};
}

template <typename Fn>
float time_us(Fn&& fn, hipStream_t stream, int warmup = 10, int iters = 100) {
    for (int i = 0; i < warmup; ++i) fn();
    RX_HIP_CHECK(hipStreamSynchronize(stream));
    hipEvent_t s, e;
    RX_HIP_CHECK(hipEventCreate(&s));
    RX_HIP_CHECK(hipEventCreate(&e));
    RX_HIP_CHECK(hipEventRecord(s, stream));
    for (int i = 0; i < iters; ++i) fn();
    RX_HIP_CHECK(hipEventRecord(e, stream));
    RX_HIP_CHECK(hipEventSynchronize(e));
    float ms = 0.0f;
    RX_HIP_CHECK(hipEventElapsedTime(&ms, s, e));
    RX_HIP_CHECK(hipEventDestroy(s));
    RX_HIP_CHECK(hipEventDestroy(e));
    return (ms * 1000.0f) / iters;
}

float silu_cpu(float x) {
    return x / (1.0f + std::exp(-x));
}

float sigmoid_cpu(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float softplus_cpu(float x) {
    return (x > 20.0f) ? x : std::log1p(std::exp(x));
}

void test_norm_swiglu_embed_argmax(hipStream_t stream, int& failures) {
    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0.0f, 0.5f);

    // 1. Fused ResidualAdd + RMSNormPlusOne
    {
        const int M = 4, D = 5120;
        const float eps = 1e-6f;
        std::vector<__half> h_x(M * D), h_res(M * D), h_w(D), h_y(M * D), h_res_out(M * D);
        std::vector<float> ref_y(M * D), ref_res(M * D);

        for (int i = 0; i < D; ++i) h_w[i] = __float2half(nd(rng) * 0.2f);
        for (int m = 0; m < M; ++m) {
            float ss = 0.0f;
            for (int d = 0; d < D; ++d) {
                h_x[m * D + d] = __float2half(nd(rng));
                h_res[m * D + d] = __float2half(nd(rng));
                const float r = __half2float(h_x[m * D + d]) + __half2float(h_res[m * D + d]);
                ref_res[m * D + d] = r;
                ss += r * r;
            }
            const float inv = 1.0f / std::sqrt(ss / D + eps);
            for (int d = 0; d < D; ++d) {
                ref_y[m * D + d] = ref_res[m * D + d] * inv * (1.0f + __half2float(h_w[d]));
            }
        }

        __half *d_x, *d_res, *d_w, *d_y;
        RX_HIP_CHECK(hipMalloc(&d_x, M * D * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_res, M * D * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_w, D * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_y, M * D * sizeof(__half)));
        RX_HIP_CHECK(hipMemcpy(d_x, h_x.data(), M * D * sizeof(__half), hipMemcpyHostToDevice));
        RX_HIP_CHECK(hipMemcpy(d_res, h_res.data(), M * D * sizeof(__half), hipMemcpyHostToDevice));
        RX_HIP_CHECK(hipMemcpy(d_w, h_w.data(), D * sizeof(__half), hipMemcpyHostToDevice));

        rxinfer::ops::fused_residual_rmsnorm(d_x, d_res, d_w, d_y, M, D, eps, true, stream);
        RX_HIP_CHECK(hipMemcpyAsync(h_y.data(), d_y, M * D * sizeof(__half), hipMemcpyDeviceToHost, stream));
        RX_HIP_CHECK(hipMemcpyAsync(h_res_out.data(), d_res, M * D * sizeof(__half), hipMemcpyDeviceToHost, stream));
        RX_HIP_CHECK(hipStreamSynchronize(stream));

        auto m_y = eval_f16_vs_f32(ref_y, h_y);
        auto m_r = eval_f16_vs_f32(ref_res, h_res_out);
        const float us_1 = time_us([&] {
            rxinfer::ops::fused_residual_rmsnorm(d_x, d_res, d_w, d_y, 1, D, eps, true, stream);
        }, stream);
        std::printf("[RMSNorm+1]   y cos=%.6f rel=%.5f | res cos=%.6f | M=1 %.2f us\n",
                    m_y.cos_sim, m_y.rel_rmse, m_r.cos_sim, us_1);
        if (m_y.cos_sim < 0.9999 || m_r.cos_sim < 0.9999) ++failures;

        RX_HIP_CHECK(hipFree(d_x));
        RX_HIP_CHECK(hipFree(d_res));
        RX_HIP_CHECK(hipFree(d_w));
        RX_HIP_CHECK(hipFree(d_y));
    }

    // 2. Packed SwiGLU
    {
        const int M = 2, I = 17408;
        std::vector<__half> h_gu(M * 2 * I), h_out(M * I);
        std::vector<float> ref(M * I);
        for (int m = 0; m < M; ++m) {
            for (int i = 0; i < I; ++i) {
                h_gu[m * 2 * I + i] = __float2half(nd(rng));
                h_gu[m * 2 * I + I + i] = __float2half(nd(rng));
                const float g = __half2float(h_gu[m * 2 * I + i]);
                const float u = __half2float(h_gu[m * 2 * I + I + i]);
                ref[m * I + i] = silu_cpu(g) * u;
            }
        }
        __half *d_gu, *d_out;
        RX_HIP_CHECK(hipMalloc(&d_gu, M * 2 * I * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_out, M * I * sizeof(__half)));
        RX_HIP_CHECK(hipMemcpy(d_gu, h_gu.data(), M * 2 * I * sizeof(__half), hipMemcpyHostToDevice));
        rxinfer::ops::fused_silu_mul_packed(d_gu, d_out, M, I, stream);
        RX_HIP_CHECK(hipMemcpyAsync(h_out.data(), d_out, M * I * sizeof(__half), hipMemcpyDeviceToHost, stream));
        RX_HIP_CHECK(hipStreamSynchronize(stream));

        auto m_s = eval_f16_vs_f32(ref, h_out);
        const float us_1 = time_us([&] {
            rxinfer::ops::fused_silu_mul_packed(d_gu, d_out, 1, I, stream);
        }, stream);
        std::printf("[SwiGLU Pack] cos=%.6f rel=%.5f | M=1 %.2f us\n", m_s.cos_sim, m_s.rel_rmse, us_1);
        if (m_s.cos_sim < 0.9999) ++failures;
        RX_HIP_CHECK(hipFree(d_gu));
        RX_HIP_CHECK(hipFree(d_out));
    }

    // 3. R4 Embedding Lookup + Argmax
    {
        const int V = 256, H = 5120;
        std::vector<float> h_tbl(V * H);
        for (float& v : h_tbl) v = nd(rng) * 0.1f;
        std::vector<uint8_t> h_qs(rxinfer::quant::r4_g64_qs_bytes(V, H));
        std::vector<__half> h_sc(rxinfer::quant::r4_g64_scales_bytes(V, H) / sizeof(__half));
        rxinfer::quant::pack_r4_gfx12_g64(h_tbl.data(), h_qs.data(), h_sc.data(), V, H);

        void* d_qs;
        __half *d_sc, *d_out;
        rxinfer::StepParams* d_step;
        RX_HIP_CHECK(hipMalloc(&d_qs, h_qs.size()));
        RX_HIP_CHECK(hipMalloc(&d_sc, h_sc.size() * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_out, H * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_step, sizeof(rxinfer::StepParams)));
        RX_HIP_CHECK(hipMemcpy(d_qs, h_qs.data(), h_qs.size(), hipMemcpyHostToDevice));
        RX_HIP_CHECK(hipMemcpy(d_sc, h_sc.data(), h_sc.size() * sizeof(__half), hipMemcpyHostToDevice));

        rxinfer::StepParams sp{};
        sp.token_id = 173;
        RX_HIP_CHECK(hipMemcpy(d_step, &sp, sizeof(sp), hipMemcpyHostToDevice));
        rxinfer::ops::embed_lookup_r4_step(d_qs, d_sc, d_step, d_out, V, H, stream);
        std::vector<__half> h_got(H);
        RX_HIP_CHECK(hipMemcpyAsync(h_got.data(), d_out, H * sizeof(__half), hipMemcpyDeviceToHost, stream));
        RX_HIP_CHECK(hipStreamSynchronize(stream));

        std::vector<float> ref_row(h_tbl.begin() + 173 * H, h_tbl.begin() + 174 * H);
        auto m_e = eval_f16_vs_f32(ref_row, h_got);
        std::printf("[Embed R4]    cos=%.6f rel=%.5f\n", m_e.cos_sim, m_e.rel_rmse);
        if (m_e.cos_sim < 0.994) ++failures;

        RX_HIP_CHECK(hipFree(d_qs));
        RX_HIP_CHECK(hipFree(d_sc));
        RX_HIP_CHECK(hipFree(d_out));
        RX_HIP_CHECK(hipFree(d_step));
    }

    // 4. Argmax on 248320 vocab
    {
        const int V = 248320;
        std::vector<__half> h_logits(V);
        for (int i = 0; i < V; ++i) h_logits[i] = __float2half(nd(rng));
        const int expected_idx = 193847;
        h_logits[expected_idx] = __float2half(15.5f);

        __half* d_logits;
        void* d_tmp;
        int32_t* d_tok;
        rxinfer::StepParams* d_step;
        RX_HIP_CHECK(hipMalloc(&d_logits, V * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_tmp, rxinfer::ops::argmax_tmp_bytes()));
        RX_HIP_CHECK(hipMalloc(&d_tok, sizeof(int32_t)));
        RX_HIP_CHECK(hipMalloc(&d_step, sizeof(rxinfer::StepParams)));
        RX_HIP_CHECK(hipMemcpy(d_logits, h_logits.data(), V * sizeof(__half), hipMemcpyHostToDevice));

        rxinfer::ops::argmax_sample(d_logits, V, d_tmp, d_tok, d_step, stream);
        rxinfer::StepParams sp{};
        RX_HIP_CHECK(hipMemcpyAsync(&sp, d_step, sizeof(sp), hipMemcpyDeviceToHost, stream));
        RX_HIP_CHECK(hipStreamSynchronize(stream));

        const float us_a = time_us([&] {
            rxinfer::ops::argmax_sample(d_logits, V, d_tmp, d_tok, d_step, stream);
        }, stream);
        std::printf("[Argmax 248k] got=%d expected=%d | %.2f us\n", sp.token_id, expected_idx, us_a);
        if (sp.token_id != expected_idx) ++failures;

        RX_HIP_CHECK(hipFree(d_logits));
        RX_HIP_CHECK(hipFree(d_tmp));
        RX_HIP_CHECK(hipFree(d_tok));
        RX_HIP_CHECK(hipFree(d_step));
    }
}

void test_attention(hipStream_t stream, int& failures) {
    std::mt19937 rng(777);
    std::normal_distribution<float> nd(0.0f, 0.5f);

    // Exact Qwen3.8-27B FullAttention parameters
    rxinfer::attn::AttentionParams p{};
    p.num_q_heads = 24;
    p.num_kv_heads = 4;
    p.head_dim = 256;
    p.rotary_dim = 64;
    p.rope_theta = 10000000.0f;
    p.rms_norm_eps = 1e-6f;
    p.rope_kind = rxinfer::RopeKind::NeoX;
    p.qk_norm = true;
    p.qk_norm_plus_one = true;
    p.attn_output_gate = true;
    p.max_seq_len = 512;

    const int M_pre = 24;
    const int M_tot = M_pre + 1; // 24 prefill + 1 hipGraph decode step!
    const int H_q = p.num_q_heads;
    const int H_kv = p.num_kv_heads;
    const int D = p.head_dim;

    std::vector<__half> h_q_in(M_tot * H_q * 2 * D);
    std::vector<__half> h_k_in(M_tot * H_kv * D);
    std::vector<__half> h_v_in(M_tot * H_kv * D);
    std::vector<__half> h_qn_w(D), h_kn_w(D);
    for (auto& v : h_q_in) v = __float2half(nd(rng));
    for (auto& v : h_k_in) v = __float2half(nd(rng));
    for (auto& v : h_v_in) v = __float2half(nd(rng));
    for (int d = 0; d < D; ++d) {
        h_qn_w[d] = __float2half(nd(rng) * 0.2f);
        h_kn_w[d] = __float2half(nd(rng) * 0.2f);
    }

    // Exact FP32 CPU reference
    std::vector<float> ref_k_cache(M_tot * H_kv * D);
    std::vector<float> ref_v_cache(M_tot * H_kv * D);
    std::vector<float> ref_pre_out(M_pre * H_q * D);
    std::vector<float> ref_dec_out(H_q * D);

    auto apply_qknorm_rope = [&](const __half* src, const __half* nw, int pos, std::vector<float>& dst) {
        dst.resize(D);
        float ss = 0.0f;
        for (int d = 0; d < D; ++d) {
            dst[d] = __half2float(src[d]);
            ss += dst[d] * dst[d];
        }
        const float inv = 1.0f / std::sqrt(ss / D + p.rms_norm_eps);
        for (int d = 0; d < D; ++d) {
            dst[d] *= inv * (1.0f + __half2float(nw[d]));
        }
        const int hr = p.rotary_dim / 2;
        for (int i = 0; i < hr; ++i) {
            const float freq = std::pow(p.rope_theta, -2.0f * i / p.rotary_dim);
            const float ang = pos * freq;
            const float c = std::cos(ang), s = std::sin(ang);
            const float x0 = dst[i], x1 = dst[i + hr];
            dst[i] = x0 * c - x1 * s;
            dst[i + hr] = x1 * c + x0 * s;
        }
    };

    std::vector<float> ref_q(M_tot * H_q * D);
    for (int m = 0; m < M_tot; ++m) {
        for (int hq = 0; hq < H_q; ++hq) {
            std::vector<float> qvec;
            apply_qknorm_rope(&h_q_in[(m * H_q + hq) * 2 * D], h_qn_w.data(), m, qvec);
            std::copy(qvec.begin(), qvec.end(), &ref_q[(m * H_q + hq) * D]);
        }
        for (int hk = 0; hk < H_kv; ++hk) {
            std::vector<float> kvec;
            apply_qknorm_rope(&h_k_in[(m * H_kv + hk) * D], h_kn_w.data(), m, kvec);
            std::copy(kvec.begin(), kvec.end(), &ref_k_cache[(m * H_kv + hk) * D]);
            for (int d = 0; d < D; ++d) {
                ref_v_cache[(m * H_kv + hk) * D + d] = __half2float(h_v_in[(m * H_kv + hk) * D + d]);
            }
        }
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    for (int m = 0; m < M_tot; ++m) {
        for (int hq = 0; hq < H_q; ++hq) {
            const int hk = hq / (H_q / H_kv);
            std::vector<float> scores(m + 1);
            float mmax = -1e30f;
            for (int t = 0; t <= m; ++t) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) {
                    dot += ref_q[(m * H_q + hq) * D + d] * ref_k_cache[(t * H_kv + hk) * D + d];
                }
                scores[t] = dot * scale;
                mmax = std::max(mmax, scores[t]);
            }
            float lsum = 0.0f;
            for (int t = 0; t <= m; ++t) {
                scores[t] = std::exp(scores[t] - mmax);
                lsum += scores[t];
            }
            for (int d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (int t = 0; t <= m; ++t) {
                    acc += (scores[t] / lsum) * ref_v_cache[(t * H_kv + hk) * D + d];
                }
                const float gate = __half2float(h_q_in[(m * H_q + hq) * 2 * D + D + d]);
                const float val = acc * sigmoid_cpu(gate);
                if (m < M_pre) ref_pre_out[(m * H_q + hq) * D + d] = val;
                else ref_dec_out[hq * D + d] = val;
            }
        }
    }

    // Test all 3 KV cache formats using paged head-major storage
    for (rxinfer::DType kv_dt : {rxinfer::DType::F16, rxinfer::DType::FP8_E4M3, rxinfer::DType::FP4_E2M1_G16}) {
        p.kv_dtype = kv_dt;
        const auto lay = rxinfer::attn::compute_kv_page_layout(p);
        const int32_t num_pages = rxinfer::div_ceil(p.max_seq_len, rxinfer::attn::KV_PAGE_TOKENS);

        __half *d_q_in, *d_k_in, *d_v_in, *d_qn_w, *d_kn_w, *d_q_out, *d_out;
        rxinfer::StepParams* d_step;
        uint8_t* d_kv_pool = nullptr;
        uint8_t** d_page_table = nullptr;
        void* d_split = nullptr;

        RX_HIP_CHECK(hipMalloc(&d_q_in, h_q_in.size() * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_k_in, h_k_in.size() * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_v_in, h_v_in.size() * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_qn_w, D * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_kn_w, D * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_q_out, M_tot * H_q * D * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_out, M_tot * H_q * D * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_step, sizeof(rxinfer::StepParams)));
        RX_HIP_CHECK(hipMalloc(&d_kv_pool, static_cast<size_t>(num_pages) * lay.page_bytes));
        RX_HIP_CHECK(hipMalloc(&d_page_table, static_cast<size_t>(num_pages) * sizeof(uint8_t*)));
        RX_HIP_CHECK(hipMemset(d_kv_pool, 0, static_cast<size_t>(num_pages) * lay.page_bytes));
        RX_HIP_CHECK(hipMalloc(&d_split, rxinfer::attn::decode_splitk_scratch_bytes(p, 1, 32)));

        std::vector<uint8_t*> h_ptrs(num_pages);
        for (int32_t pg = 0; pg < num_pages; ++pg) {
            h_ptrs[pg] = d_kv_pool + static_cast<size_t>(pg) * lay.page_bytes;
        }
        RX_HIP_CHECK(hipMemcpy(d_page_table, h_ptrs.data(), static_cast<size_t>(num_pages) * sizeof(uint8_t*),
                               hipMemcpyHostToDevice));

        RX_HIP_CHECK(hipMemcpy(d_q_in, h_q_in.data(), h_q_in.size() * sizeof(__half), hipMemcpyHostToDevice));
        RX_HIP_CHECK(hipMemcpy(d_k_in, h_k_in.data(), h_k_in.size() * sizeof(__half), hipMemcpyHostToDevice));
        RX_HIP_CHECK(hipMemcpy(d_v_in, h_v_in.data(), h_v_in.size() * sizeof(__half), hipMemcpyHostToDevice));
        RX_HIP_CHECK(hipMemcpy(d_qn_w, h_qn_w.data(), D * sizeof(__half), hipMemcpyHostToDevice));
        RX_HIP_CHECK(hipMemcpy(d_kn_w, h_kn_w.data(), D * sizeof(__half), hipMemcpyHostToDevice));

        rxinfer::attn::KVCacheView kv{
            d_page_table, num_pages, num_pages, lay.page_bytes, lay.data_sec_bytes, lay.sc_sec_bytes};

        // Prefill M_pre tokens
        rxinfer::attn::fused_qknorm_rope_kv_write(d_q_in, d_k_in, d_v_in, d_qn_w, d_kn_w, d_q_out,
                                                  kv, p, M_pre, 0, nullptr, stream);
        rxinfer::attn::flash_attention_prefill(d_q_out, d_q_in, kv, d_out, p, M_pre, 0, nullptr, stream);

        std::vector<__half> h_pre_out(M_pre * H_q * D);
        RX_HIP_CHECK(hipMemcpyAsync(h_pre_out.data(), d_out, h_pre_out.size() * sizeof(__half), hipMemcpyDeviceToHost, stream));

        // Decode token M_pre using StepParams inside a hipGraph!
        rxinfer::StepParams sp{};
        sp.pos = M_pre;
        sp.kv_seq_len = M_tot;
        RX_HIP_CHECK(hipMemcpyAsync(d_step, &sp, sizeof(sp), hipMemcpyHostToDevice, stream));

        hipGraph_t graph = nullptr;
        hipGraphExec_t graph_exec = nullptr;
        RX_HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
        rxinfer::attn::fused_qknorm_rope_kv_write(
            d_q_in + M_pre * H_q * 2 * D,
            d_k_in + M_pre * H_kv * D,
            d_v_in + M_pre * H_kv * D,
            d_qn_w, d_kn_w, d_q_out, kv, p, 1, 0, d_step, stream);
        rxinfer::attn::gqa_decode_attention(
            d_q_out, d_q_in + M_pre * H_q * 2 * D, kv, d_out, d_split, p, 1, 0, d_step, 32, stream);
        RX_HIP_CHECK(hipStreamEndCapture(stream, &graph));
        RX_HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        RX_HIP_CHECK(hipGraphLaunch(graph_exec, stream));

        std::vector<__half> h_dec_out(H_q * D);
        RX_HIP_CHECK(hipMemcpyAsync(h_dec_out.data(), d_out, h_dec_out.size() * sizeof(__half), hipMemcpyDeviceToHost, stream));
        RX_HIP_CHECK(hipStreamSynchronize(stream));

        auto m_pre = eval_f16_vs_f32(ref_pre_out, h_pre_out);
        auto m_dec = eval_f16_vs_f32(ref_dec_out, h_dec_out);

        // Benchmark decode at kv_seq_len = 512
        sp.kv_seq_len = 512;
        RX_HIP_CHECK(hipMemcpy(d_step, &sp, sizeof(sp), hipMemcpyHostToDevice));
        const float us_dec512 = time_us([&] {
            RX_HIP_CHECK(hipGraphLaunch(graph_exec, stream));
        }, stream);

        std::printf("[Attn D=256 %-12s] prefill cos=%.5f | decode(graph) cos=%.5f rel=%.4f | S=512 %.1f us\n",
                    std::string(rxinfer::dtype_name(kv_dt)).c_str(),
                    m_pre.cos_sim, m_dec.cos_sim, m_dec.rel_rmse, us_dec512);

        const double min_cos = (kv_dt == rxinfer::DType::FP4_E2M1_G16) ? 0.985 : 0.999;
        if (m_pre.cos_sim < min_cos || m_dec.cos_sim < min_cos) ++failures;

        RX_HIP_CHECK(hipGraphExecDestroy(graph_exec));
        RX_HIP_CHECK(hipGraphDestroy(graph));
        RX_HIP_CHECK(hipFree(d_q_in));
        RX_HIP_CHECK(hipFree(d_k_in));
        RX_HIP_CHECK(hipFree(d_v_in));
        RX_HIP_CHECK(hipFree(d_qn_w));
        RX_HIP_CHECK(hipFree(d_kn_w));
        RX_HIP_CHECK(hipFree(d_q_out));
        RX_HIP_CHECK(hipFree(d_out));
        RX_HIP_CHECK(hipFree(d_step));
        RX_HIP_CHECK(hipFree(d_kv_pool));
        RX_HIP_CHECK(hipFree(d_page_table));
        RX_HIP_CHECK(hipFree(d_split));
    }

    // Tiered VRAM + GTT Paged Attention Test (1 VRAM page = 64 tok + 2 GTT pages = 128 tok, split-KV + stage_buf)
    {
        const int M_t_pre = 160; // Spans page 0 (VRAM), page 1 (GTT), page 2 (GTT)
        const int M_t_tot = M_t_pre + 1;
        const int32_t num_pages = 3;
        const int32_t vram_pages = 1;
        const int32_t gtt_pages = 2;

        std::vector<__half> t_q(M_t_tot * H_q * 2 * D);
        std::vector<__half> t_k(M_t_tot * H_kv * D);
        std::vector<__half> t_v(M_t_tot * H_kv * D);
        for (auto& v : t_q) v = __float2half(nd(rng));
        for (auto& v : t_k) v = __float2half(nd(rng));
        for (auto& v : t_v) v = __float2half(nd(rng));

        std::vector<float> t_ref_k(M_t_tot * H_kv * D), t_ref_v(M_t_tot * H_kv * D);
        std::vector<float> t_ref_q(M_t_tot * H_q * D), t_ref_pre(M_t_pre * H_q * D), t_ref_dec(H_q * D);
        for (int m = 0; m < M_t_tot; ++m) {
            for (int hq = 0; hq < H_q; ++hq) {
                std::vector<float> qv;
                apply_qknorm_rope(&t_q[(m * H_q + hq) * 2 * D], h_qn_w.data(), m, qv);
                std::copy(qv.begin(), qv.end(), &t_ref_q[(m * H_q + hq) * D]);
            }
            for (int hk = 0; hk < H_kv; ++hk) {
                std::vector<float> kv_vec;
                apply_qknorm_rope(&t_k[(m * H_kv + hk) * D], h_kn_w.data(), m, kv_vec);
                std::copy(kv_vec.begin(), kv_vec.end(), &t_ref_k[(m * H_kv + hk) * D]);
                for (int d = 0; d < D; ++d) {
                    t_ref_v[(m * H_kv + hk) * D + d] = __half2float(t_v[(m * H_kv + hk) * D + d]);
                }
            }
        }
        for (int m = 0; m < M_t_tot; ++m) {
            for (int hq = 0; hq < H_q; ++hq) {
                const int hk = hq / (H_q / H_kv);
                std::vector<float> sc(m + 1);
                float mx = -1e30f;
                for (int t = 0; t <= m; ++t) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) dot += t_ref_q[(m * H_q + hq) * D + d] * t_ref_k[(t * H_kv + hk) * D + d];
                    sc[t] = dot * scale;
                    mx = std::max(mx, sc[t]);
                }
                float ls = 0.0f;
                for (int t = 0; t <= m; ++t) { sc[t] = std::exp(sc[t] - mx); ls += sc[t]; }
                for (int d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int t = 0; t <= m; ++t) acc += (sc[t] / ls) * t_ref_v[(t * H_kv + hk) * D + d];
                    const float gate = __half2float(t_q[(m * H_q + hq) * 2 * D + D + d]);
                    const float val = acc * sigmoid_cpu(gate);
                    if (m < M_t_pre) t_ref_pre[(m * H_q + hq) * D + d] = val;
                    else t_ref_dec[hq * D + d] = val;
                }
            }
        }

        for (rxinfer::DType kv_dt : {rxinfer::DType::F16, rxinfer::DType::FP4_E2M1_G16}) {
            p.kv_dtype = kv_dt;
            const auto lay = rxinfer::attn::compute_kv_page_layout(p);

            __half *d_q_in, *d_k_in, *d_v_in, *d_qn_w, *d_kn_w, *d_q_out, *d_out;
            uint8_t *d_vram_pool = nullptr, *d_gtt_pool = nullptr, *d_stage = nullptr;
            void *h_gtt_pool = nullptr, *d_split = nullptr;
            uint8_t** d_pt = nullptr;

            RX_HIP_CHECK(hipMalloc(&d_q_in, t_q.size() * sizeof(__half)));
            RX_HIP_CHECK(hipMalloc(&d_k_in, t_k.size() * sizeof(__half)));
            RX_HIP_CHECK(hipMalloc(&d_v_in, t_v.size() * sizeof(__half)));
            RX_HIP_CHECK(hipMalloc(&d_qn_w, D * sizeof(__half)));
            RX_HIP_CHECK(hipMalloc(&d_kn_w, D * sizeof(__half)));
            RX_HIP_CHECK(hipMalloc(&d_q_out, M_t_tot * H_q * D * sizeof(__half)));
            RX_HIP_CHECK(hipMalloc(&d_out, M_t_tot * H_q * D * sizeof(__half)));
            RX_HIP_CHECK(hipMalloc(&d_vram_pool, static_cast<size_t>(vram_pages) * lay.page_bytes));
            RX_HIP_CHECK(hipHostMalloc(&h_gtt_pool, static_cast<size_t>(gtt_pages) * lay.page_bytes, hipHostMallocMapped));
            RX_HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&d_gtt_pool), h_gtt_pool, 0));
            RX_HIP_CHECK(hipMalloc(&d_stage, lay.page_bytes)); // Force 1-page staging rounds across 2 GTT pages!
            RX_HIP_CHECK(hipMalloc(&d_pt, static_cast<size_t>(num_pages) * sizeof(uint8_t*)));

            const size_t dec_sb = rxinfer::attn::decode_splitk_scratch_bytes(p, 1, 64);
            const size_t pre_sb = rxinfer::attn::prefill_splitk_scratch_bytes(p, M_t_pre, 64, 4);
            RX_HIP_CHECK(hipMalloc(&d_split, std::max(dec_sb, pre_sb)));

            std::vector<uint8_t*> h_ptrs = {
                d_vram_pool,
                d_gtt_pool,
                d_gtt_pool + lay.page_bytes,
            };
            RX_HIP_CHECK(hipMemcpy(d_pt, h_ptrs.data(), static_cast<size_t>(num_pages) * sizeof(uint8_t*),
                                   hipMemcpyHostToDevice));
            RX_HIP_CHECK(hipMemcpy(d_q_in, t_q.data(), t_q.size() * sizeof(__half), hipMemcpyHostToDevice));
            RX_HIP_CHECK(hipMemcpy(d_k_in, t_k.data(), t_k.size() * sizeof(__half), hipMemcpyHostToDevice));
            RX_HIP_CHECK(hipMemcpy(d_v_in, t_v.data(), t_v.size() * sizeof(__half), hipMemcpyHostToDevice));
            RX_HIP_CHECK(hipMemcpy(d_qn_w, h_qn_w.data(), D * sizeof(__half), hipMemcpyHostToDevice));
            RX_HIP_CHECK(hipMemcpy(d_kn_w, h_kn_w.data(), D * sizeof(__half), hipMemcpyHostToDevice));

            rxinfer::attn::KVCacheView kv{
                d_pt, num_pages, vram_pages, lay.page_bytes, lay.data_sec_bytes, lay.sc_sec_bytes};
            rxinfer::attn::PrefillScratch pscratch{
                static_cast<float*>(d_split), M_t_pre, 64, 4, d_stage, 1};

            rxinfer::attn::fused_qknorm_rope_kv_write(d_q_in, d_k_in, d_v_in, d_qn_w, d_kn_w, d_q_out,
                                                      kv, p, M_t_pre, 0, nullptr, stream);
            rxinfer::attn::flash_attention_prefill(d_q_out, d_q_in, kv, d_out, p, M_t_pre, 0, &pscratch, stream);

            std::vector<__half> h_pre(M_t_pre * H_q * D);
            RX_HIP_CHECK(hipMemcpyAsync(h_pre.data(), d_out, h_pre.size() * sizeof(__half), hipMemcpyDeviceToHost, stream));

            rxinfer::attn::fused_qknorm_rope_kv_write(
                d_q_in + M_t_pre * H_q * 2 * D,
                d_k_in + M_t_pre * H_kv * D,
                d_v_in + M_t_pre * H_kv * D,
                d_qn_w, d_kn_w, d_q_out, kv, p, 1, M_t_pre, nullptr, stream);
            rxinfer::attn::gqa_decode_attention(
                d_q_out, d_q_in + M_t_pre * H_q * 2 * D, kv, d_out, d_split, p, 1, M_t_tot, nullptr, 64, stream);

            std::vector<__half> h_dec(H_q * D);
            RX_HIP_CHECK(hipMemcpyAsync(h_dec.data(), d_out, h_dec.size() * sizeof(__half), hipMemcpyDeviceToHost, stream));
            RX_HIP_CHECK(hipStreamSynchronize(stream));

            auto mt_pre = eval_f16_vs_f32(t_ref_pre, h_pre);
            auto mt_dec = eval_f16_vs_f32(t_ref_dec, h_dec);
            std::printf("[Tiered VRAM+GTT %-12s] prefill(160) cos=%.5f | decode(161) cos=%.5f\n",
                        std::string(rxinfer::dtype_name(kv_dt)).c_str(), mt_pre.cos_sim, mt_dec.cos_sim);
            const double min_cos = (kv_dt == rxinfer::DType::FP4_E2M1_G16) ? 0.985 : 0.999;
            if (mt_pre.cos_sim < min_cos || mt_dec.cos_sim < min_cos) ++failures;

            RX_HIP_CHECK(hipFree(d_q_in));
            RX_HIP_CHECK(hipFree(d_k_in));
            RX_HIP_CHECK(hipFree(d_v_in));
            RX_HIP_CHECK(hipFree(d_qn_w));
            RX_HIP_CHECK(hipFree(d_kn_w));
            RX_HIP_CHECK(hipFree(d_q_out));
            RX_HIP_CHECK(hipFree(d_out));
            RX_HIP_CHECK(hipFree(d_vram_pool));
            RX_HIP_CHECK(hipHostFree(h_gtt_pool));
            RX_HIP_CHECK(hipFree(d_stage));
            RX_HIP_CHECK(hipFree(d_pt));
            RX_HIP_CHECK(hipFree(d_split));
        }
    }

    // Long-Context RoPE Precision Test (pos = 200000, double-precision freq + 2*pi reduction)
    {
        const int32_t test_pos = 200000;
        p.kv_dtype = rxinfer::DType::F16;
        const auto lay = rxinfer::attn::compute_kv_page_layout(p);
        std::vector<float> ref_q_200k(H_q * D);
        for (int hq = 0; hq < H_q; ++hq) {
            std::vector<float> qv(D);
            float ss = 0.0f;
            for (int d = 0; d < D; ++d) {
                qv[d] = __half2float(h_q_in[hq * 2 * D + d]);
                ss += qv[d] * qv[d];
            }
            const float inv = 1.0f / std::sqrt(ss / D + p.rms_norm_eps);
            for (int d = 0; d < D; ++d) qv[d] *= inv * (1.0f + __half2float(h_qn_w[d]));
            const int hr = p.rotary_dim / 2;
            for (int i = 0; i < hr; ++i) {
                const double inv_freq = std::pow(static_cast<double>(p.rope_theta),
                                                 -2.0 * static_cast<double>(i) / static_cast<double>(p.rotary_dim));
                const double ang = static_cast<double>(test_pos) * inv_freq;
                const double c = std::cos(ang), s = std::sin(ang);
                const double x0 = qv[i], x1 = qv[i + hr];
                qv[i] = static_cast<float>(x0 * c - x1 * s);
                qv[i + hr] = static_cast<float>(x1 * c + x0 * s);
            }
            std::copy(qv.begin(), qv.end(), &ref_q_200k[hq * D]);
        }

        __half *d_q_in, *d_k_in, *d_v_in, *d_qn_w, *d_kn_w, *d_q_out;
        uint8_t* d_pg = nullptr;
        uint8_t** d_pt = nullptr;
        rxinfer::StepParams* d_step = nullptr;
        const int32_t pg_idx = test_pos / rxinfer::attn::KV_PAGE_TOKENS;
        std::vector<uint8_t*> h_pt(pg_idx + 1, nullptr);

        RX_HIP_CHECK(hipMalloc(&d_q_in, H_q * 2 * D * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_k_in, H_kv * D * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_v_in, H_kv * D * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_qn_w, D * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_kn_w, D * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_q_out, H_q * D * sizeof(__half)));
        RX_HIP_CHECK(hipMalloc(&d_pg, lay.page_bytes));
        RX_HIP_CHECK(hipMalloc(&d_pt, static_cast<size_t>(pg_idx + 1) * sizeof(uint8_t*)));
        RX_HIP_CHECK(hipMalloc(&d_step, sizeof(rxinfer::StepParams)));
        h_pt[pg_idx] = d_pg;

        RX_HIP_CHECK(hipMemcpy(d_pt, h_pt.data(), static_cast<size_t>(pg_idx + 1) * sizeof(uint8_t*),
                               hipMemcpyHostToDevice));
        RX_HIP_CHECK(hipMemcpy(d_q_in, h_q_in.data(), H_q * 2 * D * sizeof(__half), hipMemcpyHostToDevice));
        RX_HIP_CHECK(hipMemcpy(d_k_in, h_k_in.data(), H_kv * D * sizeof(__half), hipMemcpyHostToDevice));
        RX_HIP_CHECK(hipMemcpy(d_v_in, h_v_in.data(), H_kv * D * sizeof(__half), hipMemcpyHostToDevice));
        RX_HIP_CHECK(hipMemcpy(d_qn_w, h_qn_w.data(), D * sizeof(__half), hipMemcpyHostToDevice));
        RX_HIP_CHECK(hipMemcpy(d_kn_w, h_kn_w.data(), D * sizeof(__half), hipMemcpyHostToDevice));

        rxinfer::StepParams sp{};
        sp.pos = test_pos;
        sp.kv_seq_len = test_pos + 1;
        RX_HIP_CHECK(hipMemcpy(d_step, &sp, sizeof(sp), hipMemcpyHostToDevice));

        rxinfer::attn::KVCacheView kv{
            d_pt, pg_idx + 1, pg_idx + 1, lay.page_bytes, lay.data_sec_bytes, lay.sc_sec_bytes};
        rxinfer::attn::fused_qknorm_rope_kv_write(d_q_in, d_k_in, d_v_in, d_qn_w, d_kn_w, d_q_out,
                                                  kv, p, 1, 0, d_step, stream);
        std::vector<__half> h_q200k(H_q * D);
        RX_HIP_CHECK(hipMemcpyAsync(h_q200k.data(), d_q_out, h_q200k.size() * sizeof(__half),
                                    hipMemcpyDeviceToHost, stream));
        RX_HIP_CHECK(hipStreamSynchronize(stream));

        auto m_200k = eval_f16_vs_f32(ref_q_200k, h_q200k);
        std::printf("[RoPE pos=200000] cos=%.7f max_abs=%.6f\n", m_200k.cos_sim, m_200k.max_abs);
        if (m_200k.cos_sim < 0.99999) ++failures;

        RX_HIP_CHECK(hipFree(d_q_in));
        RX_HIP_CHECK(hipFree(d_k_in));
        RX_HIP_CHECK(hipFree(d_v_in));
        RX_HIP_CHECK(hipFree(d_qn_w));
        RX_HIP_CHECK(hipFree(d_kn_w));
        RX_HIP_CHECK(hipFree(d_q_out));
        RX_HIP_CHECK(hipFree(d_pg));
        RX_HIP_CHECK(hipFree(d_pt));
        RX_HIP_CHECK(hipFree(d_step));
    }
}

void test_gdn_variant(hipStream_t stream, int& failures, bool tiled_v_heads) {
    std::mt19937 rng(tiled_v_heads ? 1001u : 999u);
    std::normal_distribution<float> nd(0.0f, 0.3f);

    rxinfer::gdn::GDNParams p{};
    p.num_k_heads = 16;
    p.head_k_dim = 128;
    p.num_v_heads = 48;
    p.head_v_dim = 128;
    p.conv_kernel_size = 4;
    p.rms_norm_eps = 1e-6f;
    p.tiled_v_heads = tiled_v_heads;

    const int M_pre = 10;
    const int M_tot = M_pre + 1;
    const int conv_dim = p.conv_dim();
    const int value_dim = p.value_dim();
    const int in_stride = conv_dim + value_dim; // Test packed [qkvz] stride directly!

    std::vector<__half> h_qkvz(M_tot * in_stride);
    std::vector<__half> h_ba(M_tot * 2 * p.num_v_heads);
    std::vector<__half> h_conv_w(p.conv_kernel_size * conv_dim);
    std::vector<float> h_A_log(p.num_v_heads), h_dt_bias(p.num_v_heads);
    std::vector<__half> h_norm_w(p.head_v_dim);

    for (auto& v : h_qkvz) v = __float2half(nd(rng));
    for (auto& v : h_ba) v = __float2half(nd(rng));
    for (auto& v : h_conv_w) v = __float2half(nd(rng) * 0.25f);
    for (int h = 0; h < p.num_v_heads; ++h) {
        h_A_log[h] = nd(rng) * 0.5f;
        h_dt_bias[h] = nd(rng) * 0.5f;
    }
    for (int d = 0; d < p.head_v_dim; ++d) {
        h_norm_w[d] = __float2half(1.0f + nd(rng) * 0.2f);
    }

    // CPU FP32 reference across M_pre + 1 steps
    std::vector<float> ref_conv_state((p.conv_kernel_size - 1) * conv_dim, 0.0f);
    std::vector<float> ref_S(p.num_v_heads * p.head_k_dim * p.head_v_dim, 0.0f);
    std::vector<float> ref_out(M_tot * value_dim, 0.0f);

    for (int m = 0; m < M_tot; ++m) {
        std::vector<float> qkv_step(conv_dim);
        for (int c = 0; c < conv_dim; ++c) {
            float x_new = __half2float(h_qkvz[m * in_stride + c]);
            float acc = x_new * __half2float(h_conv_w[3 * conv_dim + c]);
            for (int k = 0; k < 3; ++k) {
                acc += ref_conv_state[k * conv_dim + c] * __half2float(h_conv_w[k * conv_dim + c]);
            }
            ref_conv_state[0 * conv_dim + c] = ref_conv_state[1 * conv_dim + c];
            ref_conv_state[1 * conv_dim + c] = ref_conv_state[2 * conv_dim + c];
            ref_conv_state[2 * conv_dim + c] = x_new;
            qkv_step[c] = silu_cpu(acc);
        }

        const int key_dim = p.key_dim();
        const int gqa = p.num_v_heads / p.num_k_heads;
        for (int h = 0; h < p.num_v_heads; ++h) {
            const int hk = tiled_v_heads ? (h % p.num_k_heads) : (h / gqa);
            float q[128], k[128], v[128], z[128];
            float q2 = 0.0f, k2 = 0.0f;
            for (int i = 0; i < 128; ++i) {
                q[i] = qkv_step[hk * 128 + i];
                k[i] = qkv_step[key_dim + hk * 128 + i];
                v[i] = qkv_step[2 * key_dim + h * 128 + i];
                z[i] = __half2float(h_qkvz[m * in_stride + conv_dim + h * 128 + i]);
                q2 += q[i] * q[i];
                k2 += k[i] * k[i];
            }
            const float iq = 1.0f / std::sqrt(q2 + 1e-6f) / std::sqrt(128.0f);
            const float ik = 1.0f / std::sqrt(k2 + 1e-6f);
            for (int i = 0; i < 128; ++i) {
                q[i] *= iq;
                k[i] *= ik;
            }

            const float b_val = __half2float(h_ba[m * 2 * p.num_v_heads + h]);
            const float a_val = __half2float(h_ba[m * 2 * p.num_v_heads + p.num_v_heads + h]);
            const float g = -std::exp(h_A_log[h]) * softplus_cpu(a_val + h_dt_bias[h]);
            const float decay = std::exp(g);
            const float beta = 1.0f / (1.0f + std::exp(-b_val));

            float* Sh = &ref_S[h * 128 * 128];
            for (int idx = 0; idx < 128 * 128; ++idx) Sh[idx] *= decay;

            float y[128]{};
            for (int j = 0; j < 128; ++j) {
                float kv_mem = 0.0f;
                for (int i = 0; i < 128; ++i) kv_mem += Sh[i * 128 + j] * k[i];
                const float delta = (v[j] - kv_mem) * beta;
                for (int i = 0; i < 128; ++i) {
                    Sh[i * 128 + j] += k[i] * delta;
                    y[j] += Sh[i * 128 + j] * q[i];
                }
            }

            float y2 = 0.0f;
            for (int j = 0; j < 128; ++j) y2 += y[j] * y[j];
            const float inv = 1.0f / std::sqrt(y2 / 128.0f + p.rms_norm_eps);
            for (int j = 0; j < 128; ++j) {
                ref_out[m * value_dim + h * 128 + j] =
                    y[j] * inv * __half2float(h_norm_w[j]) * silu_cpu(z[j]);
            }
        }
    }

    __half *d_qkvz, *d_ba, *d_conv_w, *d_conv_state, *d_qkv_out, *d_norm_w, *d_out;
    float *d_A_log, *d_dt_bias, *d_S;

    RX_HIP_CHECK(hipMalloc(&d_qkvz, h_qkvz.size() * sizeof(__half)));
    RX_HIP_CHECK(hipMalloc(&d_ba, h_ba.size() * sizeof(__half)));
    RX_HIP_CHECK(hipMalloc(&d_conv_w, h_conv_w.size() * sizeof(__half)));
    RX_HIP_CHECK(hipMalloc(&d_conv_state, rxinfer::gdn::gdn_conv_state_bytes(p)));
    RX_HIP_CHECK(hipMalloc(&d_qkv_out, M_pre * conv_dim * sizeof(__half)));
    RX_HIP_CHECK(hipMalloc(&d_norm_w, p.head_v_dim * sizeof(__half)));
    RX_HIP_CHECK(hipMalloc(&d_out, M_tot * value_dim * sizeof(__half)));
    RX_HIP_CHECK(hipMalloc(&d_A_log, p.num_v_heads * sizeof(float)));
    RX_HIP_CHECK(hipMalloc(&d_dt_bias, p.num_v_heads * sizeof(float)));
    RX_HIP_CHECK(hipMalloc(&d_S, rxinfer::gdn::gdn_recurrent_state_bytes(p)));

    RX_HIP_CHECK(hipMemset(d_conv_state, 0, rxinfer::gdn::gdn_conv_state_bytes(p)));
    RX_HIP_CHECK(hipMemset(d_S, 0, rxinfer::gdn::gdn_recurrent_state_bytes(p)));
    RX_HIP_CHECK(hipMemcpy(d_qkvz, h_qkvz.data(), h_qkvz.size() * sizeof(__half), hipMemcpyHostToDevice));
    RX_HIP_CHECK(hipMemcpy(d_ba, h_ba.data(), h_ba.size() * sizeof(__half), hipMemcpyHostToDevice));
    RX_HIP_CHECK(hipMemcpy(d_conv_w, h_conv_w.data(), h_conv_w.size() * sizeof(__half), hipMemcpyHostToDevice));
    RX_HIP_CHECK(hipMemcpy(d_norm_w, h_norm_w.data(), p.head_v_dim * sizeof(__half), hipMemcpyHostToDevice));
    RX_HIP_CHECK(hipMemcpy(d_A_log, h_A_log.data(), p.num_v_heads * sizeof(float), hipMemcpyHostToDevice));
    RX_HIP_CHECK(hipMemcpy(d_dt_bias, h_dt_bias.data(), p.num_v_heads * sizeof(float), hipMemcpyHostToDevice));

    // Step 1: Prefill M_pre tokens
    rxinfer::gdn::conv1d_silu(d_qkvz, in_stride, d_conv_w, nullptr, d_conv_state, d_qkv_out, p, M_pre, stream);
    rxinfer::gdn::fused_gdn_gated_norm(
        d_qkv_out, d_qkvz + conv_dim, in_stride,
        d_ba, d_ba + p.num_v_heads, 2 * p.num_v_heads,
        d_A_log, d_dt_bias, d_norm_w, d_S, d_out, p, M_pre, stream);

    // Step 2: Decode 1 token (M=1)
    const __half* d_qkvz_1 = d_qkvz + M_pre * in_stride;
    const __half* d_ba_1 = d_ba + M_pre * 2 * p.num_v_heads;
    rxinfer::gdn::conv1d_silu(d_qkvz_1, in_stride, d_conv_w, nullptr, d_conv_state, d_qkv_out, p, 1, stream);
    rxinfer::gdn::fused_gdn_gated_norm(
        d_qkv_out, d_qkvz_1 + conv_dim, in_stride,
        d_ba_1, d_ba_1 + p.num_v_heads, 2 * p.num_v_heads,
        d_A_log, d_dt_bias, d_norm_w, d_S, d_out + M_pre * value_dim, p, 1, stream);

    std::vector<__half> h_got_out(M_tot * value_dim);
    std::vector<float> h_got_S(ref_S.size());
    RX_HIP_CHECK(hipMemcpyAsync(h_got_out.data(), d_out, h_got_out.size() * sizeof(__half), hipMemcpyDeviceToHost, stream));
    RX_HIP_CHECK(hipMemcpyAsync(h_got_S.data(), d_S, h_got_S.size() * sizeof(float), hipMemcpyDeviceToHost, stream));
    RX_HIP_CHECK(hipStreamSynchronize(stream));

    auto m_out = eval_f16_vs_f32(ref_out, h_got_out);
    auto m_S = eval_f32_vs_f32(ref_S, h_got_S);

    const float us_conv = time_us([&] {
        rxinfer::gdn::conv1d_silu(d_qkvz_1, in_stride, d_conv_w, nullptr, d_conv_state, d_qkv_out, p, 1, stream);
    }, stream);
    const float us_gdn = time_us([&] {
        rxinfer::gdn::fused_gdn_gated_norm(
            d_qkv_out, d_qkvz_1 + conv_dim, in_stride,
            d_ba_1, d_ba_1 + p.num_v_heads, 2 * p.num_v_heads,
            d_A_log, d_dt_bias, d_norm_w, d_S, d_out, p, 1, stream);
    }, stream);

    std::printf("[GDN 48x128x128 %-7s] out cos=%.6f rel=%.5f | S cos=%.6f | M=1 conv=%.1fus + gdn=%.1fus\n",
                tiled_v_heads ? "tiled" : "grouped",
                m_out.cos_sim, m_out.rel_rmse, m_S.cos_sim, us_conv, us_gdn);
    if (m_out.cos_sim < 0.9995 || m_S.cos_sim < 0.9999) ++failures;

    RX_HIP_CHECK(hipFree(d_qkvz));
    RX_HIP_CHECK(hipFree(d_ba));
    RX_HIP_CHECK(hipFree(d_conv_w));
    RX_HIP_CHECK(hipFree(d_conv_state));
    RX_HIP_CHECK(hipFree(d_qkv_out));
    RX_HIP_CHECK(hipFree(d_norm_w));
    RX_HIP_CHECK(hipFree(d_out));
    RX_HIP_CHECK(hipFree(d_A_log));
    RX_HIP_CHECK(hipFree(d_dt_bias));
    RX_HIP_CHECK(hipFree(d_S));
}

void test_gdn(hipStream_t stream, int& failures) {
    test_gdn_variant(stream, failures, false);
    test_gdn_variant(stream, failures, true);
}

} // namespace

int main() {
    auto info = rxinfer::query_device_info(0);
    std::printf("GPU: %s (%s) | WGPs: %d\n\n", info.name.c_str(), info.gcn_arch_name.c_str(), info.multiprocessor_count);

    hipStream_t stream = nullptr;
    RX_HIP_CHECK(hipStreamCreate(&stream));

    int failures = 0;
    test_norm_swiglu_embed_argmax(stream, failures);
    test_attention(stream, failures);
    test_gdn(stream, failures);

    RX_HIP_CHECK(hipStreamDestroy(stream));
    if (failures == 0) {
        std::printf("\nALL KERNEL TESTS PASSED!\n");
        return 0;
    } else {
        std::printf("\nFAILED TESTS: %d\n", failures);
        return 1;
    }
}

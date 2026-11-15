// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/engine.hpp"
#include "rxinfer/common.hpp"
#include "rxinfer/ops.hpp"
#include "rxinfer/quant.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace rxinfer {

InferenceEngine::InferenceEngine(ModelWeights weights, EngineOptions opts)
    : weights_(std::move(weights)), opts_(opts) {
    if (opts_.max_seq_len <= 0) opts_.max_seq_len = weights_.config.max_position_embeddings;
    if (opts_.max_verify_tokens < 1) opts_.max_verify_tokens = 1;
    if (opts_.max_verify_tokens > 16) opts_.max_verify_tokens = 16;
    if (opts_.max_batch_tokens <= 0) opts_.max_batch_tokens = 256;
    if (opts_.max_batch_tokens < opts_.max_verify_tokens) opts_.max_batch_tokens = opts_.max_verify_tokens;

    RX_HIP_CHECK(hipStreamCreate(&stream_));
    init_arenas();
    reset_state();
    if (opts_.enable_hip_graph) {
        capture_decode_graph();
    }
}

InferenceEngine::~InferenceEngine() {
    for (auto& kv : verify_graph_execs_) {
        if (kv.second) (void)hipGraphExecDestroy(kv.second);
    }
    for (auto& kv : verify_graphs_) {
        if (kv.second) (void)hipGraphDestroy(kv.second);
    }
    if (decode_graph_exec_) (void)hipGraphExecDestroy(decode_graph_exec_);
    if (decode_graph_) (void)hipGraphDestroy(decode_graph_);
    if (stream_) (void)hipStreamDestroy(stream_);
}

void InferenceEngine::init_arenas() {
    const ModelConfig& c = weights_.config;
    const int32_t H = c.hidden_size;
    const int32_t I = c.intermediate_size;
    const int32_t V = c.vocab_size;
    const int32_t L = c.num_hidden_layers;
    const int32_t B = opts_.max_batch_tokens;

    attn_params_.num_q_heads = c.num_attention_heads;
    attn_params_.num_kv_heads = c.num_key_value_heads;
    attn_params_.head_dim = c.head_dim;
    attn_params_.rotary_dim = c.rotary_dim();
    attn_params_.rope_theta = c.rope_theta;
    attn_params_.rms_norm_eps = c.rms_norm_eps;
    attn_params_.attn_scale = 0.0f;
    attn_params_.rope_kind = c.rope_kind;
    attn_params_.qk_norm = c.qk_norm;
    attn_params_.qk_norm_plus_one = (c.norm_kind == NormKind::RMSNormPlusOne);
    attn_params_.attn_output_gate = c.attn_output_gate;
    attn_params_.kv_dtype = opts_.kv_dtype;
    attn_params_.max_seq_len = opts_.max_seq_len;
    attn_params_.mrope_interleaved = c.mrope_interleaved;
    for (size_t i = 0; i < 3 && i < c.mrope_section.size(); ++i) {
        attn_params_.mrope_section[i] = c.mrope_section[i];
    }

    gdn_params_.num_k_heads = c.linear_num_key_heads;
    gdn_params_.head_k_dim = c.linear_key_head_dim;
    gdn_params_.num_v_heads = c.linear_num_value_heads;
    gdn_params_.head_v_dim = c.linear_value_head_dim;
    gdn_params_.conv_kernel_size = c.linear_conv_kernel_dim;
    gdn_params_.rms_norm_eps = c.rms_norm_eps;
    gdn_params_.tiled_v_heads = c.gdn_tiled_v_heads;

    attn_idx_map_.assign(L, -1);
    gdn_idx_map_.assign(L, -1);
    gdn_layer_idx_.clear();
    int32_t n_attn = 0, n_gdn = 0;
    for (int32_t l = 0; l < L; ++l) {
        if (weights_.layers[l].kind == LayerKind::FullAttention) {
            attn_idx_map_[l] = n_attn++;
        } else {
            gdn_idx_map_[l] = n_gdn++;
            gdn_layer_idx_.push_back(l);
        }
    }

    // 1. State Arena: KV caches + contiguous GDN conv/recurrent state (+ optional speculative backup)
    const size_t kv_data_b = (n_attn > 0) ? attn::kv_cache_data_bytes(attn_params_) : 0;
    const size_t kv_sc_b = (n_attn > 0) ? attn::kv_cache_scales_bytes(attn_params_) : 0;
    const size_t gdn_conv_b = (n_gdn > 0) ? gdn::gdn_conv_state_bytes(gdn_params_) : 0;
    const size_t gdn_rec_b = (n_gdn > 0) ? gdn::gdn_recurrent_state_bytes(gdn_params_) : 0;

    const size_t per_attn_bytes = 2 * align_up(kv_data_b, 256) + 2 * align_up(kv_sc_b, 256);
    const size_t per_gdn_bytes = align_up(gdn_conv_b, 256) + align_up(gdn_rec_b, 256);
    gdn_total_bytes_ = static_cast<size_t>(n_gdn) * per_gdn_bytes;
    const bool alloc_gdn_backup = (n_gdn > 0 && opts_.max_verify_tokens > 1);
    const size_t state_total = static_cast<size_t>(n_attn) * per_attn_bytes +
                               gdn_total_bytes_ +
                               (alloc_gdn_backup ? gdn_total_bytes_ : 0) + 4096;

    state_arena_.init(state_total, "state");
    kv_caches_.resize(n_attn);
    for (int32_t i = 0; i < n_attn; ++i) {
        kv_caches_[i].k_data = state_arena_.alloc(kv_data_b, 256);
        kv_caches_[i].v_data = state_arena_.alloc(kv_data_b, 256);
        kv_caches_[i].k_scales = (kv_sc_b > 0) ? state_arena_.alloc(kv_sc_b, 256) : nullptr;
        kv_caches_[i].v_scales = (kv_sc_b > 0) ? state_arena_.alloc(kv_sc_b, 256) : nullptr;
    }
    gdn_states_.resize(n_gdn);
    if (n_gdn > 0) {
        d_gdn_base_ = state_arena_.alloc(gdn_total_bytes_, 256);
        uint8_t* ptr = static_cast<uint8_t*>(d_gdn_base_);
        for (int32_t i = 0; i < n_gdn; ++i) {
            gdn_states_[i].conv_state = reinterpret_cast<__half*>(ptr);
            ptr += align_up(gdn_conv_b, 256);
            gdn_states_[i].recurrent_state = reinterpret_cast<float*>(ptr);
            ptr += align_up(gdn_rec_b, 256);
        }
        if (alloc_gdn_backup) {
            d_gdn_backup_ = state_arena_.alloc(gdn_total_bytes_, 256);
        }
    }

    // 2. Scratch Arena: reusable activation buffers + GEMM split-K + logits + verify replay
    const int32_t MV = opts_.max_verify_tokens;
    const int32_t q_dim = (n_attn > 0) ? (c.num_attention_heads * c.head_dim) : 0;
    const int32_t q_out = q_dim * (c.attn_output_gate ? 2 : 1);
    const int32_t kv_out = (n_attn > 0) ? (2 * c.num_key_value_heads * c.head_dim) : 0;
    const int32_t gdn_conv_dim = (n_gdn > 0) ? gdn_params_.conv_dim() : 0;
    const int32_t gdn_val_dim = (n_gdn > 0) ? gdn_params_.value_dim() : 0;
    const int32_t gdn_qkvz = gdn_conv_dim + gdn_val_dim;

    const int32_t max_proj_a = std::max({q_out, gdn_qkvz, 2 * I, 1});
    const int32_t max_proj_b = std::max({kv_out, gdn_conv_dim, I, 1});
    const int32_t max_attn_dim = std::max({q_dim, gdn_val_dim, 1});
    const int32_t max_small_gemm_N = std::max({H, q_out, kv_out, 2 * I, gdn_val_dim, (V < 16384 ? V : 1), 1});

    const size_t split_bytes = (n_attn > 0) ? attn::decode_splitk_scratch_bytes(attn_params_, 16, 16) : 256;
    const size_t tmp_bytes = quant::r4_g64_tmp_bytes(16, max_small_gemm_N);
    const size_t logits_bytes = align_up(static_cast<size_t>(MV) * static_cast<size_t>(V) * sizeof(__half), 256);
    const size_t argmax_b = ops::argmax_tmp_bytes(MV);
    const size_t replay_qkv_b = alloc_gdn_backup ? (static_cast<size_t>(n_gdn) * MV * gdn_conv_dim * sizeof(__half)) : 0;
    const size_t replay_ba_b = alloc_gdn_backup ? (static_cast<size_t>(n_gdn) * MV * (2 * c.linear_num_value_heads) * sizeof(__half)) : 0;

    const size_t scratch_total =
        align_up(sizeof(StepParams), 256) +
        align_up(static_cast<size_t>(MV) * sizeof(int32_t), 256) +
        align_up(static_cast<size_t>(B) * sizeof(int32_t), 256) +
        3 * align_up(static_cast<size_t>(B) * H * sizeof(__half), 256) +
        align_up(static_cast<size_t>(B) * max_proj_a * sizeof(__half), 256) +
        align_up(static_cast<size_t>(B) * max_proj_b * sizeof(__half), 256) +
        align_up(static_cast<size_t>(B) * max_attn_dim * sizeof(__half), 256) +
        align_up(static_cast<size_t>(B) * max_attn_dim * sizeof(__half), 256) +
        align_up(static_cast<size_t>(B) * std::max(1, 2 * c.linear_num_value_heads) * sizeof(__half), 256) +
        align_up(split_bytes, 256) +
        align_up(tmp_bytes, 256) +
        align_up(logits_bytes, 256) +
        align_up(argmax_b, 256) +
        align_up(replay_qkv_b, 256) +
        align_up(replay_ba_b, 256) +
        4096;

    scratch_arena_.init(scratch_total, "scratch");
    d_step_ = static_cast<StepParams*>(scratch_arena_.alloc(sizeof(StepParams), 256));
    d_out_token_ = static_cast<int32_t*>(scratch_arena_.alloc(static_cast<size_t>(MV) * sizeof(int32_t), 256));
    d_tokens_in_ = static_cast<int32_t*>(scratch_arena_.alloc(static_cast<size_t>(B) * sizeof(int32_t), 256));
    d_x_ = static_cast<__half*>(scratch_arena_.alloc(static_cast<size_t>(B) * H * sizeof(__half), 256));
    d_residual_ = static_cast<__half*>(scratch_arena_.alloc(static_cast<size_t>(B) * H * sizeof(__half), 256));
    d_norm_out_ = static_cast<__half*>(scratch_arena_.alloc(static_cast<size_t>(B) * H * sizeof(__half), 256));
    d_proj_a_ = static_cast<__half*>(scratch_arena_.alloc(static_cast<size_t>(B) * max_proj_a * sizeof(__half), 256));
    d_proj_b_ = static_cast<__half*>(scratch_arena_.alloc(static_cast<size_t>(B) * max_proj_b * sizeof(__half), 256));
    d_proj_c_ = static_cast<__half*>(scratch_arena_.alloc(static_cast<size_t>(B) * max_attn_dim * sizeof(__half), 256));
    d_attn_out_ = static_cast<__half*>(scratch_arena_.alloc(static_cast<size_t>(B) * max_attn_dim * sizeof(__half), 256));
    d_ba_buf_ = static_cast<__half*>(
        scratch_arena_.alloc(static_cast<size_t>(B) * std::max(1, 2 * c.linear_num_value_heads) * sizeof(__half), 256));
    d_attn_split_ = scratch_arena_.alloc(split_bytes, 256);
    d_gemm_tmp_ = static_cast<float*>(scratch_arena_.alloc(tmp_bytes, 256));
    d_logits_ = static_cast<__half*>(scratch_arena_.alloc(logits_bytes, 256));
    d_argmax_tmp_ = scratch_arena_.alloc(argmax_b, 256);
    if (alloc_gdn_backup) {
        d_gdn_replay_qkv_ = static_cast<__half*>(scratch_arena_.alloc(replay_qkv_b, 256));
        d_gdn_replay_ba_ = static_cast<__half*>(scratch_arena_.alloc(replay_ba_b, 256));
    }
}

void InferenceEngine::reset_state() {
    cur_pos_ = 0;
    for (const auto& gs : gdn_states_) {
        RX_HIP_CHECK(hipMemsetAsync(gs.conv_state, 0, gdn::gdn_conv_state_bytes(gdn_params_), stream_));
        RX_HIP_CHECK(hipMemsetAsync(gs.recurrent_state, 0, gdn::gdn_recurrent_state_bytes(gdn_params_), stream_));
    }
    RX_HIP_CHECK(hipStreamSynchronize(stream_));
}

void InferenceEngine::run_layer_decode(int32_t l) {
    run_layer_decode_step(l, 1, false);
}

void InferenceEngine::run_layer_decode_step(int32_t l, int M, bool save_gdn_replay) {
    const ModelConfig& c = weights_.config;
    const LayerWeights& lw = weights_.layers[l];
    const int32_t H = c.hidden_size;
    const int32_t I = c.intermediate_size;
    const bool plus_one = (c.norm_kind == NormKind::RMSNormPlusOne);

    if (l == 0) {
        ops::rmsnorm(d_residual_, lw.input_norm, d_norm_out_, M, H, c.rms_norm_eps, plus_one, stream_);
    } else {
        ops::fused_residual_rmsnorm(d_x_, d_residual_, lw.input_norm, d_norm_out_,
                                    M, H, c.rms_norm_eps, plus_one, stream_);
    }

    if (lw.kind == LayerKind::FullAttention) {
        const int32_t kv_dim = c.num_key_value_heads * c.head_dim;
        const auto& kv = kv_caches_[attn_idx_map_[l]];

        quant::gemm_r4_gfx12_w4a16(d_norm_out_, lw.q_proj.qs, lw.q_proj.scales,
                                   d_proj_a_, d_gemm_tmp_, M, lw.q_proj.N, H, stream_);
        quant::gemm_r4_gfx12_w4a16(d_norm_out_, lw.kv_proj.qs, lw.kv_proj.scales,
                                   d_proj_b_, d_gemm_tmp_, M, lw.kv_proj.N, H, stream_);

        attn::fused_qknorm_rope_kv_write(d_proj_a_, d_proj_b_, d_proj_b_ + kv_dim,
                                         lw.q_norm, lw.k_norm, d_proj_c_, kv, attn_params_,
                                         M, 0, d_step_, stream_, 2 * kv_dim);

        attn::gqa_decode_attention(d_proj_c_, c.attn_output_gate ? d_proj_a_ : nullptr,
                                   kv, d_attn_out_, d_attn_split_, attn_params_,
                                   M, 0, d_step_, 16, stream_);

        quant::gemm_r4_gfx12_w4a16(d_attn_out_, lw.o_proj.qs, lw.o_proj.scales,
                                   d_x_, d_gemm_tmp_, M, H, lw.o_proj.K, stream_);
    } else {
        const int32_t conv_dim = gdn_params_.conv_dim();
        const int32_t value_dim = gdn_params_.value_dim();
        const int32_t Hv = gdn_params_.num_v_heads;
        const int32_t g_idx = gdn_idx_map_[l];
        const auto& gs = gdn_states_[g_idx];

        quant::gemm_r4_gfx12_w4a16(d_norm_out_, lw.in_proj_qkvz.qs, lw.in_proj_qkvz.scales,
                                   d_proj_a_, d_gemm_tmp_, M, lw.in_proj_qkvz.N, H, stream_);
        ops::gemm_f16(d_norm_out_, lw.in_proj_ba, d_ba_buf_, M, 2 * Hv, H, stream_);

        if (save_gdn_replay && d_gdn_replay_qkv_ != nullptr && d_gdn_replay_ba_ != nullptr) {
            const int32_t MV = opts_.max_verify_tokens;
            __half* qkv_dst = d_gdn_replay_qkv_ + static_cast<size_t>(g_idx) * MV * conv_dim;
            __half* ba_dst = d_gdn_replay_ba_ + static_cast<size_t>(g_idx) * MV * (2 * Hv);
            ops::copy_strided_f16(d_proj_a_, conv_dim + value_dim, qkv_dst, conv_dim, M, conv_dim, stream_);
            ops::copy_strided_f16(d_ba_buf_, 2 * Hv, ba_dst, 2 * Hv, M, 2 * Hv, stream_);
        }

        gdn::conv1d_silu(d_proj_a_, conv_dim + value_dim, lw.conv_w, nullptr,
                         gs.conv_state, d_proj_b_, gdn_params_, M, stream_);
        gdn::fused_gdn_gated_norm(d_proj_b_, d_proj_a_ + conv_dim, conv_dim + value_dim,
                                  d_ba_buf_, d_ba_buf_ + Hv, 2 * Hv,
                                  lw.A_log, lw.dt_bias, lw.gdn_norm_w,
                                  gs.recurrent_state, d_attn_out_, gdn_params_, M, stream_);

        quant::gemm_r4_gfx12_w4a16(d_attn_out_, lw.out_proj.qs, lw.out_proj.scales,
                                   d_x_, d_gemm_tmp_, M, H, value_dim, stream_);
    }

    ops::fused_residual_rmsnorm(d_x_, d_residual_, lw.post_attn_norm, d_norm_out_,
                                M, H, c.rms_norm_eps, plus_one, stream_);

    if ((I & 15) == 0 && (H & 63) == 0) {
        quant::gemv_r4_gfx12_w4a16_swiglu(d_norm_out_, lw.gate_up_proj.qs, lw.gate_up_proj.scales,
                                          d_proj_b_, I, H, stream_, M);
    } else {
        quant::gemm_r4_gfx12_w4a16(d_norm_out_, lw.gate_up_proj.qs, lw.gate_up_proj.scales,
                                   d_proj_a_, d_gemm_tmp_, M, 2 * I, H, stream_);
        ops::fused_silu_mul_packed(d_proj_a_, d_proj_b_, M, I, stream_);
    }
    quant::gemm_r4_gfx12_w4a16(d_proj_b_, lw.down_proj.qs, lw.down_proj.scales,
                               d_x_, d_gemm_tmp_, M, H, I, stream_);
}

hipGraphExec_t InferenceEngine::get_or_capture_verify_graph(int M) {
    auto it = verify_graph_execs_.find(M);
    if (it != verify_graph_execs_.end()) return it->second;

    const ModelConfig& c = weights_.config;
    const int32_t H = c.hidden_size;
    const int32_t V = c.vocab_size;
    const bool plus_one = (c.norm_kind == NormKind::RMSNormPlusOne);
    const bool save_replay = (d_gdn_backup_ != nullptr);

    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    RX_HIP_CHECK(hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal));

    ops::embed_lookup_r4_batch(weights_.embed_tokens.qs, weights_.embed_tokens.scales,
                               d_tokens_in_, d_residual_, M, V, H, stream_);

    for (int32_t l = 0; l < c.num_hidden_layers; ++l) {
        run_layer_decode_step(l, M, save_replay);
    }

    ops::fused_residual_rmsnorm(d_x_, d_residual_, weights_.final_norm, d_norm_out_,
                                M, H, c.rms_norm_eps, plus_one, stream_);

    quant::gemm_r4_gfx12_w4a16(d_norm_out_, weights_.lm_head.qs, weights_.lm_head.scales,
                               d_logits_, d_gemm_tmp_, M, V, H, stream_);

    ops::argmax_sample_batch(d_logits_, M, V, d_argmax_tmp_, d_out_token_, stream_);

    RX_HIP_CHECK(hipStreamEndCapture(stream_, &graph));
    RX_HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    verify_graphs_[M] = graph;
    verify_graph_execs_[M] = exec;
    return exec;
}

void InferenceEngine::capture_decode_graph() {
    const ModelConfig& c = weights_.config;
    const int32_t H = c.hidden_size;
    const int32_t V = c.vocab_size;
    const bool plus_one = (c.norm_kind == NormKind::RMSNormPlusOne);

    RX_HIP_CHECK(hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal));

    ops::embed_lookup_r4_step(weights_.embed_tokens.qs, weights_.embed_tokens.scales,
                              d_step_, d_residual_, V, H, stream_);

    for (int32_t l = 0; l < c.num_hidden_layers; ++l) {
        run_layer_decode(l);
    }

    ops::fused_residual_rmsnorm(d_x_, d_residual_, weights_.final_norm, d_norm_out_,
                                1, H, c.rms_norm_eps, plus_one, stream_);

    quant::gemm_r4_gfx12_w4a16(d_norm_out_, weights_.lm_head.qs, weights_.lm_head.scales,
                               d_logits_, d_gemm_tmp_, 1, V, H, stream_);

    ops::argmax_sample(d_logits_, V, d_argmax_tmp_, d_out_token_, d_step_, stream_);

    RX_HIP_CHECK(hipStreamEndCapture(stream_, &decode_graph_));
    RX_HIP_CHECK(hipGraphInstantiate(&decode_graph_exec_, decode_graph_, nullptr, nullptr, 0));
}

void InferenceEngine::run_forward_batch(const int32_t* h_tokens, int M, bool compute_logits) {
    const ModelConfig& c = weights_.config;
    const int32_t H = c.hidden_size;
    const int32_t I = c.intermediate_size;
    const int32_t V = c.vocab_size;
    const bool plus_one = (c.norm_kind == NormKind::RMSNormPlusOne);

    if (cur_pos_ + M > opts_.max_seq_len) {
        throw std::runtime_error("Context length exceeds engine max_seq_len");
    }

    RX_HIP_CHECK(hipMemcpyAsync(d_tokens_in_, h_tokens, static_cast<size_t>(M) * sizeof(int32_t),
                                hipMemcpyHostToDevice, stream_));

    ops::embed_lookup_r4_batch(weights_.embed_tokens.qs, weights_.embed_tokens.scales,
                               d_tokens_in_, d_residual_, M, V, H, stream_);

    for (int32_t l = 0; l < c.num_hidden_layers; ++l) {
        const LayerWeights& lw = weights_.layers[l];
        if (l == 0) {
            ops::rmsnorm(d_residual_, lw.input_norm, d_norm_out_, M, H, c.rms_norm_eps, plus_one, stream_);
        } else {
            ops::fused_residual_rmsnorm(d_x_, d_residual_, lw.input_norm, d_norm_out_,
                                        M, H, c.rms_norm_eps, plus_one, stream_);
        }

        if (lw.kind == LayerKind::FullAttention) {
            const int32_t kv_dim = c.num_key_value_heads * c.head_dim;
            const auto& kv = kv_caches_[attn_idx_map_[l]];

            quant::gemm_r4_gfx12_w4a16(d_norm_out_, lw.q_proj.qs, lw.q_proj.scales,
                                       d_proj_a_, d_gemm_tmp_, M, lw.q_proj.N, H, stream_);
            quant::gemm_r4_gfx12_w4a16(d_norm_out_, lw.kv_proj.qs, lw.kv_proj.scales,
                                       d_proj_b_, d_gemm_tmp_, M, lw.kv_proj.N, H, stream_);

            attn::fused_qknorm_rope_kv_write(d_proj_a_, d_proj_b_, d_proj_b_ + kv_dim,
                                             lw.q_norm, lw.k_norm, d_proj_c_, kv, attn_params_,
                                             M, cur_pos_, nullptr, stream_, 2 * kv_dim);

            attn::flash_attention_prefill(d_proj_c_, c.attn_output_gate ? d_proj_a_ : nullptr,
                                          kv, d_attn_out_, attn_params_, M, cur_pos_, stream_);

            quant::gemm_r4_gfx12_w4a16(d_attn_out_, lw.o_proj.qs, lw.o_proj.scales,
                                       d_x_, d_gemm_tmp_, M, H, lw.o_proj.K, stream_);
        } else {
            const int32_t conv_dim = gdn_params_.conv_dim();
            const int32_t value_dim = gdn_params_.value_dim();
            const int32_t Hv = gdn_params_.num_v_heads;
            const auto& gs = gdn_states_[gdn_idx_map_[l]];

            quant::gemm_r4_gfx12_w4a16(d_norm_out_, lw.in_proj_qkvz.qs, lw.in_proj_qkvz.scales,
                                       d_proj_a_, d_gemm_tmp_, M, lw.in_proj_qkvz.N, H, stream_);
            ops::gemm_f16(d_norm_out_, lw.in_proj_ba, d_ba_buf_, M, 2 * Hv, H, stream_);

            gdn::conv1d_silu(d_proj_a_, conv_dim + value_dim, lw.conv_w, nullptr,
                             gs.conv_state, d_proj_b_, gdn_params_, M, stream_);
            gdn::fused_gdn_gated_norm(d_proj_b_, d_proj_a_ + conv_dim, conv_dim + value_dim,
                                      d_ba_buf_, d_ba_buf_ + Hv, 2 * Hv,
                                      lw.A_log, lw.dt_bias, lw.gdn_norm_w,
                                      gs.recurrent_state, d_attn_out_, gdn_params_, M, stream_);

            quant::gemm_r4_gfx12_w4a16(d_attn_out_, lw.out_proj.qs, lw.out_proj.scales,
                                       d_x_, d_gemm_tmp_, M, H, value_dim, stream_);
        }

        ops::fused_residual_rmsnorm(d_x_, d_residual_, lw.post_attn_norm, d_norm_out_,
                                    M, H, c.rms_norm_eps, plus_one, stream_);

        quant::gemm_r4_gfx12_w4a16(d_norm_out_, lw.gate_up_proj.qs, lw.gate_up_proj.scales,
                                   d_proj_a_, d_gemm_tmp_, M, 2 * I, H, stream_);
        ops::fused_silu_mul_packed(d_proj_a_, d_proj_b_, M, I, stream_);
        quant::gemm_r4_gfx12_w4a16(d_proj_b_, lw.down_proj.qs, lw.down_proj.scales,
                                   d_x_, d_gemm_tmp_, M, H, I, stream_);
    }

    if (compute_logits) {
        ops::fused_residual_rmsnorm(d_x_, d_residual_, weights_.final_norm, d_norm_out_,
                                    M, H, c.rms_norm_eps, plus_one, stream_);
        const __half* last_hidden = d_norm_out_ + static_cast<size_t>(M - 1) * H;
        quant::gemm_r4_gfx12_w4a16(last_hidden, weights_.lm_head.qs, weights_.lm_head.scales,
                                   d_logits_, d_gemm_tmp_, 1, V, H, stream_);
        ops::argmax_sample(d_logits_, V, d_argmax_tmp_, d_out_token_, d_step_, stream_);
    } else {
        // Still apply residual add for correctness? No need since chunking finishes at next embed,
        // wait: each batch chunk starts at embed_lookup!
    }

    cur_pos_ += M;
}

int32_t InferenceEngine::prefill(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) throw std::runtime_error("Empty prompt in prefill");
    if (static_cast<int64_t>(cur_pos_) + static_cast<int64_t>(tokens.size()) > opts_.max_seq_len) {
        throw std::runtime_error("Context length exceeds engine max_seq_len");
    }
    const size_t total = tokens.size();
    size_t off = 0;
    while (off < total) {
        const int chunk = static_cast<int>(std::min<size_t>(opts_.max_batch_tokens, total - off));
        const bool is_last = (off + chunk == total);
        run_forward_batch(tokens.data() + off, chunk, is_last);
        off += chunk;
    }
    int32_t next_tok = 0;
    RX_HIP_CHECK(hipMemcpyAsync(&next_tok, d_out_token_, sizeof(next_tok), hipMemcpyDeviceToHost, stream_));
    RX_HIP_CHECK(hipStreamSynchronize(stream_));
    return next_tok;
}

int32_t InferenceEngine::decode_step(int32_t token_id) {
    if (cur_pos_ >= opts_.max_seq_len) {
        throw std::runtime_error("Context length exceeds engine max_seq_len");
    }

    StepParams sp{};
    sp.token_id = token_id;
    sp.pos = cur_pos_;
    sp.seq_len = 1;
    sp.kv_seq_len = cur_pos_ + 1;
    sp.mrope_pos[0] = sp.mrope_pos[1] = sp.mrope_pos[2] = cur_pos_;
    RX_HIP_CHECK(hipMemcpyAsync(d_step_, &sp, sizeof(sp), hipMemcpyHostToDevice, stream_));

    if (decode_graph_exec_) {
        RX_HIP_CHECK(hipGraphLaunch(decode_graph_exec_, stream_));
    } else {
        const ModelConfig& c = weights_.config;
        const int32_t H = c.hidden_size;
        const int32_t V = c.vocab_size;
        const bool plus_one = (c.norm_kind == NormKind::RMSNormPlusOne);
        ops::embed_lookup_r4_step(weights_.embed_tokens.qs, weights_.embed_tokens.scales,
                                  d_step_, d_residual_, V, H, stream_);
        for (int32_t l = 0; l < c.num_hidden_layers; ++l) run_layer_decode(l);
        ops::fused_residual_rmsnorm(d_x_, d_residual_, weights_.final_norm, d_norm_out_,
                                    1, H, c.rms_norm_eps, plus_one, stream_);
        quant::gemm_r4_gfx12_w4a16(d_norm_out_, weights_.lm_head.qs, weights_.lm_head.scales,
                                   d_logits_, d_gemm_tmp_, 1, V, H, stream_);
        ops::argmax_sample(d_logits_, V, d_argmax_tmp_, d_out_token_, d_step_, stream_);
    }

    int32_t next_tok = 0;
    RX_HIP_CHECK(hipMemcpyAsync(&next_tok, d_out_token_, sizeof(next_tok), hipMemcpyDeviceToHost, stream_));
    RX_HIP_CHECK(hipStreamSynchronize(stream_));
    cur_pos_++;
    return next_tok;
}

std::vector<int32_t> InferenceEngine::verify_tokens(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) throw std::runtime_error("Empty tokens in verify_tokens");
    const int M = static_cast<int>(tokens.size());
    if (M == 1) {
        return {decode_step(tokens[0])};
    }
    if (M > opts_.max_verify_tokens) {
        throw std::runtime_error("Batch size exceeds EngineOptions::max_verify_tokens");
    }
    if (cur_pos_ + M > opts_.max_seq_len) {
        throw std::runtime_error("Context length exceeds engine max_seq_len");
    }

    StepParams sp{};
    sp.token_id = tokens[0];
    sp.pos = cur_pos_;
    sp.seq_len = M;
    sp.kv_seq_len = cur_pos_ + M;
    sp.mrope_pos[0] = sp.mrope_pos[1] = sp.mrope_pos[2] = cur_pos_;
    RX_HIP_CHECK(hipMemcpyAsync(d_step_, &sp, sizeof(sp), hipMemcpyHostToDevice, stream_));
    RX_HIP_CHECK(hipMemcpyAsync(d_tokens_in_, tokens.data(), static_cast<size_t>(M) * sizeof(int32_t),
                                hipMemcpyHostToDevice, stream_));

    if (opts_.enable_hip_graph) {
        hipGraphExec_t exec = get_or_capture_verify_graph(M);
        RX_HIP_CHECK(hipGraphLaunch(exec, stream_));
    } else {
        const ModelConfig& c = weights_.config;
        const int32_t H = c.hidden_size;
        const int32_t V = c.vocab_size;
        const bool plus_one = (c.norm_kind == NormKind::RMSNormPlusOne);
        const bool save_replay = (d_gdn_backup_ != nullptr);
        ops::embed_lookup_r4_batch(weights_.embed_tokens.qs, weights_.embed_tokens.scales,
                                   d_tokens_in_, d_residual_, M, V, H, stream_);
        for (int32_t l = 0; l < c.num_hidden_layers; ++l) {
            run_layer_decode_step(l, M, save_replay);
        }
        ops::fused_residual_rmsnorm(d_x_, d_residual_, weights_.final_norm, d_norm_out_,
                                    M, H, c.rms_norm_eps, plus_one, stream_);
        quant::gemm_r4_gfx12_w4a16(d_norm_out_, weights_.lm_head.qs, weights_.lm_head.scales,
                                   d_logits_, d_gemm_tmp_, M, V, H, stream_);
        ops::argmax_sample_batch(d_logits_, M, V, d_argmax_tmp_, d_out_token_, stream_);
    }

    std::vector<int32_t> preds(M, 0);
    RX_HIP_CHECK(hipMemcpyAsync(preds.data(), d_out_token_, static_cast<size_t>(M) * sizeof(int32_t),
                                hipMemcpyDeviceToHost, stream_));
    RX_HIP_CHECK(hipStreamSynchronize(stream_));
    cur_pos_ += M;
    return preds;
}

SpeculativeVerifyResult InferenceEngine::speculative_verify(const std::vector<int32_t>& draft_tokens) {
    if (draft_tokens.empty()) throw std::runtime_error("Empty draft_tokens in speculative_verify");
    const int M = static_cast<int>(draft_tokens.size());
    if (M == 1) {
        SpeculativeVerifyResult res;
        res.accepted_tokens.push_back(decode_step(draft_tokens[0]));
        res.num_draft_accepted = 0;
        return res;
    }
    if (M > opts_.max_verify_tokens) {
        throw std::runtime_error("Draft length exceeds EngineOptions::max_verify_tokens");
    }

    const int32_t old_pos = cur_pos_;
    if (d_gdn_backup_ != nullptr && gdn_total_bytes_ > 0) {
        RX_HIP_CHECK(hipMemcpyAsync(d_gdn_backup_, d_gdn_base_, gdn_total_bytes_,
                                    hipMemcpyDeviceToDevice, stream_));
    }

    const std::vector<int32_t> preds = verify_tokens(draft_tokens);

    int32_t k_match = 0;
    while (k_match < M - 1 && preds[k_match] == draft_tokens[k_match + 1]) {
        k_match++;
    }

    SpeculativeVerifyResult res;
    res.num_draft_accepted = k_match;
    for (int32_t i = 0; i < k_match; ++i) {
        res.accepted_tokens.push_back(draft_tokens[i + 1]);
    }
    res.accepted_tokens.push_back(preds[k_match]);

    const int32_t accepted_prefix = 1 + k_match;
    if (accepted_prefix < M) {
        if (d_gdn_backup_ != nullptr && gdn_total_bytes_ > 0) {
            RX_HIP_CHECK(hipMemcpyAsync(d_gdn_base_, d_gdn_backup_, gdn_total_bytes_,
                                        hipMemcpyDeviceToDevice, stream_));
            const int32_t conv_dim = gdn_params_.conv_dim();
            const int32_t value_dim = gdn_params_.value_dim();
            const int32_t Hv = gdn_params_.num_v_heads;
            const int32_t MV = opts_.max_verify_tokens;
            const int32_t n_gdn = static_cast<int32_t>(gdn_states_.size());
            for (int32_t g = 0; g < n_gdn; ++g) {
                const LayerWeights& lw = weights_.layers[gdn_layer_idx_[g]];
                const auto& gs = gdn_states_[g];
                const __half* qkv_saved = d_gdn_replay_qkv_ + static_cast<size_t>(g) * MV * conv_dim;
                const __half* ba_saved = d_gdn_replay_ba_ + static_cast<size_t>(g) * MV * (2 * Hv);
                gdn::conv1d_silu(qkv_saved, conv_dim, lw.conv_w, nullptr,
                                 gs.conv_state, d_proj_b_, gdn_params_, accepted_prefix, stream_);
                gdn::fused_gdn_gated_norm(d_proj_b_, d_proj_a_, value_dim,
                                          ba_saved, ba_saved + Hv, 2 * Hv,
                                          lw.A_log, lw.dt_bias, lw.gdn_norm_w,
                                          gs.recurrent_state, d_attn_out_, gdn_params_, accepted_prefix, stream_);
            }
            RX_HIP_CHECK(hipStreamSynchronize(stream_));
        }
        cur_pos_ = old_pos + accepted_prefix;
    }

    return res;
}

void InferenceEngine::copy_last_logits_f16(std::vector<uint16_t>& out_bits) const {
    const int32_t V = weights_.config.vocab_size;
    out_bits.resize(static_cast<size_t>(V));
    RX_HIP_CHECK(hipMemcpyAsync(out_bits.data(), d_logits_, static_cast<size_t>(V) * sizeof(uint16_t),
                                hipMemcpyDeviceToHost, stream_));
    RX_HIP_CHECK(hipStreamSynchronize(stream_));
}

void InferenceEngine::copy_last_logits(std::vector<float>& out) const {
    const int32_t V = weights_.config.vocab_size;
    std::vector<__half> h_f16(V);
    RX_HIP_CHECK(hipMemcpyAsync(h_f16.data(), d_logits_, static_cast<size_t>(V) * sizeof(__half),
                                hipMemcpyDeviceToHost, stream_));
    RX_HIP_CHECK(hipStreamSynchronize(stream_));
    out.resize(V);
    for (int32_t i = 0; i < V; ++i) out[i] = __half2float(h_f16[i]);
}

std::vector<float> InferenceEngine::forward_logits(const std::vector<int32_t>& tokens) {
    reset_state();
    (void)prefill(tokens);
    std::vector<float> out;
    copy_last_logits(out);
    return out;
}

EngineBenchResult InferenceEngine::benchmark_decode(int steps, int warmup) {
    reset_state();
    int32_t tok = 1;
    for (int i = 0; i < warmup; ++i) {
        tok = decode_step(tok);
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i) {
        tok = decode_step(tok);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    EngineBenchResult res;
    res.tokens_generated = steps;
    res.elapsed_ms = ms;
    res.ms_per_tok = ms / steps;
    res.tok_per_sec = (steps * 1000.0) / ms;
    return res;
}

EngineBenchResult InferenceEngine::benchmark_verify(int M, int steps, int warmup) {
    reset_state();
    std::vector<int32_t> batch(M, 42);
    for (int i = 0; i < warmup; ++i) {
        auto out = verify_tokens(batch);
        batch[0] = out.back();
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i) {
        auto out = verify_tokens(batch);
        batch[0] = out.back();
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    EngineBenchResult res;
    res.tokens_generated = steps * M;
    res.elapsed_ms = ms;
    res.ms_per_tok = ms / (steps * M);
    res.tok_per_sec = (steps * M * 1000.0) / ms;
    return res;
}

EngineBenchResult InferenceEngine::benchmark_prefill(int num_tokens, int warmup) {
    std::vector<int32_t> prompt(num_tokens, 42);
    for (int i = 0; i < warmup; ++i) {
        reset_state();
        (void)prefill(prompt);
    }

    reset_state();
    const auto t0 = std::chrono::steady_clock::now();
    (void)prefill(prompt);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    EngineBenchResult res;
    res.tokens_generated = num_tokens;
    res.elapsed_ms = ms;
    res.ms_per_tok = ms / num_tokens;
    res.tok_per_sec = (num_tokens * 1000.0) / ms;
    return res;
}

DecodeBreakdown InferenceEngine::profile_decode(int32_t kv_ctx_len, int steps, int warmup) {
    DecodeBreakdown bd{};
    const ModelConfig& c = weights_.config;
    const int32_t H = c.hidden_size;
    const int32_t I = c.intermediate_size;
    const int32_t V = c.vocab_size;
    const bool plus_one = (c.norm_kind == NormKind::RMSNormPlusOne);
    const int32_t ctx = std::min(std::max(1, kv_ctx_len), opts_.max_seq_len);

    reset_state();
    StepParams sp{};
    sp.token_id = 42;
    sp.pos = ctx - 1;
    sp.seq_len = 1;
    sp.kv_seq_len = ctx;
    sp.mrope_pos[0] = sp.mrope_pos[1] = sp.mrope_pos[2] = sp.pos;
    RX_HIP_CHECK(hipMemcpy(d_step_, &sp, sizeof(sp), hipMemcpyHostToDevice));

    // Measure graphed wall time at ctx
    if (decode_graph_exec_) {
        for (int i = 0; i < warmup; ++i) RX_HIP_CHECK(hipGraphLaunch(decode_graph_exec_, stream_));
        RX_HIP_CHECK(hipStreamSynchronize(stream_));
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < steps; ++i) {
            RX_HIP_CHECK(hipMemcpyAsync(d_step_, &sp, sizeof(sp), hipMemcpyHostToDevice, stream_));
            RX_HIP_CHECK(hipGraphLaunch(decode_graph_exec_, stream_));
            int32_t tok = 0;
            RX_HIP_CHECK(hipMemcpyAsync(&tok, d_out_token_, sizeof(tok), hipMemcpyDeviceToHost, stream_));
            RX_HIP_CHECK(hipStreamSynchronize(stream_));
        }
        const auto t1 = std::chrono::steady_clock::now();
        bd.graph_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / steps;
    }

    // Measure non-graph wall time
    {
        auto run_non_graph = [&]() {
            ops::embed_lookup_r4_step(weights_.embed_tokens.qs, weights_.embed_tokens.scales,
                                      d_step_, d_residual_, V, H, stream_);
            for (int32_t l = 0; l < c.num_hidden_layers; ++l) run_layer_decode(l);
            ops::fused_residual_rmsnorm(d_x_, d_residual_, weights_.final_norm, d_norm_out_,
                                        1, H, c.rms_norm_eps, plus_one, stream_);
            quant::gemm_r4_gfx12_w4a16(d_norm_out_, weights_.lm_head.qs, weights_.lm_head.scales,
                                       d_logits_, d_gemm_tmp_, 1, V, H, stream_);
            ops::argmax_sample(d_logits_, V, d_argmax_tmp_, d_out_token_, d_step_, stream_);
        };
        for (int i = 0; i < warmup; ++i) run_non_graph();
        RX_HIP_CHECK(hipStreamSynchronize(stream_));
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < steps; ++i) {
            RX_HIP_CHECK(hipMemcpyAsync(d_step_, &sp, sizeof(sp), hipMemcpyHostToDevice, stream_));
            run_non_graph();
            int32_t tok = 0;
            RX_HIP_CHECK(hipMemcpyAsync(&tok, d_out_token_, sizeof(tok), hipMemcpyDeviceToHost, stream_));
            RX_HIP_CHECK(hipStreamSynchronize(stream_));
        }
        const auto t1 = std::chrono::steady_clock::now();
        bd.non_graph_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / steps;
    }

    // Fine-grained event profiling over 1 step (after warmup)
    const int n_events = 2 + c.num_hidden_layers * 9 + 3;
    std::vector<hipEvent_t> ev(n_events);
    for (auto& e : ev) RX_HIP_CHECK(hipEventCreate(&e));

    auto rec = [&](int idx) { RX_HIP_CHECK(hipEventRecord(ev[idx], stream_)); };
    auto dt = [&](int a, int b) -> double {
        float ms = 0.0f;
        RX_HIP_CHECK(hipEventElapsedTime(&ms, ev[a], ev[b]));
        return static_cast<double>(ms);
    };

    for (int s = 0; s < steps; ++s) {
        int ei = 0;
        rec(ei++);
        ops::embed_lookup_r4_step(weights_.embed_tokens.qs, weights_.embed_tokens.scales,
                                  d_step_, d_residual_, V, H, stream_);
        rec(ei++);

        for (int32_t l = 0; l < c.num_hidden_layers; ++l) {
            const LayerWeights& lw = weights_.layers[l];
            if (l == 0) {
                ops::rmsnorm(d_residual_, lw.input_norm, d_norm_out_, 1, H, c.rms_norm_eps, plus_one, stream_);
            } else {
                ops::fused_residual_rmsnorm(d_x_, d_residual_, lw.input_norm, d_norm_out_,
                                            1, H, c.rms_norm_eps, plus_one, stream_);
            }
            rec(ei++);

            if (lw.kind == LayerKind::FullAttention) {
                const int32_t kv_dim = c.num_key_value_heads * c.head_dim;
                const auto& kv = kv_caches_[attn_idx_map_[l]];
                quant::gemm_r4_gfx12_w4a16(d_norm_out_, lw.q_proj.qs, lw.q_proj.scales,
                                           d_proj_a_, d_gemm_tmp_, 1, lw.q_proj.N, H, stream_);
                quant::gemm_r4_gfx12_w4a16(d_norm_out_, lw.kv_proj.qs, lw.kv_proj.scales,
                                           d_proj_b_, d_gemm_tmp_, 1, lw.kv_proj.N, H, stream_);
                rec(ei++);

                attn::fused_qknorm_rope_kv_write(d_proj_a_, d_proj_b_, d_proj_b_ + kv_dim,
                                                 lw.q_norm, lw.k_norm, d_proj_c_, kv, attn_params_,
                                                 1, 0, d_step_, stream_, 2 * kv_dim);
                attn::gqa_decode_attention(d_proj_c_, c.attn_output_gate ? d_proj_a_ : nullptr,
                                           kv, d_attn_out_, d_attn_split_, attn_params_,
                                           1, 0, d_step_, 16, stream_);
                rec(ei++);

                quant::gemm_r4_gfx12_w4a16(d_attn_out_, lw.o_proj.qs, lw.o_proj.scales,
                                           d_x_, d_gemm_tmp_, 1, H, lw.o_proj.K, stream_);
                rec(ei++);
            } else {
                const int32_t conv_dim = gdn_params_.conv_dim();
                const int32_t value_dim = gdn_params_.value_dim();
                const int32_t Hv = gdn_params_.num_v_heads;
                const auto& gs = gdn_states_[gdn_idx_map_[l]];

                quant::gemm_r4_gfx12_w4a16(d_norm_out_, lw.in_proj_qkvz.qs, lw.in_proj_qkvz.scales,
                                           d_proj_a_, d_gemm_tmp_, 1, lw.in_proj_qkvz.N, H, stream_);
                rec(ei++);
                ops::gemm_f16(d_norm_out_, lw.in_proj_ba, d_ba_buf_, 1, 2 * Hv, H, stream_);
                rec(ei++);
                gdn::conv1d_silu(d_proj_a_, conv_dim + value_dim, lw.conv_w, nullptr,
                                 gs.conv_state, d_proj_b_, gdn_params_, 1, stream_);
                gdn::fused_gdn_gated_norm(d_proj_b_, d_proj_a_ + conv_dim, conv_dim + value_dim,
                                          d_ba_buf_, d_ba_buf_ + Hv, 2 * Hv,
                                          lw.A_log, lw.dt_bias, lw.gdn_norm_w,
                                          gs.recurrent_state, d_attn_out_, gdn_params_, 1, stream_);
                rec(ei++);
                quant::gemm_r4_gfx12_w4a16(d_attn_out_, lw.out_proj.qs, lw.out_proj.scales,
                                           d_x_, d_gemm_tmp_, 1, H, value_dim, stream_);
                rec(ei++);
            }

            ops::fused_residual_rmsnorm(d_x_, d_residual_, lw.post_attn_norm, d_norm_out_,
                                        1, H, c.rms_norm_eps, plus_one, stream_);
            rec(ei++);
            if ((I & 15) == 0 && (H & 63) == 0) {
                quant::gemv_r4_gfx12_w4a16_swiglu(d_norm_out_, lw.gate_up_proj.qs, lw.gate_up_proj.scales,
                                                  d_proj_b_, I, H, stream_);
                rec(ei++);
                rec(ei++); // fused mlp_act = 0
            } else {
                quant::gemm_r4_gfx12_w4a16(d_norm_out_, lw.gate_up_proj.qs, lw.gate_up_proj.scales,
                                           d_proj_a_, d_gemm_tmp_, 1, 2 * I, H, stream_);
                rec(ei++);
                ops::fused_silu_mul_packed(d_proj_a_, d_proj_b_, 1, I, stream_);
                rec(ei++);
            }
            quant::gemm_r4_gfx12_w4a16(d_proj_b_, lw.down_proj.qs, lw.down_proj.scales,
                                       d_x_, d_gemm_tmp_, 1, H, I, stream_);
            rec(ei++);
        }

        ops::fused_residual_rmsnorm(d_x_, d_residual_, weights_.final_norm, d_norm_out_,
                                    1, H, c.rms_norm_eps, plus_one, stream_);
        rec(ei++);
        quant::gemm_r4_gfx12_w4a16(d_norm_out_, weights_.lm_head.qs, weights_.lm_head.scales,
                                   d_logits_, d_gemm_tmp_, 1, V, H, stream_);
        rec(ei++);
        ops::argmax_sample(d_logits_, V, d_argmax_tmp_, d_out_token_, d_step_, stream_);
        rec(ei++);

        RX_HIP_CHECK(hipStreamSynchronize(stream_));
        int k = 0;
        bd.embed_ms += dt(k, k + 1); ++k;
        for (int32_t l = 0; l < c.num_hidden_layers; ++l) {
            bd.rmsnorm_ms += dt(k, k + 1); ++k;
            if (weights_.layers[l].kind == LayerKind::FullAttention) {
                bd.qkv_proj_ms += dt(k, k + 1); ++k;
                bd.attn_op_ms += dt(k, k + 1); ++k;
                bd.o_proj_ms += dt(k, k + 1); ++k;
            } else {
                bd.qkv_proj_ms += dt(k, k + 1); ++k;
                bd.gdn_ba_proj_ms += dt(k, k + 1); ++k;
                bd.gdn_conv_recur_ms += dt(k, k + 1); ++k;
                bd.o_proj_ms += dt(k, k + 1); ++k;
            }
            bd.rmsnorm_ms += dt(k, k + 1); ++k;
            bd.mlp_gate_up_ms += dt(k, k + 1); ++k;
            bd.mlp_act_ms += dt(k, k + 1); ++k;
            bd.mlp_down_ms += dt(k, k + 1); ++k;
        }
        bd.rmsnorm_ms += dt(k, k + 1); ++k;
        bd.lm_head_ms += dt(k, k + 1); ++k;
        bd.argmax_ms += dt(k, k + 1); ++k;
    }

    for (auto& e : ev) RX_HIP_CHECK(hipEventDestroy(e));

    bd.embed_ms /= steps;
    bd.rmsnorm_ms /= steps;
    bd.qkv_proj_ms /= steps;
    bd.attn_op_ms /= steps;
    bd.gdn_ba_proj_ms /= steps;
    bd.gdn_conv_recur_ms /= steps;
    bd.o_proj_ms /= steps;
    bd.mlp_gate_up_ms /= steps;
    bd.mlp_act_ms /= steps;
    bd.mlp_down_ms /= steps;
    bd.lm_head_ms /= steps;
    bd.argmax_ms /= steps;
    return bd;
}

} // namespace rxinfer

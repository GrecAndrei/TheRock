// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/engine.hpp"
#include "rxinfer/common.hpp"
#include "rxinfer/ops.hpp"
#include "rxinfer/quant.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace rxinfer {
namespace {

uint64_t compute_weight_fingerprint(const ModelWeights& w) {
    uint64_t h = 14695981039346656037ULL;
    std::vector<uint8_t> buf;
    auto mix_device = [&](const void* d_ptr, size_t bytes) {
        if (!d_ptr || bytes == 0) return;
        buf.resize(bytes);
        RX_HIP_CHECK(hipMemcpy(buf.data(), d_ptr, bytes, hipMemcpyDeviceToHost));
        for (uint8_t b : buf) {
            h ^= b;
            h *= 1099511628211ULL;
        }
    };
    const size_t h_bytes = static_cast<size_t>(std::max(0, w.config.hidden_size)) * sizeof(__half);
    mix_device(w.final_norm, h_bytes);
    if (w.embed_tokens.valid()) {
        const size_t sc_b = quant::r4_g64_scales_bytes(w.embed_tokens.N, w.embed_tokens.K);
        mix_device(w.embed_tokens.scales, std::min<size_t>(16384, sc_b));
    }
    if (!w.layers.empty()) {
        mix_device(w.layers.front().input_norm, h_bytes);
        mix_device(w.layers.back().post_attn_norm, h_bytes);
        if (w.layers.back().down_proj.valid()) {
            const size_t sc_b = quant::r4_g64_scales_bytes(w.layers.back().down_proj.N, w.layers.back().down_proj.K);
            mix_device(w.layers.back().down_proj.scales, std::min<size_t>(4096, sc_b));
        }
    }
    return h;
}

std::string build_cache_signature(const ModelWeights& w, DType kv_dt) {
    const ModelConfig& c = w.config;
    std::ostringstream oss;
    oss << "type=" << c.model_type
        << ";V=" << c.vocab_size
        << ";H=" << c.hidden_size
        << ";I=" << c.intermediate_size
        << ";L=" << c.num_hidden_layers
        << ";norm=" << static_cast<int>(c.norm_kind)
        << ";eps=" << c.rms_norm_eps
        << ";Hq=" << c.num_attention_heads
        << ";Hkv=" << c.num_key_value_heads
        << ";D=" << c.head_dim
        << ";qkn=" << (c.qk_norm ? 1 : 0)
        << ";og=" << (c.attn_output_gate ? 1 : 0)
        << ";rk=" << static_cast<int>(c.rope_kind)
        << ";rt=" << c.rope_theta
        << ";prf=" << c.partial_rotary_factor
        << ";mri=" << (c.mrope_interleaved ? 1 : 0)
        << ";mrs=";
    for (size_t i = 0; i < c.mrope_section.size(); ++i) {
        if (i > 0) oss << ",";
        oss << c.mrope_section[i];
    }
    oss << ";lkh=" << c.linear_num_key_heads
        << ";lkd=" << c.linear_key_head_dim
        << ";lvh=" << c.linear_num_value_heads
        << ";lvd=" << c.linear_value_head_dim
        << ";lck=" << c.linear_conv_kernel_dim
        << ";ltv=" << (c.gdn_tiled_v_heads ? 1 : 0)
        << ";kvdt=" << static_cast<int>(kv_dt)
        << ";pg=" << attn::KV_PAGE_TOKENS
        << ";lt=";
    for (int32_t l = 0; l < c.num_hidden_layers; ++l) {
        const auto k =
            (l < static_cast<int32_t>(c.layer_types.size())) ? c.layer_types[l] : LayerKind::FullAttention;
        oss << ((k == LayerKind::FullAttention) ? 'A' : 'G');
    }
    char wh_buf[24];
    std::snprintf(wh_buf, sizeof(wh_buf), ";wh=%016llx",
                  static_cast<unsigned long long>(compute_weight_fingerprint(w)));
    oss << wh_buf;
    return oss.str();
}

DType parse_kv_dtype_from_sig(const std::string& sig) {
    const size_t pos = sig.find(";kvdt=");
    if (pos != std::string::npos) {
        const int v = std::atoi(sig.c_str() + pos + 6);
        if (v == static_cast<int>(DType::F16)) return DType::F16;
        if (v == static_cast<int>(DType::FP8_E4M3)) return DType::FP8_E4M3;
        if (v == static_cast<int>(DType::FP4_E2M1_G16)) return DType::FP4_E2M1_G16;
    }
    return DType::F16;
}

} // namespace

PromptCacheInfo inspect_prompt_cache(const std::string& path, bool load_tokens) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open prompt cache file: " + path);
    }

    char magic[4]{};
    uint32_t version = 0;
    uint64_t sig_len = 0, gdn_bytes = 0, page_bytes = 0;
    int32_t pos = 0, last_tok = -1, n_attn = 0, active_pages = 0;

    ifs.read(magic, 4);
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    ifs.read(reinterpret_cast<char*>(&sig_len), sizeof(sig_len));
    ifs.read(reinterpret_cast<char*>(&pos), sizeof(pos));
    ifs.read(reinterpret_cast<char*>(&last_tok), sizeof(last_tok));
    ifs.read(reinterpret_cast<char*>(&gdn_bytes), sizeof(gdn_bytes));
    ifs.read(reinterpret_cast<char*>(&n_attn), sizeof(n_attn));
    ifs.read(reinterpret_cast<char*>(&active_pages), sizeof(active_pages));
    ifs.read(reinterpret_cast<char*>(&page_bytes), sizeof(page_bytes));

    if (!ifs || std::memcmp(magic, "RXPC", 4) != 0 || version != 1 || sig_len > (1ULL << 20) || pos < 0) {
        throw std::runtime_error("Invalid RXPC prompt cache header: " + path);
    }

    PromptCacheInfo info;
    info.signature.assign(sig_len, '\0');
    ifs.read(info.signature.data(), sig_len);
    if (!ifs) {
        throw std::runtime_error("Truncated RXPC prompt cache header: " + path);
    }
    info.position = pos;
    info.last_next_token = last_tok;
    info.kv_dtype = parse_kv_dtype_from_sig(info.signature);

    if (load_tokens && pos > 0) {
        info.tokens.resize(pos);
        ifs.read(reinterpret_cast<char*>(info.tokens.data()), static_cast<size_t>(pos) * sizeof(int32_t));
        if (!ifs) {
            throw std::runtime_error("Truncated RXPC prompt cache tokens: " + path);
        }
    }
    return info;
}

InferenceEngine::InferenceEngine(ModelWeights weights, EngineOptions opts)
    : weights_(std::move(weights)), opts_(opts) {
    if (opts_.max_seq_len <= 0) opts_.max_seq_len = weights_.config.max_position_embeddings;
    if (opts_.max_verify_tokens < 1) opts_.max_verify_tokens = 1;
    if (opts_.max_verify_tokens > 16) opts_.max_verify_tokens = 16;
    if (opts_.max_batch_tokens <= 0) opts_.max_batch_tokens = 512;
    if (opts_.max_batch_tokens < opts_.max_verify_tokens) opts_.max_batch_tokens = opts_.max_verify_tokens;
    if (opts_.max_prefix_snapshots < 0) opts_.max_prefix_snapshots = 0;

    RX_HIP_CHECK(hipStreamCreate(&stream_));
    init_arenas();
    cache_signature_ = build_cache_signature(weights_, opts_.kv_dtype);
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

    if (d_gdn_backup_) (void)hipFree(d_gdn_backup_);
    if (d_gdn_replay_qkv_) (void)hipFree(d_gdn_replay_qkv_);
    if (d_gdn_replay_ba_) (void)hipFree(d_gdn_replay_ba_);
    if (h_kv_gtt_pool_) (void)hipHostFree(h_kv_gtt_pool_);
}

void InferenceEngine::init_arenas() {
    const ModelConfig& c = weights_.config;
    const int32_t H = c.hidden_size;
    const int32_t I = c.intermediate_size;
    const int32_t V = c.vocab_size;
    const int32_t L = c.num_hidden_layers;
    const int32_t B = opts_.max_batch_tokens;
    const int32_t MV = opts_.max_verify_tokens;

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

    if (opts_.decode_attn_splits > 0) {
        decode_splits_ = std::max(1, std::min(256, opts_.decode_attn_splits));
    } else {
        decode_splits_ = (opts_.max_seq_len >= 16384) ? 64 : 32;
    }

    const int32_t prefill_sub_m = std::min<int32_t>(B, 512);
    const int32_t prefill_split_kv = (opts_.max_seq_len >= 16384) ? 8 : 4;

    // Compute GDN active state budget (speculative backup & replay buffers are allocated lazily)
    const size_t gdn_conv_b = (n_gdn > 0) ? gdn::gdn_conv_state_bytes(gdn_params_) : 0;
    const size_t gdn_rec_b = (n_gdn > 0) ? gdn::gdn_recurrent_state_bytes(gdn_params_) : 0;
    const size_t per_gdn_bytes = align_up(gdn_conv_b, 256) + align_up(gdn_rec_b, 256);
    gdn_total_bytes_ = static_cast<size_t>(n_gdn) * per_gdn_bytes;

    // Compute base Scratch Arena size (without optional GTT staging buffer)
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

    const size_t decode_split_bytes =
        (n_attn > 0) ? attn::decode_splitk_scratch_bytes(attn_params_, MV, decode_splits_) : 256;
    const size_t prefill_split_bytes =
        (n_attn > 0) ? attn::prefill_splitk_scratch_bytes(attn_params_, B, prefill_sub_m, prefill_split_kv) : 256;
    const size_t attn_split_bytes = std::max(decode_split_bytes, prefill_split_bytes);
    const size_t tmp_bytes = quant::r4_g64_tmp_bytes(MV, max_small_gemm_N);
    const size_t logits_bytes = align_up(static_cast<size_t>(MV) * static_cast<size_t>(V) * sizeof(__half), 256);
    const size_t argmax_b = ops::argmax_tmp_bytes(MV);

    const size_t scratch_base_total =
        align_up(sizeof(StepParams), 256) +
        align_up(static_cast<size_t>(MV) * sizeof(int32_t), 256) +
        align_up(static_cast<size_t>(B) * sizeof(int32_t), 256) +
        3 * align_up(static_cast<size_t>(B) * H * sizeof(__half), 256) +
        align_up(static_cast<size_t>(B) * max_proj_a * sizeof(__half), 256) +
        align_up(static_cast<size_t>(B) * max_proj_b * sizeof(__half), 256) +
        align_up(static_cast<size_t>(B) * max_attn_dim * sizeof(__half), 256) +
        align_up(static_cast<size_t>(B) * max_attn_dim * sizeof(__half), 256) +
        align_up(static_cast<size_t>(B) * std::max(1, 2 * c.linear_num_value_heads) * sizeof(__half), 256) +
        align_up(attn_split_bytes, 256) +
        align_up(tmp_bytes, 256) +
        align_up(logits_bytes, 256) +
        align_up(argmax_b, 256) +
        4096;

    // Tiered Paged KV Cache split (VRAM + mapped host GTT spillover)
    kv_layout_ = (n_attn > 0) ? attn::compute_kv_page_layout(attn_params_) : attn::KVPageLayout{};
    kv_total_pages_ = (n_attn > 0) ? static_cast<int32_t>(div_ceil(opts_.max_seq_len, attn::KV_PAGE_TOKENS)) : 0;
    vram_kv_pages_ = 0;
    gtt_kv_pages_ = 0;
    kv_vram_bytes_ = 0;
    kv_gtt_bytes_ = 0;

    const size_t page_table_sec_bytes =
        (n_attn > 0) ? static_cast<size_t>(n_attn) * align_up(static_cast<size_t>(kv_total_pages_) * sizeof(uint8_t*), 256)
                     : 0;

    int32_t stage_pages = 0;
    size_t stage_bytes = 0;

    if (n_attn > 0 && kv_total_pages_ > 0) {
        const size_t per_step_page_bytes = static_cast<size_t>(n_attn) * kv_layout_.page_bytes;
        if (opts_.kv_vram_gb == 0.0) {
            vram_kv_pages_ = 0;
        } else if (opts_.kv_vram_gb > 0.0) {
            const size_t cap_bytes = static_cast<size_t>(opts_.kv_vram_gb * double(1ULL << 30));
            vram_kv_pages_ = std::min<int32_t>(
                kv_total_pages_, static_cast<int32_t>(cap_bytes / per_step_page_bytes));
        } else {
            int cur_dev = 0;
            RX_HIP_CHECK(hipGetDevice(&cur_dev));
            const size_t free_vram = query_device_info(cur_dev).free_vram_bytes;
            const size_t fixed_non_kv = gdn_total_bytes_ + scratch_base_total + page_table_sec_bytes;
            const size_t fit_all_headroom = 192ULL << 20;
            const size_t total_kv_needed = static_cast<size_t>(kv_total_pages_) * per_step_page_bytes;
            if (free_vram > fixed_non_kv + fit_all_headroom &&
                total_kv_needed <= (free_vram - fixed_non_kv - fit_all_headroom)) {
                vram_kv_pages_ = kv_total_pages_;
            } else {
                const size_t spillover_reserve = fixed_non_kv + (24ULL << 20) + (256ULL << 20);
                const size_t avail_kv = (free_vram > spillover_reserve) ? (free_vram - spillover_reserve) : 0;
                vram_kv_pages_ = std::min<int32_t>(
                    kv_total_pages_, static_cast<int32_t>(avail_kv / per_step_page_bytes));
            }
        }
        gtt_kv_pages_ = kv_total_pages_ - vram_kv_pages_;
        kv_vram_bytes_ = static_cast<size_t>(n_attn) * vram_kv_pages_ * kv_layout_.page_bytes;
        kv_gtt_bytes_ = static_cast<size_t>(n_attn) * gtt_kv_pages_ * kv_layout_.page_bytes;

        if (gtt_kv_pages_ > 0) {
            const size_t target_stage = 20ULL << 20;
            stage_pages = std::min<int32_t>(
                gtt_kv_pages_,
                std::max<int32_t>(16, static_cast<int32_t>(target_stage / std::max<size_t>(1, kv_layout_.page_bytes))));
            stage_bytes = static_cast<size_t>(stage_pages) * kv_layout_.page_bytes;
        }
    }

    // 1. State Arena: GDN state + VRAM KV page pool + device page tables
    const size_t state_total = gdn_total_bytes_ + align_up(kv_vram_bytes_, 256) + page_table_sec_bytes + 4096;
    state_arena_.init(state_total, "state");

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
    }

    if (kv_vram_bytes_ > 0) {
        d_kv_vram_pool_ = state_arena_.alloc(kv_vram_bytes_, 256);
    }

    if (kv_gtt_bytes_ > 0) {
        const hipError_t herr = hipHostMalloc(&h_kv_gtt_pool_, kv_gtt_bytes_, hipHostMallocMapped);
        if (herr != hipSuccess) {
            throw std::runtime_error(
                "Failed to allocate " +
                std::to_string(kv_gtt_bytes_ / (1024.0 * 1024.0 * 1024.0)) +
                " GiB pinned host GTT KV pool (" + std::string(hipGetErrorString(herr)) +
                "). Try --kv-dtype fp4 or fp8, or reduce --max-seq-len.");
        }
        RX_HIP_CHECK(hipHostGetDevicePointer(&d_kv_gtt_pool_, h_kv_gtt_pool_, 0));
    }

    kv_caches_.resize(n_attn);
    d_page_tables_.resize(n_attn, nullptr);
    if (n_attn > 0) {
        std::vector<uint8_t*> h_ptrs(kv_total_pages_);
        for (int32_t i = 0; i < n_attn; ++i) {
            for (int32_t p = 0; p < kv_total_pages_; ++p) {
                if (p < vram_kv_pages_) {
                    h_ptrs[p] = static_cast<uint8_t*>(d_kv_vram_pool_) +
                                (static_cast<size_t>(i) * vram_kv_pages_ + p) * kv_layout_.page_bytes;
                } else {
                    h_ptrs[p] = static_cast<uint8_t*>(d_kv_gtt_pool_) +
                                (static_cast<size_t>(i) * gtt_kv_pages_ + (p - vram_kv_pages_)) * kv_layout_.page_bytes;
                }
            }
            const size_t pt_bytes = static_cast<size_t>(kv_total_pages_) * sizeof(uint8_t*);
            d_page_tables_[i] = static_cast<uint8_t**>(state_arena_.alloc(pt_bytes, 256));
            RX_HIP_CHECK(hipMemcpy(d_page_tables_[i], h_ptrs.data(), pt_bytes, hipMemcpyHostToDevice));

            kv_caches_[i].page_ptrs = d_page_tables_[i];
            kv_caches_[i].num_pages = kv_total_pages_;
            kv_caches_[i].num_vram_pages = vram_kv_pages_;
            kv_caches_[i].page_bytes = kv_layout_.page_bytes;
            kv_caches_[i].data_sec_bytes = kv_layout_.data_sec_bytes;
            kv_caches_[i].sc_sec_bytes = kv_layout_.sc_sec_bytes;
        }
    }

    // 2. Scratch Arena: reusable activations + split-KV attention + GEMM split-K + logits + GTT stage buffer
    const size_t scratch_total = scratch_base_total + align_up(stage_bytes, 256);
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
    d_attn_split_ = scratch_arena_.alloc(attn_split_bytes, 256);
    d_gemm_tmp_ = static_cast<float*>(scratch_arena_.alloc(tmp_bytes, 256));
    d_logits_ = static_cast<__half*>(scratch_arena_.alloc(logits_bytes, 256));
    d_argmax_tmp_ = scratch_arena_.alloc(argmax_b, 256);

    uint8_t* stage_ptr = (stage_bytes > 0) ? static_cast<uint8_t*>(scratch_arena_.alloc(stage_bytes, 256)) : nullptr;
    prefill_scratch_.split_scratch = static_cast<float*>(d_attn_split_);
    prefill_scratch_.max_M = B;
    prefill_scratch_.sub_m = prefill_sub_m;
    prefill_scratch_.split_kv = prefill_split_kv;
    prefill_scratch_.stage_buf = stage_ptr;
    prefill_scratch_.stage_pages = stage_pages;
}

void InferenceEngine::ensure_speculative_buffers() {
    const size_t n_gdn = gdn_states_.size();
    if (n_gdn == 0 || opts_.max_verify_tokens <= 1 || d_gdn_backup_ != nullptr) return;

    const int32_t MV = opts_.max_verify_tokens;
    const int32_t conv_dim = gdn_params_.conv_dim();
    const int32_t Hv = gdn_params_.num_v_heads;
    const size_t replay_qkv_b = align_up(n_gdn * MV * conv_dim * sizeof(__half), 256);
    const size_t replay_ba_b = align_up(n_gdn * MV * (2 * Hv) * sizeof(__half), 256);

    RX_HIP_CHECK(hipMalloc(&d_gdn_backup_, gdn_total_bytes_));
    RX_HIP_CHECK(hipMalloc(&d_gdn_replay_qkv_, replay_qkv_b));
    RX_HIP_CHECK(hipMalloc(&d_gdn_replay_ba_, replay_ba_b));
    speculative_vram_bytes_ = gdn_total_bytes_ + replay_qkv_b + replay_ba_b;

    // Re-capture any verify graphs previously captured without GDN replay saving.
    for (auto& kv : verify_graph_execs_) {
        if (kv.second) (void)hipGraphExecDestroy(kv.second);
    }
    for (auto& kv : verify_graphs_) {
        if (kv.second) (void)hipGraphDestroy(kv.second);
    }
    verify_graph_execs_.clear();
    verify_graphs_.clear();
}

void InferenceEngine::reset_state() {
    cur_pos_ = 0;
    last_next_token_ = -1;
    cached_tokens_.clear();
    if (gdn_total_bytes_ > 0 && d_gdn_base_ != nullptr) {
        RX_HIP_CHECK(hipMemsetAsync(d_gdn_base_, 0, gdn_total_bytes_, stream_));
    }
    RX_HIP_CHECK(hipStreamSynchronize(stream_));
}

void InferenceEngine::clear_prefix_cache() {
    prefix_snapshots_.clear();
}

std::string InferenceEngine::cache_signature() const {
    return cache_signature_;
}

void InferenceEngine::record_prefix_snapshot(bool pin) {
    if (opts_.max_prefix_snapshots <= 0 || cached_tokens_.empty() || gdn_total_bytes_ == 0) return;

    for (size_t i = 0; i < prefix_snapshots_.size(); ++i) {
        if (prefix_snapshots_[i].tokens == cached_tokens_) {
            prefix_snapshots_[i].pinned = prefix_snapshots_[i].pinned || pin;
            if (i + 1 < prefix_snapshots_.size()) {
                auto snap = std::move(prefix_snapshots_[i]);
                prefix_snapshots_.erase(prefix_snapshots_.begin() + i);
                prefix_snapshots_.push_back(std::move(snap));
            }
            return;
        }
    }

    RX_HIP_CHECK(hipStreamSynchronize(stream_));
    PrefixSnapshot snap;
    snap.tokens = cached_tokens_;
    snap.gdn_state.resize(gdn_total_bytes_);
    snap.pinned = pin;
    RX_HIP_CHECK(hipMemcpy(snap.gdn_state.data(), d_gdn_base_, gdn_total_bytes_, hipMemcpyDeviceToHost));

    if (pin) {
        for (auto& existing : prefix_snapshots_) {
            existing.pinned = false;
        }
    }

    while (static_cast<int32_t>(prefix_snapshots_.size()) >= opts_.max_prefix_snapshots) {
        auto victim = prefix_snapshots_.begin();
        while (victim != prefix_snapshots_.end() && victim->pinned) {
            ++victim;
        }
        if (victim == prefix_snapshots_.end()) {
            victim = prefix_snapshots_.begin();
        }
        prefix_snapshots_.erase(victim);
    }
    prefix_snapshots_.push_back(std::move(snap));
}

void InferenceEngine::save_prefix_checkpoint(bool pin) {
    if (opts_.max_prefix_snapshots <= 0 || cached_tokens_.empty() || gdn_total_bytes_ == 0) return;
    record_prefix_snapshot(pin);
}

void InferenceEngine::unpin_prefix_checkpoints() {
    for (auto& snap : prefix_snapshots_) {
        snap.pinned = false;
    }
}

bool InferenceEngine::restore_prefix(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) {
        reset_state();
        return true;
    }
    if (static_cast<int64_t>(tokens.size()) > opts_.max_seq_len) {
        throw std::runtime_error("Context length exceeds engine max_seq_len");
    }

    if (cur_pos_ == static_cast<int32_t>(tokens.size()) &&
        cached_tokens_.size() == tokens.size() &&
        std::equal(cached_tokens_.begin(), cached_tokens_.end(), tokens.begin())) {
        return true;
    }

    size_t kv_lcp = 0;
    while (kv_lcp < kv_resident_tokens_.size() &&
           kv_lcp < tokens.size() &&
           kv_resident_tokens_[kv_lcp] == tokens[kv_lcp]) {
        ++kv_lcp;
    }

    if (kv_lcp == tokens.size()) {
        if (gdn_total_bytes_ == 0) {
            cur_pos_ = static_cast<int32_t>(tokens.size());
            cached_tokens_ = tokens;
            last_next_token_ = -1;
            return true;
        }
        for (size_t i = 0; i < prefix_snapshots_.size(); ++i) {
            if (prefix_snapshots_[i].tokens.size() == tokens.size() &&
                std::equal(prefix_snapshots_[i].tokens.begin(), prefix_snapshots_[i].tokens.end(), tokens.begin())) {
                PrefixSnapshot snap = std::move(prefix_snapshots_[i]);
                prefix_snapshots_.erase(prefix_snapshots_.begin() + i);
                RX_HIP_CHECK(hipMemcpyAsync(d_gdn_base_, snap.gdn_state.data(), gdn_total_bytes_,
                                            hipMemcpyHostToDevice, stream_));
                RX_HIP_CHECK(hipStreamSynchronize(stream_));
                prefix_snapshots_.push_back(std::move(snap));
                cur_pos_ = static_cast<int32_t>(tokens.size());
                cached_tokens_ = tokens;
                last_next_token_ = -1;
                return true;
            }
        }
    }

    prefill_cached(tokens, false);
    return false;
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
                                   M, 0, d_step_, decode_splits_, stream_);

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
                                          kv, d_attn_out_, attn_params_, M, cur_pos_, &prefill_scratch_, stream_);

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
    }

    if (static_cast<size_t>(cur_pos_) < kv_resident_tokens_.size()) {
        kv_resident_tokens_.resize(cur_pos_);
    }
    if (static_cast<size_t>(cur_pos_) == kv_resident_tokens_.size()) {
        kv_resident_tokens_.insert(kv_resident_tokens_.end(), h_tokens, h_tokens + M);
    }
    cached_tokens_.insert(cached_tokens_.end(), h_tokens, h_tokens + M);
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
    last_next_token_ = next_tok;
    return next_tok;
}

PrefillCacheResult InferenceEngine::prefill_cached(const std::vector<int32_t>& tokens, bool save_checkpoint) {
    if (tokens.empty()) throw std::runtime_error("Empty prompt in prefill_cached");
    if (static_cast<int64_t>(tokens.size()) > opts_.max_seq_len) {
        throw std::runtime_error("Context length exceeds engine max_seq_len");
    }

    size_t kv_lcp = 0;
    while (kv_lcp < kv_resident_tokens_.size() &&
           kv_lcp < tokens.size() &&
           kv_resident_tokens_[kv_lcp] == tokens[kv_lcp]) {
        ++kv_lcp;
    }

    // Exact 100% live state match -> 0 tokens evaluated!
    if (kv_lcp == tokens.size() &&
        cur_pos_ == static_cast<int32_t>(tokens.size()) &&
        cached_tokens_.size() == tokens.size() &&
        std::equal(cached_tokens_.begin(), cached_tokens_.end(), tokens.begin()) &&
        last_next_token_ >= 0) {
        PrefillCacheResult res;
        res.next_token = last_next_token_;
        res.matched_prefix_tokens = static_cast<int32_t>(tokens.size());
        res.evaluated_tokens = 0;
        res.restored_gdn_snapshot = false;
        return res;
    }

    const size_t p_limit = std::min(kv_lcp, tokens.size() - 1);
    int32_t best_p = 0;
    int32_t snap_idx = -1;

    if (gdn_total_bytes_ == 0) {
        // Pure FullAttention: any KV prefix up to p_limit is valid without GDN state restore.
        best_p = static_cast<int32_t>(p_limit);
    } else {
        // Check live GDN + KV state first (zero restore cost).
        if (cur_pos_ > 0 &&
            static_cast<size_t>(cur_pos_) <= p_limit &&
            cached_tokens_.size() == static_cast<size_t>(cur_pos_) &&
            std::equal(cached_tokens_.begin(), cached_tokens_.end(), tokens.begin())) {
            best_p = cur_pos_;
        }
        // Check in-memory GDN snapshots whose prefix tokens are still resident in KV cache.
        for (size_t i = 0; i < prefix_snapshots_.size(); ++i) {
            const auto& snap = prefix_snapshots_[i];
            const size_t s_len = snap.tokens.size();
            if (s_len > 0 &&
                s_len <= p_limit &&
                static_cast<int32_t>(s_len) > best_p &&
                std::equal(snap.tokens.begin(), snap.tokens.end(), tokens.begin())) {
                best_p = static_cast<int32_t>(s_len);
                snap_idx = static_cast<int32_t>(i);
            }
        }
    }

    bool restored_gdn = false;
    if (best_p == 0) {
        reset_state();
    } else if (gdn_total_bytes_ == 0) {
        cur_pos_ = best_p;
        cached_tokens_.assign(tokens.begin(), tokens.begin() + best_p);
    } else if (snap_idx >= 0) {
        PrefixSnapshot snap = std::move(prefix_snapshots_[snap_idx]);
        prefix_snapshots_.erase(prefix_snapshots_.begin() + snap_idx);
        RX_HIP_CHECK(hipMemcpyAsync(d_gdn_base_, snap.gdn_state.data(), gdn_total_bytes_,
                                    hipMemcpyHostToDevice, stream_));
        prefix_snapshots_.push_back(std::move(snap)); // Touch as MRU
        cur_pos_ = best_p;
        cached_tokens_.assign(tokens.begin(), tokens.begin() + best_p);
        restored_gdn = true;
    }

    const size_t total = tokens.size();
    std::vector<size_t> milestones;
    if (save_checkpoint && opts_.max_prefix_snapshots > 0 && gdn_total_bytes_ > 0 &&
        total > static_cast<size_t>(best_p) + 4096) {
        const size_t span = total - static_cast<size_t>(best_p);
        const size_t avail_slots = std::max<size_t>(1, (opts_.max_prefix_snapshots > 2) ? (opts_.max_prefix_snapshots - 2) : 1);
        size_t stride = 4096;
        while ((span / stride) > avail_slots) {
            stride *= 2;
        }
        for (size_t m = ((static_cast<size_t>(best_p) / stride) + 1) * stride; m < total; m += stride) {
            milestones.push_back(m);
        }
        const size_t last_grid = ((total - 1) / 4096) * 4096;
        if (last_grid > static_cast<size_t>(best_p) && last_grid < total) {
            if (std::find(milestones.begin(), milestones.end(), last_grid) == milestones.end()) {
                milestones.push_back(last_grid);
                std::sort(milestones.begin(), milestones.end());
            }
        }
    }

    size_t ms_idx = 0;
    size_t off = static_cast<size_t>(best_p);
    while (off < total) {
        while (ms_idx < milestones.size() && milestones[ms_idx] <= off) {
            ++ms_idx;
        }
        const size_t stop = (ms_idx < milestones.size()) ? milestones[ms_idx] : total;
        const int chunk = static_cast<int>(std::min<size_t>(opts_.max_batch_tokens, stop - off));
        const bool is_last = (off + chunk == total);
        run_forward_batch(tokens.data() + off, chunk, is_last);
        off += chunk;
        if (off == stop && off < total) {
            record_prefix_snapshot(false);
            ++ms_idx;
        }
    }

    int32_t next_tok = 0;
    RX_HIP_CHECK(hipMemcpyAsync(&next_tok, d_out_token_, sizeof(next_tok), hipMemcpyDeviceToHost, stream_));
    RX_HIP_CHECK(hipStreamSynchronize(stream_));
    last_next_token_ = next_tok;

    if (save_checkpoint && opts_.max_prefix_snapshots > 0 && tokens.size() >= 8) {
        record_prefix_snapshot(false);
    }

    PrefillCacheResult res;
    res.next_token = next_tok;
    res.matched_prefix_tokens = best_p;
    res.evaluated_tokens = static_cast<int32_t>(total) - best_p;
    res.restored_gdn_snapshot = restored_gdn;
    return res;
}

void InferenceEngine::save_prompt_cache(const std::string& path) const {
    RX_HIP_CHECK(hipStreamSynchronize(stream_));
    const std::string tmp_path = path + ".rxtmp";
    std::ofstream ofs(tmp_path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open prompt cache output file: " + path);
    }

    const char magic[4] = {'R', 'X', 'P', 'C'};
    const uint32_t version = 1;
    const std::string& sig = cache_signature_;
    const uint64_t sig_len = sig.size();
    const int32_t pos = cur_pos_;
    const int32_t last_tok = last_next_token_;
    const uint64_t gdn_bytes = gdn_total_bytes_;
    const int32_t n_attn = static_cast<int32_t>(kv_caches_.size());
    const int32_t active_pages = (pos > 0 && n_attn > 0) ? static_cast<int32_t>(div_ceil(pos, attn::KV_PAGE_TOKENS)) : 0;
    const uint64_t page_bytes = kv_layout_.page_bytes;

    ofs.write(magic, 4);
    ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
    ofs.write(reinterpret_cast<const char*>(&sig_len), sizeof(sig_len));
    ofs.write(reinterpret_cast<const char*>(&pos), sizeof(pos));
    ofs.write(reinterpret_cast<const char*>(&last_tok), sizeof(last_tok));
    ofs.write(reinterpret_cast<const char*>(&gdn_bytes), sizeof(gdn_bytes));
    ofs.write(reinterpret_cast<const char*>(&n_attn), sizeof(n_attn));
    ofs.write(reinterpret_cast<const char*>(&active_pages), sizeof(active_pages));
    ofs.write(reinterpret_cast<const char*>(&page_bytes), sizeof(page_bytes));
    ofs.write(sig.data(), sig_len);

    if (pos > 0) {
        ofs.write(reinterpret_cast<const char*>(cached_tokens_.data()), static_cast<size_t>(pos) * sizeof(int32_t));
    }

    if (gdn_bytes > 0) {
        std::vector<uint8_t> gdn_host(gdn_bytes);
        RX_HIP_CHECK(hipMemcpy(gdn_host.data(), d_gdn_base_, gdn_bytes, hipMemcpyDeviceToHost));
        ofs.write(reinterpret_cast<const char*>(gdn_host.data()), gdn_bytes);
    }

    if (active_pages > 0 && page_bytes > 0) {
        constexpr size_t CHUNK = 16ULL << 20;
        std::vector<uint8_t> buf(std::min(CHUNK, static_cast<size_t>(active_pages) * page_bytes));
        for (int32_t a = 0; a < n_attn; ++a) {
            const int32_t vram_act = std::min(active_pages, vram_kv_pages_);
            const int32_t gtt_act = active_pages - vram_act;
            if (vram_act > 0) {
                const size_t total_vram_b = static_cast<size_t>(vram_act) * page_bytes;
                const uint8_t* src = static_cast<const uint8_t*>(d_kv_vram_pool_) +
                                     static_cast<size_t>(a) * vram_kv_pages_ * page_bytes;
                size_t off = 0;
                while (off < total_vram_b) {
                    const size_t n = std::min(CHUNK, total_vram_b - off);
                    RX_HIP_CHECK(hipMemcpy(buf.data(), src + off, n, hipMemcpyDeviceToHost));
                    ofs.write(reinterpret_cast<const char*>(buf.data()), n);
                    off += n;
                }
            }
            if (gtt_act > 0) {
                const size_t total_gtt_b = static_cast<size_t>(gtt_act) * page_bytes;
                const uint8_t* src = static_cast<const uint8_t*>(h_kv_gtt_pool_) +
                                     static_cast<size_t>(a) * gtt_kv_pages_ * page_bytes;
                ofs.write(reinterpret_cast<const char*>(src), total_gtt_b);
            }
        }
    }

    ofs.flush();
    if (!ofs) {
        ofs.close();
        (void)std::remove(tmp_path.c_str());
        throw std::runtime_error("Failed writing prompt cache file: " + path);
    }
    ofs.close();
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        (void)std::remove(tmp_path.c_str());
        throw std::runtime_error("Failed finalizing prompt cache file: " + path);
    }
}

int32_t InferenceEngine::load_prompt_cache(const std::string& path) {
    RX_HIP_CHECK(hipStreamSynchronize(stream_));
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open prompt cache file: " + path);
    }

    char magic[4]{};
    uint32_t version = 0;
    uint64_t sig_len = 0, gdn_bytes = 0, page_bytes = 0;
    int32_t pos = 0, last_tok = -1, n_attn = 0, active_pages = 0;

    ifs.read(magic, 4);
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    ifs.read(reinterpret_cast<char*>(&sig_len), sizeof(sig_len));
    ifs.read(reinterpret_cast<char*>(&pos), sizeof(pos));
    ifs.read(reinterpret_cast<char*>(&last_tok), sizeof(last_tok));
    ifs.read(reinterpret_cast<char*>(&gdn_bytes), sizeof(gdn_bytes));
    ifs.read(reinterpret_cast<char*>(&n_attn), sizeof(n_attn));
    ifs.read(reinterpret_cast<char*>(&active_pages), sizeof(active_pages));
    ifs.read(reinterpret_cast<char*>(&page_bytes), sizeof(page_bytes));

    if (!ifs || std::memcmp(magic, "RXPC", 4) != 0 || version != 1 || sig_len > (1ULL << 20)) {
        throw std::runtime_error("Invalid RXPC prompt cache header: " + path);
    }

    std::string sig(sig_len, '\0');
    ifs.read(sig.data(), sig_len);
    if (sig != cache_signature_) {
        throw std::runtime_error("Prompt cache signature mismatch (different model or --kv-dtype): " + path);
    }

    if (pos < 0 || pos > opts_.max_seq_len) {
        throw std::runtime_error("Prompt cache length (" + std::to_string(pos) +
                                 ") exceeds engine --max-seq-len (" + std::to_string(opts_.max_seq_len) + ")");
    }
    if (gdn_bytes != gdn_total_bytes_ || n_attn != static_cast<int32_t>(kv_caches_.size()) ||
        page_bytes != kv_layout_.page_bytes) {
        throw std::runtime_error("Prompt cache geometry mismatch: " + path);
    }
    const int32_t expected_pages =
        (pos > 0 && n_attn > 0) ? static_cast<int32_t>(div_ceil(pos, attn::KV_PAGE_TOKENS)) : 0;
    if (active_pages != expected_pages) {
        throw std::runtime_error("Prompt cache active_pages mismatch: " + path);
    }

    std::vector<int32_t> tokens(pos);
    if (pos > 0) {
        ifs.read(reinterpret_cast<char*>(tokens.data()), static_cast<size_t>(pos) * sizeof(int32_t));
    }

    std::vector<uint8_t> gdn_host;
    if (gdn_bytes > 0) {
        gdn_host.resize(gdn_bytes);
        ifs.read(reinterpret_cast<char*>(gdn_host.data()), gdn_bytes);
        RX_HIP_CHECK(hipMemcpy(d_gdn_base_, gdn_host.data(), gdn_bytes, hipMemcpyHostToDevice));
    }

    if (active_pages > 0 && page_bytes > 0) {
        constexpr size_t CHUNK = 16ULL << 20;
        std::vector<uint8_t> buf(std::min(CHUNK, static_cast<size_t>(active_pages) * page_bytes));
        for (int32_t a = 0; a < n_attn; ++a) {
            const int32_t vram_act = std::min(active_pages, vram_kv_pages_);
            const int32_t gtt_act = active_pages - vram_act;
            if (vram_act > 0) {
                const size_t total_vram_b = static_cast<size_t>(vram_act) * page_bytes;
                uint8_t* dst = static_cast<uint8_t*>(d_kv_vram_pool_) +
                               static_cast<size_t>(a) * vram_kv_pages_ * page_bytes;
                size_t off = 0;
                while (off < total_vram_b) {
                    const size_t n = std::min(CHUNK, total_vram_b - off);
                    ifs.read(reinterpret_cast<char*>(buf.data()), n);
                    RX_HIP_CHECK(hipMemcpy(dst + off, buf.data(), n, hipMemcpyHostToDevice));
                    off += n;
                }
            }
            if (gtt_act > 0) {
                const size_t total_gtt_b = static_cast<size_t>(gtt_act) * page_bytes;
                uint8_t* dst = static_cast<uint8_t*>(h_kv_gtt_pool_) +
                               static_cast<size_t>(a) * gtt_kv_pages_ * page_bytes;
                ifs.read(reinterpret_cast<char*>(dst), total_gtt_b);
            }
        }
    }

    if (!ifs) {
        throw std::runtime_error("Truncated prompt cache file: " + path);
    }

    cur_pos_ = pos;
    last_next_token_ = last_tok;
    cached_tokens_ = std::move(tokens);
    kv_resident_tokens_ = cached_tokens_;

    if (opts_.max_prefix_snapshots > 0 && pos > 0 && gdn_bytes > 0) {
        PrefixSnapshot snap;
        snap.tokens = cached_tokens_;
        snap.gdn_state = std::move(gdn_host);
        while (static_cast<int32_t>(prefix_snapshots_.size()) >= opts_.max_prefix_snapshots) {
            auto victim = prefix_snapshots_.begin();
            while (victim != prefix_snapshots_.end() && victim->pinned) {
                ++victim;
            }
            if (victim == prefix_snapshots_.end()) {
                victim = prefix_snapshots_.begin();
            }
            prefix_snapshots_.erase(victim);
        }
        prefix_snapshots_.push_back(std::move(snap));
    }

    return cur_pos_;
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

    if (static_cast<size_t>(cur_pos_) < kv_resident_tokens_.size()) {
        kv_resident_tokens_.resize(cur_pos_);
    }
    if (static_cast<size_t>(cur_pos_) == kv_resident_tokens_.size()) {
        kv_resident_tokens_.push_back(token_id);
    }
    cached_tokens_.push_back(token_id);
    cur_pos_++;
    last_next_token_ = next_tok;
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

    if (static_cast<size_t>(cur_pos_) < kv_resident_tokens_.size()) {
        kv_resident_tokens_.resize(cur_pos_);
    }
    if (static_cast<size_t>(cur_pos_) == kv_resident_tokens_.size()) {
        kv_resident_tokens_.insert(kv_resident_tokens_.end(), tokens.begin(), tokens.end());
    }
    cached_tokens_.insert(cached_tokens_.end(), tokens.begin(), tokens.end());
    cur_pos_ += M;
    last_next_token_ = preds.back();
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

    ensure_speculative_buffers();

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
        cached_tokens_.resize(cur_pos_);
        if (static_cast<size_t>(cur_pos_) < kv_resident_tokens_.size()) {
            kv_resident_tokens_.resize(cur_pos_);
        }
    }
    last_next_token_ = res.accepted_tokens.back();

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

    // Ensure KV pages have valid non-NaN zero contents when benchmarking arbitrary context lengths.
    kv_resident_tokens_.clear();
    prefix_snapshots_.clear();
    if (kv_vram_bytes_ > 0 && d_kv_vram_pool_ != nullptr) {
        RX_HIP_CHECK(hipMemsetAsync(d_kv_vram_pool_, 0, kv_vram_bytes_, stream_));
    }
    if (kv_gtt_bytes_ > 0 && h_kv_gtt_pool_ != nullptr) {
        std::memset(h_kv_gtt_pool_, 0, kv_gtt_bytes_);
    }

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

    // Fine-grained event profiling over `steps` steps
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
                                           1, 0, d_step_, decode_splits_, stream_);
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
                rec(ei++);
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
    reset_state();

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

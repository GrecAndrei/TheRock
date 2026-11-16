// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rxinfer/arena.hpp"
#include "rxinfer/attention.hpp"
#include "rxinfer/gdn.hpp"
#include "rxinfer/model.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace rxinfer {

struct EngineOptions {
    int32_t max_seq_len = 4096;
    int32_t max_batch_tokens = 512;
    int32_t max_verify_tokens = 8;     // Max tokens per speculative/MTP hipGraph verify step (1..16)
    DType kv_dtype = DType::F16;       // F16, FP8_E4M3, or FP4_E2M1_G16
    bool enable_hip_graph = true;
    double kv_vram_gb = -1.0;          // < 0: auto VRAM cap with GTT spillover; 0: all GTT; > 0: explicit VRAM cap (GiB)
    int32_t decode_attn_splits = 0;    // 0: auto (32 for short ctx, 64 for >= 16k ctx)
    int32_t max_prefix_snapshots = 8;  // Max host GDN checkpoints in RAM for in-memory LCP rollback
};

struct PromptCacheInfo {
    std::string signature;
    int32_t position = 0;
    int32_t last_next_token = -1;
    DType kv_dtype = DType::F16;
    std::vector<int32_t> tokens;
};

PromptCacheInfo inspect_prompt_cache(const std::string& path, bool load_tokens = false);

struct PrefillCacheResult {
    int32_t next_token = 0;
    int32_t matched_prefix_tokens = 0;
    int32_t evaluated_tokens = 0;
    bool restored_gdn_snapshot = false;
};

struct SpeculativeVerifyResult {
    std::vector<int32_t> accepted_tokens; // Accepted draft tokens + 1 bonus token
    int32_t num_draft_accepted = 0;       // Draft tokens accepted beyond tokens[0]
};

struct EngineBenchResult {
    int32_t tokens_generated = 0;
    double elapsed_ms = 0.0;
    double tok_per_sec = 0.0;
    double ms_per_tok = 0.0;
};

struct DecodeBreakdown {
    double embed_ms = 0.0;
    double rmsnorm_ms = 0.0;
    double qkv_proj_ms = 0.0;
    double attn_op_ms = 0.0;
    double gdn_ba_proj_ms = 0.0;
    double gdn_conv_recur_ms = 0.0;
    double o_proj_ms = 0.0;
    double mlp_gate_up_ms = 0.0;
    double mlp_act_ms = 0.0;
    double mlp_down_ms = 0.0;
    double lm_head_ms = 0.0;
    double argmax_ms = 0.0;
    double non_graph_wall_ms = 0.0;
    double graph_wall_ms = 0.0;
};

class InferenceEngine {
public:
    explicit InferenceEngine(ModelWeights weights, EngineOptions opts = {});
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&) = delete;
    InferenceEngine& operator=(InferenceEngine&&) = delete;

    void reset_state();

    // Prefill prompt tokens (automatically chunked by max_batch_tokens) and return next greedy token.
    int32_t prefill(const std::vector<int32_t>& tokens);

    // Prefix-cached prefill over full token history: reuses longest matching in-memory KV + GDN prefix
    // (restoring a GDN prefix snapshot if needed) and only evaluates the uncached suffix.
    PrefillCacheResult prefill_cached(const std::vector<int32_t>& tokens, bool save_checkpoint = true);

    // In-memory GDN prefix checkpoints & persistent disk prompt cache (.rxprompt)
    void save_prefix_checkpoint(bool pin = false);
    void unpin_prefix_checkpoints();
    bool restore_prefix(const std::vector<int32_t>& tokens);
    void clear_prefix_cache();
    void save_prompt_cache(const std::string& path) const;
    int32_t load_prompt_cache(const std::string& path);

    // Single-token decode step (uses captured hipGraph when enabled).
    int32_t decode_step(int32_t token_id);

    // Multi-token causal verification step (2 <= M <= max_verify_tokens, cached hipGraph per M).
    // Returns M greedy next-token predictions corresponding to tokens[0..M-1] and advances cur_pos_ += M.
    std::vector<int32_t> verify_tokens(const std::vector<int32_t>& tokens);

    // Speculative / MTP verification for draft_tokens = [cur_tok, d1, ..., d_{M-1}]:
    // Runs causal M-token verification, automatically rolls back KV + GDN state if a draft mismatches,
    // and returns newly accepted tokens ([d1..dk, bonus]).
    SpeculativeVerifyResult speculative_verify(const std::vector<int32_t>& draft_tokens);

    // Run prefill + decode and return last-token FP32 logits (for perplexity / parity verification).
    std::vector<float> forward_logits(const std::vector<int32_t>& tokens);

    // Read back the most recent step's logits without resetting state (after prefill or decode_step).
    void copy_last_logits_f16(std::vector<uint16_t>& out_bits) const;
    void copy_last_logits(std::vector<float>& out) const;

    // Benchmark steady-state decode using the captured hipGraph across `steps` tokens.
    EngineBenchResult benchmark_decode(int steps = 50, int warmup = 5);

    // Benchmark steady-state M-token hipGraph verification across `steps` verify batches.
    EngineBenchResult benchmark_verify(int M = 4, int steps = 30, int warmup = 5);

    // Benchmark prefill throughput across `num_tokens` tokens.
    EngineBenchResult benchmark_prefill(int num_tokens = 512, int warmup = 1);

    // Per-component GPU breakdown on single-token decode (at given kv_ctx_len).
    DecodeBreakdown profile_decode(int32_t kv_ctx_len = 512, int steps = 6, int warmup = 2);

    const ModelConfig& config() const { return weights_.config; }
    const ModelWeights& weights() const { return weights_; }
    const EngineOptions& options() const { return opts_; }
    int32_t position() const { return cur_pos_; }
    int32_t last_next_token() const { return last_next_token_; }
    const std::vector<int32_t>& cached_tokens() const { return cached_tokens_; }
    std::string cache_signature() const;
    int32_t num_prefix_checkpoints() const { return static_cast<int32_t>(prefix_snapshots_.size()); }
    size_t kv_vram_bytes() const { return kv_vram_bytes_; }
    size_t kv_gtt_bytes() const { return kv_gtt_bytes_; }
    int32_t vram_kv_tokens() const { return vram_kv_pages_ * attn::KV_PAGE_TOKENS; }
    int32_t gtt_kv_tokens() const { return gtt_kv_pages_ * attn::KV_PAGE_TOKENS; }
    size_t state_vram_bytes() const { return state_arena_.used() + speculative_vram_bytes_; }
    size_t scratch_vram_bytes() const { return scratch_arena_.used(); }
    size_t total_vram_used_bytes() const {
        return weights_.weight_arena.used() + state_vram_bytes() + scratch_arena_.used();
    }

private:
    struct PrefixSnapshot {
        std::vector<int32_t> tokens;
        std::vector<uint8_t> gdn_state;
        bool pinned = false;
    };

    void init_arenas();
    void ensure_speculative_buffers();
    void record_prefix_snapshot(bool pin = false);
    void capture_decode_graph();
    hipGraphExec_t get_or_capture_verify_graph(int M);
    void run_layer_decode(int32_t layer_idx);
    void run_layer_decode_step(int32_t layer_idx, int M, bool save_gdn_replay);
    void run_forward_batch(const int32_t* h_tokens, int M, bool compute_logits);

    ModelWeights weights_;
    EngineOptions opts_;

    DeviceArena state_arena_;
    DeviceArena scratch_arena_;

    std::vector<attn::KVCacheView> kv_caches_;
    std::vector<gdn::GDNStateView> gdn_states_;
    std::vector<int32_t> attn_idx_map_;  // layer -> kv_caches_ idx
    std::vector<int32_t> gdn_idx_map_;   // layer -> gdn_states_ idx
    std::vector<int32_t> gdn_layer_idx_; // gdn_states_ idx -> layer

    attn::AttentionParams attn_params_{};
    attn::KVPageLayout kv_layout_{};
    attn::PrefillScratch prefill_scratch_{};
    gdn::GDNParams gdn_params_{};

    int32_t decode_splits_ = 32;
    int32_t kv_total_pages_ = 0;
    int32_t vram_kv_pages_ = 0;
    int32_t gtt_kv_pages_ = 0;
    size_t kv_vram_bytes_ = 0;
    size_t kv_gtt_bytes_ = 0;
    void* d_kv_vram_pool_ = nullptr;
    void* h_kv_gtt_pool_ = nullptr;
    void* d_kv_gtt_pool_ = nullptr;
    std::vector<uint8_t**> d_page_tables_;

    // Contiguous GDN state + lazily allocated speculative rollback buffers
    void* d_gdn_base_ = nullptr;
    size_t gdn_total_bytes_ = 0;
    void* d_gdn_backup_ = nullptr;
    __half* d_gdn_replay_qkv_ = nullptr;
    __half* d_gdn_replay_ba_ = nullptr;
    size_t speculative_vram_bytes_ = 0;

    // Scratch device pointers
    StepParams* d_step_ = nullptr;
    int32_t* d_out_token_ = nullptr;
    int32_t* d_tokens_in_ = nullptr;
    __half* d_x_ = nullptr;
    __half* d_residual_ = nullptr;
    __half* d_norm_out_ = nullptr;
    __half* d_proj_a_ = nullptr;
    __half* d_proj_b_ = nullptr;
    __half* d_proj_c_ = nullptr;
    __half* d_attn_out_ = nullptr;
    __half* d_ba_buf_ = nullptr;
    void* d_attn_split_ = nullptr;
    float* d_gemm_tmp_ = nullptr;
    __half* d_logits_ = nullptr;
    void* d_argmax_tmp_ = nullptr;

    hipStream_t stream_ = nullptr;
    hipGraph_t decode_graph_ = nullptr;
    hipGraphExec_t decode_graph_exec_ = nullptr;
    std::unordered_map<int, hipGraph_t> verify_graphs_;
    std::unordered_map<int, hipGraphExec_t> verify_graph_execs_;
    int32_t cur_pos_ = 0;
    int32_t last_next_token_ = -1;
    std::string cache_signature_;
    std::vector<int32_t> cached_tokens_;
    std::vector<int32_t> kv_resident_tokens_;
    std::vector<PrefixSnapshot> prefix_snapshots_;
};

} // namespace rxinfer

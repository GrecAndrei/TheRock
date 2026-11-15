// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/common.hpp"
#include "rxinfer/config.hpp"
#include "rxinfer/engine.hpp"
#include "rxinfer/loader.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

rxinfer::ModelConfig make_qwen38_27b_config(int num_layers = 64) {
    rxinfer::ModelConfig cfg{};
    cfg.model_type = "qwen3_5";
    cfg.arch_name = "Qwen3_5ForConditionalGeneration";
    cfg.vocab_size = 248320;
    cfg.hidden_size = 5120;
    cfg.intermediate_size = 17408;
    cfg.num_hidden_layers = num_layers;
    cfg.max_position_embeddings = 262144;
    cfg.tie_word_embeddings = false;
    cfg.norm_kind = rxinfer::NormKind::RMSNormPlusOne;
    cfg.rms_norm_eps = 1e-6f;

    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 4;
    cfg.head_dim = 256;
    cfg.qk_norm = true;
    cfg.attn_output_gate = true;

    cfg.rope_kind = rxinfer::RopeKind::NeoX;
    cfg.rope_theta = 10000000.0f;
    cfg.partial_rotary_factor = 0.25f;
    cfg.mrope_section = {11, 11, 10};
    cfg.mrope_interleaved = true;

    cfg.linear_num_key_heads = 16;
    cfg.linear_key_head_dim = 128;
    cfg.linear_num_value_heads = 48;
    cfg.linear_value_head_dim = 128;
    cfg.linear_conv_kernel_dim = 4;

    cfg.layer_types.clear();
    for (int l = 0; l < num_layers; ++l) {
        // 3x GDN + 1x FullAttention repeating block
        cfg.layer_types.push_back((l % 4 == 3) ? rxinfer::LayerKind::FullAttention
                                               : rxinfer::LayerKind::LinearAttentionGDN);
    }
    return cfg;
}

rxinfer::ModelConfig make_small_hybrid_test_config() {
    rxinfer::ModelConfig cfg{};
    cfg.model_type = "qwen3_5";
    cfg.vocab_size = 1024;
    cfg.hidden_size = 512;
    cfg.intermediate_size = 1024;
    cfg.num_hidden_layers = 4; // 3x GDN + 1x FullAttention
    cfg.max_position_embeddings = 1024;
    cfg.tie_word_embeddings = false;
    cfg.norm_kind = rxinfer::NormKind::RMSNormPlusOne;
    cfg.rms_norm_eps = 1e-6f;

    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 128;
    cfg.qk_norm = true;
    cfg.attn_output_gate = true;

    cfg.rope_kind = rxinfer::RopeKind::NeoX;
    cfg.rope_theta = 10000000.0f;
    cfg.partial_rotary_factor = 0.5f;

    cfg.linear_num_key_heads = 2;
    cfg.linear_key_head_dim = 128;
    cfg.linear_num_value_heads = 4;
    cfg.linear_value_head_dim = 128;
    cfg.linear_conv_kernel_dim = 4;

    cfg.layer_types = {
        rxinfer::LayerKind::LinearAttentionGDN,
        rxinfer::LayerKind::LinearAttentionGDN,
        rxinfer::LayerKind::LinearAttentionGDN,
        rxinfer::LayerKind::FullAttention,
    };
    return cfg;
}

rxinfer::ModelConfig make_small_llama_test_config() {
    const char* json_str = R"({
      "model_type": "qwen3",
      "architectures": ["Qwen3ForCausalLM"],
      "vocab_size": 1024,
      "hidden_size": 512,
      "intermediate_size": 1024,
      "num_hidden_layers": 4,
      "max_position_embeddings": 1024,
      "num_attention_heads": 8,
      "num_key_value_heads": 2,
      "head_dim": 64,
      "rms_norm_eps": 1e-6,
      "rope_theta": 1000000.0,
      "tie_word_embeddings": true
    })";
    return rxinfer::ModelConfig::from_hf_json_string(json_str);
}

double cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

int test_parity(const char* label, const rxinfer::ModelConfig& cfg, rxinfer::DType kv_dt) {
    auto w = rxinfer::load_synthetic_model(cfg, 42);
    rxinfer::EngineOptions opts{};
    opts.max_seq_len = 512;
    opts.max_batch_tokens = 32;
    opts.kv_dtype = kv_dt;
    opts.enable_hip_graph = true;

    rxinfer::InferenceEngine engine(std::move(w), opts);

    // 1. Compare batched prefill([10, 25, 77, 13, 99, 5, 41, 88]) vs sequential decode_step!
    //    Wait: use M=22 tokens (>16 WMMA prefill threshold!) to exercise BOTH prefill WMMA + FlashAttn
    //    AND single-token hipGraph decode!
    std::vector<int32_t> tokens;
    for (int i = 0; i < 22; ++i) tokens.push_back((i * 37 + 11) % cfg.vocab_size);

    auto logits_prefill = engine.forward_logits(tokens);

    engine.reset_state();
    int32_t last_tok = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
        last_tok = engine.decode_step(tokens[i]);
    }
    // Re-run forward_logits for last step? Wait: decode_step doesn't return logits directly,
    // so run prefill(tokens[0..20]) + decode_step(tokens[21])? Even better: compare argmax token
    // and also compare prefill(tokens[0..19]) + 2x decode_step vs full prefill!
    int32_t pre_tok = engine.prefill(tokens); // Wait: resets needed first
    engine.reset_state();
    pre_tok = engine.prefill(tokens);

    engine.reset_state();
    int32_t seq_tok = 0;
    for (int32_t t : tokens) seq_tok = engine.decode_step(t);

    // Also test prefill(first 18) + decode(4 remaining)
    engine.reset_state();
    std::vector<int32_t> head_toks(tokens.begin(), tokens.begin() + 18);
    (void)engine.prefill(head_toks);
    int32_t mix_tok = 0;
    for (size_t i = 18; i < tokens.size(); ++i) mix_tok = engine.decode_step(tokens[i]);

    // Test M=4 verify_tokens parity vs sequential decode_step (tokens[0..19] in 5 chunks of 4 + 2 decode)
    engine.reset_state();
    int32_t ver4_tok = 0;
    for (size_t off = 0; off < 20; off += 4) {
        std::vector<int32_t> chunk(tokens.begin() + off, tokens.begin() + off + 4);
        auto preds = engine.verify_tokens(chunk);
        ver4_tok = preds.back();
    }
    for (size_t i = 20; i < tokens.size(); ++i) ver4_tok = engine.decode_step(tokens[i]);

    // Test speculative_verify with partial acceptance + KV/GDN rollback:
    // Build a sequence `spec_seq` where spec_seq[9] equals the model's greedy prediction at pos=8!
    engine.reset_state();
    std::vector<int32_t> spec_seq = tokens;
    std::vector<int32_t> spec_ref_preds;
    for (size_t i = 0; i < spec_seq.size(); ++i) {
        if (i == 9) spec_seq[9] = spec_ref_preds[8];
        spec_ref_preds.push_back(engine.decode_step(spec_seq[i]));
    }

    engine.reset_state();
    for (int i = 0; i < 8; ++i) (void)engine.decode_step(spec_seq[i]);
    const int32_t bad_tok = (spec_ref_preds[9] + 1) % cfg.vocab_size;
    auto spec_res = engine.speculative_verify({spec_seq[8], spec_seq[9], bad_tok, bad_tok});
    bool spec_ok = (spec_res.num_draft_accepted == 1) &&
                   (spec_res.accepted_tokens.size() == 2) &&
                   (spec_res.accepted_tokens[0] == spec_seq[9]) &&
                   (spec_res.accepted_tokens[1] == spec_ref_preds[9]) &&
                   (engine.position() == 10);
    int32_t spec_final_tok = 0;
    for (size_t i = 10; i < spec_seq.size(); ++i) spec_final_tok = engine.decode_step(spec_seq[i]);
    if (!spec_ok || spec_final_tok != spec_ref_preds.back()) {
        std::printf("  SPECULATIVE ROLLBACK MISMATCH on %s! spec_ok=%d pos=%d acc=%zu spec_final=%d ref=%d\n",
                    label, spec_ok ? 1 : 0, engine.position(), spec_res.accepted_tokens.size(),
                    spec_final_tok, spec_ref_preds.back());
        return 1;
    }

    const double cos_self = cosine_sim(logits_prefill, logits_prefill);
    std::printf("[Parity %-20s kv=%-12s] pre=%d dec=%d mix=%d ver4=%d spec=ok (self=%.6f)\n",
                label, std::string(rxinfer::dtype_name(kv_dt)).c_str(),
                pre_tok, seq_tok, mix_tok, ver4_tok, cos_self);

    if (pre_tok != seq_tok || pre_tok != mix_tok || pre_tok != ver4_tok) {
        std::printf("  WARNING/MISMATCH on %s!\n", label);
        return 1;
    }
    return 0;
}

int test_rxr4_roundtrip() {
    auto cfg = make_small_hybrid_test_config();
    auto w1 = rxinfer::load_synthetic_model(cfg, 123);
    const std::string tmp_path = (std::filesystem::temp_directory_path() / "rxinfer_test_model.rxr4").string();
    rxinfer::save_rxr4_checkpoint(w1, tmp_path);

    rxinfer::EngineOptions opts{};
    opts.max_seq_len = 256;
    opts.max_batch_tokens = 32;
    opts.kv_dtype = rxinfer::DType::FP8_E4M3;

    rxinfer::InferenceEngine eng1(std::move(w1), opts);
    auto w2 = rxinfer::load_from_rxr4(tmp_path);
    rxinfer::InferenceEngine eng2(std::move(w2), opts);

    std::vector<int32_t> prompt = {5, 17, 42, 108, 301};
    auto l1 = eng1.forward_logits(prompt);
    auto l2 = eng2.forward_logits(prompt);
    const double cos = cosine_sim(l1, l2);
    std::printf("[RXR4 Roundtrip] cos=%.8f\n", cos);
    return (cos > 0.999999) ? 0 : 1;
}

int test_multichunk_and_multiturn_parity(const char* label, const rxinfer::ModelConfig& cfg, rxinfer::DType kv_dt) {
    auto w_single = rxinfer::load_synthetic_model(cfg, 77);
    auto w_multi = rxinfer::load_synthetic_model(cfg, 77);

    rxinfer::EngineOptions opts_single{};
    opts_single.max_seq_len = 512;
    opts_single.max_batch_tokens = 96;
    opts_single.kv_dtype = kv_dt;
    opts_single.enable_hip_graph = true;

    rxinfer::EngineOptions opts_multi{};
    opts_multi.max_seq_len = 512;
    opts_multi.max_batch_tokens = 16; // Force multi-chunk prefill (e.g. 16+16+11)
    opts_multi.kv_dtype = kv_dt;
    opts_multi.enable_hip_graph = true;

    rxinfer::InferenceEngine eng_single(std::move(w_single), opts_single);
    rxinfer::InferenceEngine eng_multi(std::move(w_multi), opts_multi);

    // 1. Multi-chunk prefill (43 tokens across 3 chunks: 16 + 16 + 11) vs single-chunk prefill
    std::vector<int32_t> long_prompt;
    for (int i = 0; i < 43; ++i) long_prompt.push_back((i * 53 + 7) % cfg.vocab_size);
    auto logits_ref = eng_single.forward_logits(long_prompt);
    auto logits_chunked = eng_multi.forward_logits(long_prompt);
    const double cos_chunk = cosine_sim(logits_ref, logits_chunked);

    // 2. Multi-turn incremental prefill + decode_step vs single-shot full-history prefill
    eng_multi.reset_state();
    std::vector<int32_t> turn1;
    for (int i = 0; i < 21; ++i) turn1.push_back((i * 29 + 3) % cfg.vocab_size);
    int32_t tok = eng_multi.prefill(turn1);

    std::vector<int32_t> transcript = turn1;
    for (int s = 0; s < 5; ++s) {
        transcript.push_back(tok);
        tok = eng_multi.decode_step(tok);
    }
    // Simulate turn stop on `tok` (not yet fed) -> Turn 2 prepends `tok` + turn2 tokens
    std::vector<int32_t> turn2_delta = {tok};
    for (int i = 0; i < 19; ++i) turn2_delta.push_back((i * 41 + 19) % cfg.vocab_size);
    transcript.insert(transcript.end(), turn2_delta.begin(), turn2_delta.end());

    const int32_t inc_pred = eng_multi.prefill(turn2_delta);
    std::vector<float> logits_inc;
    eng_multi.copy_last_logits(logits_inc);

    eng_single.reset_state();
    const int32_t ref_pred = eng_single.prefill(transcript);
    std::vector<float> logits_full;
    eng_single.copy_last_logits(logits_full);
    const double cos_turn = cosine_sim(logits_full, logits_inc);

    std::printf("[MultiChunk/Turn %-16s kv=%-12s] chunk_cos=%.7f turn_cos=%.7f pred=(%d vs %d)\n",
                label, std::string(rxinfer::dtype_name(kv_dt)).c_str(),
                cos_chunk, cos_turn, inc_pred, ref_pred);

    if (cos_chunk < 0.99999 || cos_turn < 0.99999) {
        std::printf("  [FAIL] MultiChunk/Turn parity mismatch on %s\n", label);
        return 1;
    }
    return 0;
}

int test_prompt_cache_and_tiered_gtt(const char* label, const rxinfer::ModelConfig& cfg, rxinfer::DType kv_dt) {
    auto w1 = rxinfer::load_synthetic_model(cfg, 313);
    auto w2 = rxinfer::load_synthetic_model(cfg, 313);
    auto w3 = rxinfer::load_synthetic_model(cfg, 313);

    rxinfer::EngineOptions opts_vram{};
    opts_vram.max_seq_len = 512;
    opts_vram.max_batch_tokens = 64;
    opts_vram.kv_dtype = kv_dt;
    opts_vram.enable_hip_graph = true;

    rxinfer::EngineOptions opts_gtt = opts_vram;
    opts_gtt.kv_vram_gb = 0.0; // Force 100% mapped host GTT KV cache!

    rxinfer::InferenceEngine eng_vram(std::move(w1), opts_vram);
    rxinfer::InferenceEngine eng_gtt(std::move(w2), opts_gtt);
    rxinfer::InferenceEngine eng_ref(std::move(w3), opts_vram);

    // 1. Build shared 140-token prefix + two branching suffixes (to span multiple 64-tok pages!)
    std::vector<int32_t> prefix(140);
    for (int i = 0; i < 140; ++i) prefix[i] = (i * 29 + 5) % cfg.vocab_size;

    std::vector<int32_t> prompt_a = prefix;
    for (int i = 0; i < 36; ++i) prompt_a.push_back((i * 17 + 91) % cfg.vocab_size);

    std::vector<int32_t> prompt_b = prefix;
    for (int i = 0; i < 42; ++i) prompt_b.push_back((i * 31 + 123) % cfg.vocab_size);

    // Save prefix checkpoint into eng_vram, extend with prompt_a, generate 4 tokens, then branch to prompt_b!
    auto r_pre = eng_vram.prefill_cached(prefix, true);
    if (r_pre.matched_prefix_tokens != 0 || r_pre.evaluated_tokens != 140) {
        std::printf("  [FAIL] Unexpected initial prefill_cached stats on %s\n", label);
        return 1;
    }
    const std::string cache_path =
        (std::filesystem::temp_directory_path() / "rxinfer_test_prefix.rxprompt").string();
    eng_vram.save_prompt_cache(cache_path);

    auto r_hit = eng_vram.prefill_cached(prefix, true);
    if (r_hit.matched_prefix_tokens != 140 || r_hit.evaluated_tokens != 0 || r_hit.next_token != r_pre.next_token) {
        std::printf("  [FAIL] Exact 100%% cache hit mismatch on %s\n", label);
        return 1;
    }

    auto r_a = eng_vram.prefill_cached(prompt_a, true);
    if (r_a.matched_prefix_tokens != 140 || r_a.evaluated_tokens != 36) {
        std::printf("  [FAIL] Prefix extension mismatch on %s (matched=%d eval=%d)\n",
                    label, r_a.matched_prefix_tokens, r_a.evaluated_tokens);
        return 1;
    }
    int32_t t_a = r_a.next_token;
    for (int s = 0; s < 4; ++s) t_a = eng_vram.decode_step(t_a);

    // Branch to prompt_b (must rewind via GDN snapshot for hybrid or LCP KV for dense!)
    auto r_b = eng_vram.prefill_cached(prompt_b, true);
    std::vector<float> logits_b_cached;
    eng_vram.copy_last_logits(logits_b_cached);

    auto logits_b_ref = eng_ref.forward_logits(prompt_b);
    const double cos_branch = cosine_sim(logits_b_ref, logits_b_cached);
    if (r_b.matched_prefix_tokens != 140 || r_b.evaluated_tokens != 42 || cos_branch < 0.999999) {
        std::printf("  [FAIL] LCP branch parity on %s (matched=%d eval=%d cos=%.7f)\n",
                    label, r_b.matched_prefix_tokens, r_b.evaluated_tokens, cos_branch);
        return 1;
    }

    // 2. Load VRAM-saved .rxprompt into 100%-GTT engine (tests cross-tier .rxprompt + GTT staging prefill + decode!)
    const int32_t loaded_pos = eng_gtt.load_prompt_cache(cache_path);
    auto r_gtt_b = eng_gtt.prefill_cached(prompt_b, false);
    std::vector<float> logits_gtt_b;
    eng_gtt.copy_last_logits(logits_gtt_b);
    const double cos_gtt_b = cosine_sim(logits_b_ref, logits_gtt_b);

    const int32_t dec_gtt = eng_gtt.decode_step(r_gtt_b.next_token);
    const int32_t dec_ref = eng_ref.decode_step(r_gtt_b.next_token);
    std::vector<float> logits_dec_gtt, logits_dec_ref;
    eng_gtt.copy_last_logits(logits_dec_gtt);
    eng_ref.copy_last_logits(logits_dec_ref);
    const double cos_gtt_dec = cosine_sim(logits_dec_ref, logits_dec_gtt);

    std::printf("[PromptCache+GTT %-16s kv=%-12s] lcp_cos=%.7f disk_gtt_cos=%.7f dec_gtt_cos=%.7f tok=(%d vs %d)\n",
                label, std::string(rxinfer::dtype_name(kv_dt)).c_str(),
                cos_branch, cos_gtt_b, cos_gtt_dec, dec_gtt, dec_ref);

    if (loaded_pos != 140 || r_gtt_b.matched_prefix_tokens != 140 ||
        cos_gtt_b < 0.999999 || cos_gtt_dec < 0.999999 || dec_gtt != dec_ref) {
        std::printf("  [FAIL] Cross-tier .rxprompt/GTT mismatch on %s\n", label);
        return 1;
    }
    return 0;
}

int test_periodic_checkpoints_and_restore() {
    auto cfg = make_small_hybrid_test_config();
    rxinfer::LoaderOptions lo{};
    auto w1 = rxinfer::load_synthetic_model(cfg, 321, lo);
    auto w2 = rxinfer::load_synthetic_model(cfg, 321, lo);

    rxinfer::EngineOptions eo{};
    eo.max_seq_len = 8192;
    eo.max_batch_tokens = 256;
    eo.kv_dtype = rxinfer::DType::FP4_E2M1_G16;
    eo.enable_hip_graph = false;
    eo.max_prefix_snapshots = 4;

    rxinfer::InferenceEngine eng(std::move(w1), eo);
    rxinfer::InferenceEngine eng_ref(std::move(w2), eo);

    std::vector<int32_t> long_toks(4600);
    for (int i = 0; i < 4600; ++i) {
        long_toks[i] = (i * 19 + 5) % cfg.vocab_size;
    }
    std::vector<int32_t> sys_toks(long_toks.begin(), long_toks.begin() + 32);
    std::vector<int32_t> grid_4096(long_toks.begin(), long_toks.begin() + 4096);
    std::vector<int32_t> branch_4104(long_toks.begin(), long_toks.begin() + 4104);
    for (int i = 4096; i < 4104; ++i) branch_4104[i] = (i * 7 + 1) % cfg.vocab_size;

    // 1. Pin system prompt snapshot
    eng.prefill_cached(sys_toks, true);
    eng.save_prefix_checkpoint(true);

    // 2. Prefill > 4096 tokens: should auto-record periodic 4096-grid checkpoint
    auto r_long = eng.prefill_cached(long_toks, true);
    if (r_long.matched_prefix_tokens != 32 || r_long.evaluated_tokens != 4568) {
        std::printf("  [FAIL] System prefix reuse mismatch (matched=%d eval=%d)\n",
                    r_long.matched_prefix_tokens, r_long.evaluated_tokens);
        return 1;
    }

    // 3. Rewind directly to 4096 via restore_prefix -> must be O(1) hit!
    if (!eng.restore_prefix(grid_4096) || eng.position() != 4096) {
        std::printf("  [FAIL] Periodic 4096-grid GDN checkpoint was not hit by restore_prefix()\n");
        return 1;
    }

    // 4. Branch from 4096 to 4104 tokens -> must match 4096 cached tokens and match fresh reference logits
    auto r_branch = eng.prefill_cached(branch_4104, false);
    std::vector<float> logits_branch, logits_ref;
    eng.copy_last_logits(logits_branch);
    logits_ref = eng_ref.forward_logits(branch_4104);
    const double cos_grid = cosine_sim(logits_ref, logits_branch);
    if (r_branch.matched_prefix_tokens != 4096 || r_branch.evaluated_tokens != 8 || cos_grid < 0.999999) {
        std::printf("  [FAIL] 4096-grid branch parity failed (matched=%d eval=%d cos=%.7f)\n",
                    r_branch.matched_prefix_tokens, r_branch.evaluated_tokens, cos_grid);
        return 1;
    }

    // 5. Fill 6 more unpinned checkpoints (exceeding max_prefix_snapshots=4) and verify pinned sys_toks survives!
    for (int k = 0; k < 6; ++k) {
        std::vector<int32_t> tmp(sys_toks);
        for (int j = 0; j < 12 + k; ++j) tmp.push_back((k * 31 + j + 7) % cfg.vocab_size);
        eng.prefill_cached(tmp, true);
    }
    if (!eng.restore_prefix(sys_toks)) {
        std::printf("  [FAIL] Pinned system snapshot was evicted!\n");
        return 1;
    }

    // 6. Save .rxprompt atomically and verify inspect_prompt_cache()
    const std::string pcache = "/tmp/rxinfer_periodic_test.rxprompt";
    eng.save_prompt_cache(pcache);
    auto info = rxinfer::inspect_prompt_cache(pcache, true);
    std::remove(pcache.c_str());
    if (info.position != 32 || info.kv_dtype != rxinfer::DType::FP4_E2M1_G16 || info.tokens != sys_toks) {
        std::printf("  [FAIL] inspect_prompt_cache mismatch (pos=%d kv=%d toks=%zu)\n",
                    info.position, static_cast<int>(info.kv_dtype), info.tokens.size());
        return 1;
    }

    std::printf("  [PASS] Periodic 4096-grid GDN checkpoints + Pinned Sys + restore_prefix + inspect_prompt_cache\n");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    auto info = rxinfer::query_device_info(0);
    std::printf("GPU: %s (%s) | VRAM: %.2f GiB | WGPs: %d\n\n",
                info.name.c_str(), info.gcn_arch_name.c_str(),
                info.total_vram_bytes / (1024.0 * 1024.0 * 1024.0),
                info.multiprocessor_count);

    int failures = 0;
    failures += test_parity("Qwen3-Dense-F16", make_small_llama_test_config(), rxinfer::DType::F16);
    failures += test_parity("Hybrid-GDN-F16", make_small_hybrid_test_config(), rxinfer::DType::F16);
    failures += test_parity("Hybrid-GDN-FP8", make_small_hybrid_test_config(), rxinfer::DType::FP8_E4M3);
    failures += test_parity("Hybrid-GDN-FP4", make_small_hybrid_test_config(), rxinfer::DType::FP4_E2M1_G16);

    // Non-multiple-of-16 / 64 vocab_size parity (exercises R4 tile bounds across M=1, M=4, M=22)
    {
        auto cfg_odd_v = make_small_hybrid_test_config();
        cfg_odd_v.vocab_size = 505;
        failures += test_parity("Hybrid-OddV505-F16", cfg_odd_v, rxinfer::DType::F16);
    }

    failures += test_multichunk_and_multiturn_parity("Qwen3-Dense-F16", make_small_llama_test_config(), rxinfer::DType::F16);
    failures += test_multichunk_and_multiturn_parity("Hybrid-GDN-F16", make_small_hybrid_test_config(), rxinfer::DType::F16);
    failures += test_multichunk_and_multiturn_parity("Hybrid-GDN-FP4", make_small_hybrid_test_config(), rxinfer::DType::FP4_E2M1_G16);

    failures += test_prompt_cache_and_tiered_gtt("Qwen3-Dense-F16", make_small_llama_test_config(), rxinfer::DType::F16);
    failures += test_prompt_cache_and_tiered_gtt("Hybrid-GDN-F16", make_small_hybrid_test_config(), rxinfer::DType::F16);
    failures += test_prompt_cache_and_tiered_gtt("Hybrid-GDN-FP4", make_small_hybrid_test_config(), rxinfer::DType::FP4_E2M1_G16);

    failures += test_rxr4_roundtrip();
    failures += test_periodic_checkpoints_and_restore();

    if (failures != 0) {
        std::printf("\nParity tests failed (%d)!\n", failures);
        return 1;
    }

    // Full Qwen3.8-27B (64 layers: 48x GDN + 16x FullAttn) End-to-End Hardware Benchmark
    const bool run_27b = (argc <= 1 || std::string(argv[1]) != "--quick");
    if (run_27b) {
        std::printf("\n=== Full Qwen3.8-27B (64 Layers: 48x GDN + 16x FullAttn D=256) RX 9070 XT Benchmark ===\n");
        auto cfg27 = make_qwen38_27b_config(64);
        rxinfer::LoaderOptions lopts{};
        lopts.fast_synthetic = true;

        std::printf("Estimated weights: %.2f GiB | KV/tok (FP4): %.1f KB | KV/tok (F16): %.1f KB | GDN state: %.1f MB\n",
                    rxinfer::estimate_model_weight_bytes(cfg27) / (1024.0 * 1024.0 * 1024.0),
                    (cfg27.num_full_attention_layers() * 2.0 * cfg27.num_key_value_heads * cfg27.head_dim * 9.0 / 16.0) / 1024.0,
                    (cfg27.num_full_attention_layers() * 2.0 * cfg27.num_key_value_heads * cfg27.head_dim * 2.0) / 1024.0,
                    cfg27.estimate_recurrent_state_bytes() / (1024.0 * 1024.0));

        auto w27 = rxinfer::load_synthetic_model(cfg27, 42, lopts);
        rxinfer::EngineOptions eopts{};
        eopts.max_seq_len = 200000;
        eopts.max_batch_tokens = 512;
        eopts.kv_dtype = rxinfer::DType::FP4_E2M1_G16;
        eopts.enable_hip_graph = true;

        rxinfer::InferenceEngine eng27(std::move(w27), eopts);
        std::printf("Allocated VRAM total: %.2f GiB (weights %.2f GiB, state %.1f MB, scratch %.1f MB)\n"
                    "KV Pool (200k FP4): VRAM=%.2f GiB (%d tok) + GTT=%.2f GiB (%d tok)\n",
                    eng27.total_vram_used_bytes() / (1024.0 * 1024.0 * 1024.0),
                    eng27.weights().weight_arena.used() / (1024.0 * 1024.0 * 1024.0),
                    eng27.state_vram_bytes() / (1024.0 * 1024.0),
                    eng27.scratch_vram_bytes() / (1024.0 * 1024.0),
                    eng27.kv_vram_bytes() / (1024.0 * 1024.0 * 1024.0), eng27.vram_kv_tokens(),
                    eng27.kv_gtt_bytes() / (1024.0 * 1024.0 * 1024.0), eng27.gtt_kv_tokens());

        auto pre_res = eng27.benchmark_prefill(512, 1);
        std::printf("Prefill (512 tokens):   %.1f ms | %.1f tok/s\n", pre_res.elapsed_ms, pre_res.tok_per_sec);

        auto dec_res = eng27.benchmark_decode(60, 8);
        std::printf("Decode  (hipGraph M=1): %.2f ms/tok | %.2f tok/s\n", dec_res.ms_per_tok, dec_res.tok_per_sec);

        auto ver4_res = eng27.benchmark_verify(4, 30, 5);
        std::printf("Verify  (hipGraph M=4): %.2f ms/step | %.2f tok/s (100%% accept; ~%.2f tok/s @3/4)\n",
                    ver4_res.ms_per_tok * 4.0, ver4_res.tok_per_sec, ver4_res.tok_per_sec * 0.75);

        auto ver8_res = eng27.benchmark_verify(8, 25, 5);
        std::printf("Verify  (hipGraph M=8): %.2f ms/step | %.2f tok/s (100%% accept; ~%.2f tok/s @5/8)\n",
                    ver8_res.ms_per_tok * 8.0, ver8_res.tok_per_sec, ver8_res.tok_per_sec * 0.625);

        for (int32_t ctx_len : {512, 32768, 98304, 150000, 200000}) {
            auto bd_ctx = eng27.profile_decode(ctx_len, 4, 2);
            std::printf("Decode @ ctx=%6d: non-graph=%.3f ms | hipGraph=%.3f ms (%.2f tok/s, attn=%.3f ms)\n",
                        ctx_len, bd_ctx.non_graph_wall_ms, bd_ctx.graph_wall_ms,
                        1000.0 / bd_ctx.graph_wall_ms, bd_ctx.attn_op_ms);
        }

        auto bd = eng27.profile_decode(512, 6, 2);
        const double sum_gpu = bd.embed_ms + bd.rmsnorm_ms + bd.qkv_proj_ms + bd.attn_op_ms +
                               bd.gdn_ba_proj_ms + bd.gdn_conv_recur_ms + bd.o_proj_ms +
                               bd.mlp_gate_up_ms + bd.mlp_act_ms + bd.mlp_down_ms +
                               bd.lm_head_ms + bd.argmax_ms;
        std::printf("\n--- 27B Decode Breakdown (ctx=512) ---\n"
                    "  embed:          %6.3f ms\n"
                    "  rmsnorm (129x): %6.3f ms\n"
                    "  qkv_proj (64x): %6.3f ms\n"
                    "  attn_op (16x):  %6.3f ms\n"
                    "  gdn_ba (48x):   %6.3f ms\n"
                    "  gdn_recur(48x): %6.3f ms\n"
                    "  o_proj (64x):   %6.3f ms\n"
                    "  mlp_gu (64x):   %6.3f ms\n"
                    "  mlp_act (64x):  %6.3f ms\n"
                    "  mlp_down (64x): %6.3f ms\n"
                    "  lm_head:        %6.3f ms\n"
                    "  argmax:         %6.3f ms\n"
                    "  SUM GPU kernels:%6.3f ms\n"
                    "  Non-graph wall: %6.3f ms (%.2f tok/s)\n"
                    "  hipGraph wall:  %6.3f ms (%.2f tok/s)\n",
                    bd.embed_ms, bd.rmsnorm_ms, bd.qkv_proj_ms, bd.attn_op_ms,
                    bd.gdn_ba_proj_ms, bd.gdn_conv_recur_ms, bd.o_proj_ms,
                    bd.mlp_gate_up_ms, bd.mlp_act_ms, bd.mlp_down_ms,
                    bd.lm_head_ms, bd.argmax_ms, sum_gpu,
                    bd.non_graph_wall_ms, 1000.0 / bd.non_graph_wall_ms,
                    bd.graph_wall_ms, 1000.0 / bd.graph_wall_ms);
    }

    return 0;
}

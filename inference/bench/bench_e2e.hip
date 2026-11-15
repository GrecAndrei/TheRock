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

    failures += test_rxr4_roundtrip();

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
        eopts.max_seq_len = 4096;
        eopts.max_batch_tokens = 256;
        eopts.kv_dtype = rxinfer::DType::FP4_E2M1_G16;
        eopts.enable_hip_graph = true;

        rxinfer::InferenceEngine eng27(std::move(w27), eopts);
        std::printf("Allocated VRAM total: %.2f GiB (weights %.2f GiB, state %.1f MB, scratch %.1f MB)\n",
                    eng27.total_vram_used_bytes() / (1024.0 * 1024.0 * 1024.0),
                    eng27.weights().weight_arena.used() / (1024.0 * 1024.0 * 1024.0),
                    eng27.state_vram_bytes() / (1024.0 * 1024.0),
                    eng27.scratch_vram_bytes() / (1024.0 * 1024.0));

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

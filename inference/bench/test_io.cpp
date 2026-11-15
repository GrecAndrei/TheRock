// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/config.hpp"
#include "rxinfer/engine.hpp"
#include "rxinfer/loader.hpp"
#include "rxinfer/markdown.hpp"
#include "rxinfer/sampler.hpp"
#include "rxinfer/tokenizer.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string test_tmp_path(const std::string& name) {
    const auto base = std::filesystem::temp_directory_path() / "rxinfer_io_test";
    std::filesystem::create_directories(base);
    return (base / name).string();
}

uint16_t f32_to_bf16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    uint32_t rounding = 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>((u + rounding) >> 16);
}

uint16_t f32_to_fp16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((u >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
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

std::vector<std::string> gpt2_byte_unicode() {
    std::vector<std::string> out(256);
    std::vector<bool> direct(256, false);
    for (int b = 33; b <= 126; ++b) direct[b] = true;
    for (int b = 161; b <= 172; ++b) direct[b] = true;
    for (int b = 174; b <= 255; ++b) direct[b] = true;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = direct[b] ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + extra++);
        if (cp < 0x80u) {
            out[b].push_back(static_cast<char>(cp));
        } else {
            out[b].push_back(static_cast<char>(0xC0u | (cp >> 6)));
            out[b].push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        }
    }
    return out;
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out.push_back(static_cast<char>(c));
    }
    return out;
}

std::string build_mock_tokenizer_json(int vocab_target = 512) {
    auto b2u = gpt2_byte_unicode();
    std::vector<std::string> tokens;
    std::map<std::string, int> tok2id;
    auto add_tok = [&](const std::string& t) {
        if (tok2id.count(t)) return tok2id[t];
        int id = static_cast<int>(tokens.size());
        tokens.push_back(t);
        tok2id[t] = id;
        return id;
    };

    // Base 256 byte-level tokens
    for (int b = 0; b < 256; ++b) add_tok(b2u[b]);

    // Merged subwords (including space-prefixed Ġhello, contractions, surrogate emoji bytes)
    const std::string sp = b2u[static_cast<unsigned char>(' ')]; // Ġ
    std::vector<std::pair<std::string, std::string>> merges = {
        {"h", "e"},
        {"he", "l"},
        {"hel", "l"},
        {"hell", "o"},
        {sp, "w"},
        {sp + "w", "o"},
        {sp + "wo", "r"},
        {sp + "wor", "l"},
        {sp + "worl", "d"},
        {"'", "t"},
        {"d", "o"},
        {"do", "n"},
        {b2u[0xF0], b2u[0x9F]},
        {b2u[0x98], b2u[0x80]},
        {b2u[0xF0] + b2u[0x9F], b2u[0x98] + b2u[0x80]}, // 😀 (F0 9F 98 80)
        {b2u[0xE4], b2u[0xBD]},
        {b2u[0xE4] + b2u[0xBD], b2u[0xA0]}, // 你 (E4 BD A0)
    };
    for (const auto& m : merges) add_tok(m.first + m.second);
    while (static_cast<int>(tokens.size()) < vocab_target - 5) {
        add_tok("tok_" + std::to_string(tokens.size()));
    }

    const int id_endoftext = add_tok("<|endoftext|>");
    const int id_im_start = add_tok("<|im_start|>");
    const int id_im_end = add_tok("<|im_end|>");
    const int id_think_s = add_tok("<think>");
    const int id_think_e = add_tok("</think>");

    std::ostringstream os;
    os << "{\n"
       << "  \"version\": \"1.0\",\n"
       << "  \"added_tokens\": [\n"
       << "    {\"id\": " << id_endoftext << ", \"content\": \"<|endoftext|>\", \"special\": true},\n"
       << "    {\"id\": " << id_im_start << ", \"content\": \"<|im_start|>\", \"special\": true},\n"
       << "    {\"id\": " << id_im_end << ", \"content\": \"<|im_end|>\", \"special\": true},\n"
       << "    {\"id\": " << id_think_s << ", \"content\": \"<think>\", \"special\": false},\n"
       << "    {\"id\": " << id_think_e << ", \"content\": \"</think>\", \"special\": false}\n"
       << "  ],\n"
       << "  \"pre_tokenizer\": {\n"
       << "    \"type\": \"Sequence\",\n"
       << "    \"pretokenizers\": [\n"
       << "      {\"type\": \"Split\", \"pattern\": {\"Regex\": \"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+\"}},\n"
       << "      {\"type\": \"ByteLevel\", \"add_prefix_space\": false}\n"
       << "    ]\n"
       << "  },\n"
       << "  \"decoder\": {\"type\": \"ByteLevel\"},\n"
       << "  \"model\": {\n"
       << "    \"type\": \"BPE\",\n"
       << "    \"vocab\": {\n";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) os << ",\n";
        // Put explicit UTF-16 surrogate pair in JSON for the emoji token to verify JSON parser too!
        if (tokens[i] == (b2u[0xF0] + b2u[0x9F] + b2u[0x98] + b2u[0x80])) {
            os << "      \"" << json_escape(tokens[i]) << "\": " << i;
        } else {
            os << "      \"" << json_escape(tokens[i]) << "\": " << i;
        }
    }
    os << "\n    },\n    \"merges\": [\n";
    for (size_t i = 0; i < merges.size(); ++i) {
        if (i) os << ",\n";
        // Mix legacy string merges and HF >=0.20 2-element array merges!
        if (i % 2 == 0) {
            os << "      \"" << json_escape(merges[i].first) << " " << json_escape(merges[i].second) << "\"";
        } else {
            os << "      [\"" << json_escape(merges[i].first) << "\", \"" << json_escape(merges[i].second) << "\"]";
        }
    }
    os << "\n    ]\n  }\n}\n";
    return os.str();
}

int test_tokenizer() {
    const std::string dir = test_tmp_path("mock_tok_test");
    std::filesystem::create_directories(dir);
    {
        std::ofstream ofs(dir + "/tokenizer.json");
        ofs << build_mock_tokenizer_json(512);
    }
    {
        std::ofstream ofs(dir + "/tokenizer_config.json");
        ofs << "{\"eos_token\": \"<|im_end|>\", \"bos_token\": \"<|endoftext|>\"}\n";
    }
    // Also test UTF-16 surrogate pair decoding in JSON parser directly!
    {
        const std::string surr_json = R"({
            "added_tokens": [{"id": 0, "content": "\uD83D\uDE00", "special": false}],
            "pre_tokenizer": {"type": "ByteLevel"},
            "decoder": {"type": "ByteLevel"},
            "model": {"type": "BPE", "vocab": {"\uD83D\uDE00": 0}, "merges": []}
        })";
        auto tk_surr = rxinfer::Tokenizer::from_hf_json_string(surr_json);
        if (tk_surr.id_to_token(0) != "\xF0\x9F\x98\x80") {
            std::printf("[FAIL] UTF-16 surrogate pair decode mismatch!\n");
            return 1;
        }
    }

    auto tk = rxinfer::Tokenizer::from_hf_json_file(dir + "/tokenizer.json");
    if (tk.vocab_size() != 512 || !tk.is_eos(tk.token_to_id("<|im_end|>")) || tk.should_add_bos()) {
        std::printf("[FAIL] Tokenizer metadata mismatch (V=%d add_bos=%d)\n",
                    tk.vocab_size(), tk.should_add_bos() ? 1 : 0);
        return 1;
    }

    const std::string chat = tk.format_chatml("hello world don't 42   \xF0\x9F\x98\x80\xE4\xBD\xA0!", "sys", true, false);
    auto ids = tk.encode(chat, false);
    const std::string rt_all = tk.decode(ids, false);
    const std::string rt_nospec = tk.decode(ids, true);

    if (rt_all != chat) {
        std::printf("[FAIL] Tokenizer roundtrip mismatch:\n  exp: %s\n  got: %s\n",
                    chat.c_str(), rt_all.c_str());
        return 1;
    }
    // Verify special tokens were stripped while content + non-special <think></think> survived
    if (rt_nospec.find("<|im_start|>") != std::string::npos ||
        rt_nospec.find("hello world don't 42") == std::string::npos ||
        rt_nospec.find("\xF0\x9F\x98\x80\xE4\xBD\xA0") == std::string::npos) {
        std::printf("[FAIL] Tokenizer skip_special_tokens mismatch: %s\n", rt_nospec.c_str());
        return 1;
    }
    // Verify "hello" merged to single token and "42" split into 2 single-digit tokens ('4' and '2')!
    auto hello_ids = tk.encode("hello");
    auto num_ids = tk.encode("42");
    if (hello_ids.size() != 1 || num_ids.size() != 2) {
        std::printf("[FAIL] BPE merge/digit-split mismatch: hello=%zu num=%zu\n",
                    hello_ids.size(), num_ids.size());
        return 1;
    }

    std::printf("[PASS] Tokenizer (ByteLevel BPE + str/arr merges + UTF-16 surrogates + ChatML + CJK/Emoji)\n");
    return 0;
}

int test_safetensors_hf_qwen35_e2e() {
    const std::string dir = test_tmp_path("mock_qwen35_hf");
    std::filesystem::create_directories(dir);

    const int V = 512, H = 256, I = 512, L = 4;
    const int Hq = 4, Hkv = 2, D = 128;
    const int K_heads = 2, K_dim = 128, V_heads = 4, V_dim = 128, conv_k = 4;
    const int key_dim = K_heads * K_dim;
    const int val_dim = V_heads * V_dim;
    const int conv_dim = 2 * key_dim + val_dim;

    // Intentionally omit attn_output_gate from config.json to verify auto-detection for qwen3_5!
    const std::string cfg_json = R"({
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "tie_word_embeddings": false,
  "eos_token_id": [507, 509],
  "text_config": {
    "model_type": "qwen3_5_text",
    "vocab_size": 512,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 128,
    "linear_num_value_heads": 4,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "rope_parameters": {
      "rope_theta": 10000000.0,
      "partial_rotary_factor": 0.25,
      "mrope_interleaved": true,
      "mrope_section": [6, 5, 5]
    },
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
  }
})";
    {
        std::ofstream ofs(dir + "/config.json");
        ofs << cfg_json;
    }
    {
        std::ofstream ofs(dir + "/tokenizer.json");
        ofs << build_mock_tokenizer_json(V);
    }

    // Build safetensors binary payload
    struct TensorRec {
        std::string name;
        std::string dtype;
        std::vector<int64_t> shape;
        std::vector<uint8_t> bytes;
    };
    std::vector<TensorRec> recs;

    uint32_t seed = 1337u;
    auto next_f32 = [&](float scale) -> float {
        seed = seed * 1664525u + 1013904223u;
        int32_t s = static_cast<int32_t>((seed >> 16) & 0xFFFFu) - 32768;
        return (static_cast<float>(s) / 32768.0f) * scale;
    };

    auto add_bf16 = [&](const std::string& name, std::vector<int64_t> shape, float scale, float offset = 0.0f) {
        size_t n = 1;
        for (int64_t d : shape) n *= static_cast<size_t>(d);
        TensorRec r{name, "BF16", std::move(shape), std::vector<uint8_t>(n * 2)};
        auto* dst = reinterpret_cast<uint16_t*>(r.bytes.data());
        for (size_t i = 0; i < n; ++i) dst[i] = f32_to_bf16(offset + next_f32(scale));
        recs.push_back(std::move(r));
    };
    auto add_f32 = [&](const std::string& name, std::vector<int64_t> shape, float val) {
        size_t n = 1;
        for (int64_t d : shape) n *= static_cast<size_t>(d);
        TensorRec r{name, "F32", std::move(shape), std::vector<uint8_t>(n * 4)};
        auto* dst = reinterpret_cast<float*>(r.bytes.data());
        for (size_t i = 0; i < n; ++i) dst[i] = val + next_f32(0.05f);
        recs.push_back(std::move(r));
    };

    add_bf16("model.language_model.embed_tokens.weight", {V, H}, 0.05f);
    add_bf16("model.language_model.norm.weight", {H}, 0.02f, 0.0f);
    add_bf16("lm_head.weight", {V, H}, 0.03f);

    for (int l = 0; l < L; ++l) {
        const std::string p = "model.language_model.layers." + std::to_string(l) + ".";
        add_bf16(p + "input_layernorm.weight", {H}, 0.02f);
        add_bf16(p + "post_attention_layernorm.weight", {H}, 0.02f);
        if (l < 3) {
            add_bf16(p + "linear_attn.in_proj_qkv.weight", {conv_dim, H}, 0.03f);
            add_bf16(p + "linear_attn.in_proj_z.weight", {val_dim, H}, 0.03f);
            add_bf16(p + "linear_attn.in_proj_b.weight", {V_heads, H}, 0.02f);
            add_bf16(p + "linear_attn.in_proj_a.weight", {V_heads, H}, 0.02f);
            add_bf16(p + "linear_attn.conv1d.weight", {conv_dim, 1, conv_k}, 0.05f);
            add_f32(p + "linear_attn.A_log", {V_heads}, -1.0f);
            add_f32(p + "linear_attn.dt_bias", {V_heads}, 0.0f);
            add_bf16(p + "linear_attn.norm.weight", {V_dim}, 0.02f, 1.0f);
            add_bf16(p + "linear_attn.out_proj.weight", {H, val_dim}, 0.03f);
        } else {
            add_bf16(p + "self_attn.q_proj.weight", {Hq * D * 2, H}, 0.03f);
            add_bf16(p + "self_attn.k_proj.weight", {Hkv * D, H}, 0.03f);
            add_bf16(p + "self_attn.v_proj.weight", {Hkv * D, H}, 0.03f);
            add_bf16(p + "self_attn.o_proj.weight", {H, Hq * D}, 0.03f);
            add_bf16(p + "self_attn.q_norm.weight", {D}, 0.02f);
            add_bf16(p + "self_attn.k_norm.weight", {D}, 0.02f);
        }
        add_bf16(p + "mlp.gate_proj.weight", {I, H}, 0.03f);
        add_bf16(p + "mlp.up_proj.weight", {I, H}, 0.03f);
        add_bf16(p + "mlp.down_proj.weight", {H, I}, 0.03f);
    }

    std::ostringstream hdr;
    hdr << "{";
    size_t offset = 0;
    for (size_t i = 0; i < recs.size(); ++i) {
        if (i) hdr << ",";
        hdr << "\"" << recs[i].name << "\":{\"dtype\":\"" << recs[i].dtype << "\",\"shape\":[";
        for (size_t d = 0; d < recs[i].shape.size(); ++d) {
            if (d) hdr << ",";
            hdr << recs[i].shape[d];
        }
        const size_t end_off = offset + recs[i].bytes.size();
        hdr << "],\"data_offsets\":[" << offset << "," << end_off << "]}";
        offset = end_off;
    }
    hdr << "}";
    std::string hdr_str = hdr.str();
    while ((8 + hdr_str.size()) % 8 != 0) hdr_str.push_back(' ');

    {
        std::ofstream ofs(dir + "/model.safetensors", std::ios::binary);
        const uint64_t hlen = hdr_str.size();
        ofs.write(reinterpret_cast<const char*>(&hlen), 8);
        ofs.write(hdr_str.data(), hlen);
        for (const auto& r : recs) {
            ofs.write(reinterpret_cast<const char*>(r.bytes.data()), r.bytes.size());
        }
    }

    auto w_sf = rxinfer::load_from_safetensors_dir(dir);
    if (!w_sf.config.attn_output_gate || !w_sf.config.qk_norm ||
        w_sf.config.norm_kind != rxinfer::NormKind::RMSNormPlusOne ||
        w_sf.config.eos_token_ids.size() != 2) {
        std::printf("[FAIL] Safetensors config detection failed (gate=%d qk=%d norm=%d eos=%zu)\n",
                    w_sf.config.attn_output_gate ? 1 : 0, w_sf.config.qk_norm ? 1 : 0,
                    static_cast<int>(w_sf.config.norm_kind), w_sf.config.eos_token_ids.size());
        return 1;
    }

    const std::string rxr4_path = dir + "/model.rxr4";
    rxinfer::save_rxr4_checkpoint(w_sf, rxr4_path);
    auto w_rx = rxinfer::load_from_rxr4(rxr4_path);

    rxinfer::EngineOptions eopts{};
    eopts.max_seq_len = 256;
    eopts.max_batch_tokens = 32;
    eopts.kv_dtype = rxinfer::DType::FP8_E4M3;

    rxinfer::InferenceEngine eng_sf(std::move(w_sf), eopts);
    rxinfer::InferenceEngine eng_rx(std::move(w_rx), eopts);

    auto tk = rxinfer::Tokenizer::from_hf_json_file(dir + "/tokenizer.json");
    auto toks = tk.encode(tk.format_chatml("hello world", "sys", true, false));

    auto log_sf = eng_sf.forward_logits(toks);
    auto log_rx = eng_rx.forward_logits(toks);
    const double cos = cosine_sim(log_sf, log_rx);
    if (cos < 0.999999) {
        std::printf("[FAIL] Safetensors vs RXR4 logits mismatch cos=%.8f\n", cos);
        return 1;
    }

    // Verify qwen3_next grouped in_proj_qkvz / in_proj_ba un-grouping yields identical logits!
    {
        const std::string dir_next = test_tmp_path("mock_qwen3next_hf");
        std::filesystem::create_directories(dir_next);
        std::string next_cfg = cfg_json;
        size_t pos_mt = next_cfg.find("\"qwen3_5\"");
        if (pos_mt != std::string::npos) next_cfg.replace(pos_mt, 9, "\"qwen3_next\"");
        pos_mt = next_cfg.find("\"qwen3_5_text\"");
        if (pos_mt != std::string::npos) next_cfg.replace(pos_mt, 14, "\"qwen3_next\"");
        {
            std::ofstream ofs(dir_next + "/config.json");
            ofs << next_cfg;
        }
        std::vector<TensorRec> next_recs;
        std::map<std::string, const TensorRec*> by_name;
        for (const auto& r : recs) by_name[r.name] = &r;
        for (const auto& r : recs) {
            if (r.name.find(".linear_attn.in_proj_") != std::string::npos) continue;
            next_recs.push_back(r);
        }
        const int r_grp = V_heads / K_heads;
        const int grp_stride = 2 * K_dim + 2 * r_grp * V_dim;
        const size_t row_bytes = static_cast<size_t>(H) * 2;
        for (int l = 0; l < 3; ++l) {
            const std::string p = "model.language_model.layers." + std::to_string(l) + ".linear_attn.";
            const auto& t_qkv = *by_name.at(p + "in_proj_qkv.weight");
            const auto& t_z = *by_name.at(p + "in_proj_z.weight");
            const auto& t_b = *by_name.at(p + "in_proj_b.weight");
            const auto& t_a = *by_name.at(p + "in_proj_a.weight");

            TensorRec qkvz{p + "in_proj_qkvz.weight", "BF16", {conv_dim + val_dim, H},
                           std::vector<uint8_t>(static_cast<size_t>(conv_dim + val_dim) * row_bytes)};
            TensorRec ba{p + "in_proj_ba.weight", "BF16", {2 * V_heads, H},
                         std::vector<uint8_t>(static_cast<size_t>(2 * V_heads) * row_bytes)};
            for (int g = 0; g < K_heads; ++g) {
                const int dst0 = g * grp_stride;
                // q_g
                std::memcpy(qkvz.bytes.data() + static_cast<size_t>(dst0) * row_bytes,
                            t_qkv.bytes.data() + static_cast<size_t>(g * K_dim) * row_bytes,
                            static_cast<size_t>(K_dim) * row_bytes);
                // k_g
                std::memcpy(qkvz.bytes.data() + static_cast<size_t>(dst0 + K_dim) * row_bytes,
                            t_qkv.bytes.data() + static_cast<size_t>(key_dim + g * K_dim) * row_bytes,
                            static_cast<size_t>(K_dim) * row_bytes);
                // v_g
                std::memcpy(qkvz.bytes.data() + static_cast<size_t>(dst0 + 2 * K_dim) * row_bytes,
                            t_qkv.bytes.data() + static_cast<size_t>(2 * key_dim + g * r_grp * V_dim) * row_bytes,
                            static_cast<size_t>(r_grp * V_dim) * row_bytes);
                // z_g
                std::memcpy(qkvz.bytes.data() + static_cast<size_t>(dst0 + 2 * K_dim + r_grp * V_dim) * row_bytes,
                            t_z.bytes.data() + static_cast<size_t>(g * r_grp * V_dim) * row_bytes,
                            static_cast<size_t>(r_grp * V_dim) * row_bytes);
                // b_g, a_g
                std::memcpy(ba.bytes.data() + static_cast<size_t>(g * 2 * r_grp) * row_bytes,
                            t_b.bytes.data() + static_cast<size_t>(g * r_grp) * row_bytes,
                            static_cast<size_t>(r_grp) * row_bytes);
                std::memcpy(ba.bytes.data() + static_cast<size_t>(g * 2 * r_grp + r_grp) * row_bytes,
                            t_a.bytes.data() + static_cast<size_t>(g * r_grp) * row_bytes,
                            static_cast<size_t>(r_grp) * row_bytes);
            }
            next_recs.push_back(std::move(qkvz));
            next_recs.push_back(std::move(ba));
        }

        std::ostringstream nhdr;
        nhdr << "{";
        size_t noff = 0;
        for (size_t i = 0; i < next_recs.size(); ++i) {
            if (i) nhdr << ",";
            nhdr << "\"" << next_recs[i].name << "\":{\"dtype\":\"" << next_recs[i].dtype << "\",\"shape\":[";
            for (size_t d = 0; d < next_recs[i].shape.size(); ++d) {
                if (d) nhdr << ",";
                nhdr << next_recs[i].shape[d];
            }
            const size_t end_off = noff + next_recs[i].bytes.size();
            nhdr << "],\"data_offsets\":[" << noff << "," << end_off << "]}";
            noff = end_off;
        }
        nhdr << "}";
        std::string nhdr_str = nhdr.str();
        while ((8 + nhdr_str.size()) % 8 != 0) nhdr_str.push_back(' ');
        {
            std::ofstream ofs(dir_next + "/model.safetensors", std::ios::binary);
            const uint64_t hlen = nhdr_str.size();
            ofs.write(reinterpret_cast<const char*>(&hlen), 8);
            ofs.write(nhdr_str.data(), hlen);
            for (const auto& r : next_recs) {
                ofs.write(reinterpret_cast<const char*>(r.bytes.data()), r.bytes.size());
            }
        }
        auto w_next = rxinfer::load_from_safetensors_dir(dir_next);
        rxinfer::InferenceEngine eng_next(std::move(w_next), eopts);
        auto log_next = eng_next.forward_logits(toks);
        const double cos_next = cosine_sim(log_sf, log_next);
        if (cos_next < 0.999999) {
            std::printf("[FAIL] qwen3_next grouped vs qwen3_5 parity mismatch cos=%.8f\n", cos_next);
            return 1;
        }
    }

    std::printf("[PASS] Safetensors HF Qwen3.5 + Qwen3-Next (BF16 + GDN + FullAttn + RXR4 roundtrip cos=%.8f)\n", cos);
    return 0;
}

int test_gguf_q4k_q6k_tokenizer_e2e() {
    const std::string gguf_path = test_tmp_path("mock_qwen3.gguf");
    const int V = 320, H = 256, I = 512, L = 2, Hq = 4, Hkv = 2, D = 64;
    auto b2u = gpt2_byte_unicode();

    // Build GGUF v3 file with Q4_K, Q6_K, Q8_0, Q4_0, and F32 tensors + GPT2 tokenizer metadata
    std::vector<std::string> vocab;
    for (int b = 0; b < 256; ++b) vocab.push_back(b2u[b]);
    vocab.push_back("he");
    vocab.push_back("hel");
    vocab.push_back("hell");
    vocab.push_back("hello");
    while (static_cast<int>(vocab.size()) < V - 2) vocab.push_back("g_" + std::to_string(vocab.size()));
    vocab.push_back("<|endoftext|>");
    vocab.push_back("<|im_end|>");
    std::vector<int32_t> tok_types(V, 1);
    tok_types[V - 2] = 3;
    tok_types[V - 1] = 3;
    std::vector<std::string> merges = {"h e", "he l", "hel l", "hell o"};

    struct GRec {
        std::string name;
        uint32_t type;
        uint64_t dims[2];
        std::vector<uint8_t> data;
    };
    std::vector<GRec> recs;

    uint32_t seed = 999u;
    auto next_u8 = [&]() -> uint8_t {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<uint8_t>(seed >> 24);
    };

    auto make_f32 = [&](const std::string& name, int n, float val) {
        GRec r{name, 0, {static_cast<uint64_t>(n), 1}, std::vector<uint8_t>(n * 4)};
        auto* p = reinterpret_cast<float*>(r.data.data());
        for (int i = 0; i < n; ++i) p[i] = val;
        recs.push_back(std::move(r));
    };
    auto make_q4_k = [&](const std::string& name, int rows, int cols) {
        const int blocks = cols / 256;
        GRec r{name, 12, {static_cast<uint64_t>(cols), static_cast<uint64_t>(rows)},
               std::vector<uint8_t>(static_cast<size_t>(rows) * blocks * 144)};
        for (int row = 0; row < rows; ++row) {
            for (int b = 0; b < blocks; ++b) {
                uint8_t* blk = r.data.data() + (static_cast<size_t>(row) * blocks + b) * 144;
                const uint16_t d = f32_to_fp16(0.01f);
                const uint16_t dm = f32_to_fp16(0.005f);
                std::memcpy(blk, &d, 2);
                std::memcpy(blk + 2, &dm, 2);
                for (int i = 4; i < 16; ++i) blk[i] = 0x15;
                for (int i = 16; i < 144; ++i) blk[i] = next_u8();
            }
        }
        recs.push_back(std::move(r));
    };
    auto make_q6_k = [&](const std::string& name, int rows, int cols) {
        const int blocks = cols / 256;
        GRec r{name, 14, {static_cast<uint64_t>(cols), static_cast<uint64_t>(rows)},
               std::vector<uint8_t>(static_cast<size_t>(rows) * blocks * 210)};
        for (int row = 0; row < rows; ++row) {
            for (int b = 0; b < blocks; ++b) {
                uint8_t* blk = r.data.data() + (static_cast<size_t>(row) * blocks + b) * 210;
                for (int i = 0; i < 192; ++i) blk[i] = next_u8();
                for (int i = 192; i < 208; ++i) blk[i] = 2;
                const uint16_t d = f32_to_fp16(0.005f);
                std::memcpy(blk + 208, &d, 2);
            }
        }
        recs.push_back(std::move(r));
    };
    auto make_q8_0 = [&](const std::string& name, int rows, int cols) {
        const int blocks = cols / 32;
        GRec r{name, 8, {static_cast<uint64_t>(cols), static_cast<uint64_t>(rows)},
               std::vector<uint8_t>(static_cast<size_t>(rows) * blocks * 34)};
        for (int row = 0; row < rows; ++row) {
            for (int b = 0; b < blocks; ++b) {
                uint8_t* blk = r.data.data() + (static_cast<size_t>(row) * blocks + b) * 34;
                const uint16_t d = f32_to_fp16(0.002f);
                std::memcpy(blk, &d, 2);
                for (int i = 2; i < 34; ++i) blk[i] = static_cast<uint8_t>(static_cast<int8_t>((next_u8() % 17) - 8));
            }
        }
        recs.push_back(std::move(r));
    };

    make_q6_k("token_embd.weight", V, H);
    make_f32("output_norm.weight", H, 1.0f);
    make_q6_k("output.weight", V, H);
    for (int l = 0; l < L; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        make_f32(p + "attn_norm.weight", H, 1.0f);
        make_f32(p + "ffn_norm.weight", H, 1.0f);
        make_q4_k(p + "attn_q.weight", Hq * D, H);
        make_q8_0(p + "attn_k.weight", Hkv * D, H);
        make_q8_0(p + "attn_v.weight", Hkv * D, H);
        make_q4_k(p + "attn_output.weight", H, Hq * D);
        make_f32(p + "attn_q_norm.weight", D, 1.0f);
        make_f32(p + "attn_k_norm.weight", D, 1.0f);
        make_q4_k(p + "ffn_gate.weight", I, H);
        make_q4_k(p + "ffn_up.weight", I, H);
        make_q6_k(p + "ffn_down.weight", H, I);
    }

    std::ofstream ofs(gguf_path, std::ios::binary);
    auto w_u32 = [&](uint32_t v) { ofs.write(reinterpret_cast<const char*>(&v), 4); };
    auto w_i32 = [&](int32_t v) { ofs.write(reinterpret_cast<const char*>(&v), 4); };
    auto w_u64 = [&](uint64_t v) { ofs.write(reinterpret_cast<const char*>(&v), 8); };
    auto w_f32 = [&](float v) { ofs.write(reinterpret_cast<const char*>(&v), 4); };
    auto w_str = [&](const std::string& s) {
        w_u64(s.size());
        ofs.write(s.data(), s.size());
    };
    auto kv_str = [&](const std::string& k, const std::string& v) {
        w_str(k); w_u32(8); w_str(v);
    };
    auto kv_u32 = [&](const std::string& k, uint32_t v) {
        w_str(k); w_u32(4); w_u32(v);
    };
    auto kv_f32 = [&](const std::string& k, float v) {
        w_str(k); w_u32(6); w_f32(v);
    };

    w_u32(0x46554747u); // GGUF magic
    w_u32(3);           // version
    w_u64(recs.size());
    w_u64(17);          // n_kv

    kv_str("general.architecture", "qwen3");
    kv_u32("qwen3.block_count", L);
    kv_u32("qwen3.embedding_length", H);
    kv_u32("qwen3.feed_forward_length", I);
    kv_u32("qwen3.attention.head_count", Hq);
    kv_u32("qwen3.attention.head_count_kv", Hkv);
    // Intentionally place rope.dimension_count BEFORE key_length to test ordering!
    kv_u32("qwen3.rope.dimension_count", 32);
    kv_u32("qwen3.attention.key_length", D);
    kv_u32("qwen3.context_length", 1024);
    kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
    kv_f32("qwen3.rope.freq_base", 1000000.0f);
    kv_str("tokenizer.ggml.model", "gpt2");
    kv_str("tokenizer.ggml.pre", "qwen2");
    kv_u32("tokenizer.ggml.eos_token_id", static_cast<uint32_t>(V - 1));

    w_str("tokenizer.ggml.tokens"); w_u32(9); w_u32(8); w_u64(vocab.size());
    for (const auto& t : vocab) w_str(t);

    w_str("tokenizer.ggml.token_type"); w_u32(9); w_u32(5); w_u64(tok_types.size());
    for (int32_t tt : tok_types) w_i32(tt);

    w_str("tokenizer.ggml.merges"); w_u32(9); w_u32(8); w_u64(merges.size());
    for (const auto& m : merges) w_str(m);

    // Note: header says 16 KVs; let's count carefully! (14 above + tokens + merges = 16!)
    // Write tensor info table
    uint64_t cur_off = 0;
    for (const auto& r : recs) {
        w_str(r.name);
        const uint32_t nd = (r.dims[1] == 1 && r.type == 0) ? 1u : 2u;
        w_u32(nd);
        w_u64(r.dims[0]);
        if (nd == 2) w_u64(r.dims[1]);
        w_u32(r.type);
        w_u64(cur_off);
        cur_off = (cur_off + r.data.size() + 31ULL) & ~31ULL;
    }
    // Pad header to 32-byte alignment
    while (static_cast<uint64_t>(ofs.tellp()) % 32ULL != 0) ofs.put(0);
    for (const auto& r : recs) {
        ofs.write(reinterpret_cast<const char*>(r.data.data()), r.data.size());
        const size_t pad = ((r.data.size() + 31ULL) & ~31ULL) - r.data.size();
        for (size_t p = 0; p < pad; ++p) ofs.put(0);
    }
    ofs.close();

    auto w_gguf = rxinfer::load_from_gguf(gguf_path);
    auto tk_gguf = rxinfer::Tokenizer::from_gguf(gguf_path);

    if (std::abs(w_gguf.config.partial_rotary_factor - 0.5f) > 1e-5f ||
        !w_gguf.config.qk_norm ||
        tk_gguf.vocab_size() != V) {
        std::printf("[FAIL] GGUF metadata mismatch (rot=%.3f qk=%d V=%d)\n",
                    w_gguf.config.partial_rotary_factor, w_gguf.config.qk_norm ? 1 : 0,
                    tk_gguf.vocab_size());
        return 1;
    }

    auto ids = tk_gguf.encode("hello");
    if (ids.size() != 1 || tk_gguf.decode(ids) != "hello") {
        std::printf("[FAIL] GGUF tokenizer encode/decode mismatch!\n");
        return 1;
    }

    rxinfer::EngineOptions eopts{};
    eopts.max_seq_len = 256;
    rxinfer::InferenceEngine eng(std::move(w_gguf), eopts);
    auto tok = eng.prefill({1, 2, 3});
    auto tok2 = eng.decode_step(tok);
    (void)tok2;

    std::printf("[PASS] GGUF v3 (Q4_K + Q6_K + Q8_0 + F32 + embedded GPT2 tokenizer)\n");
    return 0;
}

int test_gguf_qwen35_iq_and_rxr4_e2e() {
    const std::string gguf_path = test_tmp_path("mock_qwen35_iq.gguf");
    const int V = 320, H = 256, I = 256, L = 2, Hq = 4, Hkv = 2, D = 64;
    const int Kh = 2, Kd = 128, Vh = 4, Vd = 128;
    const int key_dim = Kh * Kd;
    const int val_dim = Vh * Vd;
    const int conv_dim = 2 * key_dim + val_dim;
    auto b2u = gpt2_byte_unicode();

    std::vector<std::string> vocab;
    for (int b = 0; b < 256; ++b) vocab.push_back(b2u[b]);
    vocab.push_back("hello");
    while (static_cast<int>(vocab.size()) < V - 5) vocab.push_back("q35_" + std::to_string(vocab.size()));
    vocab.push_back("<|endoftext|>");
    vocab.push_back("<|im_start|>");
    vocab.push_back("<|im_end|>");
    vocab.push_back("<think>");
    vocab.push_back("</think>");
    std::vector<int32_t> tok_types(V, 1);
    tok_types[V - 5] = 3;
    tok_types[V - 4] = 3;
    tok_types[V - 3] = 3;
    tok_types[V - 2] = 4;
    tok_types[V - 1] = 4;

    struct GRec {
        std::string name;
        uint32_t type;
        std::vector<uint64_t> dims;
        std::vector<uint8_t> data;
    };
    std::vector<GRec> recs;

    uint32_t seed = 2026u;
    auto next_u8 = [&]() -> uint8_t {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<uint8_t>(seed >> 24);
    };

    auto add_f32_vec = [&](const std::string& name, int n, float val) {
        GRec r{name, 0, {static_cast<uint64_t>(n)}, std::vector<uint8_t>(static_cast<size_t>(n) * 4)};
        auto* p = reinterpret_cast<float*>(r.data.data());
        for (int i = 0; i < n; ++i) p[i] = val;
        recs.push_back(std::move(r));
    };
    auto add_f32_2d = [&](const std::string& name, int c0, int c1, float val) {
        GRec r{name, 0, {static_cast<uint64_t>(c0), static_cast<uint64_t>(c1)},
               std::vector<uint8_t>(static_cast<size_t>(c0) * c1 * 4)};
        auto* p = reinterpret_cast<float*>(r.data.data());
        for (int i = 0; i < c0 * c1; ++i) p[i] = val;
        recs.push_back(std::move(r));
    };
    auto add_f16_vec = [&](const std::string& name, int n, float val) {
        GRec r{name, 1, {static_cast<uint64_t>(n)}, std::vector<uint8_t>(static_cast<size_t>(n) * 2)};
        auto* p = reinterpret_cast<uint16_t*>(r.data.data());
        for (int i = 0; i < n; ++i) p[i] = f32_to_fp16(val);
        recs.push_back(std::move(r));
    };
    auto add_bf16_vec = [&](const std::string& name, int n, float val) {
        GRec r{name, 30, {static_cast<uint64_t>(n)}, std::vector<uint8_t>(static_cast<size_t>(n) * 2)};
        auto* p = reinterpret_cast<uint16_t*>(r.data.data());
        for (int i = 0; i < n; ++i) p[i] = f32_to_bf16(val);
        recs.push_back(std::move(r));
    };
    auto add_qblock = [&](const std::string& name, uint32_t ggml_type, int rows, int cols, int blk_elems, int blk_bytes) {
        const int blocks = cols / blk_elems;
        GRec r{name, ggml_type, {static_cast<uint64_t>(cols), static_cast<uint64_t>(rows)},
               std::vector<uint8_t>(static_cast<size_t>(rows) * blocks * blk_bytes)};
        const uint16_t d16 = f32_to_fp16(0.005f);
        for (int row = 0; row < rows; ++row) {
            for (int b = 0; b < blocks; ++b) {
                uint8_t* blk = r.data.data() + (static_cast<size_t>(row) * blocks + b) * blk_bytes;
                for (int i = 0; i < blk_bytes; ++i) blk[i] = next_u8();
                if (ggml_type == 10) { // Q2_K: d @ 80, dmin @ 82
                    std::memcpy(blk + 80, &d16, 2);
                    std::memcpy(blk + 82, &d16, 2);
                } else if (ggml_type == 11) { // Q3_K: d @ 108
                    std::memcpy(blk + 108, &d16, 2);
                } else if (ggml_type == 12 || ggml_type == 13) { // Q4_K / Q5_K: d @ 0, dmin @ 2
                    std::memcpy(blk + 0, &d16, 2);
                    std::memcpy(blk + 2, &d16, 2);
                } else if (ggml_type == 14) { // Q6_K: d @ 208
                    std::memcpy(blk + 208, &d16, 2);
                } else if (ggml_type == 29) { // IQ1_M: 4x uint16 scales at +48..54 whose top nibbles encode d16
                    for (int s = 0; s < 4; ++s) {
                        const uint16_t sc_lo = static_cast<uint16_t>(next_u8() | ((next_u8() & 0x0Fu) << 8));
                        const uint16_t nib = static_cast<uint16_t>(((d16 >> (4 * s)) & 0x0Fu) << 12);
                        const uint16_t scv = static_cast<uint16_t>(sc_lo | nib);
                        std::memcpy(blk + 48 + s * 2, &scv, 2);
                    }
                } else { // Q4_0, Q5_0, Q8_0, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS: d @ 0
                    std::memcpy(blk, &d16, 2);
                }
            }
        }
        recs.push_back(std::move(r));
    };

    // Exercise ALL GGML quant types: Q4_0(2), Q5_0(6), Q8_0(8), Q2_K(10), Q3_K(11), Q4_K(12), Q5_K(13), Q6_K(14),
    // IQ2_XXS(16), IQ2_XS(17), IQ3_XXS(18), IQ1_S(19), IQ4_NL(20), IQ3_S(21), IQ2_S(22), IQ4_XS(23), IQ1_M(29),
    // plus F32(0), F16(1), BF16(30)!
    add_qblock("token_embd.weight", 23, V, H, 256, 136); // IQ4_XS
    add_f32_vec("output_norm.weight", H, 1.0f);
    add_qblock("output.weight", 29, V, H, 256, 56);      // IQ1_M

    // blk.0: GDN layer
    add_f32_vec("blk.0.attn_norm.weight", H, 1.0f);
    add_f32_vec("blk.0.post_attention_norm.weight", H, 1.0f);
    add_qblock("blk.0.attn_qkv.weight", 21, conv_dim, H, 256, 110); // IQ3_S
    add_qblock("blk.0.attn_gate.weight", 22, val_dim, H, 256, 82);  // IQ2_S
    add_qblock("blk.0.ssm_beta.weight", 20, Vh, H, 32, 18);         // IQ4_NL
    add_qblock("blk.0.ssm_alpha.weight", 10, Vh, H, 256, 84);       // Q2_K
    add_f32_2d("blk.0.ssm_conv1d.weight", 4, conv_dim, 0.02f);      // F32
    add_f32_vec("blk.0.ssm_a", Vh, -0.5f);                          // F32
    add_f32_vec("blk.0.ssm_dt.bias", Vh, 0.1f);                     // F32
    add_bf16_vec("blk.0.ssm_norm.weight", Vd, 1.0f);                // BF16
    add_qblock("blk.0.ssm_out.weight", 18, H, val_dim, 256, 98);    // IQ3_XXS
    add_qblock("blk.0.ffn_gate.weight", 11, I, H, 256, 110);        // Q3_K
    add_qblock("blk.0.ffn_up.weight", 17, I, H, 256, 74);           // IQ2_XS
    add_qblock("blk.0.ffn_down.weight", 16, H, I, 256, 66);         // IQ2_XXS

    // blk.1: FullAttention layer (with attn_output_gate=true + qk_norm=true)
    add_f32_vec("blk.1.attn_norm.weight", H, 1.0f);
    add_f32_vec("blk.1.post_attention_norm.weight", H, 1.0f);
    add_qblock("blk.1.attn_q.weight", 19, 2 * Hq * D, H, 256, 50);  // IQ1_S
    add_qblock("blk.1.attn_k.weight", 13, Hkv * D, H, 256, 176);    // Q5_K
    add_qblock("blk.1.attn_v.weight", 2, Hkv * D, H, 32, 18);       // Q4_0
    add_qblock("blk.1.attn_output.weight", 6, H, Hq * D, 32, 22);   // Q5_0
    add_f16_vec("blk.1.attn_q_norm.weight", D, 1.0f);               // F16
    add_f32_vec("blk.1.attn_k_norm.weight", D, 1.0f);               // F32
    add_qblock("blk.1.ffn_gate.weight", 12, I, H, 256, 144);        // Q4_K
    add_qblock("blk.1.ffn_up.weight", 14, I, H, 256, 210);          // Q6_K
    add_qblock("blk.1.ffn_down.weight", 8, H, I, 32, 34);           // Q8_0

    // blk.2: MTP/NextN block (must be ignored by trunk!)
    add_f32_vec("blk.2.attn_norm.weight", H, 1.0f);

    auto write_q35_gguf = [&](const std::string& out_path, const std::vector<GRec>& grecs, bool use_merged_gdn) {
        std::ofstream ofs(out_path, std::ios::binary);
        auto w_u8 = [&](uint8_t v) { ofs.write(reinterpret_cast<const char*>(&v), 1); };
        auto w_u32 = [&](uint32_t v) { ofs.write(reinterpret_cast<const char*>(&v), 4); };
        auto w_i32 = [&](int32_t v) { ofs.write(reinterpret_cast<const char*>(&v), 4); };
        auto w_u64 = [&](uint64_t v) { ofs.write(reinterpret_cast<const char*>(&v), 8); };
        auto w_f32 = [&](float v) { ofs.write(reinterpret_cast<const char*>(&v), 4); };
        auto w_str = [&](const std::string& s) {
            w_u64(s.size());
            ofs.write(s.data(), s.size());
        };
        auto kv_str = [&](const std::string& k, const std::string& v) { w_str(k); w_u32(8); w_str(v); };
        auto kv_u32 = [&](const std::string& k, uint32_t v) { w_str(k); w_u32(4); w_u32(v); };
        auto kv_f32 = [&](const std::string& k, float v) { w_str(k); w_u32(6); w_f32(v); };
        auto kv_bool = [&](const std::string& k, bool v) { w_str(k); w_u32(7); w_u8(v ? 1 : 0); };

        w_u32(0x46554747u);
        w_u32(3);
        w_u64(grecs.size());
        w_u64(28); // n_kv

        kv_str("general.architecture", "qwen35");
        kv_u32("qwen35.block_count", L + 1); // includes 1 MTP layer
        kv_u32("qwen35.nextn_predict_layers", 1);
        kv_u32("qwen35.embedding_length", H);
        kv_u32("qwen35.feed_forward_length", I);
        kv_u32("qwen35.attention.head_count", Hq);
        kv_u32("qwen35.attention.head_count_kv", Hkv);
        kv_u32("qwen35.attention.key_length", D);
        kv_u32("qwen35.context_length", 1024);
        kv_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
        kv_f32("qwen35.rope.freq_base", 10000000.0f);
        kv_u32("qwen35.rope.dimension_count", 32);
        w_str("qwen35.rope.dimension_sections"); w_u32(9); w_u32(4); w_u64(4);
        w_u32(6); w_u32(5); w_u32(5); w_u32(0);
        kv_u32("qwen35.full_attention_interval", 2);
        w_str("qwen35.attention.recurrent_layers"); w_u32(9); w_u32(7); w_u64(3);
        w_u8(1); w_u8(0); w_u8(0);
        kv_u32("qwen35.ssm.conv_kernel", 4);
        kv_u32("qwen35.ssm.state_size", Kd);
        kv_u32("qwen35.ssm.group_count", Kh);
        kv_u32("qwen35.ssm.time_step_rank", Vh);
        kv_u32("qwen35.ssm.inner_size", val_dim);
        kv_str("tokenizer.ggml.model", "gpt2");
        kv_str("tokenizer.ggml.pre", "qwen35");
        kv_bool("tokenizer.ggml.add_bos_token", false);
        kv_u32("tokenizer.ggml.bos_token_id", static_cast<uint32_t>(V - 5));
        if (use_merged_gdn) {
            w_str("tokenizer.ggml.eos_token_id"); w_u32(9); w_u32(4); w_u64(2);
            w_u32(static_cast<uint32_t>(V - 5));
            w_u32(static_cast<uint32_t>(V - 3));
        } else {
            kv_u32("tokenizer.ggml.eos_token_id", static_cast<uint32_t>(V - 3));
        }
        kv_u32("tokenizer.ggml.eot_token_id", static_cast<uint32_t>(V - 5));

        w_str("tokenizer.ggml.tokens"); w_u32(9); w_u32(8); w_u64(vocab.size());
        for (const auto& t : vocab) w_str(t);
        w_str("tokenizer.ggml.token_type"); w_u32(9); w_u32(5); w_u64(tok_types.size());
        for (int32_t tt : tok_types) w_i32(tt);

        uint64_t cur_off = 0;
        for (const auto& r : grecs) {
            w_str(r.name);
            w_u32(static_cast<uint32_t>(r.dims.size()));
            for (uint64_t d : r.dims) w_u64(d);
            w_u32(r.type);
            w_u64(cur_off);
            cur_off = (cur_off + r.data.size() + 31ULL) & ~31ULL;
        }
        while (static_cast<uint64_t>(ofs.tellp()) % 32ULL != 0) ofs.put(0);
        for (const auto& r : grecs) {
            ofs.write(reinterpret_cast<const char*>(r.data.data()), r.data.size());
            const size_t pad = ((r.data.size() + 31ULL) & ~31ULL) - r.data.size();
            for (size_t p = 0; p < pad; ++p) ofs.put(0);
        }
    };

    write_q35_gguf(gguf_path, recs, false);

    auto w_q35 = rxinfer::load_from_gguf(gguf_path);
    auto tk_q35 = rxinfer::Tokenizer::from_gguf(gguf_path);
    const auto& cfg = w_q35.config;
    if (cfg.num_hidden_layers != 2 || cfg.mtp_num_hidden_layers != 1 ||
        !cfg.gdn_tiled_v_heads || !cfg.mrope_interleaved ||
        !cfg.attn_output_gate || !cfg.qk_norm ||
        cfg.norm_kind != rxinfer::NormKind::RMSNorm ||
        !tk_q35.is_eos(V - 3) || !tk_q35.is_eos(V - 5)) {
        std::printf("[FAIL] qwen35 GGUF metadata/config mismatch (L=%d mtp=%d tiled=%d mrope=%d gate=%d eot=%d)\n",
                    cfg.num_hidden_layers, cfg.mtp_num_hidden_layers,
                    cfg.gdn_tiled_v_heads ? 1 : 0, cfg.mrope_interleaved ? 1 : 0,
                    cfg.attn_output_gate ? 1 : 0, tk_q35.is_eos(V - 5) ? 1 : 0);
        return 1;
    }

    const std::string rxr4_path = test_tmp_path("mock_qwen35_iq.rxr4");
    rxinfer::save_rxr4_checkpoint(w_q35, rxr4_path);
    auto w_rx = rxinfer::load_from_rxr4(rxr4_path);
    if (!w_rx.config.gdn_tiled_v_heads || !w_rx.config.mrope_interleaved ||
        w_rx.config.norm_kind != rxinfer::NormKind::RMSNorm) {
        std::printf("[FAIL] qwen35 RXR4 config roundtrip lost gdn_tiled_v_heads/mrope\n");
        return 1;
    }

    rxinfer::EngineOptions eopts{};
    eopts.max_seq_len = 256;
    rxinfer::InferenceEngine eng_g(std::move(w_q35), eopts);
    rxinfer::InferenceEngine eng_r(std::move(w_rx), eopts);
    const std::vector<int32_t> toks = {10, 42, 77, 128};
    const auto logits_g = eng_g.forward_logits(toks);
    const auto logits_r = eng_r.forward_logits(toks);
    for (float v : logits_g) {
        if (!std::isfinite(v)) {
            std::printf("[FAIL] qwen35 GGUF produced non-finite logit!\n");
            return 1;
        }
    }
    const double cos_rx = cosine_sim(logits_g, logits_r);
    if (cos_rx < 0.999999) {
        std::printf("[FAIL] qwen35 GGUF vs RXR4 logits mismatch cos=%.8f\n", cos_rx);
        return 1;
    }

    // Verify merged attn_qkv_gate.weight + ssm_ba.weight + array eos_token_id path too (all F32 so exact!)
    const std::string gguf_merged_path = test_tmp_path("mock_qwen35_merged.gguf");
    std::vector<GRec> mrecs;
    for (const auto& r : recs) {
        if (r.name == "blk.0.attn_qkv.weight" || r.name == "blk.0.attn_gate.weight" ||
            r.name == "blk.0.ssm_beta.weight" || r.name == "blk.0.ssm_alpha.weight") {
            continue;
        }
        mrecs.push_back(r);
    }
    GRec qkv_gate{"blk.0.attn_qkv_gate.weight", 0, {static_cast<uint64_t>(H), static_cast<uint64_t>(conv_dim + val_dim)},
                  std::vector<uint8_t>(static_cast<size_t>(conv_dim + val_dim) * H * 4, 0)};
    GRec ssm_ba{"blk.0.ssm_ba.weight", 0, {static_cast<uint64_t>(H), static_cast<uint64_t>(2 * Vh)},
                std::vector<uint8_t>(static_cast<size_t>(2 * Vh) * H * 4, 0)};
    mrecs.push_back(std::move(qkv_gate));
    mrecs.push_back(std::move(ssm_ba));
    write_q35_gguf(gguf_merged_path, mrecs, true);
    auto w_m = rxinfer::load_from_gguf(gguf_merged_path);
    auto tk_m = rxinfer::Tokenizer::from_gguf(gguf_merged_path);
    if (!tk_m.is_eos(V - 5) || !tk_m.is_eos(V - 3) || !w_m.config.gdn_tiled_v_heads) {
        std::printf("[FAIL] qwen35 merged GGUF array eos or tiled check failed\n");
        return 1;
    }
    rxinfer::InferenceEngine eng_m(std::move(w_m), eopts);
    const auto logits_m = eng_m.forward_logits(toks);
    for (float v : logits_m) {
        if (!std::isfinite(v)) {
            std::printf("[FAIL] qwen35 merged GGUF produced non-finite logit!\n");
            return 1;
        }
    }

    std::printf("[PASS] GGUF v3 Qwen3.5 (all K/IQ/F16/BF16/Q4_0/Q5_0 quants + MTP + GDN tiled + RXR4 cos=%.6f)\n",
                cos_rx);
    return 0;
}

int test_sampler_and_multiturn_chat() {
    // 1. Sampler tests: greedy, FP16 bits LUT, repeat penalty, seeded stochastic top_k/top_p/min_p
    {
        rxinfer::SamplerConfig gcfg{};
        gcfg.temperature = 0.0f;
        rxinfer::Sampler gs(gcfg);

        std::vector<uint16_t> f16(64, f32_to_fp16(-2.0f));
        f16[17] = f32_to_fp16(5.5f);
        f16[33] = f32_to_fp16(3.0f);
        if (gs.sample_from_f16_bits(f16.data(), 64) != 17) {
            std::printf("[FAIL] Sampler FP16 greedy argmax failed\n");
            return 1;
        }

        // Repeat penalty suppresses token 17 (5.5 / 2.5 = 2.2 < 3.0) -> picks 33
        rxinfer::SamplerConfig rcfg{};
        rcfg.temperature = 0.0f;
        rcfg.repeat_penalty = 2.5f;
        rcfg.repeat_last_n = 16;
        rxinfer::Sampler rs(rcfg);
        rs.record_token(17);
        if (rs.sample_from_f16_bits(f16.data(), 64) != 33) {
            std::printf("[FAIL] Sampler repeat penalty failed\n");
            return 1;
        }

        // Seeded stochastic sampling with top_k=3, top_p=0.95, min_p=0.05 is deterministic and in top set
        rxinfer::SamplerConfig scfg{};
        scfg.temperature = 0.8f;
        scfg.top_k = 3;
        scfg.top_p = 0.95f;
        scfg.min_p = 0.05f;
        scfg.seed = 12345ULL;
        rxinfer::Sampler s1(scfg), s2(scfg);
        std::vector<float> f32(256, -50.0f);
        f32[10] = 6.0f;
        f32[20] = 5.8f;
        f32[30] = 5.6f;
        f32[40] = 5.4f; // excluded by top_k=3
        for (int i = 0; i < 12; ++i) {
            const int32_t t1 = s1.sample_from_f32(f32.data(), 256);
            const int32_t t2 = s2.sample_from_f32(f32.data(), 256);
            if (t1 != t2 || (t1 != 10 && t1 != 20 && t1 != 30)) {
                std::printf("[FAIL] Seeded stochastic sampler mismatch t1=%d t2=%d\n", t1, t2);
                return 1;
            }
        }
    }

    // 2. Multi-turn ChatML incremental delta tokenization == full single-shot transcript
    {
        const std::string dir = test_tmp_path("mock_tok_multiturn");
        std::filesystem::create_directories(dir);
        {
            std::ofstream ofs(dir + "/tokenizer.json");
            ofs << build_mock_tokenizer_json(512);
        }
        {
            std::ofstream ofs(dir + "/generation_config.json");
            ofs << "{\"eos_token_id\": [505, 504]}\n";
        }
        auto tk = rxinfer::Tokenizer::from_hf_json_file(dir + "/tokenizer.json");
        if (tk.detect_chat_format() != rxinfer::Tokenizer::ChatFormat::ChatML || !tk.is_eos(504)) {
            std::printf("[FAIL] generation_config.json EOS or ChatML format detection failed\n");
            return 1;
        }

        const std::string sys = "Be concise.";
        const std::string u1 = "hello world";
        const std::string a1 = "hi!";
        const std::string u2 = "42";

        const std::string full_str =
            "<|im_start|>system\n" + sys + "<|im_end|>\n"
            "<|im_start|>user\n" + u1 + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n" + a1 + "<|im_end|>\n"
            "<|im_start|>user\n" + u2 + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        const auto full_ids = tk.encode(full_str, false);

        const auto t1_ids = tk.encode(tk.format_chat_turn(u1, sys, true, false, false), false);
        const auto a1_ids = tk.encode(a1, false);
        const auto t2_ids = tk.encode(tk.format_chat_turn(u2, sys, false, true, false), false);

        std::vector<int32_t> inc_ids = t1_ids;
        inc_ids.insert(inc_ids.end(), a1_ids.begin(), a1_ids.end());
        inc_ids.insert(inc_ids.end(), t2_ids.begin(), t2_ids.end());

        if (inc_ids != full_ids) {
            std::printf("[FAIL] Incremental multi-turn ChatML tokens != full transcript (%zu vs %zu)\n",
                        inc_ids.size(), full_ids.size());
            return 1;
        }

        const std::string th_on_1 = tk.format_chatml(u1, sys, true, true);
        const std::string th_on_t = tk.format_chat_turn(u1, sys, true, false, true);
        const std::string th_off_t = tk.format_chat_turn(u1, sys, true, false, false);
        if (th_on_1 != th_on_t ||
            th_on_t.find("<|im_start|>assistant\n<think>\n") == std::string::npos ||
            th_off_t.find("<|im_start|>assistant\n<think>\n\n</think>\n\n") == std::string::npos) {
            std::printf("[FAIL] --think ChatML prefix mismatch\n");
            return 1;
        }
    }

    // 3. Llama3 chat format detection & multi-turn delta test
    {
        const std::string llama3_tok = R"({
            "added_tokens": [
                {"id": 0, "content": "<|begin_of_text|>", "special": true},
                {"id": 1, "content": "<|start_header_id|>", "special": true},
                {"id": 2, "content": "<|end_header_id|>", "special": true},
                {"id": 3, "content": "<|eot_id|>", "special": true}
            ],
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
            "decoder": {"type": "ByteLevel"},
            "post_processor": {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}]},
            "model": {"type": "BPE", "vocab": {"<|begin_of_text|>": 0, "<|start_header_id|>": 1, "<|end_header_id|>": 2, "<|eot_id|>": 3, "u": 4, "s": 5, "e": 6, "r": 7, "a": 8, "i": 9, "t": 10, "n": 11, "h": 12, "o": 13, "Ċ": 14}, "merges": []}
        })";
        auto tk = rxinfer::Tokenizer::from_hf_json_string(llama3_tok);
        tk.set_bos_id(0);
        tk.add_eos_id(3);
        if (tk.detect_chat_format() != rxinfer::Tokenizer::ChatFormat::Llama3 || !tk.should_add_bos()) {
            std::printf("[FAIL] Llama3 format/BOS detection failed\n");
            return 1;
        }
        const std::string d0 = tk.format_chat_turn("hi", "sys", true, false, false);
        const std::string d1 = tk.format_chat_turn("next", "sys", false, true, false);
        if (d0.find("<|start_header_id|>system<|end_header_id|>") == std::string::npos ||
            d1.rfind("<|eot_id|><|start_header_id|>user", 0) != 0) {
            std::printf("[FAIL] Llama3 chat turn formatting mismatch\n");
            return 1;
        }
    }

    std::printf("[PASS] Sampler + Multi-turn ChatML/Llama3 + generation_config.json\n");
    return 0;
}

std::string strip_ansi_sgr(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        if (s[i] == '\x1b' && i + 1 < s.size() && s[i + 1] == '[') {
            size_t j = i + 2;
            while (j < s.size() && s[j] != 'm') ++j;
            if (j < s.size() && s[j] == 'm') {
                i = j + 1;
                continue;
            }
        }
        out.push_back(s[i++]);
    }
    return out;
}

std::string render_markdown_chunks(const std::string& src, bool enabled, size_t chunk_sz) {
    std::string out;
    rxinfer::MarkdownStreamFormatter fmt(enabled, [&](std::string_view sv) { out.append(sv); });
    if (chunk_sz == 0) {
        fmt.push(src);
    } else {
        for (size_t i = 0; i < src.size(); i += chunk_sz) {
            fmt.push(std::string_view(src).substr(i, std::min(chunk_sz, src.size() - i)));
        }
    }
    fmt.finish();
    return out;
}

int test_markdown_formatter() {
    const std::string doc =
        "<think>\nLet's check *reasoning* and **nested bold** plus 3 * 4.\n</think>\n\n"
        "# Heading One\n"
        "## Sub Heading\n\n"
        "Here is *italic*, **bold**, ***bold italic***, ~~strike~~, `inline_code`, and \\*escaped\\*.\n"
        "Math: 3 * 4 = 12 and 2*x + 3*y and snake_case_var stay untouched.\n\n"
        "*   **Item One:** Bullet with nested **bold**.\n"
        "    - Sub-bullet with *emphasis*.\n"
        "1. Ordered first\n"
        "2. Ordered second\n\n"
        "> Quoted *text* here\n"
        "---\n"
        "```cpp\n"
        "int a = 3 * 4; // **not bold** inside code fence\n"
        "```\n"
        "Unicode UTF-8: café 🚀 日本語\n";

    // 1. Raw pass-through mode across 1-byte chunks preserves exact UTF-8 bytes
    const std::string raw_1 = render_markdown_chunks(doc, false, 1);
    const std::string raw_all = render_markdown_chunks(doc, false, 0);
    if (raw_1 != doc || raw_all != doc) {
        std::printf("[FAIL] Markdown disabled pass-through altered text\n");
        return 1;
    }

    // 2. Formatted mode across whole-doc, 1-byte, 2-byte, 3-byte, and 5-byte streams
    const std::string f_all = render_markdown_chunks(doc, true, 0);
    const std::string f_1 = render_markdown_chunks(doc, true, 1);
    const std::string f_2 = render_markdown_chunks(doc, true, 2);
    const std::string f_3 = render_markdown_chunks(doc, true, 3);
    const std::string f_5 = render_markdown_chunks(doc, true, 5);

    if (f_1 != f_all || f_2 != f_all || f_3 != f_all || f_5 != f_all) {
        std::printf("[FAIL] Markdown streaming chunk-size invariance mismatch!\n");
        std::printf("--- f_all ---\n%s\n--- f_1 ---\n%s\n", strip_ansi_sgr(f_all).c_str(), strip_ansi_sgr(f_1).c_str());
        return 1;
    }

    const std::string plain = strip_ansi_sgr(f_all);
    // Check ANSI codes are present and end cleanly with SGR reset (if any style was used)
    if (f_all.find("\x1b[") == std::string::npos) {
        std::printf("[FAIL] Markdown formatter produced no ANSI SGR escapes\n");
        return 1;
    }
    // Verify ** / * markers inside prose were stripped while literal math/code/escapes remain intact
    if (plain.find("italic, bold, bold italic, strike, inline_code, and *escaped*.") == std::string::npos) {
        std::printf("[FAIL] Markdown inline formatting plain text mismatch:\n%s\n", plain.c_str());
        return 1;
    }
    if (plain.find("3 * 4 = 12 and 2*x + 3*y and snake_case_var") == std::string::npos) {
        std::printf("[FAIL] Markdown math/snake_case got mangled:\n%s\n", plain.c_str());
        return 1;
    }
    if (plain.find("int a = 3 * 4; // **not bold** inside code fence") == std::string::npos) {
        std::printf("[FAIL] Markdown code fence altered code content:\n%s\n", plain.c_str());
        return 1;
    }
    if (plain.find("• Item One: Bullet with nested bold.") == std::string::npos ||
        plain.find("▪ Sub-bullet with emphasis.") == std::string::npos) {
        std::printf("[FAIL] Markdown bullet + bold colon formatting mismatch:\n%s\n", plain.c_str());
        return 1;
    }
    if (plain.find("café 🚀 日本語") == std::string::npos) {
        std::printf("[FAIL] Markdown UTF-8 text corrupted:\n%s\n", plain.c_str());
        return 1;
    }
    // Check unclosed italic SGR cleanup + pending trailing delimiter flush on finish()
    const std::string unclosed_italic = render_markdown_chunks("hello *world", true, 1);
    if (strip_ansi_sgr(unclosed_italic) != "hello world" ||
        unclosed_italic.rfind("\x1b[0m") == std::string::npos) {
        std::printf("[FAIL] Unclosed italic SGR cleanup mismatch: '%s'\n", unclosed_italic.c_str());
        return 1;
    }
    const std::string trailing_star = strip_ansi_sgr(render_markdown_chunks("hello *", true, 1));
    if (trailing_star != "hello *") {
        std::printf("[FAIL] Trailing asterisk flush mismatch: '%s'\n", trailing_star.c_str());
        return 1;
    }

    std::printf("[PASS] MarkdownStreamFormatter (streaming byte/chunk invariance, SGR, UTF-8, math, fences, <think>)\n");
    return 0;
}

} // namespace

int main() {
    if (test_tokenizer() != 0) return 1;
    if (test_sampler_and_multiturn_chat() != 0) return 1;
    if (test_markdown_formatter() != 0) return 1;
    if (test_safetensors_hf_qwen35_e2e() != 0) return 1;
    if (test_gguf_q4k_q6k_tokenizer_e2e() != 0) return 1;
    if (test_gguf_qwen35_iq_and_rxr4_e2e() != 0) return 1;
    std::printf("\nALL IO / TOKENIZER TESTS PASSED!\n");
    return 0;
}

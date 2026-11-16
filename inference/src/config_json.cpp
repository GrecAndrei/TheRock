// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "config_json.hpp"
#include "rxinfer/config.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace rxinfer::json {
namespace {

class Parser {
public:
    explicit Parser(const std::string& src) : src_(src) {}

    Value parse_root() {
        skip_ws();
        Value v = parse_value();
        skip_ws();
        if (pos_ != src_.size()) {
            throw std::runtime_error("Trailing characters in JSON");
        }
        return v;
    }

private:
    const std::string& src_;
    size_t pos_ = 0;

    void skip_ws() {
        while (pos_ < src_.size() && std::isspace(static_cast<unsigned char>(src_[pos_]))) {
            ++pos_;
        }
    }

    char peek() const {
        if (pos_ >= src_.size()) throw std::runtime_error("Unexpected end of JSON");
        return src_[pos_];
    }

    char take() {
        char c = peek();
        ++pos_;
        return c;
    }

    void expect(char c) {
        if (take() != c) {
            throw std::runtime_error(std::string("Expected '") + c + "' in JSON");
        }
    }

    Value parse_value() {
        skip_ws();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return Value{parse_string()};
        if (c == 't') {
            consume_literal("true");
            return Value{true};
        }
        if (c == 'f') {
            consume_literal("false");
            return Value{false};
        }
        if (c == 'n') {
            consume_literal("null");
            return Value{nullptr};
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            return Value{parse_number()};
        }
        throw std::runtime_error("Invalid JSON token");
    }

    void consume_literal(const char* lit) {
        for (size_t i = 0; lit[i] != '\0'; ++i) {
            if (take() != lit[i]) {
                throw std::runtime_error(std::string("Invalid JSON literal: ") + lit);
            }
        }
    }

    uint32_t parse_hex4() {
        if (pos_ + 4 > src_.size()) throw std::runtime_error("Bad \\u escape");
        uint32_t val = 0;
        for (int i = 0; i < 4; ++i) {
            const char ch = src_[pos_++];
            val <<= 4;
            if (ch >= '0' && ch <= '9') val |= static_cast<uint32_t>(ch - '0');
            else if (ch >= 'a' && ch <= 'f') val |= static_cast<uint32_t>(ch - 'a' + 10);
            else if (ch >= 'A' && ch <= 'F') val |= static_cast<uint32_t>(ch - 'A' + 10);
            else throw std::runtime_error("Invalid hex digit in \\u escape");
        }
        return val;
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80u) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800u) {
            out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        } else if (cp < 0x10000u) {
            out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        } else {
            out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        }
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            char c = take();
            if (c == '"') break;
            if (c == '\\') {
                char esc = take();
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        uint32_t cp = parse_hex4();
                        if (cp >= 0xD800u && cp <= 0xDBFFu &&
                            pos_ + 2 <= src_.size() && src_[pos_] == '\\' && src_[pos_ + 1] == 'u') {
                            const size_t saved = pos_;
                            pos_ += 2;
                            const uint32_t low = parse_hex4();
                            if (low >= 0xDC00u && low <= 0xDFFFu) {
                                cp = 0x10000u + (((cp - 0xD800u) << 10) | (low - 0xDC00u));
                            } else {
                                pos_ = saved;
                            }
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default:
                        throw std::runtime_error("Unsupported escape in JSON string");
                }
            } else {
                out.push_back(c);
            }
        }
        return out;
    }

    double parse_number() {
        const char* start = src_.c_str() + pos_;
        char* end = nullptr;
        double val = std::strtod(start, &end);
        if (end == start) throw std::runtime_error("Invalid JSON number");
        pos_ = static_cast<size_t>(end - src_.c_str());
        return val;
    }

    Value parse_array() {
        expect('[');
        Array arr;
        skip_ws();
        if (peek() == ']') {
            take();
            return Value{std::move(arr)};
        }
        while (true) {
            arr.push_back(parse_value());
            skip_ws();
            char c = take();
            if (c == ']') break;
            if (c != ',') throw std::runtime_error("Expected ',' or ']' in JSON array");
        }
        return Value{std::move(arr)};
    }

    Value parse_object() {
        expect('{');
        Object obj;
        skip_ws();
        if (peek() == '}') {
            take();
            return Value{std::move(obj)};
        }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            obj.emplace(std::move(key), parse_value());
            skip_ws();
            char c = take();
            if (c == '}') break;
            if (c != ',') throw std::runtime_error("Expected ',' or '}' in JSON object");
        }
        return Value{std::move(obj)};
    }
};

} // namespace

Value parse(const std::string& text) {
    return Parser(text).parse_root();
}

void append_escaped_string(std::string& out, const std::string& s) {
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    out.push_back('"');
}

std::string escape_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    append_escaped_string(out, s);
    return out;
}

namespace {

void dump_impl(const Value& v, std::string& out, int indent, int depth) {
    if (v.is_null()) {
        out += "null";
    } else if (v.is_bool()) {
        out += std::get<bool>(v.data) ? "true" : "false";
    } else if (v.is_number()) {
        double d = std::get<double>(v.data);
        if (!std::isfinite(d)) {
            out += "null";
        } else if (d == 0.0) {
            out.push_back('0');
        } else if (std::fabs(d) < 9007199254740992.0 && d == std::trunc(d)) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(d));
            out += buf;
        } else {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.17g", d);
            out += buf;
        }
    } else if (v.is_string()) {
        append_escaped_string(out, std::get<std::string>(v.data));
    } else if (v.is_array()) {
        const auto& arr = std::get<Array>(v.data);
        if (arr.empty()) {
            out += "[]";
            return;
        }
        out.push_back('[');
        for (size_t i = 0; i < arr.size(); ++i) {
            if (i > 0) out.push_back(',');
            if (indent >= 0) {
                out.push_back('\n');
                out.append(static_cast<size_t>((depth + 1) * indent), ' ');
            }
            dump_impl(arr[i], out, indent, depth + 1);
        }
        if (indent >= 0) {
            out.push_back('\n');
            out.append(static_cast<size_t>(depth * indent), ' ');
        }
        out.push_back(']');
    } else if (v.is_object()) {
        const auto& obj = std::get<Object>(v.data);
        if (obj.empty()) {
            out += "{}";
            return;
        }
        out.push_back('{');
        bool first = true;
        for (const auto& kv : obj) {
            if (!first) out.push_back(',');
            first = false;
            if (indent >= 0) {
                out.push_back('\n');
                out.append(static_cast<size_t>((depth + 1) * indent), ' ');
            }
            append_escaped_string(out, kv.first);
            out.push_back(':');
            if (indent >= 0) out.push_back(' ');
            dump_impl(kv.second, out, indent, depth + 1);
        }
        if (indent >= 0) {
            out.push_back('\n');
            out.append(static_cast<size_t>(depth * indent), ' ');
        }
        out.push_back('}');
    }
}

} // namespace

std::string dump(const Value& v, int indent) {
    std::string out;
    dump_impl(v, out, indent, 0);
    return out;
}

} // namespace rxinfer::json

namespace rxinfer {

int32_t ModelConfig::num_full_attention_layers() const {
    if (layer_types.empty()) return num_hidden_layers;
    int32_t count = 0;
    for (LayerKind kind : layer_types) {
        if (kind == LayerKind::FullAttention) ++count;
    }
    return count;
}

int32_t ModelConfig::num_linear_attention_layers() const {
    int32_t count = 0;
    for (LayerKind kind : layer_types) {
        if (kind == LayerKind::LinearAttentionGDN) ++count;
    }
    return count;
}

uint64_t ModelConfig::estimate_kv_cache_bytes_per_token() const {
    const uint64_t attn_layers = static_cast<uint64_t>(num_full_attention_layers());
    const uint64_t vec_dim = static_cast<uint64_t>(num_key_value_heads) * static_cast<uint64_t>(head_dim);
    if (kv_dtype == DType::FP4_E2M1_G16) {
        // 4-bit E2M1 payload + 1 byte FP8 E4M3 scale per 16 channels = 9 bytes per 16 channels
        const uint64_t bytes_per_vec = (vec_dim / 2ULL) + (vec_dim / 16ULL);
        return attn_layers * 2ULL * bytes_per_vec;
    }
    const uint64_t elem_bytes = dtype_element_bits(kv_dtype) / 8;
    // K and V per layer per token
    return attn_layers * 2ULL * vec_dim * elem_bytes;
}

uint64_t ModelConfig::estimate_recurrent_state_bytes() const {
    const uint64_t gdn_layers = static_cast<uint64_t>(num_linear_attention_layers());
    if (gdn_layers == 0) return 0;
    // FP32 recurrent matrix S[v_heads, k_dim, v_dim] + conv1d circular buffer in FP16
    const uint64_t s_bytes = static_cast<uint64_t>(linear_num_value_heads) *
                             static_cast<uint64_t>(linear_key_head_dim) *
                             static_cast<uint64_t>(linear_value_head_dim) * sizeof(float);
    const uint64_t conv_dim = static_cast<uint64_t>(linear_num_key_heads) * linear_key_head_dim * 2ULL +
                              static_cast<uint64_t>(linear_num_value_heads) * linear_value_head_dim;
    const uint64_t conv_bytes = conv_dim * static_cast<uint64_t>(linear_conv_kernel_dim) * 2ULL;
    return gdn_layers * (s_bytes + conv_bytes);
}

ModelConfig ModelConfig::from_hf_json_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("Failed to open config JSON: " + path);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return from_hf_json_string(oss.str());
}

ModelConfig ModelConfig::from_hf_json_string(const std::string& json_text) {
    const json::Value root = json::parse(json_text);
    const json::Value* cfg = root.get("text_config");
    if (!cfg || !cfg->is_object()) {
        cfg = &root;
    }

    auto pick = [&](const char* key) -> const json::Value* {
        if (const auto* v = cfg->get(key)) return v;
        return root.get(key);
    };

    ModelConfig mc;
    if (const auto* v = pick("model_type")) mc.model_type = v->as_str();
    if (const auto* v = root.get("architectures"); v && v->is_array()) {
        const auto& arr = std::get<json::Array>(v->data);
        if (!arr.empty()) mc.arch_name = arr[0].as_str();
    }

    if (const auto* v = pick("vocab_size")) mc.vocab_size = static_cast<int32_t>(v->as_i64());
    if (const auto* v = pick("hidden_size")) mc.hidden_size = static_cast<int32_t>(v->as_i64());
    if (const auto* v = pick("intermediate_size")) mc.intermediate_size = static_cast<int32_t>(v->as_i64());
    if (const auto* v = pick("num_hidden_layers")) mc.num_hidden_layers = static_cast<int32_t>(v->as_i64());
    if (const auto* v = pick("max_position_embeddings")) mc.max_position_embeddings = static_cast<int32_t>(v->as_i64(32768));
    if (const auto* v = pick("tie_word_embeddings")) mc.tie_word_embeddings = v->as_bool(false);
    if (const auto* v = pick("rms_norm_eps")) mc.rms_norm_eps = static_cast<float>(v->as_f64(1e-6));

    if (const auto* v = pick("num_attention_heads")) mc.num_attention_heads = static_cast<int32_t>(v->as_i64());
    if (const auto* v = pick("num_key_value_heads")) {
        mc.num_key_value_heads = static_cast<int32_t>(v->as_i64());
    } else {
        mc.num_key_value_heads = mc.num_attention_heads;
    }
    if (const auto* v = pick("head_dim")) {
        mc.head_dim = static_cast<int32_t>(v->as_i64());
    } else if (mc.num_attention_heads > 0) {
        mc.head_dim = mc.hidden_size / mc.num_attention_heads;
    }
    // Architecture defaults (applied before explicit JSON overrides)
    if (mc.model_type.find("qwen3_5") != std::string::npos ||
        mc.model_type.find("qwen3_8") != std::string::npos ||
        mc.model_type.find("qwen3_next") != std::string::npos) {
        mc.norm_kind = NormKind::RMSNormPlusOne;
        mc.qk_norm = true;
        mc.attn_output_gate = true;
    } else if (mc.model_type == "qwen3") {
        mc.norm_kind = NormKind::RMSNorm;
        mc.qk_norm = true;
    }

    if (const auto* v = pick("attention_bias")) mc.attention_bias = v->as_bool(false);
    if (const auto* v = pick("attn_output_gate")) mc.attn_output_gate = v->as_bool(mc.attn_output_gate);
    if (const auto* v = pick("qk_norm")) mc.qk_norm = v->as_bool(mc.qk_norm);
    if (const auto* v = pick("mtp_num_hidden_layers")) mc.mtp_num_hidden_layers = static_cast<int32_t>(v->as_i64(0));
    if (const auto* v = pick("bos_token_id"); v && v->is_number()) mc.bos_token_id = static_cast<int32_t>(v->as_i64(-1));
    if (const auto* v = pick("eos_token_id")) {
        if (v->is_number()) {
            mc.eos_token_ids.push_back(static_cast<int32_t>(v->as_i64()));
        } else if (v->is_array()) {
            for (const auto& elem : std::get<json::Array>(v->data)) {
                if (elem.is_number()) mc.eos_token_ids.push_back(static_cast<int32_t>(elem.as_i64()));
            }
        }
    }

    // RoPE parameters (top-level or nested under rope_parameters / rope_scaling)
    if (const auto* v = pick("rope_theta")) mc.rope_theta = static_cast<float>(v->as_f64(10000.0));
    if (const auto* v = pick("partial_rotary_factor")) mc.partial_rotary_factor = static_cast<float>(v->as_f64(1.0));
    if (const auto* v = pick("mrope_interleaved")) mc.mrope_interleaved = v->as_bool(false);
    if (const auto* v = pick("mrope_section"); v && v->is_array()) {
        for (const auto& elem : std::get<json::Array>(v->data)) {
            mc.mrope_section.push_back(static_cast<int32_t>(elem.as_i64()));
        }
    }
    const json::Value* rope_p = pick("rope_parameters");
    if (!rope_p || !rope_p->is_object()) rope_p = pick("rope_scaling");
    if (rope_p && rope_p->is_object()) {
        if (const auto* v = rope_p->get("rope_theta")) mc.rope_theta = static_cast<float>(v->as_f64(mc.rope_theta));
        if (const auto* v = rope_p->get("partial_rotary_factor")) {
            mc.partial_rotary_factor = static_cast<float>(v->as_f64(mc.partial_rotary_factor));
        }
        if (const auto* v = rope_p->get("mrope_interleaved")) {
            mc.mrope_interleaved = v->as_bool(false);
        }
        if (const auto* v = rope_p->get("mrope_section"); v && v->is_array()) {
            mc.mrope_section.clear();
            for (const auto& elem : std::get<json::Array>(v->data)) {
                mc.mrope_section.push_back(static_cast<int32_t>(elem.as_i64()));
            }
        }
    }
    if (const auto* v = pick("rope_kind")) {
        const std::string rk = v->as_str();
        if (rk == "interleaved") mc.rope_kind = RopeKind::Interleaved;
        else if (rk == "none") mc.rope_kind = RopeKind::None;
        else mc.rope_kind = RopeKind::NeoX;
    }
    if (const auto* v = pick("norm_kind")) {
        const std::string nk = v->as_str();
        if (nk == "rmsnorm_plus_one") mc.norm_kind = NormKind::RMSNormPlusOne;
        else if (nk == "layernorm") mc.norm_kind = NormKind::LayerNorm;
        else mc.norm_kind = NormKind::RMSNorm;
    }

    // Linear attention / GDN params
    if (const auto* v = pick("linear_num_key_heads")) mc.linear_num_key_heads = static_cast<int32_t>(v->as_i64());
    if (const auto* v = pick("linear_key_head_dim")) mc.linear_key_head_dim = static_cast<int32_t>(v->as_i64());
    if (const auto* v = pick("linear_num_value_heads")) mc.linear_num_value_heads = static_cast<int32_t>(v->as_i64());
    if (const auto* v = pick("linear_value_head_dim")) mc.linear_value_head_dim = static_cast<int32_t>(v->as_i64());
    if (const auto* v = pick("linear_conv_kernel_dim")) mc.linear_conv_kernel_dim = static_cast<int32_t>(v->as_i64(4));
    if (const auto* v = pick("gdn_tiled_v_heads")) mc.gdn_tiled_v_heads = v->as_bool(false);

    // Per-layer schedule
    if (const auto* v = pick("layer_types"); v && v->is_array()) {
        for (const auto& elem : std::get<json::Array>(v->data)) {
            const std::string s = elem.as_str();
            if (s == "linear_attention") {
                mc.layer_types.push_back(LayerKind::LinearAttentionGDN);
            } else {
                mc.layer_types.push_back(LayerKind::FullAttention);
            }
        }
    } else if (mc.num_hidden_layers > 0) {
        int32_t full_interval = 0;
        if (const auto* v = pick("full_attention_interval")) full_interval = static_cast<int32_t>(v->as_i64(0));
        if (full_interval > 0 && mc.linear_num_value_heads > 0) {
            for (int32_t l = 0; l < mc.num_hidden_layers; ++l) {
                mc.layer_types.push_back(((l + 1) % full_interval == 0)
                                             ? LayerKind::FullAttention
                                             : LayerKind::LinearAttentionGDN);
            }
        } else {
            mc.layer_types.assign(mc.num_hidden_layers, LayerKind::FullAttention);
        }
    }

    return mc;
}

} // namespace rxinfer

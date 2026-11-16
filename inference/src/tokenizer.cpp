// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/tokenizer.hpp"
#include "config_json.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rxinfer {
namespace {

inline size_t decode_utf8(const std::string& s, size_t pos, uint32_t& cp) {
    const size_t n = s.size();
    const uint8_t b0 = static_cast<uint8_t>(s[pos]);
    if (b0 < 0x80u) {
        cp = b0;
        return pos + 1;
    }
    if ((b0 & 0xE0u) == 0xC0u && pos + 1 < n) {
        const uint8_t b1 = static_cast<uint8_t>(s[pos + 1]);
        if ((b1 & 0xC0u) == 0x80u) {
            cp = ((b0 & 0x1Fu) << 6) | (b1 & 0x3Fu);
            return pos + 2;
        }
    } else if ((b0 & 0xF0u) == 0xE0u && pos + 2 < n) {
        const uint8_t b1 = static_cast<uint8_t>(s[pos + 1]);
        const uint8_t b2 = static_cast<uint8_t>(s[pos + 2]);
        if ((b1 & 0xC0u) == 0x80u && (b2 & 0xC0u) == 0x80u) {
            cp = ((b0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
            return pos + 3;
        }
    } else if ((b0 & 0xF8u) == 0xF0u && pos + 3 < n) {
        const uint8_t b1 = static_cast<uint8_t>(s[pos + 1]);
        const uint8_t b2 = static_cast<uint8_t>(s[pos + 2]);
        const uint8_t b3 = static_cast<uint8_t>(s[pos + 3]);
        if ((b1 & 0xC0u) == 0x80u && (b2 & 0xC0u) == 0x80u && (b3 & 0xC0u) == 0x80u) {
            cp = ((b0 & 0x07u) << 18) | ((b1 & 0x3Fu) << 12) | ((b2 & 0x3Fu) << 6) | (b3 & 0x3Fu);
            return pos + 4;
        }
    }
    cp = b0;
    return pos + 1;
}

inline void append_utf8_cp(std::string& out, uint32_t cp) {
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

inline bool is_unicode_number(uint32_t cp) {
    if (cp >= '0' && cp <= '9') return true;
    if (cp == 0x00B2u || cp == 0x00B3u || cp == 0x00B9u) return true;
    if (cp >= 0x00BCu && cp <= 0x00BEu) return true;
    if (cp >= 0x0660u && cp <= 0x0669u) return true;
    if (cp >= 0x06F0u && cp <= 0x06F9u) return true;
    if (cp >= 0xFF10u && cp <= 0xFF19u) return true;
    return false;
}

inline bool is_unicode_space(uint32_t cp) {
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' || cp == '\f') return true;
    if (cp == 0x0085u || cp == 0x00A0u || cp == 0x1680u || cp == 0x2028u ||
        cp == 0x2029u || cp == 0x202Fu || cp == 0x205Fu || cp == 0x3000u) {
        return true;
    }
    if (cp >= 0x2000u && cp <= 0x200Au) return true;
    return false;
}

inline bool is_unicode_letter(uint32_t cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    if (cp < 0x80u) return false;
    if (is_unicode_number(cp) || is_unicode_space(cp)) return false;
    if (cp >= 0x00C0u && cp <= 0x02AFu) return (cp != 0x00D7u && cp != 0x00F7u);
    if (cp >= 0x0370u && cp <= 0x052Fu) return (cp != 0x037Eu && cp != 0x0387u);
    if (cp >= 0x0530u && cp <= 0x06FFu) return (cp != 0x060Cu && cp != 0x061Bu && cp != 0x061Fu);
    if (cp >= 0x0700u && cp <= 0x109Fu) return true;
    if (cp >= 0x1100u && cp <= 0x13FFu) return true;
    if (cp >= 0x1E00u && cp <= 0x1FFFu) return true;
    if (cp >= 0x3040u && cp <= 0x31FFu) return true;
    if (cp >= 0x3400u && cp <= 0x4DBFu) return true;
    if (cp >= 0x4E00u && cp <= 0x9FFFu) return true;
    if (cp >= 0xAC00u && cp <= 0xD7AFu) return true;
    if (cp >= 0xF900u && cp <= 0xFAFFu) return true;
    if ((cp >= 0xFF21u && cp <= 0xFF3Au) || (cp >= 0xFF41u && cp <= 0xFF5Au) ||
        (cp >= 0xFF66u && cp <= 0xFFDCu)) {
        return true;
    }
    if (cp >= 0x20000u && cp <= 0x2EBFFu) return true;
    return false;
}

std::string read_file_str(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open file: " + path);
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

std::string parent_dir(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    return (slash == std::string::npos) ? "." : path.substr(0, slash);
}

bool file_exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

struct MappedHeader {
    int fd = -1;
    size_t size = 0;
    const uint8_t* data = nullptr;

    explicit MappedHeader(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Failed to open file: " + path);
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error("fstat failed: " + path);
        }
        size = static_cast<size_t>(st.st_size);
        if (size > 0) {
            void* p = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (p == MAP_FAILED) {
                ::close(fd);
                throw std::runtime_error("mmap failed: " + path);
            }
            data = static_cast<const uint8_t*>(p);
        }
    }
    ~MappedHeader() {
        if (data && size > 0) ::munmap(const_cast<uint8_t*>(data), size);
        if (fd >= 0) ::close(fd);
    }
};

} // namespace

void Tokenizer::init_byte_tables() {
    std::fill(std::begin(cp_to_byte_), std::end(cp_to_byte_), static_cast<int16_t>(-1));
    std::array<bool, 256> direct{};
    for (int b = 33; b <= 126; ++b) direct[b] = true;
    for (int b = 161; b <= 172; ++b) direct[b] = true;
    for (int b = 174; b <= 255; ++b) direct[b] = true;

    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        const uint32_t cp = direct[b] ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + extra++);
        byte_to_utf8_[b].clear();
        append_utf8_cp(byte_to_utf8_[b], cp);
        if (cp < 512) {
            cp_to_byte_[cp] = static_cast<int16_t>(b);
        }
    }
}

void Tokenizer::finalize_added_tokens() {
    std::sort(added_tokens_.begin(), added_tokens_.end(),
              [](const AddedToken& a, const AddedToken& b) {
                  if (a.content.size() != b.content.size()) {
                      return a.content.size() > b.content.size();
                  }
                  return a.id < b.id;
              });
    added_by_first_byte_.assign(256, {});
    for (size_t i = 0; i < added_tokens_.size(); ++i) {
        if (!added_tokens_[i].content.empty()) {
            const uint8_t first = static_cast<uint8_t>(added_tokens_[i].content[0]);
            added_by_first_byte_[first].push_back(i);
        }
    }
}

void Tokenizer::add_eos_id(int32_t id) {
    if (id < 0) return;
    if (std::find(eos_ids_.begin(), eos_ids_.end(), id) == eos_ids_.end()) {
        eos_ids_.push_back(id);
    }
}

bool Tokenizer::is_eos(int32_t id) const {
    return std::find(eos_ids_.begin(), eos_ids_.end(), id) != eos_ids_.end();
}

int32_t Tokenizer::token_to_id(const std::string& tok) const {
    auto it = token_to_id_.find(tok);
    return (it == token_to_id_.end()) ? -1 : it->second;
}

const std::string& Tokenizer::id_to_token(int32_t id) const {
    static const std::string empty;
    if (id < 0 || id >= static_cast<int32_t>(id_to_token_.size())) return empty;
    return id_to_token_[id];
}

Tokenizer Tokenizer::from_hf_json_string(const std::string& json_text) {
    Tokenizer tk;
    tk.init_byte_tables();

    const json::Value root = json::parse(json_text);
    const json::Value* model = root.get("model");
    if (!model || !model->is_object()) {
        throw std::runtime_error("tokenizer.json missing 'model' object");
    }

    if (const auto* bf = model->get("byte_fallback")) {
        tk.byte_fallback_ = bf->as_bool(false);
    }

    auto inspect_node = [&](auto&& self, const json::Value* node) -> void {
        if (!node || !node->is_object()) return;
        const std::string t = node->get("type") ? node->get("type")->as_str() : "";
        if (t == "ByteLevel") {
            tk.byte_level_ = true;
            if (const auto* aps = node->get("add_prefix_space")) {
                tk.add_prefix_space_ = tk.add_prefix_space_ || aps->as_bool(false);
            }
        } else if (t == "Metaspace") {
            tk.metaspace_ = true;
            if (const auto* aps = node->get("add_prefix_space")) {
                tk.add_prefix_space_ = tk.add_prefix_space_ || aps->as_bool(false);
            }
        } else if (t == "Split") {
            if (const auto* pat = node->get("pattern")) {
                if (const auto* rgx = pat->get("Regex")) {
                    const std::string r = rgx->as_str();
                    if (r.find("\\p{N}{1,3}") != std::string::npos ||
                        r.find("{1,3}") != std::string::npos) {
                        tk.max_digit_chunk_ = 3;
                    }
                }
            }
        }
        for (const char* arr_key : {"pretokenizers", "decoders", "normalizers"}) {
            if (const auto* arr = node->get(arr_key); arr && arr->is_array()) {
                for (const auto& elem : std::get<json::Array>(arr->data)) {
                    self(self, &elem);
                }
            }
        }
    };

    inspect_node(inspect_node, root.get("pre_tokenizer"));
    inspect_node(inspect_node, root.get("decoder"));

    // Determine max vocab ID across model.vocab and added_tokens
    int32_t max_id = -1;
    const json::Value* vocab = model->get("vocab");
    if (!vocab || !vocab->is_object()) {
        throw std::runtime_error("tokenizer.json missing 'model.vocab'");
    }
    for (const auto& kv : std::get<json::Object>(vocab->data)) {
        const int32_t id = static_cast<int32_t>(kv.second.as_i64(-1));
        if (id > max_id) max_id = id;
    }

    const json::Value* added = root.get("added_tokens");
    if (added && added->is_array()) {
        for (const auto& item : std::get<json::Array>(added->data)) {
            if (!item.is_object()) continue;
            if (const auto* iv = item.get("id")) {
                const int32_t id = static_cast<int32_t>(iv->as_i64(-1));
                if (id > max_id) max_id = id;
            }
        }
    }

    if (max_id < 0) throw std::runtime_error("Empty vocabulary in tokenizer.json");
    tk.id_to_token_.assign(static_cast<size_t>(max_id) + 1, "");
    tk.token_to_id_.reserve(static_cast<size_t>(max_id) + 1);

    for (const auto& kv : std::get<json::Object>(vocab->data)) {
        const int32_t id = static_cast<int32_t>(kv.second.as_i64(-1));
        if (id >= 0) {
            tk.id_to_token_[id] = kv.first;
            tk.token_to_id_[kv.first] = id;
        }
    }

    if (added && added->is_array()) {
        for (const auto& item : std::get<json::Array>(added->data)) {
            if (!item.is_object()) continue;
            const auto* id_v = item.get("id");
            const auto* content_v = item.get("content");
            if (!id_v || !content_v) continue;
            const int32_t id = static_cast<int32_t>(id_v->as_i64(-1));
            const std::string content = content_v->as_str();
            const bool is_spec = item.get("special") ? item.get("special")->as_bool(false) : false;
            if (id >= 0 && id < static_cast<int32_t>(tk.id_to_token_.size())) {
                tk.id_to_token_[id] = content;
                tk.token_to_id_[content] = id;
                if (is_spec) tk.special_ids_.insert(id);
                tk.added_tokens_.push_back(AddedToken{content, id, is_spec});
            }
        }
    }

    if (!tk.byte_level_ && !tk.metaspace_) {
        if (tk.token_to_id_.count("\xC4\xA0") > 0) {
            tk.byte_level_ = true;
        } else if (tk.token_to_id_.count("\xE2\x96\x81") > 0) {
            tk.metaspace_ = true;
        }
    }

    // Build BPE merge table (supports both "tokA tokB" strings and ["tokA", "tokB"] arrays)
    if (const auto* merges = model->get("merges"); merges && merges->is_array()) {
        const auto& marr = std::get<json::Array>(merges->data);
        tk.merge_ranks_.reserve(marr.size());
        int32_t rank = 0;
        for (const auto& entry : marr) {
            std::string left, right;
            if (entry.is_string()) {
                const std::string s = entry.as_str();
                const size_t sp = s.find(' ');
                if (sp == std::string::npos) continue;
                left = s.substr(0, sp);
                right = s.substr(sp + 1);
            } else if (entry.is_array()) {
                const auto& pair = std::get<json::Array>(entry.data);
                if (pair.size() != 2) continue;
                left = pair[0].as_str();
                right = pair[1].as_str();
            } else {
                continue;
            }
            const auto it_l = tk.token_to_id_.find(left);
            const auto it_r = tk.token_to_id_.find(right);
            const auto it_m = tk.token_to_id_.find(left + right);
            if (it_l != tk.token_to_id_.end() && it_r != tk.token_to_id_.end() &&
                it_m != tk.token_to_id_.end()) {
                const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(it_l->second)) << 32) |
                                     static_cast<uint32_t>(it_r->second);
                tk.merge_ranks_.emplace(key, MergeEntry{rank, it_m->second});
            }
            ++rank;
        }
    }

    if (const auto* unk = model->get("unk_token"); unk && unk->is_string()) {
        tk.unk_id_ = tk.token_to_id(unk->as_str());
    }
    for (const char* eos_cand : {"<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"}) {
        const int32_t eid = tk.token_to_id(eos_cand);
        if (eid >= 0) tk.add_eos_id(eid);
    }
    if (tk.bos_id_ < 0) {
        for (const char* bos_cand : {"<|begin_of_text|>", "<s>"}) {
            const int32_t bid = tk.token_to_id(bos_cand);
            if (bid >= 0) {
                tk.bos_id_ = bid;
                break;
            }
        }
    }
    if (const auto* pp = root.get("post_processor"); pp && pp->is_object()) {
        if (const auto* single = pp->get("single"); single && single->is_array()) {
            const auto& sarr = std::get<json::Array>(single->data);
            if (!sarr.empty() && sarr[0].is_object() && sarr[0].get("SpecialToken") != nullptr) {
                tk.add_bos_token_ = true;
            }
        }
    } else if (tk.metaspace_ && tk.bos_id_ >= 0 && !tk.is_eos(tk.bos_id_)) {
        tk.add_bos_token_ = true;
    }

    tk.finalize_added_tokens();
    return tk;
}

Tokenizer Tokenizer::from_hf_json_file(const std::string& tokenizer_json_path) {
    Tokenizer tk = from_hf_json_string(read_file_str(tokenizer_json_path));
    const std::string dir = parent_dir(tokenizer_json_path);

    auto apply_token_ref = [&](const json::Value* v, bool is_eos_field) {
        if (!v) return;
        auto resolve_one = [&](const json::Value& item) -> int32_t {
            if (item.is_number()) return static_cast<int32_t>(item.as_i64(-1));
            if (item.is_string()) return tk.token_to_id(item.as_str());
            if (item.is_object()) {
                if (const auto* c = item.get("content")) return tk.token_to_id(c->as_str());
            }
            return -1;
        };
        if (v->is_array()) {
            for (const auto& el : std::get<json::Array>(v->data)) {
                const int32_t id = resolve_one(el);
                if (id >= 0 && is_eos_field) tk.add_eos_id(id);
            }
        } else {
            const int32_t id = resolve_one(*v);
            if (id >= 0) {
                if (is_eos_field) tk.add_eos_id(id);
                else tk.set_bos_id(id);
            }
        }
    };

    const std::string tok_cfg_path = dir + "/tokenizer_config.json";
    if (file_exists(tok_cfg_path)) {
        try {
            const json::Value tc = json::parse(read_file_str(tok_cfg_path));
            apply_token_ref(tc.get("bos_token"), false);
            apply_token_ref(tc.get("eos_token"), true);
            apply_token_ref(tc.get("bos_token_id"), false);
            apply_token_ref(tc.get("eos_token_id"), true);
            if (const auto* ab = tc.get("add_bos_token")) {
                tk.set_add_bos_token(ab->as_bool(false));
            }
        } catch (...) {
        }
    }

    const std::string cfg_path = dir + "/config.json";
    if (file_exists(cfg_path)) {
        try {
            const json::Value cfg = json::parse(read_file_str(cfg_path));
            const json::Value* tcfg = cfg.get("text_config");
            apply_token_ref(cfg.get("bos_token_id"), false);
            apply_token_ref(cfg.get("eos_token_id"), true);
            if (tcfg && tcfg->is_object()) {
                apply_token_ref(tcfg->get("bos_token_id"), false);
                apply_token_ref(tcfg->get("eos_token_id"), true);
            }
        } catch (...) {
        }
    }

    const std::string gen_cfg_path = dir + "/generation_config.json";
    if (file_exists(gen_cfg_path)) {
        try {
            const json::Value gcfg = json::parse(read_file_str(gen_cfg_path));
            apply_token_ref(gcfg.get("bos_token_id"), false);
            apply_token_ref(gcfg.get("eos_token_id"), true);
        } catch (...) {
        }
    }

    return tk;
}

Tokenizer Tokenizer::from_gguf(const std::string& gguf_path) {
    MappedHeader mf(gguf_path);
    size_t pos = 0;

    auto read_raw = [&](void* dst, size_t bytes) {
        if (pos + bytes > mf.size) throw std::runtime_error("GGUF unexpected EOF");
        std::memcpy(dst, mf.data + pos, bytes);
        pos += bytes;
    };
    auto read_u32 = [&]() { uint32_t v; read_raw(&v, 4); return v; };
    auto read_i32 = [&]() { int32_t v; read_raw(&v, 4); return v; };
    auto read_u64 = [&]() { uint64_t v; read_raw(&v, 8); return v; };
    auto read_f32 = [&]() { float v; read_raw(&v, 4); return v; };
    auto read_str = [&]() {
        const uint64_t len = read_u64();
        if (pos + len > mf.size) throw std::runtime_error("GGUF string EOF");
        std::string s(reinterpret_cast<const char*>(mf.data + pos), len);
        pos += len;
        return s;
    };
    auto skip_val = [&](auto&& self, uint32_t vt) -> void {
        switch (vt) {
            case 0: case 1: case 7: pos += 1; break;
            case 2: case 3: pos += 2; break;
            case 4: case 5: case 6: pos += 4; break;
            case 10: case 11: case 12: pos += 8; break;
            case 8: (void)read_str(); break;
            case 9: {
                const uint32_t et = read_u32();
                const uint64_t n = read_u64();
                for (uint64_t i = 0; i < n; ++i) self(self, et);
                break;
            }
            default: throw std::runtime_error("Unknown GGUF metadata type");
        }
    };
    auto read_u32_any = [&](uint32_t vt) -> int32_t {
        if (vt == 4) return static_cast<int32_t>(read_u32());
        if (vt == 5) return read_i32();
        if (vt == 10) return static_cast<int32_t>(read_u64());
        skip_val(skip_val, vt);
        return -1;
    };

    if (read_u32() != 0x46554747u) throw std::runtime_error("Not a GGUF file: " + gguf_path);
    const uint32_t ver = read_u32();
    if (ver < 2 || ver > 3) throw std::runtime_error("Unsupported GGUF version");
    (void)read_u64(); // n_tensors
    const uint64_t n_kv = read_u64();

    Tokenizer tk;
    tk.init_byte_tables();

    std::string ggml_model = "gpt2";
    std::string ggml_pre;
    bool has_add_bos = false;
    bool add_bos_val = false;
    std::vector<std::string> merges_raw;
    std::vector<int32_t> token_types;

    for (uint64_t i = 0; i < n_kv; ++i) {
        const std::string key = read_str();
        const uint32_t vt = read_u32();
        if (key == "tokenizer.ggml.model" && vt == 8) {
            ggml_model = read_str();
        } else if (key == "tokenizer.ggml.pre" && vt == 8) {
            ggml_pre = read_str();
        } else if (key == "tokenizer.ggml.add_bos_token" && vt == 7) {
            uint8_t bv = 0;
            read_raw(&bv, 1);
            has_add_bos = true;
            add_bos_val = (bv != 0);
        } else if (key == "tokenizer.ggml.tokens" && vt == 9) {
            const uint32_t et = read_u32();
            const uint64_t n = read_u64();
            if (et == 8) {
                tk.id_to_token_.reserve(n);
                for (uint64_t j = 0; j < n; ++j) tk.id_to_token_.push_back(read_str());
            } else {
                for (uint64_t j = 0; j < n; ++j) skip_val(skip_val, et);
            }
        } else if (key == "tokenizer.ggml.merges" && vt == 9) {
            const uint32_t et = read_u32();
            const uint64_t n = read_u64();
            if (et == 8) {
                merges_raw.reserve(n);
                for (uint64_t j = 0; j < n; ++j) merges_raw.push_back(read_str());
            } else {
                for (uint64_t j = 0; j < n; ++j) skip_val(skip_val, et);
            }
        } else if (key == "tokenizer.ggml.scores" && vt == 9) {
            const uint32_t et = read_u32();
            const uint64_t n = read_u64();
            if (et == 6) {
                tk.scores_.reserve(n);
                for (uint64_t j = 0; j < n; ++j) tk.scores_.push_back(read_f32());
            } else {
                for (uint64_t j = 0; j < n; ++j) skip_val(skip_val, et);
            }
        } else if (key == "tokenizer.ggml.token_type" && vt == 9) {
            const uint32_t et = read_u32();
            const uint64_t n = read_u64();
            token_types.reserve(n);
            for (uint64_t j = 0; j < n; ++j) token_types.push_back(read_u32_any(et));
        } else if (key == "tokenizer.ggml.bos_token_id") {
            tk.bos_id_ = read_u32_any(vt);
        } else if (key == "tokenizer.ggml.eos_token_id" || key == "tokenizer.ggml.eot_token_id") {
            if (vt == 9) {
                const uint32_t et = read_u32();
                const uint64_t n = read_u64();
                for (uint64_t j = 0; j < n; ++j) tk.add_eos_id(read_u32_any(et));
            } else {
                tk.add_eos_id(read_u32_any(vt));
            }
        } else if (key == "tokenizer.ggml.unknown_token_id") {
            tk.unk_id_ = read_u32_any(vt);
        } else {
            skip_val(skip_val, vt);
        }
    }

    if (tk.id_to_token_.empty()) {
        throw std::runtime_error("No tokenizer.ggml.tokens found in GGUF: " + gguf_path);
    }

    tk.token_to_id_.reserve(tk.id_to_token_.size());
    for (size_t id = 0; id < tk.id_to_token_.size(); ++id) {
        tk.token_to_id_[tk.id_to_token_[id]] = static_cast<int32_t>(id);
        if (id < token_types.size()) {
            const int32_t tt = token_types[id];
            if (tt == 3 || tt == 4) { // control or user_defined
                if (tt == 3) tk.special_ids_.insert(static_cast<int32_t>(id));
                tk.added_tokens_.push_back(AddedToken{tk.id_to_token_[id], static_cast<int32_t>(id), tt == 3});
            }
        }
    }

    if (ggml_model == "gpt2") {
        tk.byte_level_ = true;
        if (ggml_pre.find("llama") != std::string::npos ||
            ggml_pre.find("deepseek") != std::string::npos) {
            tk.max_digit_chunk_ = 3;
        } else {
            tk.max_digit_chunk_ = 1;
        }
        tk.merge_ranks_.reserve(merges_raw.size());
        for (size_t r = 0; r < merges_raw.size(); ++r) {
            const size_t sp = merges_raw[r].find(' ');
            if (sp == std::string::npos) continue;
            const std::string left = merges_raw[r].substr(0, sp);
            const std::string right = merges_raw[r].substr(sp + 1);
            const auto it_l = tk.token_to_id_.find(left);
            const auto it_r = tk.token_to_id_.find(right);
            const auto it_m = tk.token_to_id_.find(left + right);
            if (it_l != tk.token_to_id_.end() && it_r != tk.token_to_id_.end() &&
                it_m != tk.token_to_id_.end()) {
                const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(it_l->second)) << 32) |
                                     static_cast<uint32_t>(it_r->second);
                tk.merge_ranks_.emplace(key, MergeEntry{static_cast<int32_t>(r), it_m->second});
            }
        }
    } else {
        tk.metaspace_ = true;
        tk.byte_fallback_ = true;
        tk.use_scores_bpe_ = !tk.scores_.empty();
    }

    if (has_add_bos) {
        tk.add_bos_token_ = add_bos_val;
    } else {
        tk.add_bos_token_ = (!tk.byte_level_ && tk.bos_id_ >= 0 && !tk.is_eos(tk.bos_id_));
    }

    // Also register standard end-of-sequence tokens present in vocab (matching HF tokenizer loader),
    // applied after add_bos_token_ is finalized so byte-level BOS/pad sharing doesn't trigger add_bos.
    for (const char* eos_cand : {"<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"}) {
        const int32_t eid = tk.token_to_id(eos_cand);
        if (eid >= 0) tk.add_eos_id(eid);
    }

    // Ensure well-known chat/reasoning delimiter tokens present in vocab are matched verbatim in encode().
    const char* known_delims[] = {
        "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
        "<think>", "</think>",
    };
    for (const char* tok_str : known_delims) {
        const auto it = tk.token_to_id_.find(tok_str);
        if (it == tk.token_to_id_.end()) continue;
        bool already = false;
        for (const auto& at : tk.added_tokens_) {
            if (at.content == tok_str) {
                already = true;
                break;
            }
        }
        if (!already) {
            const bool is_spec = tk.special_ids_.count(it->second) > 0;
            tk.added_tokens_.push_back(AddedToken{tok_str, it->second, is_spec});
        }
    }

    tk.finalize_added_tokens();
    return tk;
}

std::vector<std::string> Tokenizer::pretokenize_bytelevel(const std::string& text) const {
    std::vector<std::string> pieces;
    const size_t n = text.size();
    size_t pos = 0;

    while (pos < n) {
        uint32_t cp0 = 0;
        const size_t p1 = decode_utf8(text, pos, cp0);

        // 1. Contractions (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cp0 == '\'' && p1 < n) {
            const char c1 = static_cast<char>(std::tolower(static_cast<unsigned char>(text[p1])));
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                pieces.emplace_back(text, pos, 2);
                pos += 2;
                continue;
            }
            if (p1 + 1 < n) {
                const char c2 = static_cast<char>(std::tolower(static_cast<unsigned char>(text[p1 + 1])));
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    pieces.emplace_back(text, pos, 3);
                    pos += 3;
                    continue;
                }
            }
        }

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        if (is_unicode_letter(cp0)) {
            size_t p = p1;
            while (p < n) {
                uint32_t cp = 0;
                const size_t np = decode_utf8(text, p, cp);
                if (!is_unicode_letter(cp)) break;
                p = np;
            }
            pieces.emplace_back(text, pos, p - pos);
            pos = p;
            continue;
        }
        if (cp0 != '\r' && cp0 != '\n' && !is_unicode_number(cp0) && p1 < n) {
            uint32_t cp1 = 0;
            const size_t p2 = decode_utf8(text, p1, cp1);
            if (is_unicode_letter(cp1)) {
                size_t p = p2;
                while (p < n) {
                    uint32_t cp = 0;
                    const size_t np = decode_utf8(text, p, cp);
                    if (!is_unicode_letter(cp)) break;
                    p = np;
                }
                pieces.emplace_back(text, pos, p - pos);
                pos = p;
                continue;
            }
        }

        // 3. \p{N} (or \p{N}{1,3})
        if (is_unicode_number(cp0)) {
            size_t p = p1;
            int cnt = 1;
            while (cnt < max_digit_chunk_ && p < n) {
                uint32_t cp = 0;
                const size_t np = decode_utf8(text, p, cp);
                if (!is_unicode_number(cp)) break;
                p = np;
                ++cnt;
            }
            pieces.emplace_back(text, pos, p - pos);
            pos = p;
            continue;
        }

        // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            size_t p = pos;
            if (cp0 == ' ' && p1 < n) {
                uint32_t cp1 = 0;
                (void)decode_utf8(text, p1, cp1);
                if (!is_unicode_space(cp1) && !is_unicode_letter(cp1) && !is_unicode_number(cp1)) {
                    p = p1;
                }
            }
            uint32_t cpx = 0;
            size_t px = decode_utf8(text, p, cpx);
            if (!is_unicode_space(cpx) && !is_unicode_letter(cpx) && !is_unicode_number(cpx)) {
                p = px;
                while (p < n) {
                    uint32_t cp = 0;
                    const size_t np = decode_utf8(text, p, cp);
                    if (is_unicode_space(cp) || is_unicode_letter(cp) || is_unicode_number(cp)) break;
                    p = np;
                }
                while (p < n && (text[p] == '\r' || text[p] == '\n')) ++p;
                pieces.emplace_back(text, pos, p - pos);
                pos = p;
                continue;
            }
        }

        // 5/6/7. Whitespace runs: \s*[\r\n]+ | \s+(?!\S) | \s+
        if (is_unicode_space(cp0)) {
            size_t ws_end = pos;
            size_t last_rn_end = std::string::npos;
            size_t prev_ws_start = pos;
            int ws_count = 0;
            while (ws_end < n) {
                uint32_t cp = 0;
                const size_t np = decode_utf8(text, ws_end, cp);
                if (!is_unicode_space(cp)) break;
                prev_ws_start = ws_end;
                if (cp == '\r' || cp == '\n') last_rn_end = np;
                ws_end = np;
                ++ws_count;
            }
            if (last_rn_end != std::string::npos) {
                pieces.emplace_back(text, pos, last_rn_end - pos);
                pos = last_rn_end;
            } else if (ws_end < n && ws_count > 1) {
                pieces.emplace_back(text, pos, prev_ws_start - pos);
                pos = prev_ws_start;
            } else {
                pieces.emplace_back(text, pos, ws_end - pos);
                pos = ws_end;
            }
            continue;
        }

        pieces.emplace_back(text, pos, p1 - pos);
        pos = p1;
    }

    return pieces;
}

void Tokenizer::encode_bpe_segment(const std::string& segment, std::vector<int32_t>& out) const {
    if (segment.empty()) return;

    auto run_bpe_on_piece = [&](const std::string& piece) {
        std::string mapped;
        if (byte_level_) {
            mapped.reserve(piece.size() * 2);
            for (unsigned char b : piece) {
                mapped += byte_to_utf8_[b];
            }
        } else {
            mapped = piece;
        }

        std::vector<int32_t> ids;
        size_t p = 0;
        while (p < mapped.size()) {
            uint32_t cp = 0;
            const size_t np = decode_utf8(mapped, p, cp);
            const std::string ch = mapped.substr(p, np - p);
            const auto it = token_to_id_.find(ch);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
            } else if (byte_fallback_) {
                for (size_t k = p; k < np; ++k) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned char>(mapped[k]));
                    const int32_t bid = token_to_id(buf);
                    if (bid >= 0) ids.push_back(bid);
                    else if (unk_id_ >= 0) ids.push_back(unk_id_);
                }
            } else if (unk_id_ >= 0) {
                ids.push_back(unk_id_);
            }
            p = np;
        }

        while (ids.size() >= 2) {
            int32_t best_rank = std::numeric_limits<int32_t>::max();
            int32_t best_a = -1, best_b = -1, best_m = -1;
            for (size_t i = 0; i + 1 < ids.size(); ++i) {
                const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(ids[i])) << 32) |
                                     static_cast<uint32_t>(ids[i + 1]);
                const auto it = merge_ranks_.find(key);
                if (it != merge_ranks_.end() && it->second.rank < best_rank) {
                    best_rank = it->second.rank;
                    best_a = ids[i];
                    best_b = ids[i + 1];
                    best_m = it->second.merged_id;
                }
            }
            if (best_m < 0) break;

            std::vector<int32_t> next;
            next.reserve(ids.size());
            for (size_t i = 0; i < ids.size();) {
                if (i + 1 < ids.size() && ids[i] == best_a && ids[i + 1] == best_b) {
                    next.push_back(best_m);
                    i += 2;
                } else {
                    next.push_back(ids[i]);
                    ++i;
                }
            }
            ids.swap(next);
        }

        out.insert(out.end(), ids.begin(), ids.end());
    };

    if (byte_level_) {
        std::string buf = segment;
        if (add_prefix_space_ && !buf.empty() && buf[0] != ' ') {
            buf.insert(buf.begin(), ' ');
        }
        const auto pieces = pretokenize_bytelevel(buf);
        for (const auto& piece : pieces) {
            run_bpe_on_piece(piece);
        }
    } else if (metaspace_) {
        std::string replaced;
        if (add_prefix_space_ && !segment.empty() && segment[0] != ' ') {
            replaced += "\xE2\x96\x81";
        }
        for (char c : segment) {
            if (c == ' ') replaced += "\xE2\x96\x81";
            else replaced.push_back(c);
        }
        run_bpe_on_piece(replaced);
    } else {
        run_bpe_on_piece(segment);
    }
}

void Tokenizer::encode_sp_scores_segment(const std::string& segment, bool is_start_of_text,
                                         std::vector<int32_t>& out) const {
    if (segment.empty()) return;
    std::string text;
    if (is_start_of_text && add_prefix_space_ && segment[0] != ' ') {
        text += "\xE2\x96\x81";
    } else if (is_start_of_text && token_to_id_.count("\xE2\x96\x81") > 0 && segment[0] != ' ') {
        text += "\xE2\x96\x81";
    }
    for (char c : segment) {
        if (c == ' ') text += "\xE2\x96\x81";
        else text.push_back(c);
    }

    struct Sym {
        size_t pos;
        size_t len;
        int prev;
        int next;
    };
    std::vector<Sym> syms;
    size_t p = 0;
    while (p < text.size()) {
        uint32_t cp = 0;
        const size_t np = decode_utf8(text, p, cp);
        const int idx = static_cast<int>(syms.size());
        syms.push_back(Sym{p, np - p, idx - 1, idx + 1});
        p = np;
    }
    if (syms.empty()) return;
    syms.back().next = -1;

    while (true) {
        float best_score = -std::numeric_limits<float>::infinity();
        int best_i = -1;
        for (int i = 0; i >= 0; i = syms[i].next) {
            const int j = syms[i].next;
            if (j < 0) break;
            const std::string merged = text.substr(syms[i].pos, syms[i].len + syms[j].len);
            const auto it = token_to_id_.find(merged);
            if (it != token_to_id_.end() && static_cast<size_t>(it->second) < scores_.size()) {
                const float sc = scores_[it->second];
                if (sc > best_score) {
                    best_score = sc;
                    best_i = i;
                }
            }
        }
        if (best_i < 0) break;
        const int j = syms[best_i].next;
        syms[best_i].len += syms[j].len;
        syms[best_i].next = syms[j].next;
        if (syms[j].next >= 0) syms[syms[j].next].prev = best_i;
    }

    for (int i = 0; i >= 0; i = syms[i].next) {
        const std::string s = text.substr(syms[i].pos, syms[i].len);
        const auto it = token_to_id_.find(s);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
        } else if (byte_fallback_) {
            for (size_t k = 0; k < s.size(); ++k) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned char>(s[k]));
                const int32_t bid = token_to_id(buf);
                if (bid >= 0) out.push_back(bid);
                else if (unk_id_ >= 0) out.push_back(unk_id_);
            }
        } else if (unk_id_ >= 0) {
            out.push_back(unk_id_);
        }
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
    if (!loaded()) throw std::runtime_error("Tokenizer not loaded");
    std::vector<int32_t> out;
    if (add_bos && bos_id_ >= 0) {
        out.push_back(bos_id_);
    }

    size_t seg_start = 0;
    size_t pos = 0;
    bool first_seg = true;
    auto flush_seg = [&](size_t end) {
        if (end <= seg_start) return;
        const std::string seg = text.substr(seg_start, end - seg_start);
        if (use_scores_bpe_) encode_sp_scores_segment(seg, first_seg, out);
        else encode_bpe_segment(seg, out);
        first_seg = false;
    };

    while (pos < text.size()) {
        const uint8_t fb = static_cast<uint8_t>(text[pos]);
        int32_t matched_id = -1;
        size_t matched_len = 0;
        if (fb < added_by_first_byte_.size()) {
            for (size_t idx : added_by_first_byte_[fb]) {
                const auto& at = added_tokens_[idx];
                if (pos + at.content.size() <= text.size() &&
                    text.compare(pos, at.content.size(), at.content) == 0) {
                    matched_id = at.id;
                    matched_len = at.content.size();
                    break;
                }
            }
        }
        if (matched_id >= 0) {
            flush_seg(pos);
            out.push_back(matched_id);
            pos += matched_len;
            seg_start = pos;
            first_seg = false;
        } else {
            ++pos;
        }
    }
    flush_seg(text.size());
    return out;
}

std::string Tokenizer::decode_token(int32_t id, bool skip_special_tokens) const {
    if (id < 0 || id >= static_cast<int32_t>(id_to_token_.size())) return "";
    if (skip_special_tokens && is_special(id)) return "";

    const std::string& tok = id_to_token_[id];
    if (is_special(id)) return tok;

    if (byte_level_) {
        std::string out;
        out.reserve(tok.size());
        size_t p = 0;
        while (p < tok.size()) {
            uint32_t cp = 0;
            const size_t np = decode_utf8(tok, p, cp);
            if (cp < 512 && cp_to_byte_[cp] >= 0) {
                out.push_back(static_cast<char>(cp_to_byte_[cp]));
            } else {
                out.append(tok, p, np - p);
            }
            p = np;
        }
        return out;
    }

    if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>') {
        auto hex_n = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        const int hi = hex_n(tok[3]);
        const int lo = hex_n(tok[4]);
        if (hi >= 0 && lo >= 0) {
            return std::string(1, static_cast<char>((hi << 4) | lo));
        }
    }

    if (metaspace_ || use_scores_bpe_) {
        std::string out;
        out.reserve(tok.size());
        for (size_t i = 0; i < tok.size();) {
            if (i + 3 <= tok.size() &&
                static_cast<uint8_t>(tok[i]) == 0xE2 &&
                static_cast<uint8_t>(tok[i + 1]) == 0x96 &&
                static_cast<uint8_t>(tok[i + 2]) == 0x81) {
                out.push_back(' ');
                i += 3;
            } else {
                out.push_back(tok[i]);
                ++i;
            }
        }
        return out;
    }

    return tok;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special_tokens) const {
    std::string out;
    for (int32_t id : ids) {
        out += decode_token(id, skip_special_tokens);
    }
    return out;
}

std::string Tokenizer::format_chatml(const std::string& user_prompt,
                                     const std::string& system_prompt,
                                     bool add_generation_prompt,
                                     bool enable_thinking) const {
    std::string out;
    if (!system_prompt.empty()) {
        out += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    }
    out += "<|im_start|>user\n" + user_prompt + "<|im_end|>\n";
    if (add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        if (!enable_thinking && token_to_id("</think>") >= 0) {
            out += "<think>\n\n</think>\n\n";
        } else if (enable_thinking && token_to_id("<think>") >= 0 && token_to_id("</think>") >= 0) {
            out += "<think>\n";
        }
    }
    return out;
}

Tokenizer::ChatFormat Tokenizer::detect_chat_format() const {
    if (token_to_id("<|im_start|>") >= 0 && token_to_id("<|im_end|>") >= 0) {
        return ChatFormat::ChatML;
    }
    if (token_to_id("<|start_header_id|>") >= 0 && token_to_id("<|eot_id|>") >= 0) {
        return ChatFormat::Llama3;
    }
    return ChatFormat::ChatML;
}

const char* Tokenizer::chat_format_name(ChatFormat fmt) {
    switch (fmt) {
        case ChatFormat::ChatML: return "ChatML";
        case ChatFormat::Llama3: return "Llama3";
    }
    return "ChatML";
}

std::string Tokenizer::format_system_prefix(const std::string& system_prompt) const {
    if (system_prompt.empty()) return {};
    if (detect_chat_format() == ChatFormat::Llama3) {
        return "<|start_header_id|>system<|end_header_id|>\n\n" + system_prompt + "<|eot_id|>";
    }
    return "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
}

std::string Tokenizer::format_chat_turn(const std::string& user_prompt,
                                        const std::string& system_prompt,
                                        bool is_first_turn,
                                        bool close_previous_turn,
                                        bool enable_thinking) const {
    const ChatFormat fmt = detect_chat_format();
    std::string out;
    if (fmt == ChatFormat::Llama3) {
        if (is_first_turn) {
            if (!system_prompt.empty()) {
                out += "<|start_header_id|>system<|end_header_id|>\n\n" + system_prompt + "<|eot_id|>";
            }
        } else if (close_previous_turn) {
            out += "<|eot_id|>";
        }
        out += "<|start_header_id|>user<|end_header_id|>\n\n" + user_prompt +
               "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        return out;
    }

    if (is_first_turn) {
        if (!system_prompt.empty()) {
            out += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
        }
    } else if (close_previous_turn) {
        out += "<|im_end|>\n";
    }
    out += "<|im_start|>user\n" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n";
    if (!enable_thinking && token_to_id("</think>") >= 0) {
        out += "<think>\n\n</think>\n\n";
    } else if (enable_thinking && token_to_id("<think>") >= 0 && token_to_id("</think>") >= 0) {
        out += "<think>\n";
    }
    return out;
}

} // namespace rxinfer

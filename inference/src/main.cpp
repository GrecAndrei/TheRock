// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/common.hpp"
#include "rxinfer/config.hpp"
#include "rxinfer/engine.hpp"
#include "rxinfer/gdn.hpp"
#include "rxinfer/loader.hpp"
#include "rxinfer/markdown.hpp"
#include "rxinfer/sampler.hpp"
#include "rxinfer/tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <vector>

namespace {

volatile std::sig_atomic_t g_interrupted = 0;

void handle_sigint(int) {
    g_interrupted = 1;
}

void install_interactive_sigint() {
    struct sigaction sa{};
    sa.sa_handler = handle_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
}

class BracketedPasteGuard {
public:
    explicit BracketedPasteGuard(bool enable) : active_(enable) {
        if (active_) {
            std::fprintf(stderr, "\x1b[?2004h");
            std::fflush(stderr);
        }
    }
    ~BracketedPasteGuard() {
        if (active_) {
            std::fprintf(stderr, "\x1b[?2004l");
            std::fflush(stderr);
        }
    }
    BracketedPasteGuard(const BracketedPasteGuard&) = delete;
    BracketedPasteGuard& operator=(const BracketedPasteGuard&) = delete;

private:
    bool active_ = false;
};

bool path_is_dir(const std::string& p) {
    struct stat st{};
    return (::stat(p.c_str(), &st) == 0) && S_ISDIR(st.st_mode);
}

bool path_exists(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string parent_dir(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    return (slash == std::string::npos) ? "." : path.substr(0, slash);
}

std::string expand_user_path(const std::string& p) {
    if (!p.empty() && p[0] == '~' && (p.size() == 1 || p[1] == '/')) {
        const char* home = std::getenv("HOME");
        if (home && home[0] != '\0') {
            return std::string(home) + p.substr(1);
        }
    }
    return p;
}

bool ensure_dir_recursive(const std::string& dir) {
    if (dir.empty()) return false;
    if (path_is_dir(dir)) return true;
    std::string cur;
    size_t i = 0;
    if (dir[0] == '/') {
        cur = "/";
        i = 1;
    }
    while (i <= dir.size()) {
        const size_t next = dir.find('/', i);
        const std::string part = (next == std::string::npos) ? dir.substr(i) : dir.substr(i, next - i);
        if (!part.empty()) {
            if (!cur.empty() && cur.back() != '/') cur.push_back('/');
            cur += part;
            if (!path_exists(cur)) {
                if (::mkdir(cur.c_str(), 0755) != 0 && !path_is_dir(cur)) {
                    return false;
                }
            } else if (!path_is_dir(cur)) {
                return false;
            }
        }
        if (next == std::string::npos) break;
        i = next + 1;
    }
    return path_is_dir(dir);
}

std::string default_auto_cache_dir() {
    const char* xdg = std::getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0] != '\0') {
        return std::string(xdg) + "/rxinfer/prompts";
    }
    const char* home = std::getenv("HOME");
    if (home && home[0] != '\0') {
        return std::string(home) + "/.cache/rxinfer/prompts";
    }
    return "/tmp/rxinfer_cache/prompts";
}

std::string read_text_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open file: " + path);
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

std::string trim_ws(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

std::string format_byte_size(size_t bytes) {
    char buf[64];
    if (bytes >= (1ULL << 30)) {
        std::snprintf(buf, sizeof(buf), "%.2f GiB", bytes / double(1ULL << 30));
    } else if (bytes >= (1ULL << 20)) {
        std::snprintf(buf, sizeof(buf), "%.2f MiB", bytes / double(1ULL << 20));
    } else if (bytes >= (1ULL << 10)) {
        std::snprintf(buf, sizeof(buf), "%.1f KiB", bytes / double(1ULL << 10));
    } else {
        std::snprintf(buf, sizeof(buf), "%zu B", bytes);
    }
    return std::string(buf);
}

rxinfer::DType parse_kv_dtype(const std::string& s) {
    if (s == "f16" || s == "fp16") return rxinfer::DType::F16;
    if (s == "fp8" || s == "fp8_e4m3") return rxinfer::DType::FP8_E4M3;
    if (s == "fp4" || s == "fp4_e2m1" || s == "fp4_e2m1_g16") return rxinfer::DType::FP4_E2M1_G16;
    throw std::runtime_error("Unsupported --kv-dtype '" + s + "' (expected f16, fp8, or fp4)");
}

void query_host_mem_bytes(size_t& total_bytes, size_t& avail_bytes) {
    total_bytes = 0;
    avail_bytes = 0;
    std::ifstream ifs("/proc/meminfo");
    std::string line;
    while (std::getline(ifs, line)) {
        unsigned long long kb = 0;
        if (std::sscanf(line.c_str(), "MemTotal: %llu kB", &kb) == 1) {
            total_bytes = static_cast<size_t>(kb) * 1024ULL;
        } else if (std::sscanf(line.c_str(), "MemAvailable: %llu kB", &kb) == 1) {
            avail_bytes = static_cast<size_t>(kb) * 1024ULL;
        }
        if (total_bytes > 0 && avail_bytes > 0) break;
    }
    if (avail_bytes == 0) avail_bytes = total_bytes / 2;
}

size_t estimate_kv_bytes_per_page(const rxinfer::ModelConfig& c, rxinfer::DType kv_dt, int32_t& out_n_attn) {
    out_n_attn = 0;
    for (int32_t l = 0; l < c.num_hidden_layers; ++l) {
        const auto k = (l < static_cast<int32_t>(c.layer_types.size()))
                           ? c.layer_types[l]
                           : rxinfer::LayerKind::FullAttention;
        if (k == rxinfer::LayerKind::FullAttention) ++out_n_attn;
    }
    if (out_n_attn == 0) return 0;
    rxinfer::attn::AttentionParams ap{};
    ap.num_q_heads = c.num_attention_heads;
    ap.num_kv_heads = c.num_key_value_heads;
    ap.head_dim = c.head_dim;
    ap.kv_dtype = kv_dt;
    return static_cast<size_t>(out_n_attn) * rxinfer::attn::compute_kv_page_layout(ap).page_bytes;
}

size_t estimate_non_kv_vram_bytes(const rxinfer::ModelConfig& c, int32_t max_batch_tokens) {
    int32_t n_gdn = 0;
    for (int32_t l = 0; l < c.num_hidden_layers; ++l) {
        const auto k = (l < static_cast<int32_t>(c.layer_types.size()))
                           ? c.layer_types[l]
                           : rxinfer::LayerKind::FullAttention;
        if (k == rxinfer::LayerKind::LinearAttentionGDN) ++n_gdn;
    }
    size_t gdn_bytes = 0;
    if (n_gdn > 0) {
        rxinfer::gdn::GDNParams gp{};
        gp.num_k_heads = c.linear_num_key_heads;
        gp.head_k_dim = c.linear_key_head_dim;
        gp.num_v_heads = c.linear_num_value_heads;
        gp.head_v_dim = c.linear_value_head_dim;
        gp.conv_kernel_size = c.linear_conv_kernel_dim;
        gp.rms_norm_eps = c.rms_norm_eps;
        gp.tiled_v_heads = c.gdn_tiled_v_heads;
        const size_t per_gdn = rxinfer::align_up(rxinfer::gdn::gdn_conv_state_bytes(gp), 256) +
                               rxinfer::align_up(rxinfer::gdn::gdn_recurrent_state_bytes(gp), 256);
        gdn_bytes = static_cast<size_t>(n_gdn) * per_gdn;
    }
    const size_t approx_scratch = (160ULL << 20) + static_cast<size_t>(std::max(1, max_batch_tokens)) * 128ULL * 1024ULL;
    return gdn_bytes + approx_scratch;
}

uint64_t hash_tokens_fnv1a(const std::vector<int32_t>& tokens) {
    uint64_t h = 14695981039346656037ULL;
    for (int32_t t : tokens) {
        const uint32_t u = static_cast<uint32_t>(t);
        for (int b = 0; b < 4; ++b) {
            h ^= (u >> (b * 8)) & 0xFFu;
            h *= 1099511628211ULL;
        }
    }
    return h;
}

struct AutoCacheEntry {
    std::string path;
    time_t mtime = 0;
    size_t size_bytes = 0;
};

std::vector<AutoCacheEntry> list_auto_cache_files(const std::string& dir) {
    std::vector<AutoCacheEntry> entries;
    DIR* d = ::opendir(dir.c_str());
    if (!d) return entries;
    struct dirent* ent = nullptr;
    while ((ent = ::readdir(d)) != nullptr) {
        const std::string name = ent->d_name;
        if (!ends_with(name, ".rxprompt")) continue;
        const std::string full = dir + "/" + name;
        struct stat st{};
        if (::stat(full.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            entries.push_back({full, st.st_mtime, static_cast<size_t>(st.st_size)});
        }
    }
    ::closedir(d);
    return entries;
}

std::string find_best_auto_cache(const std::string& dir,
                                 const std::string& expected_sig,
                                 int32_t max_seq_len,
                                 bool has_gdn,
                                 const std::vector<int32_t>& target_tokens,
                                 size_t& out_lcp,
                                 int32_t& out_pos) {
    out_lcp = 0;
    out_pos = 0;
    if (dir.empty() || target_tokens.empty() || !path_is_dir(dir)) return {};

    std::string best_path;
    const auto files = list_auto_cache_files(dir);
    for (const auto& f : files) {
        try {
            rxinfer::PromptCacheInfo info = rxinfer::inspect_prompt_cache(f.path, true);
            if (info.signature != expected_sig) continue;
            if (info.position <= 0 || info.position > max_seq_len) continue;
            if (info.tokens.size() != static_cast<size_t>(info.position)) continue;

            const size_t min_len = std::min(info.tokens.size(), target_tokens.size());
            size_t lcp = 0;
            while (lcp < min_len && info.tokens[lcp] == target_tokens[lcp]) ++lcp;
            if (lcp < 64) continue;
            if (has_gdn && lcp != info.tokens.size()) continue;

            if (lcp > out_lcp || (lcp == out_lcp && info.position == static_cast<int32_t>(target_tokens.size()))) {
                out_lcp = lcp;
                out_pos = info.position;
                best_path = f.path;
            }
        } catch (...) {
            // Skip unreadable or incompatible cache entries silently.
        }
    }
    return best_path;
}

std::string save_to_auto_cache(const std::string& dir,
                               const rxinfer::InferenceEngine& engine,
                               bool quiet) {
    if (engine.position() < 1024) return {};
    if (!ensure_dir_recursive(dir)) return {};

    constexpr size_t kMaxAutoCacheBytes = 6ULL << 30;  // 6 GiB total auto-cache cap
    constexpr size_t kMinFreeDiskBytes = 8ULL << 30;   // Keep >= 8 GiB free on filesystem

    const auto& toks = engine.cached_tokens();
    const uint64_t h = hash_tokens_fnv1a(toks);
    char fname[96];
    std::snprintf(fname, sizeof(fname), "rx_%d_%016llx.rxprompt",
                  engine.position(), static_cast<unsigned long long>(h));
    const std::string out_path = dir + "/" + fname;

    // Estimate payload size
    int32_t n_attn = 0;
    const size_t page_bytes = estimate_kv_bytes_per_page(engine.config(), engine.options().kv_dtype, n_attn);
    const size_t active_pages = (engine.position() > 0)
                                    ? rxinfer::div_ceil(engine.position(), rxinfer::attn::KV_PAGE_TOKENS)
                                    : 0;
    const size_t est_bytes = (engine.state_vram_bytes() > 0 ? (192ULL << 20) : 0ULL) + active_pages * page_bytes;

    struct statvfs vfs{};
    if (::statvfs(dir.c_str(), &vfs) == 0) {
        const size_t avail_disk = static_cast<size_t>(vfs.f_bavail) * static_cast<size_t>(vfs.f_frsize);
        if (avail_disk < kMinFreeDiskBytes + est_bytes) {
            if (!quiet) {
                std::fprintf(stderr, "[Auto-cache skip: low disk headroom (%s free)]\n",
                             format_byte_size(avail_disk).c_str());
            }
            return {};
        }
    }

    if (est_bytes > kMaxAutoCacheBytes) {
        return {};
    }

    auto entries = list_auto_cache_files(dir);
    size_t total_bytes = 0;
    for (const auto& e : entries) {
        if (e.path != out_path) total_bytes += e.size_bytes;
    }
    std::sort(entries.begin(), entries.end(),
              [](const AutoCacheEntry& a, const AutoCacheEntry& b) { return a.mtime < b.mtime; });
    for (const auto& e : entries) {
        if (total_bytes + est_bytes <= kMaxAutoCacheBytes) break;
        if (e.path == out_path) continue;
        if (std::remove(e.path.c_str()) == 0) {
            total_bytes -= std::min(total_bytes, e.size_bytes);
        }
    }

    engine.save_prompt_cache(out_path);
    return out_path;
}

void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s --model <hf_dir|model.gguf|model.rxr4|synthetic> [options]\n"
                 "Options:\n"
                 "  --tokenizer <path>        Path to tokenizer.json (auto-detected if omitted)\n"
                 "  --synthetic-config <json> HF config.json used when --model synthetic\n"
                 "  -i, --interactive         Interactive multi-turn chat (auto large context up to 200k)\n"
                 "  --prompt <text>           Prompt text to generate from (or Turn 1 when paired with -i)\n"
                 "  --prompt-file <path>      Read prompt/context text from file\n"
                 "  --system <text>           Optional system prompt (cached across /reset)\n"
                 "  --chat                    Wrap prompt in chat template (default ON in -i)\n"
                 "  --no-chat                 Disable chat template (raw continuation mode)\n"
                 "  --think                   Enable <think> mode in chat template\n"
                 "  --format                  Force ANSI Markdown terminal formatting (default: auto on TTY)\n"
                 "  --no-format, --raw        Disable ANSI Markdown formatting (plain UTF-8 output)\n"
                 "  --max-tokens <N>          Max new tokens per turn (default: 256, or 1024 in -i)\n"
                 "  --max-seq-len <N>         Max context length (default: auto up to 200k in -i, or prompt-fit)\n"
                 "  --max-batch-tokens <N>    Prefill chunk size (default: 512)\n"
                 "  --kv-dtype <f16|fp8|fp4>  KV cache dtype (default: auto based on VRAM/context)\n"
                 "  --kv-vram-gb <F>          KV VRAM cap in GiB (-1=auto GTT spillover, 0=all GTT, default: -1)\n"
                 "  --decode-splits <N>       Split-K partitions for decode attention (0=auto, default: 0)\n"
                 "  --prompt-cache <path|auto> Persistent .rxprompt cache file (or 'auto' for ~/.cache/rxinfer LRU)\n"
                 "  --auto-cache              Shorthand for --prompt-cache auto\n"
                 "  --save-prompt-cache <p>   Always save prompt cache to <p> (after prefill or on -i exit)\n"
                 "  --temp <F>                Sampling temperature (0 = greedy GPU argmax, default: 0.0)\n"
                 "  --top-k <N>               Top-K cutoff when --temp > 0 (0 = off, default: 40)\n"
                 "  --top-p <F>               Nucleus Top-P when --temp > 0 (default: 0.95)\n"
                 "  --min-p <F>               Min-P relative cutoff when --temp > 0 (default: 0.0)\n"
                 "  --repeat-penalty <F>      Repetition penalty over recent tokens (default: 1.0)\n"
                 "  --presence-penalty <F>    Presence penalty over recent tokens (default: 0.0)\n"
                 "  --repeat-last-n <N>       History window for penalties (default: 64)\n"
                 "  --seed <N>                RNG seed when sampling (0 = random, default: 0)\n"
                 "  --threads <N>             CPU threads for R4 packing (default: auto)\n"
                 "  --save-rxr4 <path>        Export packed R4 weights to .rxr4 checkpoint\n"
                 "  --bench                   Run prefill/decode/verify throughput benchmark\n"
                 "  --no-graph                Disable hipGraph capture\n"
                 "  --ignore-eos              Ignore EOS stop tokens\n"
                 "  --quiet                   Suppress diagnostics on stderr\n"
                 "  -h, --help                Show this help\n",
                 argv0);
}

void print_interactive_help() {
    std::fprintf(stderr,
                 "\nInteractive commands:\n"
                 "  /help, /?               Show this help\n"
                 "  /file <path> [prompt]   Attach a text/code file into context (send immediately if prompt given)\n"
                 "  /paste                  Start multi-line block input (end with '/end' or '\"\"\"')\n"
                 "  /undo [N]               Rewind conversation by N turns (default 1, uses cached snapshots)\n"
                 "  /retry                  Rewind last turn and regenerate response\n"
                 "  /reset, /clear          Clear active state and start a new conversation (keeps LCP snapshots)\n"
                 "  /save-cache [path]      Save live KV + GDN prompt state to .rxprompt (or auto-cache)\n"
                 "  /load-cache <path>      Restore KV + GDN prompt state from .rxprompt file\n"
                 "  /cache                  Show prompt cache, LCP snapshots, and VRAM/GTT KV tier status\n"
                 "  /clear-cache            Free in-memory GDN prefix checkpoints\n"
                 "  /stats                  Show context usage, VRAM/GTT, and true TPS stats\n"
                 "  /format [on|off]        Toggle or set terminal Markdown formatting (*bold*, headings, code)\n"
                 "  /think [on|off]         Toggle or set <think> mode\n"
                 "  /system <text>          Set system prompt (resets conversation; auto-cached across /reset)\n"
                 "  /temp <F>               Set sampling temperature (0 = greedy GPU argmax)\n"
                 "  /top-k <N>              Set top-k (0 = off)\n"
                 "  /top-p <F>              Set top-p (1.0 = off)\n"
                 "  /min-p <F>              Set min-p (0.0 = off)\n"
                 "  /repeat-penalty <F>     Set repetition penalty (1.0 = off)\n"
                 "  /max-tokens <N>         Set max tokens generated per turn\n"
                 "  /seed <N>               Set sampling seed (0 = random)\n"
                 "  /bench                  Run prefill/decode/verify benchmark (resets state)\n"
                 "  /quit, /exit, /q        Exit (or Ctrl-D; Ctrl-C interrupts generation)\n"
                 "Tips: Paste multi-line blocks directly, wrap blocks in '\"\"\"', or end lines with '\\'.\n\n");
}

rxinfer::ModelConfig default_qwen38_27b_config() {
    rxinfer::ModelConfig c;
    c.model_type = "qwen3_5";
    c.arch_name = "Qwen3_5ForCausalLM";
    c.vocab_size = 248320;
    c.hidden_size = 5120;
    c.intermediate_size = 17408;
    c.num_hidden_layers = 64;
    c.max_position_embeddings = 262144;
    c.tie_word_embeddings = false;
    c.norm_kind = rxinfer::NormKind::RMSNormPlusOne;
    c.rms_norm_eps = 1e-6f;
    c.num_attention_heads = 24;
    c.num_key_value_heads = 4;
    c.head_dim = 256;
    c.qk_norm = true;
    c.attn_output_gate = true;
    c.rope_kind = rxinfer::RopeKind::NeoX;
    c.rope_theta = 10000000.0f;
    c.partial_rotary_factor = 0.25f;
    c.mrope_interleaved = true;
    c.mrope_section = {11, 11, 10};
    c.linear_num_key_heads = 16;
    c.linear_key_head_dim = 128;
    c.linear_num_value_heads = 48;
    c.linear_value_head_dim = 128;
    c.linear_conv_kernel_dim = 4;
    for (int l = 0; l < 64; ++l) {
        c.layer_types.push_back(((l + 1) % 4 == 0) ? rxinfer::LayerKind::FullAttention
                                                   : rxinfer::LayerKind::LinearAttentionGDN);
    }
    return c;
}

bool is_stop_token(int32_t tok, const rxinfer::Tokenizer& tokenizer, const rxinfer::ModelConfig& cfg) {
    if (tokenizer.is_eos(tok)) return true;
    return std::find(cfg.eos_token_ids.begin(), cfg.eos_token_ids.end(), tok) != cfg.eos_token_ids.end();
}

struct TurnRecord {
    std::vector<int32_t> tokens_before;
    std::vector<int32_t> pending_before;
    bool was_first_turn = true;
    bool was_close_previous = false;
    std::string user_input;
};

struct PendingAttachment {
    std::string label;
    std::string content;
};

struct ChatSessionState {
    bool is_first_turn = true;
    bool close_previous_turn = false;
    std::vector<int32_t> pending_tokens;
    std::vector<PendingAttachment> pending_attachments;
    std::vector<TurnRecord> history;

    int32_t turns = 0;
    int64_t total_prompt_tokens = 0;
    int64_t total_generated_tokens = 0;
    int64_t total_decode_calls = 0;
    double total_prefill_ms = 0.0;
    double total_decode_ms = 0.0;

    int32_t last_prompt_tokens = 0;
    int32_t last_generated_tokens = 0;
    int32_t last_decode_calls = 0;
    double last_prefill_ms = 0.0;
    double last_decode_ms = 0.0;

    void reset_conversation(rxinfer::InferenceEngine& engine, rxinfer::Sampler& sampler) {
        engine.reset_state();
        sampler.clear_history();
        is_first_turn = true;
        close_previous_turn = false;
        pending_tokens.clear();
        pending_attachments.clear();
        history.clear();
    }
};

struct TurnOptions {
    bool chat_mode = false;
    bool think_mode = false;
    bool format_markdown = true;
    bool ignore_eos = false;
    bool quiet = false;
    bool interactive = false;
    bool auto_prompt_cache = false;
    int32_t max_tokens = 256;
    int32_t max_seq_len = 4096;
    std::string system_prompt;
    std::string save_prompt_cache_after_prefill;
    std::string auto_cache_dir;
    std::string shared_doc_prefix; // Optional non-interactive --prompt-file content when paired with --prompt
};

void print_session_stats(const ChatSessionState& st,
                           const rxinfer::InferenceEngine& engine,
                           const rxinfer::Sampler& sampler) {
    const auto& sc = sampler.config();
    const double sess_prefill_tps =
        (st.total_prefill_ms > 0.0) ? (st.total_prompt_tokens * 1000.0 / st.total_prefill_ms) : 0.0;
    const double sess_decode_tps =
        (st.total_decode_ms > 0.0) ? (st.total_decode_calls * 1000.0 / st.total_decode_ms) : 0.0;
    const double sess_ms_tok =
        (st.total_decode_calls > 0) ? (st.total_decode_ms / st.total_decode_calls) : 0.0;

    std::fprintf(stderr,
                 "\n[Stats] Context: %d / %d tokens (%.1f%%) | VRAM: %.2f GiB (W=%.2f S=%.2f X=%.2f)\n"
                 "        KV Pool: VRAM=%.2f GiB (%d tok) + GTT=%.2f GiB (%d tok) | Prefix snaps: %d\n"
                 "        Sampler: temp=%.2f top_k=%d top_p=%.2f min_p=%.2f rep=%.2f path=%s\n"
                 "        Session: %d turns | prompt=%lld tok (%.1f tok/s) | gen=%lld tok | decode=%lld steps (%.2f tok/s, %.2f ms/tok)\n",
                 engine.position(), engine.options().max_seq_len,
                 100.0 * engine.position() / std::max(1, engine.options().max_seq_len),
                 engine.total_vram_used_bytes() / double(1ULL << 30),
                 engine.weights().weight_arena.used() / double(1ULL << 30),
                 engine.state_vram_bytes() / double(1ULL << 30),
                 engine.scratch_vram_bytes() / double(1ULL << 30),
                 engine.kv_vram_bytes() / double(1ULL << 30), engine.vram_kv_tokens(),
                 engine.kv_gtt_bytes() / double(1ULL << 30), engine.gtt_kv_tokens(),
                 engine.num_prefix_checkpoints(),
                 sc.temperature, sc.top_k, sc.top_p, sc.min_p, sc.repeat_penalty,
                 sc.uses_host_logits() ? "host-sample" : "gpu-argmax",
                 st.turns,
                 static_cast<long long>(st.total_prompt_tokens), sess_prefill_tps,
                 static_cast<long long>(st.total_generated_tokens),
                 static_cast<long long>(st.total_decode_calls), sess_decode_tps, sess_ms_tok);
    if (st.last_prompt_tokens > 0 || st.last_generated_tokens > 0) {
        const double l_p_tps = (st.last_prefill_ms > 0.0) ? (st.last_prompt_tokens * 1000.0 / st.last_prefill_ms) : 0.0;
        const double l_d_tps = (st.last_decode_ms > 0.0) ? (st.last_decode_calls * 1000.0 / st.last_decode_ms) : 0.0;
        std::fprintf(stderr,
                     "        Last turn: prompt=%d tok (%.1f ms, %.1f tok/s) | gen=%d tok | decode=%d steps (%.1f ms, %.2f tok/s)\n\n",
                     st.last_prompt_tokens, st.last_prefill_ms, l_p_tps,
                     st.last_generated_tokens, st.last_decode_calls, st.last_decode_ms, l_d_tps);
    } else {
        std::fprintf(stderr, "\n");
    }
}

std::vector<int32_t> encode_with_optional_bos(const rxinfer::Tokenizer& tokenizer,
                                              const std::string& text,
                                              bool add_bos) {
    std::vector<int32_t> ids = tokenizer.encode(text, false);
    if (add_bos && tokenizer.should_add_bos() && tokenizer.bos_id() >= 0 &&
        (ids.empty() || ids[0] != tokenizer.bos_id())) {
        ids.insert(ids.begin(), tokenizer.bos_id());
    }
    return ids;
}

bool run_turn(rxinfer::InferenceEngine& engine,
              const rxinfer::Tokenizer& tokenizer,
              rxinfer::Sampler& sampler,
              ChatSessionState& state,
              const std::string& raw_input_text,
              const TurnOptions& opts,
              std::vector<uint16_t>& logits_f16_scratch) {
    std::string input_text;
    if (!state.pending_attachments.empty()) {
        for (const auto& att : state.pending_attachments) {
            input_text += "[File: " + att.label + "]\n";
            input_text += att.content;
            if (!att.content.empty() && att.content.back() != '\n') input_text.push_back('\n');
            input_text.push_back('\n');
        }
        input_text += raw_input_text;
    } else {
        input_text = raw_input_text;
    }

    std::string delta_text;
    if (opts.chat_mode) {
        delta_text = tokenizer.format_chat_turn(
            input_text, opts.system_prompt, state.is_first_turn, state.close_previous_turn, opts.think_mode);
    } else {
        delta_text = (state.is_first_turn || input_text.empty()) ? input_text : ("\n" + input_text);
    }

    const bool add_bos = state.is_first_turn && state.pending_tokens.empty() && tokenizer.should_add_bos();
    std::vector<int32_t> encoded = encode_with_optional_bos(tokenizer, delta_text, add_bos);

    std::vector<int32_t> turn_ids = state.pending_tokens;
    turn_ids.insert(turn_ids.end(), encoded.begin(), encoded.end());

    if (state.is_first_turn) {
        if (turn_ids.empty()) {
            if (opts.interactive) {
                std::fprintf(stderr, "[Empty turn; nothing to prefill]\n");
                return false;
            }
            throw std::runtime_error("Prompt tokenized to 0 tokens");
        }
        if (static_cast<int32_t>(turn_ids.size()) >= opts.max_seq_len) {
            if (opts.interactive) {
                std::fprintf(stderr,
                             "[Context limit reached (%zu >= %d). Use /reset to start a new conversation.]\n",
                             turn_ids.size(), opts.max_seq_len);
                return false;
            }
            throw std::runtime_error("Prompt length exceeds --max-seq-len");
        }
    } else {
        if (turn_ids.empty()) {
            if (opts.interactive) {
                std::fprintf(stderr, "[Empty turn; nothing to prefill]\n");
                return false;
            }
            throw std::runtime_error("Prompt tokenized to 0 tokens");
        }
        if (engine.position() + static_cast<int32_t>(turn_ids.size()) >= opts.max_seq_len) {
            if (opts.interactive) {
                std::fprintf(stderr,
                             "[Context limit reached (%d + %zu >= %d). Use /reset to start a new conversation.]\n",
                             engine.position(), turn_ids.size(), opts.max_seq_len);
                return false;
            }
            throw std::runtime_error("Prompt length exceeds --max-seq-len");
        }
    }

    TurnRecord turn_rec;
    turn_rec.tokens_before = engine.cached_tokens();
    turn_rec.pending_before = state.pending_tokens;
    turn_rec.was_first_turn = state.is_first_turn;
    turn_rec.was_close_previous = state.close_previous_turn;
    turn_rec.user_input = input_text;

    std::vector<int32_t> full_prompt_ids;
    if (state.is_first_turn) {
        full_prompt_ids = turn_ids;
    } else {
        full_prompt_ids = engine.cached_tokens();
        full_prompt_ids.insert(full_prompt_ids.end(), turn_ids.begin(), turn_ids.end());
    }

    // If auto-cache is enabled and engine is empty on Turn 1, look for longest matching prefix on disk.
    bool loaded_disk_auto_cache = false;
    if (opts.auto_prompt_cache && state.is_first_turn && engine.position() == 0 && !opts.auto_cache_dir.empty()) {
        size_t auto_lcp = 0;
        int32_t auto_pos = 0;
        const bool has_gdn = (engine.state_vram_bytes() > 0 && engine.config().num_hidden_layers > 0 &&
                              std::any_of(engine.config().layer_types.begin(), engine.config().layer_types.end(),
                                          [](rxinfer::LayerKind k) { return k == rxinfer::LayerKind::LinearAttentionGDN; }));
        const std::string best_file = find_best_auto_cache(
            opts.auto_cache_dir, engine.cache_signature(), opts.max_seq_len, has_gdn,
            full_prompt_ids, auto_lcp, auto_pos);
        if (!best_file.empty()) {
            const auto tlc0 = std::chrono::steady_clock::now();
            const int32_t loaded_pos = engine.load_prompt_cache(best_file);
            const auto tlc1 = std::chrono::steady_clock::now();
            loaded_disk_auto_cache = (loaded_pos > 0);
            if (!opts.quiet) {
                std::fprintf(stderr, "[Auto-cache hit: loaded %d tok from %s in %.1f ms]\n",
                             loaded_pos, best_file.c_str(),
                             std::chrono::duration<double, std::milli>(tlc1 - tlc0).count());
            }
        }
    }

    sampler.record_tokens(encoded);
    const int32_t vocab_size = engine.config().vocab_size;

    const auto tp0 = std::chrono::steady_clock::now();
    bool evaluated_cold_prefix = false;
    int32_t prior_cached_tokens = loaded_disk_auto_cache ? engine.position() : 0;

    // Automatically cache and pin system prompt prefix on Turn 1 so /reset keeps it hot.
    if (opts.chat_mode && !opts.system_prompt.empty() && state.is_first_turn && !loaded_disk_auto_cache) {
        const std::string sys_text = tokenizer.format_system_prefix(opts.system_prompt);
        const std::vector<int32_t> sys_ids = encode_with_optional_bos(tokenizer, sys_text, tokenizer.should_add_bos());
        if (!sys_ids.empty() && sys_ids.size() < full_prompt_ids.size() &&
            std::equal(sys_ids.begin(), sys_ids.end(), full_prompt_ids.begin())) {
            const bool restored_from_cache = engine.restore_prefix(sys_ids);
            engine.save_prefix_checkpoint(true);
            if (restored_from_cache) {
                prior_cached_tokens = static_cast<int32_t>(sys_ids.size());
            } else {
                evaluated_cold_prefix = true;
            }
        }
    }

    // If --prompt-file + --prompt with --auto-cache, pre-cache the shared document prefix too.
    if (!opts.interactive && opts.auto_prompt_cache && !opts.shared_doc_prefix.empty() &&
        !opts.auto_cache_dir.empty() && !loaded_disk_auto_cache) {
        std::string doc_prefix_text;
        if (opts.chat_mode) {
            doc_prefix_text = tokenizer.format_system_prefix(opts.system_prompt) + "<|im_start|>user\n";
            if (tokenizer.detect_chat_format() == rxinfer::Tokenizer::ChatFormat::Llama3) {
                doc_prefix_text = tokenizer.format_system_prefix(opts.system_prompt) +
                                  "<|start_header_id|>user<|end_header_id|>\n\n";
            }
        }
        doc_prefix_text += opts.shared_doc_prefix;
        const std::vector<int32_t> doc_ids =
            encode_with_optional_bos(tokenizer, doc_prefix_text, tokenizer.should_add_bos());
        size_t shared = 0;
        const size_t min_len = std::min(doc_ids.size(), full_prompt_ids.size());
        while (shared < min_len && doc_ids[shared] == full_prompt_ids[shared]) ++shared;
        if (shared >= 1024 && shared + 16 < full_prompt_ids.size()) {
            std::vector<int32_t> shared_ids(full_prompt_ids.begin(), full_prompt_ids.begin() + shared);
            const auto shared_res = engine.prefill_cached(shared_ids, true);
            if (shared_res.evaluated_tokens > 0) {
                evaluated_cold_prefix = true;
            } else if (!evaluated_cold_prefix) {
                prior_cached_tokens = static_cast<int32_t>(shared);
            }
            const auto tsc0 = std::chrono::steady_clock::now();
            const std::string saved = save_to_auto_cache(opts.auto_cache_dir, engine, opts.quiet);
            const auto tsc1 = std::chrono::steady_clock::now();
            if (!saved.empty() && !opts.quiet) {
                std::fprintf(stderr, "[Auto-cached shared prefix (%zu tok) to %s in %.1f ms]\n",
                             shared, saved.c_str(),
                             std::chrono::duration<double, std::milli>(tsc1 - tsc0).count());
            }
        }
    }

    rxinfer::PrefillCacheResult cache_res = engine.prefill_cached(full_prompt_ids);
    if (evaluated_cold_prefix) {
        cache_res.matched_prefix_tokens = prior_cached_tokens;
        cache_res.evaluated_tokens = static_cast<int32_t>(full_prompt_ids.size()) - prior_cached_tokens;
    }

    int32_t next_tok = cache_res.next_token;
    if (sampler.config().uses_host_logits() && (cache_res.evaluated_tokens > 0 || loaded_disk_auto_cache)) {
        engine.copy_last_logits_f16(logits_f16_scratch);
        next_tok = sampler.sample_from_f16_bits(logits_f16_scratch.data(), vocab_size);
    }
    const auto tp1 = std::chrono::steady_clock::now();
    const double prefill_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    if (!opts.save_prompt_cache_after_prefill.empty()) {
        const auto tsc0 = std::chrono::steady_clock::now();
        engine.save_prompt_cache(opts.save_prompt_cache_after_prefill);
        const auto tsc1 = std::chrono::steady_clock::now();
        if (!opts.quiet) {
            std::fprintf(stderr, "[Saved prompt cache (%d tok) to %s in %.1f ms]\n",
                         engine.position(), opts.save_prompt_cache_after_prefill.c_str(),
                         std::chrono::duration<double, std::milli>(tsc1 - tsc0).count());
        }
    } else if (!opts.interactive && opts.auto_prompt_cache && !opts.auto_cache_dir.empty() &&
               opts.shared_doc_prefix.empty() && cache_res.evaluated_tokens >= 1024) {
        const auto tsc0 = std::chrono::steady_clock::now();
        const std::string saved = save_to_auto_cache(opts.auto_cache_dir, engine, opts.quiet);
        const auto tsc1 = std::chrono::steady_clock::now();
        if (!saved.empty() && !opts.quiet) {
            std::fprintf(stderr, "[Auto-cached prompt (%d tok) to %s in %.1f ms]\n",
                         engine.position(), saved.c_str(),
                         std::chrono::duration<double, std::milli>(tsc1 - tsc0).count());
        }
    }

    state.pending_attachments.clear();
    state.pending_tokens.clear();
    state.is_first_turn = false;
    state.close_previous_turn = opts.chat_mode;

    rxinfer::MarkdownStreamFormatter printer(opts.format_markdown);
    if (opts.chat_mode && opts.think_mode &&
        tokenizer.token_to_id("<think>") >= 0 && tokenizer.token_to_id("</think>") >= 0) {
        printer.push("<think>\n");
    }
    int32_t generated = 0;
    int32_t decode_calls = 0;
    double decode_ms = 0.0;
    bool hit_ctx_limit = false;
    bool interrupted = false;

    for (int32_t step = 0; step < opts.max_tokens; ++step) {
        if (g_interrupted) {
            interrupted = true;
            break;
        }
        if (!opts.ignore_eos && is_stop_token(next_tok, tokenizer, engine.config())) {
            break;
        }

        const std::string& raw_tok = tokenizer.id_to_token(next_tok);
        if (opts.think_mode && (raw_tok == "<think>" || raw_tok == "</think>")) {
            printer.push(raw_tok);
        } else {
            printer.push(tokenizer.decode_token(next_tok, true));
        }
        sampler.record_token(next_tok);
        ++generated;

        if (engine.position() >= opts.max_seq_len) {
            hit_ctx_limit = true;
            state.pending_tokens.push_back(next_tok);
            break;
        }
        if (step + 1 >= opts.max_tokens) {
            state.pending_tokens.push_back(next_tok);
            break;
        }
        if (g_interrupted) {
            interrupted = true;
            state.pending_tokens.push_back(next_tok);
            break;
        }

        const auto td0 = std::chrono::steady_clock::now();
        int32_t cand_tok = engine.decode_step(next_tok);
        if (sampler.config().uses_host_logits()) {
            engine.copy_last_logits_f16(logits_f16_scratch);
            cand_tok = sampler.sample_from_f16_bits(logits_f16_scratch.data(), vocab_size);
        }
        const auto td1 = std::chrono::steady_clock::now();
        decode_ms += std::chrono::duration<double, std::milli>(td1 - td0).count();
        ++decode_calls;
        next_tok = cand_tok;
    }

    printer.finish();
    std::printf("\n");
    std::fflush(stdout);

    // Save post-turn prefix checkpoint so /undo, /retry, and branching stay instant.
    if (opts.interactive && engine.position() >= 8) {
        engine.save_prefix_checkpoint(false);
    }
    state.history.push_back(std::move(turn_rec));

    if (interrupted) {
        g_interrupted = 0;
        if (!opts.quiet) std::fprintf(stderr, "[interrupted]\n");
    } else if (hit_ctx_limit && !opts.quiet) {
        std::fprintf(stderr, "[context limit %d reached; use /reset for a new conversation]\n", opts.max_seq_len);
    }

    state.turns++;
    state.last_prompt_tokens = cache_res.evaluated_tokens;
    state.last_generated_tokens = generated;
    state.last_decode_calls = decode_calls;
    state.last_prefill_ms = prefill_ms;
    state.last_decode_ms = decode_ms;

    state.total_prompt_tokens += cache_res.evaluated_tokens;
    state.total_generated_tokens += generated;
    state.total_decode_calls += decode_calls;
    state.total_prefill_ms += prefill_ms;
    state.total_decode_ms += decode_ms;

    if (!opts.quiet) {
        const double p_tps = (cache_res.evaluated_tokens * 1000.0) / std::max(0.001, prefill_ms);
        const double total_ms = prefill_ms + decode_ms;
        const double e2e_tps = (generated > 0) ? (generated * 1000.0 / std::max(0.001, total_ms)) : 0.0;
        std::string prompt_desc;
        if (cache_res.matched_prefix_tokens > 0) {
            char pbuf[160];
            std::snprintf(pbuf, sizeof(pbuf), "%d new + %d cached%s",
                          cache_res.evaluated_tokens, cache_res.matched_prefix_tokens,
                          cache_res.restored_gdn_snapshot ? " (GDN restored)" : "");
            prompt_desc = pbuf;
        } else {
            prompt_desc = std::to_string(cache_res.evaluated_tokens) + " tok";
        }
        if (decode_calls > 0) {
            const double d_tps = (decode_calls * 1000.0) / std::max(0.001, decode_ms);
            const double ms_per_tok = decode_ms / decode_calls;
            std::fprintf(stderr,
                         "[Prompt: %s in %.1f ms (%.1f tok/s, TTFT %.1f ms) | Decode: %d steps in %.1f ms (%.2f tok/s, %.2f ms/tok) | Out: %d tok (E2E %.2f tok/s) | Ctx: %d/%d]\n",
                         prompt_desc.c_str(), prefill_ms, p_tps, prefill_ms,
                         decode_calls, decode_ms, d_tps, ms_per_tok,
                         generated, e2e_tps, engine.position(), opts.max_seq_len);
        } else {
            std::fprintf(stderr,
                         "[Prompt: %s in %.1f ms (%.1f tok/s, TTFT %.1f ms) | Out: %d tok (0 decode steps) | Ctx: %d/%d]\n",
                         prompt_desc.c_str(), prefill_ms, p_tps, prefill_ms,
                         generated, engine.position(), opts.max_seq_len);
        }
    }

    return true;
}

bool parse_file_command_args(const std::string& arg, std::string& out_path, std::string& out_question) {
    const std::string trimmed = trim_ws(arg);
    if (trimmed.empty()) return false;
    if (trimmed[0] == '"' || trimmed[0] == '\'') {
        const char q = trimmed[0];
        const size_t close = trimmed.find(q, 1);
        if (close == std::string::npos) {
            out_path = expand_user_path(trimmed.substr(1));
            out_question.clear();
            return !out_path.empty();
        }
        out_path = expand_user_path(trimmed.substr(1, close - 1));
        out_question = trim_ws(trimmed.substr(close + 1));
        return !out_path.empty();
    }
    const size_t sp = trimmed.find_first_of(" \t");
    if (sp == std::string::npos) {
        out_path = expand_user_path(trimmed);
        out_question.clear();
    } else {
        out_path = expand_user_path(trimmed.substr(0, sp));
        out_question = trim_ws(trimmed.substr(sp + 1));
    }
    return !out_path.empty();
}

void strip_all_substr(std::string& s, const std::string& pat) {
    size_t pos = 0;
    while ((pos = s.find(pat, pos)) != std::string::npos) {
        s.erase(pos, pat.size());
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string model_path;
        std::string tokenizer_path;
        std::string synthetic_cfg_path;
        std::string cli_prompt;
        std::string prompt_file;
        std::string system_prompt;
        std::string save_rxr4_path;
        std::string prompt_cache_path;
        std::string save_prompt_cache_path;
        bool has_cli_prompt = false;
        bool has_prompt_file = false;
        bool interactive = false;
        bool chat_flag_set = false;
        bool chat_flag_val = false;
        bool format_flag_set = false;
        bool format_flag_val = true;
        bool think_mode = false;
        bool run_bench = false;
        bool no_graph = false;
        bool ignore_eos = false;
        bool quiet = false;
        bool has_max_tokens = false;
        bool has_max_seq_len = false;
        bool has_kv_dtype = false;
        int32_t max_tokens = 256;
        int32_t max_seq_len = 4096;
        int32_t max_batch_tokens = 512;
        int32_t decode_attn_splits = 0;
        double kv_vram_gb = -1.0;
        int threads = 0;
        rxinfer::DType kv_dtype = rxinfer::DType::F16;
        rxinfer::SamplerConfig sampler_cfg{};

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto need_val = [&](const char* flag) -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + flag);
                return argv[++i];
            };
            if (arg == "-h" || arg == "--help") {
                print_usage(argv[0]);
                return 0;
            } else if (arg == "--model") {
                model_path = need_val("--model");
            } else if (arg == "--tokenizer") {
                tokenizer_path = need_val("--tokenizer");
            } else if (arg == "--synthetic-config") {
                synthetic_cfg_path = need_val("--synthetic-config");
            } else if (arg == "-i" || arg == "--interactive") {
                interactive = true;
            } else if (arg == "--prompt") {
                cli_prompt = need_val("--prompt");
                has_cli_prompt = true;
            } else if (arg == "--prompt-file") {
                prompt_file = need_val("--prompt-file");
                has_prompt_file = true;
            } else if (arg == "--system") {
                system_prompt = need_val("--system");
            } else if (arg == "--chat") {
                chat_flag_set = true;
                chat_flag_val = true;
            } else if (arg == "--no-chat") {
                chat_flag_set = true;
                chat_flag_val = false;
            } else if (arg == "--format") {
                format_flag_set = true;
                format_flag_val = true;
            } else if (arg == "--no-format" || arg == "--raw") {
                format_flag_set = true;
                format_flag_val = false;
            } else if (arg == "--think") {
                think_mode = true;
            } else if (arg == "--max-tokens") {
                max_tokens = std::stoi(need_val("--max-tokens"));
                has_max_tokens = true;
            } else if (arg == "--max-seq-len") {
                max_seq_len = std::stoi(need_val("--max-seq-len"));
                has_max_seq_len = true;
            } else if (arg == "--max-batch-tokens") {
                max_batch_tokens = std::stoi(need_val("--max-batch-tokens"));
            } else if (arg == "--kv-dtype") {
                kv_dtype = parse_kv_dtype(need_val("--kv-dtype"));
                has_kv_dtype = true;
            } else if (arg == "--kv-vram-gb") {
                kv_vram_gb = std::stod(need_val("--kv-vram-gb"));
            } else if (arg == "--decode-splits") {
                decode_attn_splits = std::stoi(need_val("--decode-splits"));
            } else if (arg == "--prompt-cache") {
                prompt_cache_path = need_val("--prompt-cache");
            } else if (arg == "--auto-cache") {
                prompt_cache_path = "auto";
            } else if (arg == "--save-prompt-cache") {
                save_prompt_cache_path = need_val("--save-prompt-cache");
            } else if (arg == "--temp") {
                sampler_cfg.temperature = std::stof(need_val("--temp"));
            } else if (arg == "--top-k") {
                sampler_cfg.top_k = std::stoi(need_val("--top-k"));
            } else if (arg == "--top-p") {
                sampler_cfg.top_p = std::stof(need_val("--top-p"));
            } else if (arg == "--min-p") {
                sampler_cfg.min_p = std::stof(need_val("--min-p"));
            } else if (arg == "--repeat-penalty") {
                sampler_cfg.repeat_penalty = std::stof(need_val("--repeat-penalty"));
            } else if (arg == "--presence-penalty") {
                sampler_cfg.presence_penalty = std::stof(need_val("--presence-penalty"));
            } else if (arg == "--repeat-last-n") {
                sampler_cfg.repeat_last_n = std::stoi(need_val("--repeat-last-n"));
            } else if (arg == "--seed") {
                sampler_cfg.seed = std::stoull(need_val("--seed"));
            } else if (arg == "--threads") {
                threads = std::stoi(need_val("--threads"));
            } else if (arg == "--save-rxr4") {
                save_rxr4_path = need_val("--save-rxr4");
            } else if (arg == "--bench") {
                run_bench = true;
            } else if (arg == "--no-graph") {
                no_graph = true;
            } else if (arg == "--ignore-eos") {
                ignore_eos = true;
            } else if (arg == "--quiet") {
                quiet = true;
            } else {
                throw std::runtime_error("Unknown option: " + arg);
            }
        }

        if (model_path.empty()) {
            print_usage(argv[0]);
            return 1;
        }

        std::string prompt_file_text;
        if (has_prompt_file) {
            prompt_file_text = read_text_file(prompt_file);
        }
        const bool has_prompt = has_cli_prompt || has_prompt_file;
        std::string prompt;
        if (has_prompt_file && has_cli_prompt) {
            prompt = prompt_file_text;
            if (!prompt.empty() && prompt.back() != '\n') prompt.push_back('\n');
            prompt += "\n" + cli_prompt;
        } else if (has_prompt_file) {
            prompt = prompt_file_text;
        } else if (has_cli_prompt) {
            prompt = cli_prompt;
        }

        if (interactive && !has_max_tokens) {
            max_tokens = 1024;
        }
        if (max_tokens < 1) {
            throw std::runtime_error("--max-tokens must be >= 1");
        }

        const bool chat_mode = chat_flag_set ? chat_flag_val : interactive;
        const bool stdout_is_tty = (::isatty(STDOUT_FILENO) != 0);
        const bool no_color_env = (std::getenv("NO_COLOR") != nullptr && std::getenv("NO_COLOR")[0] != '\0');
        const bool format_markdown = format_flag_set ? format_flag_val : (stdout_is_tty && !no_color_env);

        const bool auto_prompt_cache = (prompt_cache_path == "auto");
        const std::string auto_cache_dir = auto_prompt_cache ? default_auto_cache_dir() : "";
        if (!auto_prompt_cache && !prompt_cache_path.empty()) {
            prompt_cache_path = expand_user_path(prompt_cache_path);
        }

        const auto dev = rxinfer::query_device_info(0);
        if (!quiet) {
            std::fprintf(stderr, "rxinfer | GPU: %s (%s) | VRAM: %.2f GiB (%.2f GiB free) | WGPs: %d\n",
                         dev.name.c_str(), dev.gcn_arch_name.c_str(),
                         dev.total_vram_bytes / double(1ULL << 30),
                         dev.free_vram_bytes / double(1ULL << 30),
                         dev.multiprocessor_count);
        }

        const auto t_load0 = std::chrono::steady_clock::now();
        rxinfer::LoaderOptions lopts{};
        lopts.num_threads = threads;

        rxinfer::ModelWeights weights;
        rxinfer::Tokenizer tokenizer;

        if (model_path == "synthetic") {
            rxinfer::ModelConfig cfg = synthetic_cfg_path.empty()
                                           ? default_qwen38_27b_config()
                                           : rxinfer::ModelConfig::from_hf_json_file(synthetic_cfg_path);
            lopts.fast_synthetic = true;
            weights = rxinfer::load_synthetic_model(cfg, 42, lopts);
        } else if (path_is_dir(model_path)) {
            weights = rxinfer::load_from_safetensors_dir(model_path, nullptr, lopts);
            if (tokenizer_path.empty() && path_exists(model_path + "/tokenizer.json")) {
                tokenizer_path = model_path + "/tokenizer.json";
            }
        } else if (ends_with(model_path, ".rxr4")) {
            weights = rxinfer::load_from_rxr4(model_path);
            if (tokenizer_path.empty()) {
                const std::string stem = model_path.substr(0, model_path.size() - 5);
                const std::string stem_tok = stem + ".tokenizer.json";
                const std::string stem_gguf = stem + ".gguf";
                const std::string dir_tok = parent_dir(model_path) + "/tokenizer.json";
                if (path_exists(stem_tok)) tokenizer_path = stem_tok;
                else if (path_exists(stem_gguf)) tokenizer_path = stem_gguf;
                else if (path_exists(dir_tok)) tokenizer_path = dir_tok;
            }
        } else if (ends_with(model_path, ".gguf")) {
            weights = rxinfer::load_from_gguf(model_path, nullptr, lopts);
            if (tokenizer_path.empty()) {
                tokenizer = rxinfer::Tokenizer::from_gguf(model_path);
            }
        } else {
            throw std::runtime_error("Unrecognized model path (expected directory, .rxr4, .gguf, or synthetic): " +
                                     model_path);
        }

        if (!tokenizer_path.empty()) {
            if (ends_with(tokenizer_path, ".gguf")) {
                tokenizer = rxinfer::Tokenizer::from_gguf(tokenizer_path);
            } else {
                tokenizer = rxinfer::Tokenizer::from_hf_json_file(tokenizer_path);
            }
        }
        if (tokenizer.loaded()) {
            if (tokenizer.bos_id() < 0 && weights.config.bos_token_id >= 0) {
                tokenizer.set_bos_id(weights.config.bos_token_id);
            }
            for (int32_t eid : weights.config.eos_token_ids) {
                tokenizer.add_eos_id(eid);
            }
        }

        const auto t_load1 = std::chrono::steady_clock::now();
        const double load_ms = std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();

        if (!quiet) {
            std::fprintf(stderr,
                         "Loaded %s (%d layers, H=%d, V=%d, weights=%.2f GiB) in %.1f ms\n",
                         weights.config.model_type.empty() ? "model" : weights.config.model_type.c_str(),
                         weights.config.num_hidden_layers, weights.config.hidden_size,
                         weights.config.vocab_size,
                         weights.weight_arena.used() / double(1ULL << 30), load_ms);
        }

        if (!save_rxr4_path.empty()) {
            const auto ts0 = std::chrono::steady_clock::now();
            rxinfer::save_rxr4_checkpoint(weights, save_rxr4_path);
            const auto ts1 = std::chrono::steady_clock::now();
            if (!quiet) {
                std::fprintf(stderr, "Saved RXR4 checkpoint to %s in %.1f ms\n",
                             save_rxr4_path.c_str(),
                             std::chrono::duration<double, std::milli>(ts1 - ts0).count());
            }
            if (!has_prompt && !run_bench && !interactive) return 0;
        }

        // Auto-configure context length & KV dtype when omitted, matching explicit prompt caches too.
        rxinfer::PromptCacheInfo explicit_cache_info{};
        bool has_explicit_cache_file = false;
        if (!auto_prompt_cache && !prompt_cache_path.empty() && path_exists(prompt_cache_path)) {
            explicit_cache_info = rxinfer::inspect_prompt_cache(prompt_cache_path, false);
            has_explicit_cache_file = true;
            if (!has_kv_dtype) {
                kv_dtype = explicit_cache_info.kv_dtype;
            }
        }

        const int32_t model_max_pos =
            (weights.config.max_position_embeddings > 0) ? weights.config.max_position_embeddings : 200000;
        const size_t free_vram_now = rxinfer::query_device_info(0).free_vram_bytes;
        const size_t non_kv_vram = estimate_non_kv_vram_bytes(weights.config, max_batch_tokens);
        const size_t safe_kv_vram =
            (free_vram_now > non_kv_vram + (256ULL << 20)) ? (free_vram_now - non_kv_vram - (256ULL << 20)) : 0;

        size_t host_total_b = 0, host_avail_b = 0;
        query_host_mem_bytes(host_total_b, host_avail_b);
        const size_t safe_host_gtt =
            (host_total_b > 0) ? std::min(host_total_b * 45 / 100, host_avail_b * 60 / 100) : (8ULL << 30);

        if (!has_max_seq_len) {
            if (interactive) {
                max_seq_len = std::min<int32_t>(200000, model_max_pos);
                if (has_explicit_cache_file && explicit_cache_info.position + max_tokens + 256 > max_seq_len) {
                    max_seq_len = std::min<int32_t>(model_max_pos, explicit_cache_info.position + max_tokens + 256);
                }
                // Guard against tiny host RAM when auto-sizing interactive large context
                const rxinfer::DType probe_dt =
                    has_kv_dtype ? kv_dtype
                                 : ((weights.config.head_dim % 16 == 0) ? rxinfer::DType::FP4_E2M1_G16
                                                                        : rxinfer::DType::FP8_E4M3);
                int32_t n_attn = 0;
                const size_t pg_b = estimate_kv_bytes_per_page(weights.config, probe_dt, n_attn);
                if (pg_b > 0 && kv_vram_gb != 0.0) {
                    const size_t vram_budget =
                        (kv_vram_gb > 0.0) ? static_cast<size_t>(kv_vram_gb * double(1ULL << 30)) : safe_kv_vram;
                    const size_t max_bytes = vram_budget + safe_host_gtt;
                    const int64_t max_pages = static_cast<int64_t>(max_bytes / pg_b);
                    const int32_t safe_tok =
                        static_cast<int32_t>(std::max<int64_t>(8192, max_pages * rxinfer::attn::KV_PAGE_TOKENS));
                    max_seq_len = std::min(max_seq_len, safe_tok);
                }
            } else if (has_prompt && tokenizer.loaded()) {
                const std::string first_txt =
                    chat_mode ? tokenizer.format_chat_turn(prompt, system_prompt, true, false, think_mode) : prompt;
                const size_t p_toks =
                    encode_with_optional_bos(tokenizer, first_txt, tokenizer.should_add_bos()).size();
                int64_t needed = static_cast<int64_t>(p_toks) + max_tokens + 512;
                if (has_explicit_cache_file) {
                    needed = std::max<int64_t>(needed, static_cast<int64_t>(explicit_cache_info.position) + max_tokens + 256);
                }
                const int64_t aligned = static_cast<int64_t>(rxinfer::align_up(std::max<int64_t>(8192, needed), 4096));
                max_seq_len = static_cast<int32_t>(std::min<int64_t>(model_max_pos, aligned));
            } else {
                max_seq_len = std::min<int32_t>(8192, model_max_pos);
            }
        }

        if (!has_kv_dtype && !has_explicit_cache_file) {
            int32_t n_attn = 0;
            const size_t pages = rxinfer::div_ceil(max_seq_len, rxinfer::attn::KV_PAGE_TOKENS);
            const size_t f16_bytes = pages * estimate_kv_bytes_per_page(weights.config, rxinfer::DType::F16, n_attn);
            const size_t fp8_bytes = pages * estimate_kv_bytes_per_page(weights.config, rxinfer::DType::FP8_E4M3, n_attn);
            if (n_attn == 0 || f16_bytes <= safe_kv_vram ||
                (max_seq_len <= 8192 && f16_bytes <= safe_kv_vram + (512ULL << 20))) {
                kv_dtype = rxinfer::DType::F16;
            } else if (fp8_bytes <= safe_kv_vram) {
                kv_dtype = rxinfer::DType::FP8_E4M3;
            } else if (weights.config.head_dim % 16 == 0) {
                kv_dtype = rxinfer::DType::FP4_E2M1_G16;
            } else {
                kv_dtype = rxinfer::DType::FP8_E4M3;
            }
        }

        rxinfer::EngineOptions eopts{};
        eopts.max_seq_len = max_seq_len;
        eopts.max_batch_tokens = max_batch_tokens;
        eopts.kv_dtype = kv_dtype;
        eopts.enable_hip_graph = !no_graph;
        eopts.kv_vram_gb = kv_vram_gb;
        eopts.decode_attn_splits = decode_attn_splits;

        rxinfer::InferenceEngine engine(std::move(weights), eopts);
        max_seq_len = engine.options().max_seq_len;

        if (!quiet) {
            std::fprintf(stderr,
                         "Total allocated VRAM: %.2f GiB | KV Pool: VRAM=%.2f GiB (%d tok) + GTT=%.2f GiB (%d tok) | kv=%s%s | hipGraph=%s | max_seq=%d%s | batch=%d\n",
                         engine.total_vram_used_bytes() / double(1ULL << 30),
                         engine.kv_vram_bytes() / double(1ULL << 30), engine.vram_kv_tokens(),
                         engine.kv_gtt_bytes() / double(1ULL << 30), engine.gtt_kv_tokens(),
                         std::string(rxinfer::dtype_name(kv_dtype)).c_str(),
                         (has_kv_dtype || has_explicit_cache_file) ? "" : " (auto)",
                         eopts.enable_hip_graph ? "on" : "off",
                         max_seq_len,
                         has_max_seq_len ? "" : " (auto)",
                         engine.options().max_batch_tokens);
        }

        if (run_bench) {
            const int pre_len = std::min(512, std::max(32, max_seq_len / 2));
            auto pre = engine.benchmark_prefill(pre_len, 1);
            auto dec = engine.benchmark_decode(60, 8);
            auto ver4 = engine.benchmark_verify(4, 30, 5);
            std::fprintf(stderr,
                         "Benchmark | prefill(%d)=%.1f tok/s (%.1f ms) | decode(M=1)=%.2f tok/s (%.2f ms/tok) | verify(M=4)=%.2f tok/s\n",
                         pre_len, pre.tok_per_sec, pre.elapsed_ms,
                         dec.tok_per_sec, dec.ms_per_tok, ver4.tok_per_sec);
            if (!has_prompt && !interactive) return 0;
            engine.reset_state();
        }

        if (!has_prompt && !interactive) {
            std::fprintf(stderr, "Nothing to run: pass -i/--interactive, --prompt, --bench, or --save-rxr4.\n");
            return 1;
        }
        if (!tokenizer.loaded()) {
            throw std::runtime_error("Tokenizer not loaded: pass --tokenizer <path/to/tokenizer.json>");
        }

        rxinfer::Sampler sampler(sampler_cfg);
        ChatSessionState state;
        TurnOptions topts;
        topts.chat_mode = chat_mode;
        topts.think_mode = think_mode;
        topts.format_markdown = format_markdown;
        topts.ignore_eos = ignore_eos;
        topts.quiet = quiet;
        topts.interactive = interactive;
        topts.auto_prompt_cache = auto_prompt_cache;
        topts.max_tokens = max_tokens;
        topts.max_seq_len = max_seq_len;
        topts.system_prompt = system_prompt;
        topts.auto_cache_dir = auto_cache_dir;
        if (!interactive && has_prompt_file && has_cli_prompt) {
            topts.shared_doc_prefix = prompt_file_text;
        }
        if (!interactive) {
            if (!save_prompt_cache_path.empty()) {
                topts.save_prompt_cache_after_prefill = expand_user_path(save_prompt_cache_path);
            } else if (!auto_prompt_cache && !prompt_cache_path.empty() && !path_exists(prompt_cache_path)) {
                topts.save_prompt_cache_after_prefill = prompt_cache_path;
            }
        }
        std::vector<uint16_t> logits_f16_scratch;

        if (!auto_prompt_cache && !prompt_cache_path.empty() && path_exists(prompt_cache_path)) {
            const auto tlc0 = std::chrono::steady_clock::now();
            const int32_t loaded_pos = engine.load_prompt_cache(prompt_cache_path);
            const auto tlc1 = std::chrono::steady_clock::now();
            if (!quiet) {
                std::fprintf(stderr, "Loaded prompt cache from %s (%d tok) in %.1f ms\n",
                             prompt_cache_path.c_str(), loaded_pos,
                             std::chrono::duration<double, std::milli>(tlc1 - tlc0).count());
            }
            if (loaded_pos > 0) {
                bool continue_loaded_turn = interactive;
                if (!interactive && has_prompt) {
                    const std::string first_text =
                        chat_mode ? tokenizer.format_chat_turn(prompt, system_prompt, true, false, think_mode) : prompt;
                    const std::vector<int32_t> first_ids =
                        encode_with_optional_bos(tokenizer, first_text, tokenizer.should_add_bos());
                    const auto& cached = engine.cached_tokens();
                    const size_t min_len = std::min(first_ids.size(), cached.size());
                    size_t lcp = 0;
                    while (lcp < min_len && first_ids[lcp] == cached[lcp]) ++lcp;
                    if (!(lcp == first_ids.size() || lcp == cached.size() || lcp > 16)) {
                        continue_loaded_turn = true;
                    }
                }
                if (continue_loaded_turn) {
                    state.is_first_turn = false;
                    if (chat_mode && !engine.cached_tokens().empty()) {
                        const int32_t last_id = engine.cached_tokens().back();
                        const int32_t im_end = tokenizer.token_to_id("<|im_end|>");
                        const int32_t eot = tokenizer.token_to_id("<|eot_id|>");
                        state.close_previous_turn = (last_id != im_end && last_id != eot);
                    }
                    sampler.record_tokens(engine.cached_tokens());
                }
            }
        }

        const bool stdin_is_tty = (::isatty(STDIN_FILENO) != 0);
        const bool stderr_is_tty = (::isatty(STDERR_FILENO) != 0);

        if (interactive) {
            install_interactive_sigint();
            if (!quiet) {
                const char* tpl_name =
                    chat_mode ? rxinfer::Tokenizer::chat_format_name(tokenizer.detect_chat_format()) : "raw";
                std::fprintf(stderr,
                             "Interactive chat ready | template=%s | format=%s | think=%s | temp=%.2f top_k=%d top_p=%.2f rep=%.2f | max_tokens=%d | cache=%s\n"
                             "Type your message, /file <path> to attach large files, /help for commands, /quit or Ctrl-D to exit.\n",
                             tpl_name,
                             topts.format_markdown ? "markdown" : "raw",
                             topts.think_mode ? "on" : "off",
                             sampler.config().temperature, sampler.config().top_k,
                             sampler.config().top_p, sampler.config().repeat_penalty,
                             topts.max_tokens,
                             auto_prompt_cache ? "auto" : (prompt_cache_path.empty() ? "in-mem LCP" : prompt_cache_path.c_str()));
            }
        }

        if (has_prompt) {
            run_turn(engine, tokenizer, sampler, state, prompt, topts, logits_f16_scratch);
            if (!interactive) return 0;
        }

        BracketedPasteGuard paste_guard(interactive && stdin_is_tty && stderr_is_tty);

        std::string pending_multiline;
        std::string line;
        bool in_bracketed_paste = false;
        bool in_block_paste = false;
        std::string block_end_marker;

        while (true) {
            if (!quiet && !in_bracketed_paste) {
                if (pending_multiline.empty() && !in_block_paste) {
                    if (!state.pending_attachments.empty()) {
                        std::fprintf(stderr, "\n[%zu attachment(s) ready]> ", state.pending_attachments.size());
                    } else {
                        std::fprintf(stderr, "\n> ");
                    }
                } else {
                    std::fprintf(stderr, ".. ");
                }
                std::fflush(stderr);
            }

            g_interrupted = 0;
            if (!std::getline(std::cin, line)) {
                if (g_interrupted) {
                    g_interrupted = 0;
                    pending_multiline.clear();
                    in_bracketed_paste = false;
                    in_block_paste = false;
                    std::cin.clear();
                    std::fprintf(stderr, "\n");
                    continue;
                }
                if (!pending_multiline.empty()) {
                    const std::string final_msg = pending_multiline;
                    pending_multiline.clear();
                    run_turn(engine, tokenizer, sampler, state, final_msg, topts, logits_f16_scratch);
                }
                break;
            }

            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }

            // Detect bracketed paste escape sequences from terminal emulators (\x1b[200~ ... \x1b[201~)
            if (stdin_is_tty) {
                const bool saw_start = (line.find("\x1b[200~") != std::string::npos);
                const bool saw_end = (line.find("\x1b[201~") != std::string::npos);
                if (saw_start || in_bracketed_paste) {
                    strip_all_substr(line, "\x1b[200~");
                    strip_all_substr(line, "\x1b[201~");
                    if (saw_start && !saw_end) {
                        in_bracketed_paste = true;
                        pending_multiline += line;
                        pending_multiline.push_back('\n');
                        continue;
                    }
                    if (in_bracketed_paste && !saw_end) {
                        pending_multiline += line;
                        pending_multiline.push_back('\n');
                        continue;
                    }
                    in_bracketed_paste = false;
                    if (!pending_multiline.empty()) {
                        pending_multiline += line;
                        std::string pasted = std::move(pending_multiline);
                        pending_multiline.clear();
                        if (!pasted.empty()) {
                            run_turn(engine, tokenizer, sampler, state, pasted, topts, logits_f16_scratch);
                        }
                        continue;
                    }
                    // Single-line paste falls through cleanly
                }
            }

            if (in_block_paste) {
                const std::string trimmed_line = trim_ws(line);
                if (trimmed_line == block_end_marker || trimmed_line == "/end") {
                    in_block_paste = false;
                    std::string full_block = std::move(pending_multiline);
                    pending_multiline.clear();
                    if (!full_block.empty() && full_block.back() == '\n') full_block.pop_back();
                    if (!full_block.empty()) {
                        run_turn(engine, tokenizer, sampler, state, full_block, topts, logits_f16_scratch);
                    }
                } else {
                    pending_multiline += line;
                    pending_multiline.push_back('\n');
                }
                continue;
            }

            if (pending_multiline.empty()) {
                const std::string trimmed_line = trim_ws(line);
                if (trimmed_line == "\"\"\"" || trimmed_line == "/paste") {
                    in_block_paste = true;
                    block_end_marker = (trimmed_line == "\"\"\"") ? "\"\"\"" : "/end";
                    pending_multiline.clear();
                    if (!quiet) {
                        std::fprintf(stderr, "[Multi-line mode: paste or type lines, end with '%s']\n",
                                     block_end_marker.c_str());
                    }
                    continue;
                }
                if (trimmed_line.rfind("\"\"\"", 0) == 0 && trimmed_line.size() > 3) {
                    const size_t close_pos = trimmed_line.rfind("\"\"\"");
                    if (close_pos > 3 && close_pos + 3 == trimmed_line.size()) {
                        line = trimmed_line.substr(3, close_pos - 3);
                    } else {
                        in_block_paste = true;
                        block_end_marker = "\"\"\"";
                        pending_multiline = line.substr(line.find("\"\"\"") + 3) + "\n";
                        continue;
                    }
                }
            }

            // Support trailing '\' line continuation for multi-line messages.
            if (!line.empty() && line.back() == '\\') {
                line.pop_back();
                pending_multiline += line;
                pending_multiline.push_back('\n');
                continue;
            }

            std::string full_input = pending_multiline + line;
            pending_multiline.clear();

            if (full_input.empty() && state.pending_attachments.empty()) continue;

            // Escape leading slash with "\/..." to send literal slash text.
            if (full_input.size() >= 2 && full_input[0] == '\\' && full_input[1] == '/') {
                full_input.erase(0, 1);
            } else if (!full_input.empty() && full_input[0] == '/') {
                const std::string cmd_trimmed = trim_ws(full_input);
                const size_t sp = cmd_trimmed.find_first_of(" \t");
                const std::string cmd = (sp == std::string::npos) ? cmd_trimmed : cmd_trimmed.substr(0, sp);
                const std::string arg = (sp == std::string::npos) ? "" : trim_ws(cmd_trimmed.substr(sp + 1));

                try {
                    if (cmd == "/quit" || cmd == "/exit" || cmd == "/q") {
                        break;
                    } else if (cmd == "/help" || cmd == "/?") {
                        print_interactive_help();
                    } else if (cmd == "/file" || cmd == "/attach") {
                        std::string fpath, fquestion;
                        if (!parse_file_command_args(arg, fpath, fquestion)) {
                            std::fprintf(stderr, "Usage: /file <path> [optional prompt]\n");
                        } else {
                            std::string content = read_text_file(fpath);
                            const size_t nlines = std::count(content.begin(), content.end(), '\n') + (content.empty() ? 0 : 1);
                            state.pending_attachments.push_back({fpath, std::move(content)});
                            std::fprintf(stderr, "[Attached %s (%zu lines) from %s]\n",
                                         format_byte_size(state.pending_attachments.back().content.size()).c_str(),
                                         nlines, fpath.c_str());
                            if (!fquestion.empty()) {
                                run_turn(engine, tokenizer, sampler, state, fquestion, topts, logits_f16_scratch);
                            }
                        }
                    } else if (cmd == "/undo") {
                        const int32_t steps = arg.empty() ? 1 : std::max(1, std::stoi(arg));
                        if (state.history.empty()) {
                            std::fprintf(stderr, "[Nothing to undo]\n");
                        } else {
                            const int32_t actual = std::min<int32_t>(steps, static_cast<int32_t>(state.history.size()));
                            TurnRecord target = state.history[state.history.size() - actual];
                            state.history.resize(state.history.size() - actual);
                            const auto t0 = std::chrono::steady_clock::now();
                            const bool fast = engine.restore_prefix(target.tokens_before);
                            const auto t1 = std::chrono::steady_clock::now();
                            state.pending_tokens = std::move(target.pending_before);
                            state.is_first_turn = target.was_first_turn;
                            state.close_previous_turn = target.was_close_previous;
                            state.pending_attachments.clear();
                            sampler.clear_history();
                            sampler.record_tokens(engine.cached_tokens());
                            std::fprintf(stderr, "[Rewound %d turn(s) to ctx=%d/%d in %.1f ms%s]\n",
                                         actual, engine.position(), topts.max_seq_len,
                                         std::chrono::duration<double, std::milli>(t1 - t0).count(),
                                         fast ? " (cached snapshot)" : " (replayed prefix)");
                        }
                    } else if (cmd == "/retry") {
                        if (state.history.empty()) {
                            std::fprintf(stderr, "[Nothing to retry]\n");
                        } else {
                            TurnRecord last = std::move(state.history.back());
                            state.history.pop_back();
                            engine.restore_prefix(last.tokens_before);
                            state.pending_tokens = std::move(last.pending_before);
                            state.is_first_turn = last.was_first_turn;
                            state.close_previous_turn = last.was_close_previous;
                            state.pending_attachments.clear();
                            sampler.clear_history();
                            sampler.record_tokens(engine.cached_tokens());
                            run_turn(engine, tokenizer, sampler, state, last.user_input, topts, logits_f16_scratch);
                        }
                    } else if (cmd == "/reset" || cmd == "/clear") {
                        state.reset_conversation(engine, sampler);
                        std::fprintf(stderr, "[Conversation reset. Context 0 / %d (LCP snapshots=%d)]\n",
                                     topts.max_seq_len, engine.num_prefix_checkpoints());
                    } else if (cmd == "/save-cache") {
                        const std::string target = arg.empty() ? (auto_prompt_cache ? "" : prompt_cache_path) : expand_user_path(arg);
                        if (target.empty()) {
                            if (engine.position() <= 0) {
                                std::fprintf(stderr, "[Nothing to save; context is empty]\n");
                            } else {
                                const std::string dir = auto_cache_dir.empty() ? default_auto_cache_dir() : auto_cache_dir;
                                const auto t0 = std::chrono::steady_clock::now();
                                const std::string saved = save_to_auto_cache(dir, engine, false);
                                const auto t1 = std::chrono::steady_clock::now();
                                if (!saved.empty()) {
                                    std::fprintf(stderr, "[Saved prompt cache (%d tok) to %s in %.1f ms]\n",
                                                 engine.position(), saved.c_str(),
                                                 std::chrono::duration<double, std::milli>(t1 - t0).count());
                                }
                            }
                        } else {
                            const auto t0 = std::chrono::steady_clock::now();
                            engine.save_prompt_cache(target);
                            const auto t1 = std::chrono::steady_clock::now();
                            std::fprintf(stderr, "[Saved prompt cache (%d tok) to %s in %.1f ms]\n",
                                         engine.position(), target.c_str(),
                                         std::chrono::duration<double, std::milli>(t1 - t0).count());
                        }
                    } else if (cmd == "/load-cache") {
                        const std::string target = arg.empty() ? (auto_prompt_cache ? "" : prompt_cache_path) : expand_user_path(arg);
                        if (target.empty()) {
                            std::fprintf(stderr, "Usage: /load-cache <path>\n");
                        } else {
                            const auto t0 = std::chrono::steady_clock::now();
                            const int32_t loaded_pos = engine.load_prompt_cache(target);
                            const auto t1 = std::chrono::steady_clock::now();
                            state.pending_tokens.clear();
                            state.pending_attachments.clear();
                            state.history.clear();
                            state.is_first_turn = (loaded_pos == 0);
                            state.close_previous_turn = false;
                            if (topts.chat_mode && loaded_pos > 0 && !engine.cached_tokens().empty()) {
                                const int32_t last_id = engine.cached_tokens().back();
                                const int32_t im_end = tokenizer.token_to_id("<|im_end|>");
                                const int32_t eot = tokenizer.token_to_id("<|eot_id|>");
                                state.close_previous_turn = (last_id != im_end && last_id != eot);
                            }
                            sampler.clear_history();
                            sampler.record_tokens(engine.cached_tokens());
                            std::fprintf(stderr, "[Loaded prompt cache (%d tok) from %s in %.1f ms]\n",
                                         loaded_pos, target.c_str(),
                                         std::chrono::duration<double, std::milli>(t1 - t0).count());
                        }
                    } else if (cmd == "/cache") {
                        std::fprintf(stderr,
                                     "[Cache] Live ctx: %d/%d tok | Snapshots: %d | KV Pool: VRAM=%.2f GiB (%d tok) + GTT=%.2f GiB (%d tok) | Mode: %s\n",
                                     engine.position(), topts.max_seq_len,
                                     engine.num_prefix_checkpoints(),
                                     engine.kv_vram_bytes() / double(1ULL << 30), engine.vram_kv_tokens(),
                                     engine.kv_gtt_bytes() / double(1ULL << 30), engine.gtt_kv_tokens(),
                                     auto_prompt_cache
                                         ? ("auto (" + auto_cache_dir + ")").c_str()
                                         : (prompt_cache_path.empty() ? "in-mem LCP" : prompt_cache_path.c_str()));
                    } else if (cmd == "/clear-cache") {
                        engine.clear_prefix_cache();
                        std::fprintf(stderr, "[Cleared in-memory GDN prefix checkpoints]\n");
                    } else if (cmd == "/stats") {
                        print_session_stats(state, engine, sampler);
                    } else if (cmd == "/format") {
                        if (arg == "on" || arg == "1" || arg == "true" || arg == "markdown") topts.format_markdown = true;
                        else if (arg == "off" || arg == "0" || arg == "false" || arg == "raw") topts.format_markdown = false;
                        else if (arg.empty()) topts.format_markdown = !topts.format_markdown;
                        else {
                            std::fprintf(stderr, "Usage: /format [on|off]\n");
                            continue;
                        }
                        std::fprintf(stderr, "[Output formatting: %s]\n",
                                     topts.format_markdown ? "markdown" : "raw");
                    } else if (cmd == "/think") {
                        if (arg == "on" || arg == "1" || arg == "true") topts.think_mode = true;
                        else if (arg == "off" || arg == "0" || arg == "false") topts.think_mode = false;
                        else if (arg.empty()) topts.think_mode = !topts.think_mode;
                        else {
                            std::fprintf(stderr, "Usage: /think [on|off]\n");
                            continue;
                        }
                        std::fprintf(stderr, "[Thinking mode: %s]\n", topts.think_mode ? "on" : "off");
                    } else if (cmd == "/system") {
                        topts.system_prompt = arg;
                        engine.unpin_prefix_checkpoints();
                        state.reset_conversation(engine, sampler);
                        std::fprintf(stderr, "[System prompt updated (%zu chars); conversation reset]\n",
                                     topts.system_prompt.size());
                    } else if (cmd == "/temp") {
                        if (arg.empty()) {
                            std::fprintf(stderr, "[temp=%.3f]\n", sampler.config().temperature);
                        } else {
                            auto sc = sampler.config();
                            sc.temperature = std::stof(arg);
                            sampler.set_config(sc);
                            std::fprintf(stderr, "[temp=%.3f (%s)]\n", sc.temperature,
                                         sc.uses_host_logits() ? "host-sample" : "gpu-argmax");
                        }
                    } else if (cmd == "/top-k") {
                        if (arg.empty()) {
                            std::fprintf(stderr, "[top_k=%d]\n", sampler.config().top_k);
                        } else {
                            auto sc = sampler.config();
                            sc.top_k = std::stoi(arg);
                            sampler.set_config(sc);
                            std::fprintf(stderr, "[top_k=%d]\n", sc.top_k);
                        }
                    } else if (cmd == "/top-p") {
                        if (arg.empty()) {
                            std::fprintf(stderr, "[top_p=%.3f]\n", sampler.config().top_p);
                        } else {
                            auto sc = sampler.config();
                            sc.top_p = std::stof(arg);
                            sampler.set_config(sc);
                            std::fprintf(stderr, "[top_p=%.3f]\n", sc.top_p);
                        }
                    } else if (cmd == "/min-p") {
                        if (arg.empty()) {
                            std::fprintf(stderr, "[min_p=%.3f]\n", sampler.config().min_p);
                        } else {
                            auto sc = sampler.config();
                            sc.min_p = std::stof(arg);
                            sampler.set_config(sc);
                            std::fprintf(stderr, "[min_p=%.3f]\n", sc.min_p);
                        }
                    } else if (cmd == "/repeat-penalty") {
                        if (arg.empty()) {
                            std::fprintf(stderr, "[repeat_penalty=%.3f]\n", sampler.config().repeat_penalty);
                        } else {
                            auto sc = sampler.config();
                            sc.repeat_penalty = std::stof(arg);
                            sampler.set_config(sc);
                            std::fprintf(stderr, "[repeat_penalty=%.3f]\n", sc.repeat_penalty);
                        }
                    } else if (cmd == "/max-tokens") {
                        if (arg.empty()) {
                            std::fprintf(stderr, "[max_tokens=%d]\n", topts.max_tokens);
                        } else {
                            const int32_t v = std::stoi(arg);
                            if (v < 1) {
                                std::fprintf(stderr, "[max_tokens must be >= 1]\n");
                            } else {
                                topts.max_tokens = v;
                                std::fprintf(stderr, "[max_tokens=%d]\n", topts.max_tokens);
                            }
                        }
                    } else if (cmd == "/seed") {
                        if (arg.empty()) {
                            std::fprintf(stderr, "[seed=%llu]\n",
                                         static_cast<unsigned long long>(sampler.config().seed));
                        } else {
                            sampler.reseed(std::stoull(arg));
                            std::fprintf(stderr, "[seed=%llu]\n",
                                         static_cast<unsigned long long>(sampler.config().seed));
                        }
                    } else if (cmd == "/bench") {
                        const int pre_len = std::min(512, std::max(32, max_seq_len / 2));
                        auto pre = engine.benchmark_prefill(pre_len, 1);
                        auto dec = engine.benchmark_decode(60, 8);
                        auto ver4 = engine.benchmark_verify(4, 30, 5);
                        std::fprintf(stderr,
                                     "[Benchmark | prefill(%d)=%.1f tok/s (%.1f ms) | decode(M=1)=%.2f tok/s (%.2f ms/tok) | verify(M=4)=%.2f tok/s]\n",
                                     pre_len, pre.tok_per_sec, pre.elapsed_ms,
                                     dec.tok_per_sec, dec.ms_per_tok, ver4.tok_per_sec);
                        state.reset_conversation(engine, sampler);
                        std::fprintf(stderr, "[Benchmark complete; conversation reset]\n");
                    } else {
                        std::fprintf(stderr, "[Unknown command '%s'. Type /help for commands.]\n", cmd.c_str());
                    }
                } catch (const std::exception& ex) {
                    std::fprintf(stderr, "[Command error: %s]\n", ex.what());
                }
                continue;
            }

            run_turn(engine, tokenizer, sampler, state, full_input, topts, logits_f16_scratch);
        }

        if (interactive && engine.position() > 0) {
            if (!save_prompt_cache_path.empty()) {
                const std::string target = expand_user_path(save_prompt_cache_path);
                const auto tsc0 = std::chrono::steady_clock::now();
                engine.save_prompt_cache(target);
                const auto tsc1 = std::chrono::steady_clock::now();
                if (!quiet) {
                    std::fprintf(stderr, "[Saved prompt cache (%d tok) to %s in %.1f ms]\n",
                                 engine.position(), target.c_str(),
                                 std::chrono::duration<double, std::milli>(tsc1 - tsc0).count());
                }
            } else if (auto_prompt_cache && !auto_cache_dir.empty()) {
                const auto tsc0 = std::chrono::steady_clock::now();
                const std::string saved = save_to_auto_cache(auto_cache_dir, engine, quiet);
                const auto tsc1 = std::chrono::steady_clock::now();
                if (!saved.empty() && !quiet) {
                    std::fprintf(stderr, "[Auto-cached session (%d tok) to %s in %.1f ms]\n",
                                 engine.position(), saved.c_str(),
                                 std::chrono::duration<double, std::milli>(tsc1 - tsc0).count());
                }
            } else if (!prompt_cache_path.empty()) {
                const auto tsc0 = std::chrono::steady_clock::now();
                engine.save_prompt_cache(prompt_cache_path);
                const auto tsc1 = std::chrono::steady_clock::now();
                if (!quiet) {
                    std::fprintf(stderr, "[Saved prompt cache (%d tok) to %s in %.1f ms]\n",
                                 engine.position(), prompt_cache_path.c_str(),
                                 std::chrono::duration<double, std::milli>(tsc1 - tsc0).count());
                }
            }
        }

        if (!quiet && interactive && state.turns > 0) {
            const double sess_decode_tps =
                (state.total_decode_ms > 0.0) ? (state.total_decode_calls * 1000.0 / state.total_decode_ms) : 0.0;
            std::fprintf(stderr,
                         "[Session: %d turns | prompt=%lld tok | gen=%lld tok | avg decode=%.2f tok/s]\n",
                         state.turns,
                         static_cast<long long>(state.total_prompt_tokens),
                         static_cast<long long>(state.total_generated_tokens),
                         sess_decode_tps);
        }

        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "rxinfer error: %s\n", ex.what());
        return 1;
    }
}

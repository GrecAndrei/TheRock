// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/common.hpp"
#include "rxinfer/config.hpp"
#include "rxinfer/engine.hpp"
#include "rxinfer/gdn.hpp"
#include "rxinfer/loader.hpp"
#include "rxinfer/markdown.hpp"
#include "rxinfer/sampler.hpp"
#include "rxinfer/server.hpp"
#include "rxinfer/tokenizer.hpp"
#include "rxinfer/tools.hpp"
#include "rxinfer/tui.hpp"

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
    std::signal(SIGPIPE, SIG_IGN);
    struct sigaction sa{};
    sa.sa_handler = handle_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
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
                 "  --think                   Enable <think> reasoning mode in chat template\n"
                 "  --no-think                Disable <think> reasoning mode (even in --serve / tool mode)\n"
                 "  --reasoning-effort <lvl>  Reasoning effort hint (xhigh|medium|low|none)\n"
                 "  --preserve-thinking       Serve prior <think> reasoning back across turns & tool calls (default: on)\n"
                 "  --no-preserve-thinking    Strip prior-turn <think> on new user turns (still kept across tool calls)\n"
                 "  --tools <path.json>       Load OpenAI-format tool definitions JSON file\n"
                 "  --builtin-tools           Enable built-in workspace tools (read_file, list_dir, search_text, write_file, run_command)\n"
                 "  --auto-approve            Auto-approve mutating built-in tools (write_file, run_command)\n"
                 "  --tool-exec <cmd>         External command receiving {id,name,arguments} JSON on stdin for custom tools\n"
                 "  --max-tool-rounds <N>     Max tool-calling rounds per turn (default: 16)\n"
                 "  --serve                   Start OpenAI-compatible HTTP server (/v1/chat/completions, /v1/models, /health)\n"
                 "  --host <addr>             Server bind address (default: 127.0.0.1)\n"
                 "  --port <N>                Server TCP port (default: 8080)\n"
                 "  --api-key <key>           Optional Bearer API key required by --serve\n"
                 "  --format                  Force ANSI Markdown terminal formatting (default: auto on TTY)\n"
                 "  --no-format, --raw        Disable ANSI Markdown formatting (plain UTF-8 output)\n"
                 "  --max-tokens <N>          Max new tokens per turn (0 = unlimited, default: 0)\n"
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

std::vector<rxinfer::tui::InteractiveLineEditor::CommandSpec> build_interactive_commands() {
    return {
        {"/help", "", "Show interactive commands", false},
        {"/?", "", "Alias for /help", false},
        {"/file", "<path> [prompt]", "Attach file into context (send if prompt given)", true},
        {"/attach", "<path> [prompt]", "Alias for /file", true},
        {"/paste", "", "Start multi-line block input (/end or \"\"\")", false},
        {"/undo", "[N]", "Rewind conversation by N turns (default 1)", false},
        {"/retry", "", "Rewind last turn and regenerate response", false},
        {"/reset", "", "Reset conversation (keeps LCP snapshots)", false},
        {"/clear", "", "Alias for /reset", false},
        {"/stats", "", "Show context, VRAM/GTT, sampler, and speed stats", false},
        {"/cache", "", "Show prompt cache, LCP snapshots, and KV tiers", false},
        {"/save-cache", "[path]", "Save live KV + GDN state to .rxprompt", true},
        {"/load-cache", "<path>", "Restore KV + GDN state from .rxprompt", true},
        {"/clear-cache", "", "Free in-memory GDN prefix checkpoints", false},
        {"/format", "[on|off]", "Toggle terminal Markdown formatting", false},
        {"/think", "[on|off]", "Toggle <think> reasoning mode", false},
        {"/tools", "[builtin|load <p>|clear]", "Inspect or manage function-calling tools", true},
        {"/approve", "[on|off]", "Toggle auto-approval for mutating tools", false},
        {"/history", "", "Show structured turn, reasoning & tool history", false},
        {"/system", "<text>", "Set system prompt (resets conversation)", false},
        {"/temp", "<F>", "Set sampling temperature (0 = greedy GPU argmax)", false},
        {"/top-k", "<N>", "Set top-k cutoff (0 = off)", false},
        {"/top-p", "<F>", "Set top-p cutoff (1.0 = off)", false},
        {"/min-p", "<F>", "Set min-p cutoff (0.0 = off)", false},
        {"/repeat-penalty", "<F>", "Set repetition penalty (1.0 = off)", false},
        {"/max-tokens", "[N|0]", "Set max tokens generated per turn (0 = unlimited)", false},
        {"/seed", "<N>", "Set sampling seed (0 = random)", false},
        {"/bench", "", "Run prefill/decode/verify benchmark", false},
        {"/quit", "", "Exit interactive session", false},
        {"/exit", "", "Alias for /quit", false},
        {"/q", "", "Alias for /quit", false},
    };
}

void print_interactive_help(const rxinfer::tui::UiTheme& theme) {
    const auto cmds = build_interactive_commands();
    std::vector<std::pair<std::string, std::string>> rows;
    auto add_cmd_row = [&](const char* group, std::initializer_list<const char*> names) {
        std::string list;
        for (const char* n : names) {
            for (const auto& c : cmds) {
                if (c.name == n) {
                    std::string left = c.name;
                    if (!c.args_hint.empty()) left += " " + c.args_hint;
                    while (left.size() < 24) left.push_back(' ');
                    std::string line = std::string(theme.accent()) + left + theme.reset() +
                                       theme.muted() + c.desc + theme.reset();
                    rows.emplace_back(list.empty() ? group : "", std::move(line));
                    list = "1";
                    break;
                }
            }
        }
    };
    add_cmd_row("Session", {"/undo", "/retry", "/reset", "/quit"});
    add_cmd_row("Input", {"/file", "/paste", "/system", "/think", "/format"});
    add_cmd_row("Cache", {"/cache", "/save-cache", "/load-cache", "/clear-cache"});
    add_cmd_row("Sampler", {"/temp", "/top-k", "/top-p", "/min-p", "/repeat-penalty", "/max-tokens", "/seed"});
    add_cmd_row("Tools", {"/tools", "/approve", "/history", "/stats", "/bench", "/help"});
    rows.emplace_back("Tips",
                      std::string(theme.muted()) +
                          "Tab completes commands & paths · ↑↓ history · Alt+Enter / \\ newline · Ctrl-C interrupts" +
                          theme.reset());

    std::fprintf(stderr, "\n");
    const std::string title = std::string(theme.accent()) + "◆" + theme.reset() + " " +
                              theme.bold() + "Interactive Commands" + theme.reset();
    rxinfer::tui::print_box_card(stderr, theme, title, rows, 92);
    std::fprintf(stderr, "\n");
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
    size_t messages_count_before = 0;
    int next_tool_call_id_before = 1;
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
    std::vector<rxinfer::ChatMessage> messages;
    int next_tool_call_id = 1;
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
        messages.clear();
        next_tool_call_id = 1;
        history.clear();
    }
};

struct TurnOptions {
    bool chat_mode = false;
    bool think_mode = false;
    bool preserve_thinking = true;
    bool format_markdown = true;
    bool ignore_eos = false;
    bool quiet = false;
    bool interactive = false;
    bool auto_prompt_cache = false;
    int32_t max_tokens = 0;
    int32_t max_seq_len = 4096;
    int32_t max_tool_rounds = 16;
    std::string system_prompt;
    std::string reasoning_effort;
    std::string save_prompt_cache_after_prefill;
    std::string auto_cache_dir;
    std::string shared_doc_prefix; // Optional non-interactive --prompt-file content when paired with --prompt
    std::string model_display_name = "Assistant";
    std::vector<rxinfer::ToolFunctionDef> tools;
    rxinfer::tools::ToolExecOptions tool_exec{};
    rxinfer::tui::UiTheme theme{};
};

void print_session_stats(const ChatSessionState& st,
                           const rxinfer::InferenceEngine& engine,
                           const rxinfer::Sampler& sampler,
                           const rxinfer::tui::UiTheme& theme) {
    const auto& sc = sampler.config();
    const double sess_prefill_tps =
        (st.total_prefill_ms > 0.0) ? (st.total_prompt_tokens * 1000.0 / st.total_prefill_ms) : 0.0;
    const double sess_decode_tps =
        (st.total_decode_ms > 0.0) ? (st.total_decode_calls * 1000.0 / st.total_decode_ms) : 0.0;
    const double sess_ms_tok =
        (st.total_decode_calls > 0) ? (st.total_decode_ms / st.total_decode_calls) : 0.0;
    const double ctx_frac =
        static_cast<double>(engine.position()) / static_cast<double>(std::max(1, engine.options().max_seq_len));

    char buf_ctx[512], buf_vram[512], buf_kv[512], buf_samp[512], buf_sess[512];
    std::snprintf(buf_ctx, sizeof(buf_ctx), "%s  %s%d / %d tok%s %s(%.1f%%)%s",
                  rxinfer::tui::render_progress_bar(ctx_frac, 18, theme).c_str(),
                  theme.bold(), engine.position(), engine.options().max_seq_len, theme.reset(),
                  theme.muted(), ctx_frac * 100.0, theme.reset());
    std::snprintf(buf_vram, sizeof(buf_vram), "%.2f GiB %s(weights %.2f · state %.2f · scratch %.2f GiB)%s",
                  engine.total_vram_used_bytes() / double(1ULL << 30),
                  theme.muted(),
                  engine.weights().weight_arena.used() / double(1ULL << 30),
                  engine.state_vram_bytes() / double(1ULL << 30),
                  engine.scratch_vram_bytes() / double(1ULL << 30),
                  theme.reset());
    std::snprintf(buf_kv, sizeof(buf_kv), "VRAM %.2f GiB (%s tok) + GTT %.2f GiB (%s tok) · %d snap%s",
                  engine.kv_vram_bytes() / double(1ULL << 30),
                  rxinfer::tui::format_tokens_compact(engine.vram_kv_tokens()).c_str(),
                  engine.kv_gtt_bytes() / double(1ULL << 30),
                  rxinfer::tui::format_tokens_compact(engine.gtt_kv_tokens()).c_str(),
                  engine.num_prefix_checkpoints(),
                  engine.num_prefix_checkpoints() == 1 ? "" : "s");
    std::snprintf(buf_samp, sizeof(buf_samp), "temp %.2f · top_k %d · top_p %.2f · min_p %.2f · rep %.2f · %s",
                  sc.temperature, sc.top_k, sc.top_p, sc.min_p, sc.repeat_penalty,
                  sc.uses_host_logits() ? "host-sample" : "gpu-argmax");
    std::snprintf(buf_sess, sizeof(buf_sess),
                  "%d turn%s · prompt %lld tok (%.1f/s) · gen %lld tok · decode %s%.2f tok/s%s (%.2f ms/tok)",
                  st.turns, st.turns == 1 ? "" : "s",
                  static_cast<long long>(st.total_prompt_tokens), sess_prefill_tps,
                  static_cast<long long>(st.total_generated_tokens),
                  theme.ok(), sess_decode_tps, theme.reset(), sess_ms_tok);

    std::vector<std::pair<std::string, std::string>> rows = {
        {"Context", buf_ctx},
        {"VRAM", buf_vram},
        {"KV Pool", buf_kv},
        {"Sampler", buf_samp},
        {"Session", buf_sess},
    };
    if (st.last_prompt_tokens > 0 || st.last_generated_tokens > 0) {
        const double l_p_tps = (st.last_prefill_ms > 0.0) ? (st.last_prompt_tokens * 1000.0 / st.last_prefill_ms) : 0.0;
        const double l_d_tps = (st.last_decode_ms > 0.0) ? (st.last_decode_calls * 1000.0 / st.last_decode_ms) : 0.0;
        char buf_last[192];
        std::snprintf(buf_last, sizeof(buf_last),
                      "prompt %d tok (%.0f ms · %.1f/s) · gen %d tok · decode %.2f tok/s",
                      st.last_prompt_tokens, st.last_prefill_ms, l_p_tps,
                      st.last_generated_tokens, l_d_tps);
        rows.emplace_back("Last turn", buf_last);
    }

    std::fprintf(stderr, "\n");
    const std::string title = std::string(theme.accent()) + "◆" + theme.reset() + " " +
                              theme.bold() + "Session Stats" + theme.reset();
    rxinfer::tui::print_box_card(stderr, theme, title, rows, 90);
    std::fprintf(stderr, "\n");
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

std::string format_first_turn_prompt(const rxinfer::Tokenizer& tokenizer,
                                     const std::string& user_prompt,
                                     const TurnOptions& opts) {
    if (opts.tools.empty() && opts.reasoning_effort.empty()) {
        return tokenizer.format_chat_turn(user_prompt, opts.system_prompt, true, false, opts.think_mode);
    }
    rxinfer::ChatTemplateOptions tpl_opts;
    tpl_opts.add_generation_prompt = true;
    tpl_opts.enable_thinking = opts.think_mode;
    tpl_opts.preserve_thinking = opts.preserve_thinking;
    tpl_opts.reasoning_effort = opts.reasoning_effort;

    std::string out = tokenizer.format_system_prompt(opts.system_prompt, opts.tools, tpl_opts);
    const auto fmt = tokenizer.detect_chat_format();
    const bool has_think = (tokenizer.token_to_id("<think>") >= 0 && tokenizer.token_to_id("</think>") >= 0);
    if (fmt == rxinfer::Tokenizer::ChatFormat::Llama3) {
        out += "<|start_header_id|>user<|end_header_id|>\n\n" + user_prompt + "<|eot_id|>";
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        if (opts.think_mode && has_think) out += "<think>\n";
    } else {
        out += "<|im_start|>user\n" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n";
        if (!opts.think_mode && tokenizer.token_to_id("</think>") >= 0) {
            out += "<think>\n\n</think>\n\n";
        } else if (opts.think_mode && has_think) {
            out += "<think>\n";
        }
    }
    return out;
}

std::string truncate_inline_preview(const std::string& s, size_t max_chars = 96) {
    std::string clean;
    clean.reserve(std::min(s.size(), max_chars + 4));
    for (char c : s) {
        if (c == '\n' || c == '\r' || c == '\t') clean.push_back(' ');
        else clean.push_back(c);
        if (clean.size() >= max_chars) {
            clean += "...";
            break;
        }
    }
    return clean;
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

    rxinfer::ChatTemplateOptions tpl_opts;
    tpl_opts.add_generation_prompt = true;
    tpl_opts.enable_thinking = opts.think_mode;
    tpl_opts.preserve_thinking = opts.preserve_thinking;
    tpl_opts.reasoning_effort = opts.reasoning_effort;

    const bool rebuild_without_prior_thinking =
        (!state.is_first_turn && opts.chat_mode && !opts.preserve_thinking && !state.messages.empty());

    std::string delta_text;
    std::vector<int32_t> encoded;
    std::vector<int32_t> turn_ids;
    std::vector<int32_t> full_prompt_ids;

    if (rebuild_without_prior_thinking) {
        std::vector<rxinfer::ChatMessage> rebuilt = state.messages;
        if (!opts.system_prompt.empty() &&
            (rebuilt.empty() || (rebuilt[0].role != "system" && rebuilt[0].role != "developer"))) {
            rxinfer::ChatMessage sys_msg;
            sys_msg.role = "system";
            sys_msg.content = opts.system_prompt;
            rebuilt.insert(rebuilt.begin(), std::move(sys_msg));
        }
        rxinfer::ChatMessage umsg;
        umsg.role = "user";
        umsg.content = input_text;
        rebuilt.push_back(std::move(umsg));

        const std::string full_text = tokenizer.format_chat_messages(rebuilt, opts.tools, tpl_opts);
        full_prompt_ids = encode_with_optional_bos(tokenizer, full_text, tokenizer.should_add_bos());
        turn_ids = full_prompt_ids;
        encoded = full_prompt_ids;
    } else {
        if (opts.chat_mode) {
            if (state.is_first_turn) {
                delta_text = format_first_turn_prompt(tokenizer, input_text, opts);
            } else {
                delta_text = tokenizer.format_chat_turn(
                    input_text, opts.system_prompt, false, state.close_previous_turn, opts.think_mode);
            }
        } else {
            delta_text = (state.is_first_turn || input_text.empty()) ? input_text : ("\n" + input_text);
        }

        const bool add_bos = state.is_first_turn && state.pending_tokens.empty() && tokenizer.should_add_bos();
        encoded = encode_with_optional_bos(tokenizer, delta_text, add_bos);

        turn_ids = state.pending_tokens;
        turn_ids.insert(turn_ids.end(), encoded.begin(), encoded.end());
    }

    if (state.is_first_turn || rebuild_without_prior_thinking) {
        if (turn_ids.empty()) {
            if (opts.interactive) {
                std::fprintf(stderr, "[Empty turn; nothing to prefill]\n");
                return false;
            }
            throw std::runtime_error("Prompt tokenized to 0 tokens");
        }
        if (static_cast<int32_t>(turn_ids.size()) >= opts.max_seq_len) {
            if (opts.interactive) {
                if (opts.theme.color) {
                    char msg[160];
                    std::snprintf(msg, sizeof(msg), "Context limit reached (%zu >= %d) — use /reset to start fresh",
                                  turn_ids.size(), opts.max_seq_len);
                    rxinfer::tui::print_status_badge(stderr, opts.theme, "warn", msg);
                } else {
                    std::fprintf(stderr,
                                 "[Context limit reached (%zu >= %d). Use /reset to start a new conversation.]\n",
                                 turn_ids.size(), opts.max_seq_len);
                }
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
                if (opts.theme.color) {
                    char msg[160];
                    std::snprintf(msg, sizeof(msg),
                                  "Context limit reached (%d + %zu >= %d) — use /reset to start fresh",
                                  engine.position(), turn_ids.size(), opts.max_seq_len);
                    rxinfer::tui::print_status_badge(stderr, opts.theme, "warn", msg);
                } else {
                    std::fprintf(stderr,
                                 "[Context limit reached (%d + %zu >= %d). Use /reset to start a new conversation.]\n",
                                 engine.position(), turn_ids.size(), opts.max_seq_len);
                }
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
    turn_rec.messages_count_before = state.messages.size();
    turn_rec.next_tool_call_id_before = state.next_tool_call_id;
    turn_rec.user_input = input_text;

    if (!rebuild_without_prior_thinking) {
        if (state.is_first_turn) {
            full_prompt_ids = turn_ids;
        } else {
            full_prompt_ids = engine.cached_tokens();
            full_prompt_ids.insert(full_prompt_ids.end(), turn_ids.begin(), turn_ids.end());
        }
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
                if (opts.theme.color) {
                    char msg[256];
                    std::snprintf(msg, sizeof(msg), "Auto-cache hit · %s tok loaded in %s",
                                  rxinfer::tui::format_tokens_compact(loaded_pos).c_str(),
                                  rxinfer::tui::format_duration_compact(
                                      std::chrono::duration<double, std::milli>(tlc1 - tlc0).count())
                                      .c_str());
                    rxinfer::tui::print_status_badge(stderr, opts.theme, "info", msg);
                } else {
                    std::fprintf(stderr, "[Auto-cache hit: loaded %d tok from %s in %.1f ms]\n",
                                 loaded_pos, best_file.c_str(),
                                 std::chrono::duration<double, std::milli>(tlc1 - tlc0).count());
                }
            }
        }
    }

    if (rebuild_without_prior_thinking) {
        sampler.clear_history();
        sampler.record_tokens(full_prompt_ids);
    } else {
        sampler.record_tokens(encoded);
    }
    const int32_t vocab_size = engine.config().vocab_size;

    const auto tp0 = std::chrono::steady_clock::now();
    bool evaluated_cold_prefix = false;
    int32_t prior_cached_tokens = loaded_disk_auto_cache ? engine.position() : 0;

    std::string spin_label = "Thinking";
    const size_t approx_new =
        (full_prompt_ids.size() > static_cast<size_t>(prior_cached_tokens))
            ? (full_prompt_ids.size() - static_cast<size_t>(prior_cached_tokens))
            : full_prompt_ids.size();
    if (approx_new >= 256) {
        spin_label = "Reading " + rxinfer::tui::format_tokens_compact(static_cast<int64_t>(approx_new)) + " tokens";
    }
    rxinfer::tui::PrefillSpinner prefill_spinner(opts.theme, !opts.quiet && opts.theme.tty_stderr, spin_label);

    // Automatically cache and pin system/tools prompt prefix on Turn 1 so /reset keeps it hot.
    if (opts.chat_mode && (!opts.system_prompt.empty() || !opts.tools.empty() || !opts.reasoning_effort.empty()) &&
        state.is_first_turn && !loaded_disk_auto_cache) {
        const std::string sys_text = tokenizer.format_system_prompt(opts.system_prompt, opts.tools, tpl_opts);
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
            const std::string sys_part = tokenizer.format_system_prompt(opts.system_prompt, opts.tools, tpl_opts);
            doc_prefix_text = sys_part + "<|im_start|>user\n";
            if (tokenizer.detect_chat_format() == rxinfer::Tokenizer::ChatFormat::Llama3) {
                doc_prefix_text = sys_part + "<|start_header_id|>user<|end_header_id|>\n\n";
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
                if (opts.theme.color) {
                    char msg[256];
                    std::snprintf(msg, sizeof(msg), "Auto-cached shared prefix (%s tok) in %s",
                                  rxinfer::tui::format_tokens_compact(static_cast<int64_t>(shared)).c_str(),
                                  rxinfer::tui::format_duration_compact(
                                      std::chrono::duration<double, std::milli>(tsc1 - tsc0).count())
                                      .c_str());
                    rxinfer::tui::print_status_badge(stderr, opts.theme, "info", msg);
                } else {
                    std::fprintf(stderr, "[Auto-cached shared prefix (%zu tok) to %s in %.1f ms]\n",
                                 shared, saved.c_str(),
                                 std::chrono::duration<double, std::milli>(tsc1 - tsc0).count());
                }
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
    prefill_spinner.stop();

    if (!opts.save_prompt_cache_after_prefill.empty()) {
        const auto tsc0 = std::chrono::steady_clock::now();
        engine.save_prompt_cache(opts.save_prompt_cache_after_prefill);
        const auto tsc1 = std::chrono::steady_clock::now();
        if (!opts.quiet) {
            if (opts.theme.color) {
                char msg[256];
                std::snprintf(msg, sizeof(msg), "Saved prompt cache (%d tok) in %s",
                              engine.position(),
                              rxinfer::tui::format_duration_compact(
                                  std::chrono::duration<double, std::milli>(tsc1 - tsc0).count())
                                  .c_str());
                rxinfer::tui::print_status_badge(stderr, opts.theme, "ok", msg);
            } else {
                std::fprintf(stderr, "[Saved prompt cache (%d tok) to %s in %.1f ms]\n",
                             engine.position(), opts.save_prompt_cache_after_prefill.c_str(),
                             std::chrono::duration<double, std::milli>(tsc1 - tsc0).count());
            }
        }
    } else if (!opts.interactive && opts.auto_prompt_cache && !opts.auto_cache_dir.empty() &&
               opts.shared_doc_prefix.empty() && cache_res.evaluated_tokens >= 1024) {
        const auto tsc0 = std::chrono::steady_clock::now();
        const std::string saved = save_to_auto_cache(opts.auto_cache_dir, engine, opts.quiet);
        const auto tsc1 = std::chrono::steady_clock::now();
        if (!saved.empty() && !opts.quiet) {
            if (opts.theme.color) {
                char msg[256];
                std::snprintf(msg, sizeof(msg), "Auto-cached prompt (%d tok) in %s",
                              engine.position(),
                              rxinfer::tui::format_duration_compact(
                                  std::chrono::duration<double, std::milli>(tsc1 - tsc0).count())
                                  .c_str());
                rxinfer::tui::print_status_badge(stderr, opts.theme, "info", msg);
            } else {
                std::fprintf(stderr, "[Auto-cached prompt (%d tok) to %s in %.1f ms]\n",
                             engine.position(), saved.c_str(),
                             std::chrono::duration<double, std::milli>(tsc1 - tsc0).count());
            }
        }
    }

    state.pending_attachments.clear();
    state.pending_tokens.clear();
    state.is_first_turn = false;
    state.close_previous_turn = opts.chat_mode;

    if (opts.chat_mode) {
        rxinfer::ChatMessage umsg;
        umsg.role = "user";
        umsg.content = input_text;
        state.messages.push_back(std::move(umsg));
    }

    if (opts.interactive && opts.chat_mode && !opts.quiet && opts.format_markdown && opts.theme.tty_stdout) {
        std::printf("\n%s%s◆%s %s%s%s\n",
                    opts.theme.bold(), opts.theme.accent(), opts.theme.reset(),
                    opts.theme.bold(), opts.model_display_name.c_str(), opts.theme.reset());
        std::fflush(stdout);
    }

    const bool has_think_tokens =
        (tokenizer.token_to_id("<think>") >= 0 && tokenizer.token_to_id("</think>") >= 0);

    int32_t total_turn_prompt_tokens = cache_res.evaluated_tokens;
    int32_t generated = 0;
    int32_t decode_calls = 0;
    double total_turn_prefill_ms = prefill_ms;
    double decode_ms = 0.0;
    bool hit_ctx_limit = false;
    bool hit_tok_limit = false;
    bool hit_tool_round_limit = false;
    bool interrupted = false;

    for (int32_t tool_round = 0;; ++tool_round) {
        const bool started_inside_think = opts.chat_mode && opts.think_mode && has_think_tokens;
        const bool use_structured_tool_display = (!opts.tools.empty() && opts.format_markdown);

        rxinfer::MarkdownStreamFormatter printer(opts.format_markdown);
        rxinfer::tools::StreamingAssistantParser tool_parser(
            started_inside_think, opts.tools, state.next_tool_call_id);
        bool printer_in_think = false;
        bool printed_to_stdout = false;
        if (started_inside_think && !use_structured_tool_display) {
            printer.push("<think>\n");
            printer_in_think = true;
            printed_to_stdout = true;
        }

        std::string raw_generated;
        for (int32_t step = 0; opts.max_tokens <= 0 || step < opts.max_tokens; ++step) {
            if (g_interrupted) {
                interrupted = true;
                break;
            }
            if (!opts.ignore_eos && is_stop_token(next_tok, tokenizer, engine.config())) {
                break;
            }

            const std::string& raw_tok = tokenizer.id_to_token(next_tok);
            const std::string piece =
                (raw_tok == "<think>" || raw_tok == "</think>" || raw_tok == "<tool_call>" ||
                 raw_tok == "</tool_call>" || raw_tok == "<|python_tag|>" || raw_tok == "<|eom_id|>")
                    ? raw_tok
                    : tokenizer.decode_token(next_tok, true);
            raw_generated += piece;

            if (use_structured_tool_display) {
                auto d = tool_parser.feed(piece);
                if (!d.reasoning_content.empty()) {
                    if (!printer_in_think) {
                        printer.push("<think>\n");
                        printer_in_think = true;
                    }
                    printer.push(d.reasoning_content);
                    printed_to_stdout = true;
                }
                if (!d.content.empty() || !d.tool_calls.empty()) {
                    if (printer_in_think) {
                        printer.push("</think>\n\n");
                        printer_in_think = false;
                    }
                    if (!d.content.empty()) {
                        printer.push(d.content);
                        printed_to_stdout = true;
                    }
                }
            } else {
                printer.push(piece);
                printed_to_stdout = true;
            }

            sampler.record_token(next_tok);
            ++generated;

            if (engine.position() >= opts.max_seq_len) {
                hit_ctx_limit = true;
                state.pending_tokens.push_back(next_tok);
                break;
            }
            if (opts.max_tokens > 0 && step + 1 >= opts.max_tokens) {
                hit_tok_limit = true;
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

        if (use_structured_tool_display) {
            auto tail = tool_parser.finish();
            if (!tail.reasoning_content.empty()) {
                if (!printer_in_think) {
                    printer.push("<think>\n");
                    printer_in_think = true;
                }
                printer.push(tail.reasoning_content);
                printed_to_stdout = true;
            }
            if (printer_in_think && (!tail.content.empty() || !tool_parser.tool_calls().empty())) {
                printer.push("</think>\n\n");
                printer_in_think = false;
            }
            if (!tail.content.empty()) {
                printer.push(tail.content);
                printed_to_stdout = true;
            }
        }

        printer.finish();
        if (printed_to_stdout || !use_structured_tool_display) {
            std::printf("\n");
        }
        std::fflush(stdout);

        rxinfer::tools::ParsedAssistantOutput parsed = rxinfer::tools::parse_assistant_output(
            raw_generated, started_inside_think, opts.tools, state.next_tool_call_id);
        state.next_tool_call_id += static_cast<int>(parsed.tool_calls.size());

        if (opts.chat_mode) {
            rxinfer::ChatMessage amsg;
            amsg.role = "assistant";
            amsg.content = parsed.content;
            amsg.reasoning_content = parsed.reasoning_content;
            amsg.tool_calls = parsed.tool_calls;
            state.messages.push_back(std::move(amsg));
        }

        if (opts.tools.empty() || parsed.tool_calls.empty() || interrupted || hit_ctx_limit || hit_tok_limit) {
            break;
        }
        if (tool_round >= opts.max_tool_rounds) {
            hit_tool_round_limit = true;
            break;
        }

        // Execute requested tool calls and feed <tool_response> back WITH prior reasoning preserved in KV + GDN cache.
        rxinfer::tools::ToolExecOptions exec_opts = opts.tool_exec;
        if (!exec_opts.allow_mutating && opts.interactive && ::isatty(STDIN_FILENO) != 0) {
            exec_opts.confirm_cb = [&](const rxinfer::ToolCall& tc) -> bool {
                std::fprintf(stderr, "\n%s[Approve mutating tool %s(%s)? y/N]: %s",
                             opts.theme.warn(), tc.name.c_str(),
                             truncate_inline_preview(tc.arguments, 80).c_str(),
                             opts.theme.reset());
                std::fflush(stderr);
                std::string ans;
                if (!std::getline(std::cin, ans)) return false;
                const std::string t = trim_ws(ans);
                return (t == "y" || t == "Y" || t == "yes" || t == "YES");
            };
        }

        std::vector<rxinfer::ChatMessage> tool_msgs;
        tool_msgs.reserve(parsed.tool_calls.size());
        for (const auto& tc : parsed.tool_calls) {
            if (!opts.quiet) {
                const std::string arg_preview = truncate_inline_preview(tc.arguments, 92);
                if (opts.theme.color) {
                    std::fprintf(stderr, "  %s⚙ %s%s(%s)%s\n",
                                 opts.theme.accent(), tc.name.c_str(),
                                 opts.theme.muted(), arg_preview.c_str(), opts.theme.reset());
                } else {
                    std::fprintf(stderr, "[Tool call: %s(%s)]\n", tc.name.c_str(), arg_preview.c_str());
                }
            }
            const auto tx0 = std::chrono::steady_clock::now();
            const rxinfer::tools::ToolExecResult res = rxinfer::tools::execute_tool_call(tc, exec_opts);
            const auto tx1 = std::chrono::steady_clock::now();
            const double tx_ms = std::chrono::duration<double, std::milli>(tx1 - tx0).count();
            if (!opts.quiet) {
                if (opts.theme.color) {
                    std::fprintf(stderr, "  %s%s %s%s · %s in %s%s\n",
                                 res.ok ? opts.theme.ok() : opts.theme.err(),
                                 res.ok ? "✓" : "✗",
                                 tc.name.c_str(),
                                 opts.theme.muted(),
                                 format_byte_size(res.output.size()).c_str(),
                                 rxinfer::tui::format_duration_compact(tx_ms).c_str(),
                                 opts.theme.reset());
                } else {
                    std::fprintf(stderr, "[Tool result: %s -> %s (%zu bytes, %.1f ms)]\n",
                                 tc.name.c_str(), res.ok ? "ok" : "error", res.output.size(), tx_ms);
                }
            }

            rxinfer::ChatMessage tm;
            tm.role = "tool";
            tm.tool_call_id = tc.id;
            tm.name = tc.name;
            tm.content = res.output;
            tool_msgs.push_back(tm);
            if (opts.chat_mode) {
                state.messages.push_back(std::move(tm));
            }
        }

        const std::string tool_delta =
            tokenizer.format_tool_response_turn(tool_msgs, /*close_previous_turn=*/true, opts.think_mode);
        const std::vector<int32_t> tool_delta_ids = encode_with_optional_bos(tokenizer, tool_delta, false);
        if (engine.position() + static_cast<int32_t>(tool_delta_ids.size()) >= opts.max_seq_len) {
            hit_ctx_limit = true;
            break;
        }

        std::vector<int32_t> next_prompt_ids = engine.cached_tokens();
        next_prompt_ids.insert(next_prompt_ids.end(), tool_delta_ids.begin(), tool_delta_ids.end());
        sampler.record_tokens(tool_delta_ids);

        rxinfer::tui::PrefillSpinner tool_spinner(opts.theme, !opts.quiet && opts.theme.tty_stderr, "Thinking");
        const auto ttp0 = std::chrono::steady_clock::now();
        const rxinfer::PrefillCacheResult tool_cache_res = engine.prefill_cached(next_prompt_ids, true);
        next_tok = tool_cache_res.next_token;
        if (sampler.config().uses_host_logits() && tool_cache_res.evaluated_tokens > 0) {
            engine.copy_last_logits_f16(logits_f16_scratch);
            next_tok = sampler.sample_from_f16_bits(logits_f16_scratch.data(), vocab_size);
        }
        const auto ttp1 = std::chrono::steady_clock::now();
        tool_spinner.stop();

        total_turn_prefill_ms += std::chrono::duration<double, std::milli>(ttp1 - ttp0).count();
        total_turn_prompt_tokens += tool_cache_res.evaluated_tokens;
    }

    // Save post-turn prefix checkpoint so /undo, /retry, and branching stay instant.
    if (opts.interactive && engine.position() >= 8) {
        engine.save_prefix_checkpoint(false);
    }
    state.history.push_back(std::move(turn_rec));

    if (interrupted) {
        g_interrupted = 0;
        if (!opts.quiet) {
            if (opts.theme.color) {
                rxinfer::tui::print_status_badge(stderr, opts.theme, "warn", "interrupted");
            } else {
                std::fprintf(stderr, "[interrupted]\n");
            }
        }
    } else if (hit_tool_round_limit && !opts.quiet) {
        if (opts.theme.color) {
            char msg[160];
            std::snprintf(msg, sizeof(msg), "Reached max tool rounds (%d) — pass --max-tool-rounds to raise",
                          opts.max_tool_rounds);
            rxinfer::tui::print_status_badge(stderr, opts.theme, "warn", msg);
        } else {
            std::fprintf(stderr, "[reached max tool rounds (%d)]\n", opts.max_tool_rounds);
        }
    } else if (hit_ctx_limit && !opts.quiet) {
        if (opts.theme.color) {
            char msg[160];
            std::snprintf(msg, sizeof(msg), "Context limit %d reached — use /reset for a new conversation",
                          opts.max_seq_len);
            rxinfer::tui::print_status_badge(stderr, opts.theme, "warn", msg);
        } else {
            std::fprintf(stderr, "[context limit %d reached; use /reset for a new conversation]\n", opts.max_seq_len);
        }
    } else if (hit_tok_limit && !opts.quiet) {
        if (opts.theme.color) {
            char msg[160];
            std::snprintf(msg, sizeof(msg), "Token limit %d reached — use /max-tokens 0 for unlimited",
                          opts.max_tokens);
            rxinfer::tui::print_status_badge(stderr, opts.theme, "warn", msg);
        } else {
            std::fprintf(stderr, "[max_tokens limit %d reached; use /max-tokens 0 for unlimited]\n", opts.max_tokens);
        }
    }

    state.turns++;
    state.last_prompt_tokens = total_turn_prompt_tokens;
    state.last_generated_tokens = generated;
    state.last_decode_calls = decode_calls;
    state.last_prefill_ms = total_turn_prefill_ms;
    state.last_decode_ms = decode_ms;

    state.total_prompt_tokens += total_turn_prompt_tokens;
    state.total_generated_tokens += generated;
    state.total_decode_calls += decode_calls;
    state.total_prefill_ms += total_turn_prefill_ms;
    state.total_decode_ms += decode_ms;

    if (!opts.quiet) {
        const double p_tps = (total_turn_prompt_tokens * 1000.0) / std::max(0.001, total_turn_prefill_ms);
        const double total_ms = total_turn_prefill_ms + decode_ms;
        const double e2e_tps = (generated > 0) ? (generated * 1000.0 / std::max(0.001, total_ms)) : 0.0;
        if (opts.theme.color) {
            std::string speed_part;
            if (decode_calls > 0) {
                const double d_tps = (decode_calls * 1000.0) / std::max(0.001, decode_ms);
                char sb[96];
                std::snprintf(sb, sizeof(sb), "%s⚡ %.1f tok/s%s · ", opts.theme.ok(), d_tps, opts.theme.reset());
                speed_part = sb;
            }
            std::string cache_part;
            if (cache_res.matched_prefix_tokens > 0) {
                char cb[128];
                std::snprintf(cb, sizeof(cb), " · %scached %s +%d%s",
                              opts.theme.accent(),
                              rxinfer::tui::format_tokens_compact(cache_res.matched_prefix_tokens).c_str(),
                              total_turn_prompt_tokens,
                              opts.theme.reset());
                cache_part = cb;
            }
            std::fprintf(stderr, "  %s%s%d tok · TTFT %s%s · ctx %s/%s%s\n",
                         speed_part.c_str(),
                         opts.theme.muted(),
                         generated,
                         rxinfer::tui::format_duration_compact(total_turn_prefill_ms).c_str(),
                         cache_part.c_str(),
                         rxinfer::tui::format_tokens_compact(engine.position()).c_str(),
                         rxinfer::tui::format_tokens_compact(opts.max_seq_len).c_str(),
                         opts.theme.reset());
        } else {
            std::string prompt_desc;
            if (cache_res.matched_prefix_tokens > 0) {
                char pbuf[160];
                std::snprintf(pbuf, sizeof(pbuf), "%d new + %d cached%s",
                              total_turn_prompt_tokens, cache_res.matched_prefix_tokens,
                              cache_res.restored_gdn_snapshot ? " (GDN restored)" : "");
                prompt_desc = pbuf;
            } else {
                prompt_desc = std::to_string(total_turn_prompt_tokens) + " tok";
            }
            if (decode_calls > 0) {
                const double d_tps = (decode_calls * 1000.0) / std::max(0.001, decode_ms);
                const double ms_per_tok = decode_ms / decode_calls;
                std::fprintf(stderr,
                             "[Prompt: %s in %.1f ms (%.1f tok/s, TTFT %.1f ms) | Decode: %d steps in %.1f ms (%.2f tok/s, %.2f ms/tok) | Out: %d tok (E2E %.2f tok/s) | Ctx: %d/%d]\n",
                             prompt_desc.c_str(), total_turn_prefill_ms, p_tps, total_turn_prefill_ms,
                             decode_calls, decode_ms, d_tps, ms_per_tok,
                             generated, e2e_tps, engine.position(), opts.max_seq_len);
            } else {
                std::fprintf(stderr,
                             "[Prompt: %s in %.1f ms (%.1f tok/s, TTFT %.1f ms) | Out: %d tok (0 decode steps) | Ctx: %d/%d]\n",
                             prompt_desc.c_str(), total_turn_prefill_ms, p_tps, total_turn_prefill_ms,
                             generated, engine.position(), opts.max_seq_len);
            }
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
        std::vector<std::string> tools_files;
        std::string tool_exec_cmd;
        std::string reasoning_effort;
        std::string server_host = "127.0.0.1";
        std::string server_api_key;
        int server_port = 8080;
        bool has_cli_prompt = false;
        bool has_prompt_file = false;
        bool interactive = false;
        bool serve_mode = false;
        bool chat_flag_set = false;
        bool chat_flag_val = false;
        bool format_flag_set = false;
        bool format_flag_val = true;
        bool think_flag_set = false;
        bool think_mode = false;
        bool preserve_thinking = true;
        bool enable_builtin_tools = false;
        bool auto_approve = false;
        int32_t max_tool_rounds = 16;
        bool run_bench = false;
        bool no_graph = false;
        bool ignore_eos = false;
        bool quiet = false;
        bool has_max_seq_len = false;
        bool has_kv_dtype = false;
        int32_t max_tokens = 0;
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
                think_flag_set = true;
            } else if (arg == "--no-think") {
                think_mode = false;
                think_flag_set = true;
            } else if (arg == "--reasoning-effort") {
                reasoning_effort = need_val("--reasoning-effort");
                if (reasoning_effort == "none") {
                    think_mode = false;
                    think_flag_set = true;
                }
            } else if (arg == "--preserve-thinking") {
                preserve_thinking = true;
            } else if (arg == "--no-preserve-thinking") {
                preserve_thinking = false;
            } else if (arg == "--tools") {
                tools_files.push_back(expand_user_path(need_val("--tools")));
            } else if (arg == "--builtin-tools") {
                enable_builtin_tools = true;
            } else if (arg == "--auto-approve") {
                auto_approve = true;
            } else if (arg == "--tool-exec") {
                tool_exec_cmd = need_val("--tool-exec");
            } else if (arg == "--max-tool-rounds") {
                max_tool_rounds = std::max(1, std::stoi(need_val("--max-tool-rounds")));
            } else if (arg == "--serve") {
                serve_mode = true;
            } else if (arg == "--host") {
                server_host = need_val("--host");
            } else if (arg == "--port") {
                server_port = std::stoi(need_val("--port"));
                if (server_port <= 0 || server_port > 65535) {
                    throw std::runtime_error("--port must be in [1, 65535]");
                }
            } else if (arg == "--api-key") {
                server_api_key = need_val("--api-key");
            } else if (arg == "--max-tokens") {
                const int32_t v = std::stoi(need_val("--max-tokens"));
                if (v < -1) {
                    throw std::runtime_error("--max-tokens must be >= 0 (or -1 for unlimited)");
                }
                max_tokens = (v <= 0) ? 0 : v;
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
        if (serve_mode && interactive) {
            throw std::runtime_error("Cannot combine --serve and -i/--interactive");
        }

        std::vector<rxinfer::ToolFunctionDef> configured_tools;
        auto add_unique_tools = [&](const std::vector<rxinfer::ToolFunctionDef>& defs) {
            for (const auto& def : defs) {
                auto it = std::find_if(configured_tools.begin(), configured_tools.end(),
                                       [&](const rxinfer::ToolFunctionDef& e) { return e.name == def.name; });
                if (it != configured_tools.end()) {
                    *it = def;
                } else {
                    configured_tools.push_back(def);
                }
            }
        };
        if (enable_builtin_tools) {
            add_unique_tools(rxinfer::tools::builtin_workspace_tools());
        }
        for (const auto& tf : tools_files) {
            add_unique_tools(rxinfer::tools::load_tools_file(tf));
        }

        const bool tools_enabled = !configured_tools.empty();
        if (!think_flag_set) {
            if (serve_mode || tools_enabled) {
                think_mode = true;
            }
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

        const bool chat_mode = chat_flag_set ? chat_flag_val : (interactive || serve_mode || tools_enabled);
        const bool stdout_is_tty = (::isatty(STDOUT_FILENO) != 0);
        const bool no_color_env = (std::getenv("NO_COLOR") != nullptr && std::getenv("NO_COLOR")[0] != '\0');
        const bool format_markdown = format_flag_set ? format_flag_val : (stdout_is_tty && !no_color_env);
        const bool raw_flag_off = (format_flag_set && !format_flag_val);
        const rxinfer::tui::UiTheme theme = rxinfer::tui::UiTheme::detect(!raw_flag_off);
        const std::string model_display_name = rxinfer::tui::clean_model_display_name(model_path, "");

        const bool auto_prompt_cache = (prompt_cache_path == "auto");
        const std::string auto_cache_dir = auto_prompt_cache ? default_auto_cache_dir() : "";
        if (!auto_prompt_cache && !prompt_cache_path.empty()) {
            prompt_cache_path = expand_user_path(prompt_cache_path);
        }

        const auto dev = rxinfer::query_device_info(0);
        if (!quiet && !theme.color) {
            std::fprintf(stderr, "rxinfer | GPU: %s (%s) | VRAM: %.2f GiB (%.2f GiB free) | WGPs: %d\n",
                         dev.name.c_str(), dev.gcn_arch_name.c_str(),
                         dev.total_vram_bytes / double(1ULL << 30),
                         dev.free_vram_bytes / double(1ULL << 30),
                         dev.multiprocessor_count);
        }

        const auto t_load0 = std::chrono::steady_clock::now();
        rxinfer::tui::LoadProgressReporter load_prog(theme, !quiet && theme.tty_stderr, model_display_name);
        rxinfer::LoaderOptions lopts{};
        lopts.num_threads = threads;
        if (!quiet && theme.tty_stderr) {
            lopts.progress_cb = [&](size_t done, size_t total) { load_prog.update(done, total); };
        }

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
            weights = rxinfer::load_from_rxr4(model_path, lopts);
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
        load_prog.finish();

        const auto t_load1 = std::chrono::steady_clock::now();
        const double load_ms = std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();

        if (!quiet && !theme.color) {
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
                if (theme.color) {
                    char msg[256];
                    std::snprintf(msg, sizeof(msg), "Saved RXR4 checkpoint to %s in %s",
                                  save_rxr4_path.c_str(),
                                  rxinfer::tui::format_duration_compact(
                                      std::chrono::duration<double, std::milli>(ts1 - ts0).count())
                                      .c_str());
                    rxinfer::tui::print_status_badge(stderr, theme, "ok", msg);
                } else {
                    std::fprintf(stderr, "Saved RXR4 checkpoint to %s in %.1f ms\n",
                                 save_rxr4_path.c_str(),
                                 std::chrono::duration<double, std::milli>(ts1 - ts0).count());
                }
            }
            if (!has_prompt && !run_bench && !interactive && !serve_mode) return 0;
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

        const int64_t gen_reserve =
            (max_tokens > 0) ? static_cast<int64_t>(max_tokens) : (tools_enabled ? 16384 : 8192);
        if (!has_max_seq_len) {
            if (interactive || serve_mode) {
                max_seq_len = std::min<int32_t>(200000, model_max_pos);
                if (has_explicit_cache_file && explicit_cache_info.position + gen_reserve + 256 > max_seq_len) {
                    max_seq_len = std::min<int32_t>(model_max_pos,
                                                    static_cast<int32_t>(explicit_cache_info.position + gen_reserve + 256));
                }
                // Guard against tiny host RAM when auto-sizing interactive/server large context
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
                std::string first_txt = prompt;
                if (chat_mode) {
                    rxinfer::ChatTemplateOptions sz_opts;
                    sz_opts.add_generation_prompt = true;
                    sz_opts.enable_thinking = think_mode;
                    sz_opts.preserve_thinking = preserve_thinking;
                    sz_opts.reasoning_effort = reasoning_effort;
                    std::vector<rxinfer::ChatMessage> sz_msgs;
                    if (!system_prompt.empty()) {
                        sz_msgs.push_back(rxinfer::ChatMessage{"system", system_prompt, "", {}, "", ""});
                    }
                    sz_msgs.push_back(rxinfer::ChatMessage{"user", prompt, "", {}, "", ""});
                    first_txt = tokenizer.format_chat_messages(sz_msgs, configured_tools, sz_opts);
                }
                const size_t p_toks =
                    encode_with_optional_bos(tokenizer, first_txt, tokenizer.should_add_bos()).size();
                int64_t needed = static_cast<int64_t>(p_toks) + gen_reserve + 512;
                if (has_explicit_cache_file) {
                    needed = std::max<int64_t>(needed, static_cast<int64_t>(explicit_cache_info.position) + gen_reserve + 256);
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
            if (!theme.color) {
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
            } else if (!interactive) {
                char msg[256];
                std::snprintf(msg, sizeof(msg),
                              "%s · %s · VRAM %.2f GiB · ctx %s%s · KV %s · loaded in %s",
                              model_display_name.c_str(),
                              dev.gcn_arch_name.c_str(),
                              engine.total_vram_used_bytes() / double(1ULL << 30),
                              rxinfer::tui::format_tokens_compact(max_seq_len).c_str(),
                              has_max_seq_len ? "" : " auto",
                              std::string(rxinfer::dtype_name(kv_dtype)).c_str(),
                              rxinfer::tui::format_duration_compact(load_ms).c_str());
                rxinfer::tui::print_status_badge(stderr, theme, "ok", msg);
            }
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
            if (!has_prompt && !interactive && !serve_mode) return 0;
            engine.reset_state();
        }

        if (!has_prompt && !interactive && !serve_mode) {
            std::fprintf(stderr, "Nothing to run: pass -i/--interactive, --serve, --prompt, --bench, or --save-rxr4.\n");
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
        topts.preserve_thinking = preserve_thinking;
        topts.format_markdown = format_markdown;
        topts.ignore_eos = ignore_eos;
        topts.quiet = quiet;
        topts.interactive = interactive;
        topts.auto_prompt_cache = auto_prompt_cache;
        topts.max_tokens = max_tokens;
        topts.max_seq_len = max_seq_len;
        topts.max_tool_rounds = max_tool_rounds;
        topts.system_prompt = system_prompt;
        topts.reasoning_effort = reasoning_effort;
        topts.auto_cache_dir = auto_cache_dir;
        topts.model_display_name = model_display_name;
        topts.tools = std::move(configured_tools);
        topts.tool_exec.allow_mutating = auto_approve;
        topts.tool_exec.external_exec_cmd = tool_exec_cmd;
        topts.theme = theme;
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
                if (theme.color) {
                    char msg[256];
                    std::snprintf(msg, sizeof(msg), "Loaded prompt cache · %s tok in %s",
                                  rxinfer::tui::format_tokens_compact(loaded_pos).c_str(),
                                  rxinfer::tui::format_duration_compact(
                                      std::chrono::duration<double, std::milli>(tlc1 - tlc0).count())
                                      .c_str());
                    rxinfer::tui::print_status_badge(stderr, theme, "info", msg);
                } else {
                    std::fprintf(stderr, "Loaded prompt cache from %s (%d tok) in %.1f ms\n",
                                 prompt_cache_path.c_str(), loaded_pos,
                                 std::chrono::duration<double, std::milli>(tlc1 - tlc0).count());
                }
            }
            if (loaded_pos > 0) {
                if (serve_mode && engine.position() >= 8) {
                    engine.save_prefix_checkpoint(false);
                }
                bool continue_loaded_turn = interactive;
                if (!interactive && has_prompt) {
                    const std::string first_text =
                        chat_mode ? format_first_turn_prompt(tokenizer, prompt, topts) : prompt;
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

        if (serve_mode) {
            install_interactive_sigint();
            rxinfer::server::ServerOptions srv_opts;
            srv_opts.host = server_host;
            srv_opts.port = server_port;
            srv_opts.api_key = server_api_key;
            srv_opts.model_id = model_display_name;
            srv_opts.default_tools = topts.tools;
            srv_opts.default_system_prompt = system_prompt;
            srv_opts.default_enable_thinking = think_mode;
            srv_opts.default_preserve_thinking = preserve_thinking;
            srv_opts.default_reasoning_effort = reasoning_effort;
            srv_opts.default_max_tokens = max_tokens;
            srv_opts.ignore_eos = ignore_eos;
            srv_opts.quiet = quiet;
            srv_opts.stop_flag = &g_interrupted;
            srv_opts.theme = theme;
            return rxinfer::server::run_openai_server(engine, tokenizer, sampler_cfg, srv_opts);
        }

        const bool stdin_is_tty = (::isatty(STDIN_FILENO) != 0);
        const bool stderr_is_tty = (::isatty(STDERR_FILENO) != 0);

        if (interactive) {
            install_interactive_sigint();
            if (!quiet) {
                const char* tpl_name =
                    chat_mode ? rxinfer::Tokenizer::chat_format_name(tokenizer.detect_chat_format()) : "raw";
                const std::string max_tok_str =
                    (topts.max_tokens > 0) ? std::to_string(topts.max_tokens) : "unlimited";
                const char* cache_desc =
                    auto_prompt_cache ? "auto" : (prompt_cache_path.empty() ? "in-mem LCP" : prompt_cache_path.c_str());
                const std::string tools_desc =
                    topts.tools.empty()
                        ? "off"
                        : (std::to_string(topts.tools.size()) + " (" +
                           std::string(topts.tool_exec.allow_mutating ? "auto-approve" : "confirm-mutating") + ")");

                if (theme.color) {
                    const auto& c = engine.config();
                    char r_model[256], r_dev[256], r_ctx[256], r_chat[320];
                    std::snprintf(r_model, sizeof(r_model), "%s · %dL H=%d · %.2f GiB · loaded in %s",
                                  c.model_type.empty() ? "model" : c.model_type.c_str(),
                                  c.num_hidden_layers, c.hidden_size,
                                  engine.weights().weight_arena.used() / double(1ULL << 30),
                                  rxinfer::tui::format_duration_compact(load_ms).c_str());
                    std::snprintf(r_dev, sizeof(r_dev), "%s (%s) · %d WGPs · %.2f / %.2f GiB VRAM",
                                  dev.name.c_str(), dev.gcn_arch_name.c_str(),
                                  dev.multiprocessor_count,
                                  engine.total_vram_used_bytes() / double(1ULL << 30),
                                  dev.total_vram_bytes / double(1ULL << 30));
                    std::snprintf(r_ctx, sizeof(r_ctx),
                                  "%s tok%s · KV %s (%.2f GiB VRAM + %.2f GiB GTT) · hipGraph %s",
                                  rxinfer::tui::format_tokens_compact(max_seq_len).c_str(),
                                  has_max_seq_len ? "" : " auto",
                                  std::string(rxinfer::dtype_name(kv_dtype)).c_str(),
                                  engine.kv_vram_bytes() / double(1ULL << 30),
                                  engine.kv_gtt_bytes() / double(1ULL << 30),
                                  eopts.enable_hip_graph ? "on" : "off");
                    std::snprintf(r_chat, sizeof(r_chat),
                                  "%s · %s · think %s · tools %s · temp %.2f · max %s · cache %s",
                                  tpl_name,
                                  topts.format_markdown ? "markdown" : "raw",
                                  topts.think_mode ? "on" : "off",
                                  tools_desc.c_str(),
                                  sampler.config().temperature,
                                  max_tok_str.c_str(),
                                  cache_desc);

                    std::vector<std::pair<std::string, std::string>> rows = {
                        {"Model", r_model},
                        {"Device", r_dev},
                        {"Context", r_ctx},
                        {"Chat", r_chat},
                        {"Controls",
                         std::string(theme.muted()) +
                             "Tab complete · ↑↓ history · Alt+Enter newline · /tools · /file · /help" +
                             theme.reset()},
                    };
                    const std::string title = std::string(theme.accent()) + "◆" + theme.reset() + " " +
                                              theme.bold() + "rxinfer" + theme.reset() + " " +
                                              theme.muted() + model_display_name + theme.reset();
                    rxinfer::tui::print_box_card(stderr, theme, title, rows, 92);
                } else {
                    std::fprintf(stderr,
                                 "Interactive chat ready | template=%s | format=%s | think=%s | tools=%s | temp=%.2f top_k=%d top_p=%.2f rep=%.2f | max_tokens=%s | cache=%s\n"
                                 "Type your message, /file <path> to attach large files, /help for commands, /quit or Ctrl-D to exit.\n",
                                 tpl_name,
                                 topts.format_markdown ? "markdown" : "raw",
                                 topts.think_mode ? "on" : "off",
                                 tools_desc.c_str(),
                                 sampler.config().temperature, sampler.config().top_k,
                                 sampler.config().top_p, sampler.config().repeat_penalty,
                                 max_tok_str.c_str(),
                                 cache_desc);
                }
            }
        }

        if (has_prompt) {
            run_turn(engine, tokenizer, sampler, state, prompt, topts, logits_f16_scratch);
            if (!interactive) return 0;
        }

        BracketedPasteGuard paste_guard(interactive && stdin_is_tty && stderr_is_tty);
        rxinfer::tui::InteractiveLineEditor editor(build_interactive_commands());

        auto badge = [&](const char* kind, const std::string& styled_msg, const std::string& plain_msg) {
            if (theme.color) {
                rxinfer::tui::print_status_badge(stderr, theme, kind, styled_msg);
            } else {
                std::fprintf(stderr, "%s\n", plain_msg.c_str());
            }
        };

        std::string pending_multiline;
        std::string line;
        bool in_bracketed_paste = false;
        bool in_block_paste = false;
        std::string block_end_marker;

        while (true) {
            if (stdin_is_tty && stderr_is_tty && !in_block_paste && pending_multiline.empty()) {
                if (!quiet) {
                    std::fputc('\n', stderr);
                    std::fflush(stderr);
                }
                g_interrupted = 0;
                const auto st = editor.read_line(theme, static_cast<int>(state.pending_attachments.size()), line);
                if (st == rxinfer::tui::InteractiveLineEditor::Status::Interrupted) {
                    g_interrupted = 0;
                    pending_multiline.clear();
                    in_bracketed_paste = false;
                    in_block_paste = false;
                    continue;
                }
                if (st == rxinfer::tui::InteractiveLineEditor::Status::Eof) {
                    break;
                }
            } else {
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
                                editor.add_history(pasted);
                                run_turn(engine, tokenizer, sampler, state, pasted, topts, logits_f16_scratch);
                            }
                            continue;
                        }
                        // Single-line paste falls through cleanly
                    }
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
                        editor.add_history(full_block);
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
                        badge("info",
                              "Multi-line mode — paste or type lines, end with '" + block_end_marker + "'",
                              "[Multi-line mode: paste or type lines, end with '" + block_end_marker + "']");
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
            if (!full_input.empty()) editor.add_history(full_input);

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
                        print_interactive_help(theme);
                    } else if (cmd == "/file" || cmd == "/attach") {
                        std::string fpath, fquestion;
                        if (!parse_file_command_args(arg, fpath, fquestion)) {
                            badge("info", "Usage: /file <path> [optional prompt]", "Usage: /file <path> [optional prompt]");
                        } else {
                            std::string content = read_text_file(fpath);
                            const size_t nlines = std::count(content.begin(), content.end(), '\n') + (content.empty() ? 0 : 1);
                            state.pending_attachments.push_back({fpath, std::move(content)});
                            char mbuf[512], pbuf[512];
                            std::snprintf(mbuf, sizeof(mbuf), "Attached %s (%zu lines) from %s",
                                          format_byte_size(state.pending_attachments.back().content.size()).c_str(),
                                          nlines, fpath.c_str());
                            std::snprintf(pbuf, sizeof(pbuf), "[Attached %s (%zu lines) from %s]",
                                          format_byte_size(state.pending_attachments.back().content.size()).c_str(),
                                          nlines, fpath.c_str());
                            badge("ok", mbuf, pbuf);
                            if (!fquestion.empty()) {
                                run_turn(engine, tokenizer, sampler, state, fquestion, topts, logits_f16_scratch);
                            }
                        }
                    } else if (cmd == "/undo") {
                        const int32_t steps = arg.empty() ? 1 : std::max(1, std::stoi(arg));
                        if (state.history.empty()) {
                            badge("info", "Nothing to undo", "[Nothing to undo]");
                        } else {
                            const int32_t actual = std::min<int32_t>(steps, static_cast<int32_t>(state.history.size()));
                            TurnRecord target = state.history[state.history.size() - actual];
                            state.history.resize(state.history.size() - actual);
                            const auto t0 = std::chrono::steady_clock::now();
                            const bool fast = engine.restore_prefix(target.tokens_before);
                            const auto t1 = std::chrono::steady_clock::now();
                            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                            state.pending_tokens = std::move(target.pending_before);
                            state.is_first_turn = target.was_first_turn;
                            state.close_previous_turn = target.was_close_previous;
                            if (target.messages_count_before <= state.messages.size()) {
                                state.messages.resize(target.messages_count_before);
                            }
                            state.next_tool_call_id = target.next_tool_call_id_before;
                            state.pending_attachments.clear();
                            sampler.clear_history();
                            sampler.record_tokens(engine.cached_tokens());
                            char mbuf[256], pbuf[256];
                            std::snprintf(mbuf, sizeof(mbuf), "Rewound %d turn(s) · ctx %d/%d · %s (%s)",
                                          actual, engine.position(), topts.max_seq_len,
                                          rxinfer::tui::format_duration_compact(ms).c_str(),
                                          fast ? "cached snapshot" : "replayed prefix");
                            std::snprintf(pbuf, sizeof(pbuf), "[Rewound %d turn(s) to ctx=%d/%d in %.1f ms%s]",
                                          actual, engine.position(), topts.max_seq_len, ms,
                                          fast ? " (cached snapshot)" : " (replayed prefix)");
                            badge("ok", mbuf, pbuf);
                        }
                    } else if (cmd == "/retry") {
                        if (state.history.empty()) {
                            badge("info", "Nothing to retry", "[Nothing to retry]");
                        } else {
                            TurnRecord last = std::move(state.history.back());
                            state.history.pop_back();
                            engine.restore_prefix(last.tokens_before);
                            state.pending_tokens = std::move(last.pending_before);
                            state.is_first_turn = last.was_first_turn;
                            state.close_previous_turn = last.was_close_previous;
                            if (last.messages_count_before <= state.messages.size()) {
                                state.messages.resize(last.messages_count_before);
                            }
                            state.next_tool_call_id = last.next_tool_call_id_before;
                            state.pending_attachments.clear();
                            sampler.clear_history();
                            sampler.record_tokens(engine.cached_tokens());
                            run_turn(engine, tokenizer, sampler, state, last.user_input, topts, logits_f16_scratch);
                        }
                    } else if (cmd == "/reset" || cmd == "/clear") {
                        state.reset_conversation(engine, sampler);
                        char mbuf[192], pbuf[192];
                        std::snprintf(mbuf, sizeof(mbuf), "Conversation reset · ctx 0/%s (%d LCP snapshot%s)",
                                      rxinfer::tui::format_tokens_compact(topts.max_seq_len).c_str(),
                                      engine.num_prefix_checkpoints(),
                                      engine.num_prefix_checkpoints() == 1 ? "" : "s");
                        std::snprintf(pbuf, sizeof(pbuf), "[Conversation reset. Context 0 / %d (LCP snapshots=%d)]",
                                      topts.max_seq_len, engine.num_prefix_checkpoints());
                        badge("ok", mbuf, pbuf);
                    } else if (cmd == "/save-cache") {
                        const std::string target = arg.empty() ? (auto_prompt_cache ? "" : prompt_cache_path) : expand_user_path(arg);
                        if (target.empty()) {
                            if (engine.position() <= 0) {
                                badge("info", "Nothing to save — context is empty", "[Nothing to save; context is empty]");
                            } else {
                                const std::string dir = auto_cache_dir.empty() ? default_auto_cache_dir() : auto_cache_dir;
                                const auto t0 = std::chrono::steady_clock::now();
                                const std::string saved = save_to_auto_cache(dir, engine, false);
                                const auto t1 = std::chrono::steady_clock::now();
                                const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                                if (!saved.empty()) {
                                    char mbuf[512], pbuf[512];
                                    std::snprintf(mbuf, sizeof(mbuf), "Saved prompt cache (%d tok) to %s in %s",
                                                  engine.position(), saved.c_str(),
                                                  rxinfer::tui::format_duration_compact(ms).c_str());
                                    std::snprintf(pbuf, sizeof(pbuf), "[Saved prompt cache (%d tok) to %s in %.1f ms]",
                                                  engine.position(), saved.c_str(), ms);
                                    badge("ok", mbuf, pbuf);
                                }
                            }
                        } else {
                            const auto t0 = std::chrono::steady_clock::now();
                            engine.save_prompt_cache(target);
                            const auto t1 = std::chrono::steady_clock::now();
                            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                            char mbuf[512], pbuf[512];
                            std::snprintf(mbuf, sizeof(mbuf), "Saved prompt cache (%d tok) to %s in %s",
                                          engine.position(), target.c_str(),
                                          rxinfer::tui::format_duration_compact(ms).c_str());
                            std::snprintf(pbuf, sizeof(pbuf), "[Saved prompt cache (%d tok) to %s in %.1f ms]",
                                          engine.position(), target.c_str(), ms);
                            badge("ok", mbuf, pbuf);
                        }
                    } else if (cmd == "/load-cache") {
                        const std::string target = arg.empty() ? (auto_prompt_cache ? "" : prompt_cache_path) : expand_user_path(arg);
                        if (target.empty()) {
                            badge("info", "Usage: /load-cache <path>", "Usage: /load-cache <path>");
                        } else {
                            const auto t0 = std::chrono::steady_clock::now();
                            const int32_t loaded_pos = engine.load_prompt_cache(target);
                            const auto t1 = std::chrono::steady_clock::now();
                            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                            state.pending_tokens.clear();
                            state.pending_attachments.clear();
                            state.messages.clear();
                            state.next_tool_call_id = 1;
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
                            char mbuf[512], pbuf[512];
                            std::snprintf(mbuf, sizeof(mbuf), "Loaded prompt cache (%d tok) from %s in %s",
                                          loaded_pos, target.c_str(),
                                          rxinfer::tui::format_duration_compact(ms).c_str());
                            std::snprintf(pbuf, sizeof(pbuf), "[Loaded prompt cache (%d tok) from %s in %.1f ms]",
                                          loaded_pos, target.c_str(), ms);
                            badge("ok", mbuf, pbuf);
                        }
                    } else if (cmd == "/cache") {
                        const std::string mode_str =
                            auto_prompt_cache ? ("auto (" + auto_cache_dir + ")")
                                              : (prompt_cache_path.empty() ? "in-mem LCP" : prompt_cache_path);
                        if (theme.color) {
                            const double ctx_frac =
                                static_cast<double>(engine.position()) / static_cast<double>(std::max(1, topts.max_seq_len));
                            char r_ctx[256], r_kv[256], r_snap[128];
                            std::snprintf(r_ctx, sizeof(r_ctx), "%s  %s%d / %d tok%s (%.1f%%)",
                                          rxinfer::tui::render_progress_bar(ctx_frac, 18, theme).c_str(),
                                          theme.bold(), engine.position(), topts.max_seq_len, theme.reset(),
                                          ctx_frac * 100.0);
                            std::snprintf(r_kv, sizeof(r_kv), "VRAM %.2f GiB (%s tok) + GTT %.2f GiB (%s tok)",
                                          engine.kv_vram_bytes() / double(1ULL << 30),
                                          rxinfer::tui::format_tokens_compact(engine.vram_kv_tokens()).c_str(),
                                          engine.kv_gtt_bytes() / double(1ULL << 30),
                                          rxinfer::tui::format_tokens_compact(engine.gtt_kv_tokens()).c_str());
                            std::snprintf(r_snap, sizeof(r_snap), "%d in-memory checkpoint%s",
                                          engine.num_prefix_checkpoints(),
                                          engine.num_prefix_checkpoints() == 1 ? "" : "s");
                            std::vector<std::pair<std::string, std::string>> rows = {
                                {"Live ctx", r_ctx},
                                {"Snapshots", r_snap},
                                {"KV Pool", r_kv},
                                {"Mode", mode_str},
                            };
                            std::fprintf(stderr, "\n");
                            const std::string title = std::string(theme.accent()) + "◆" + theme.reset() + " " +
                                                      theme.bold() + "Prompt Cache" + theme.reset();
                            rxinfer::tui::print_box_card(stderr, theme, title, rows, 90);
                            std::fprintf(stderr, "\n");
                        } else {
                            std::fprintf(stderr,
                                         "[Cache] Live ctx: %d/%d tok | Snapshots: %d | KV Pool: VRAM=%.2f GiB (%d tok) + GTT=%.2f GiB (%d tok) | Mode: %s\n",
                                         engine.position(), topts.max_seq_len,
                                         engine.num_prefix_checkpoints(),
                                         engine.kv_vram_bytes() / double(1ULL << 30), engine.vram_kv_tokens(),
                                         engine.kv_gtt_bytes() / double(1ULL << 30), engine.gtt_kv_tokens(),
                                         mode_str.c_str());
                        }
                    } else if (cmd == "/clear-cache") {
                        engine.clear_prefix_cache();
                        badge("ok", "Cleared in-memory GDN prefix checkpoints", "[Cleared in-memory GDN prefix checkpoints]");
                    } else if (cmd == "/stats") {
                        print_session_stats(state, engine, sampler, theme);
                    } else if (cmd == "/format") {
                        if (arg == "on" || arg == "1" || arg == "true" || arg == "markdown") topts.format_markdown = true;
                        else if (arg == "off" || arg == "0" || arg == "false" || arg == "raw") topts.format_markdown = false;
                        else if (arg.empty()) topts.format_markdown = !topts.format_markdown;
                        else {
                            badge("info", "Usage: /format [on|off]", "Usage: /format [on|off]");
                            continue;
                        }
                        badge("ok",
                              std::string("Output formatting: ") + (topts.format_markdown ? "markdown" : "raw"),
                              std::string("[Output formatting: ") + (topts.format_markdown ? "markdown]" : "raw]"));
                    } else if (cmd == "/think") {
                        if (arg == "on" || arg == "1" || arg == "true") topts.think_mode = true;
                        else if (arg == "off" || arg == "0" || arg == "false") topts.think_mode = false;
                        else if (arg.empty()) topts.think_mode = !topts.think_mode;
                        else {
                            badge("info", "Usage: /think [on|off]", "Usage: /think [on|off]");
                            continue;
                        }
                        badge("ok",
                              std::string("Thinking mode: ") + (topts.think_mode ? "on" : "off"),
                              std::string("[Thinking mode: ") + (topts.think_mode ? "on]" : "off]"));
                    } else if (cmd == "/tools") {
                        const size_t sp2 = arg.find_first_of(" \t");
                        const std::string sub = (sp2 == std::string::npos) ? arg : arg.substr(0, sp2);
                        const std::string subarg = (sp2 == std::string::npos) ? "" : trim_ws(arg.substr(sp2 + 1));
                        auto merge_tools = [&](const std::vector<rxinfer::ToolFunctionDef>& defs) {
                            for (const auto& d : defs) {
                                auto it = std::find_if(topts.tools.begin(), topts.tools.end(),
                                                       [&](const rxinfer::ToolFunctionDef& e) { return e.name == d.name; });
                                if (it != topts.tools.end()) *it = d;
                                else topts.tools.push_back(d);
                            }
                        };
                        if (sub.empty()) {
                            if (topts.tools.empty()) {
                                badge("info",
                                      "No tools enabled — use '/tools builtin' or '/tools load <file.json>'",
                                      "[No tools enabled. Use '/tools builtin' or '/tools load <file.json>']");
                            } else if (theme.color) {
                                std::vector<std::pair<std::string, std::string>> rows;
                                rows.push_back({"Approval",
                                                topts.tool_exec.allow_mutating ? "auto-approve mutating" : "confirm mutating"});
                                rows.push_back({"Max rounds", std::to_string(topts.max_tool_rounds)});
                                for (const auto& t : topts.tools) {
                                    rows.push_back({t.name, truncate_inline_preview(t.description, 72)});
                                }
                                std::fprintf(stderr, "\n");
                                const std::string title = std::string(theme.accent()) + "◆" + theme.reset() + " " +
                                                          theme.bold() + "Active Tools" + theme.reset();
                                rxinfer::tui::print_box_card(stderr, theme, title, rows, 92);
                                std::fprintf(stderr, "\n");
                            } else {
                                std::fprintf(stderr, "[Tools (%zu, mutating=%s, max_rounds=%d)]\n",
                                             topts.tools.size(),
                                             topts.tool_exec.allow_mutating ? "auto-approve" : "confirm",
                                             topts.max_tool_rounds);
                                for (const auto& t : topts.tools) {
                                    std::fprintf(stderr, "  - %s: %s\n", t.name.c_str(),
                                                 truncate_inline_preview(t.description, 72).c_str());
                                }
                            }
                        } else if (sub == "builtin") {
                            merge_tools(rxinfer::tools::builtin_workspace_tools());
                            topts.chat_mode = true;
                            if (!state.is_first_turn || engine.position() > 0) {
                                engine.unpin_prefix_checkpoints();
                                state.reset_conversation(engine, sampler);
                                badge("ok",
                                      "Enabled built-in workspace tools (" + std::to_string(topts.tools.size()) +
                                          ") · conversation reset to apply system prompt",
                                      "[Enabled built-in workspace tools (" + std::to_string(topts.tools.size()) +
                                          "); conversation reset]");
                            } else {
                                badge("ok",
                                      "Enabled built-in workspace tools (" + std::to_string(topts.tools.size()) + ")",
                                      "[Enabled built-in workspace tools (" + std::to_string(topts.tools.size()) + ")]");
                            }
                        } else if (sub == "load") {
                            if (subarg.empty()) {
                                badge("info", "Usage: /tools load <path.json>", "Usage: /tools load <path.json>");
                            } else {
                                auto defs = rxinfer::tools::load_tools_file(expand_user_path(subarg));
                                merge_tools(defs);
                                topts.chat_mode = true;
                                if (!state.is_first_turn || engine.position() > 0) {
                                    engine.unpin_prefix_checkpoints();
                                    state.reset_conversation(engine, sampler);
                                    badge("ok",
                                          "Loaded " + std::to_string(defs.size()) + " tool(s) (" +
                                              std::to_string(topts.tools.size()) + " total) · conversation reset",
                                          "[Loaded " + std::to_string(defs.size()) + " tool(s); conversation reset]");
                                } else {
                                    badge("ok",
                                          "Loaded " + std::to_string(defs.size()) + " tool(s) (" +
                                              std::to_string(topts.tools.size()) + " total)",
                                          "[Loaded " + std::to_string(defs.size()) + " tool(s)]");
                                }
                            }
                        } else if (sub == "clear") {
                            topts.tools.clear();
                            if (!state.is_first_turn || engine.position() > 0) {
                                engine.unpin_prefix_checkpoints();
                                state.reset_conversation(engine, sampler);
                                badge("ok", "Cleared tools · conversation reset", "[Cleared tools; conversation reset]");
                            } else {
                                badge("ok", "Cleared tools", "[Cleared tools]");
                            }
                        } else {
                            badge("info", "Usage: /tools [builtin|load <path.json>|clear]",
                                  "Usage: /tools [builtin|load <path.json>|clear]");
                        }
                    } else if (cmd == "/approve") {
                        if (arg == "on" || arg == "1" || arg == "true") topts.tool_exec.allow_mutating = true;
                        else if (arg == "off" || arg == "0" || arg == "false") topts.tool_exec.allow_mutating = false;
                        else if (arg.empty()) topts.tool_exec.allow_mutating = !topts.tool_exec.allow_mutating;
                        else {
                            badge("info", "Usage: /approve [on|off]", "Usage: /approve [on|off]");
                            continue;
                        }
                        badge("ok",
                              std::string("Auto-approve mutating tools: ") +
                                  (topts.tool_exec.allow_mutating ? "on" : "off"),
                              std::string("[Auto-approve mutating tools: ") +
                                  (topts.tool_exec.allow_mutating ? "on]" : "off]"));
                    } else if (cmd == "/history") {
                        if (state.messages.empty()) {
                            badge("info", "No messages in structured history yet", "[No messages in structured history yet]");
                        } else {
                            std::fprintf(stderr, "\n");
                            for (size_t mi = 0; mi < state.messages.size(); ++mi) {
                                const auto& m = state.messages[mi];
                                if (m.role == "assistant") {
                                    std::fprintf(stderr, "  [%zu] assistant", mi);
                                    if (!m.reasoning_content.empty()) {
                                        std::fprintf(stderr, " | think(%zuB): \"%s\"", m.reasoning_content.size(),
                                                     truncate_inline_preview(m.reasoning_content, 60).c_str());
                                    }
                                    if (!m.tool_calls.empty()) {
                                        std::fprintf(stderr, " | tool_calls=%zu [", m.tool_calls.size());
                                        for (size_t ti = 0; ti < m.tool_calls.size(); ++ti) {
                                            if (ti > 0) std::fprintf(stderr, ", ");
                                            std::fprintf(stderr, "%s", m.tool_calls[ti].name.c_str());
                                        }
                                        std::fprintf(stderr, "]");
                                    }
                                    if (!m.content.empty()) {
                                        std::fprintf(stderr, " | text: \"%s\"",
                                                     truncate_inline_preview(m.content, 60).c_str());
                                    }
                                    std::fprintf(stderr, "\n");
                                } else if (m.role == "tool") {
                                    std::fprintf(stderr, "  [%zu] tool(%s): \"%s\"\n", mi, m.name.c_str(),
                                                 truncate_inline_preview(m.content, 72).c_str());
                                } else {
                                    std::fprintf(stderr, "  [%zu] %s: \"%s\"\n", mi, m.role.c_str(),
                                                 truncate_inline_preview(m.content, 72).c_str());
                                }
                            }
                            std::fprintf(stderr, "\n");
                        }
                    } else if (cmd == "/system") {
                        topts.system_prompt = arg;
                        engine.unpin_prefix_checkpoints();
                        state.reset_conversation(engine, sampler);
                        char mbuf[192], pbuf[192];
                        std::snprintf(mbuf, sizeof(mbuf), "System prompt updated (%zu chars) · conversation reset",
                                      topts.system_prompt.size());
                        std::snprintf(pbuf, sizeof(pbuf), "[System prompt updated (%zu chars); conversation reset]",
                                      topts.system_prompt.size());
                        badge("ok", mbuf, pbuf);
                    } else if (cmd == "/temp") {
                        if (arg.empty()) {
                            char mbuf[96], pbuf[96];
                            std::snprintf(mbuf, sizeof(mbuf), "temp = %.3f", sampler.config().temperature);
                            std::snprintf(pbuf, sizeof(pbuf), "[temp=%.3f]", sampler.config().temperature);
                            badge("info", mbuf, pbuf);
                        } else {
                            auto sc = sampler.config();
                            sc.temperature = std::stof(arg);
                            sampler.set_config(sc);
                            char mbuf[96], pbuf[96];
                            std::snprintf(mbuf, sizeof(mbuf), "temp = %.3f (%s)", sc.temperature,
                                          sc.uses_host_logits() ? "host-sample" : "gpu-argmax");
                            std::snprintf(pbuf, sizeof(pbuf), "[temp=%.3f (%s)]", sc.temperature,
                                          sc.uses_host_logits() ? "host-sample" : "gpu-argmax");
                            badge("ok", mbuf, pbuf);
                        }
                    } else if (cmd == "/top-k") {
                        if (arg.empty()) {
                            char mbuf[96], pbuf[96];
                            std::snprintf(mbuf, sizeof(mbuf), "top_k = %d", sampler.config().top_k);
                            std::snprintf(pbuf, sizeof(pbuf), "[top_k=%d]", sampler.config().top_k);
                            badge("info", mbuf, pbuf);
                        } else {
                            auto sc = sampler.config();
                            sc.top_k = std::stoi(arg);
                            sampler.set_config(sc);
                            char mbuf[96], pbuf[96];
                            std::snprintf(mbuf, sizeof(mbuf), "top_k = %d", sc.top_k);
                            std::snprintf(pbuf, sizeof(pbuf), "[top_k=%d]", sc.top_k);
                            badge("ok", mbuf, pbuf);
                        }
                    } else if (cmd == "/top-p") {
                        if (arg.empty()) {
                            char mbuf[96], pbuf[96];
                            std::snprintf(mbuf, sizeof(mbuf), "top_p = %.3f", sampler.config().top_p);
                            std::snprintf(pbuf, sizeof(pbuf), "[top_p=%.3f]", sampler.config().top_p);
                            badge("info", mbuf, pbuf);
                        } else {
                            auto sc = sampler.config();
                            sc.top_p = std::stof(arg);
                            sampler.set_config(sc);
                            char mbuf[96], pbuf[96];
                            std::snprintf(mbuf, sizeof(mbuf), "top_p = %.3f", sc.top_p);
                            std::snprintf(pbuf, sizeof(pbuf), "[top_p=%.3f]", sc.top_p);
                            badge("ok", mbuf, pbuf);
                        }
                    } else if (cmd == "/min-p") {
                        if (arg.empty()) {
                            char mbuf[96], pbuf[96];
                            std::snprintf(mbuf, sizeof(mbuf), "min_p = %.3f", sampler.config().min_p);
                            std::snprintf(pbuf, sizeof(pbuf), "[min_p=%.3f]", sampler.config().min_p);
                            badge("info", mbuf, pbuf);
                        } else {
                            auto sc = sampler.config();
                            sc.min_p = std::stof(arg);
                            sampler.set_config(sc);
                            char mbuf[96], pbuf[96];
                            std::snprintf(mbuf, sizeof(mbuf), "min_p = %.3f", sc.min_p);
                            std::snprintf(pbuf, sizeof(pbuf), "[min_p=%.3f]", sc.min_p);
                            badge("ok", mbuf, pbuf);
                        }
                    } else if (cmd == "/repeat-penalty") {
                        if (arg.empty()) {
                            char mbuf[96], pbuf[96];
                            std::snprintf(mbuf, sizeof(mbuf), "repeat_penalty = %.3f", sampler.config().repeat_penalty);
                            std::snprintf(pbuf, sizeof(pbuf), "[repeat_penalty=%.3f]", sampler.config().repeat_penalty);
                            badge("info", mbuf, pbuf);
                        } else {
                            auto sc = sampler.config();
                            sc.repeat_penalty = std::stof(arg);
                            sampler.set_config(sc);
                            char mbuf[96], pbuf[96];
                            std::snprintf(mbuf, sizeof(mbuf), "repeat_penalty = %.3f", sc.repeat_penalty);
                            std::snprintf(pbuf, sizeof(pbuf), "[repeat_penalty=%.3f]", sc.repeat_penalty);
                            badge("ok", mbuf, pbuf);
                        }
                    } else if (cmd == "/max-tokens") {
                        auto fmt_lim = [](int32_t v) -> std::string {
                            return (v > 0) ? std::to_string(v) : "unlimited";
                        };
                        if (arg.empty()) {
                            const std::string ls = fmt_lim(topts.max_tokens);
                            badge("info", "max_tokens = " + ls, "[max_tokens=" + ls + "]");
                        } else if (arg == "unlimited" || arg == "off" || arg == "none" || arg == "inf" ||
                                   arg == "0" || arg == "-1") {
                            topts.max_tokens = 0;
                            badge("ok", "max_tokens = unlimited", "[max_tokens=unlimited]");
                        } else {
                            const int32_t v = std::stoi(arg);
                            if (v < -1) {
                                badge("warn", "max_tokens must be >= 0 (or 'unlimited')", "[max_tokens must be >= 0]");
                            } else {
                                topts.max_tokens = v;
                                const std::string ls = fmt_lim(topts.max_tokens);
                                badge("ok", "max_tokens = " + ls, "[max_tokens=" + ls + "]");
                            }
                        }
                    } else if (cmd == "/seed") {
                        if (arg.empty()) {
                            char mbuf[96], pbuf[96];
                            std::snprintf(mbuf, sizeof(mbuf), "seed = %llu",
                                          static_cast<unsigned long long>(sampler.config().seed));
                            std::snprintf(pbuf, sizeof(pbuf), "[seed=%llu]",
                                          static_cast<unsigned long long>(sampler.config().seed));
                            badge("info", mbuf, pbuf);
                        } else {
                            sampler.reseed(std::stoull(arg));
                            char mbuf[96], pbuf[96];
                            std::snprintf(mbuf, sizeof(mbuf), "seed = %llu",
                                          static_cast<unsigned long long>(sampler.config().seed));
                            std::snprintf(pbuf, sizeof(pbuf), "[seed=%llu]",
                                          static_cast<unsigned long long>(sampler.config().seed));
                            badge("ok", mbuf, pbuf);
                        }
                    } else if (cmd == "/bench") {
                        const int pre_len = std::min(512, std::max(32, max_seq_len / 2));
                        auto pre = engine.benchmark_prefill(pre_len, 1);
                        auto dec = engine.benchmark_decode(60, 8);
                        auto ver4 = engine.benchmark_verify(4, 30, 5);
                        if (theme.color) {
                            char r_pre[128], r_dec[128], r_ver[128];
                            std::snprintf(r_pre, sizeof(r_pre), "%.1f tok/s (%.1f ms for %d tok)",
                                          pre.tok_per_sec, pre.elapsed_ms, pre_len);
                            std::snprintf(r_dec, sizeof(r_dec), "%s%.2f tok/s%s (%.2f ms/tok)",
                                          theme.ok(), dec.tok_per_sec, theme.reset(), dec.ms_per_tok);
                            std::snprintf(r_ver, sizeof(r_ver), "%.2f tok/s (M=4)", ver4.tok_per_sec);
                            std::vector<std::pair<std::string, std::string>> rows = {
                                {"Prefill", r_pre},
                                {"Decode", r_dec},
                                {"Verify", r_ver},
                            };
                            std::fprintf(stderr, "\n");
                            const std::string title = std::string(theme.accent()) + "◆" + theme.reset() + " " +
                                                      theme.bold() + "Benchmark" + theme.reset();
                            rxinfer::tui::print_box_card(stderr, theme, title, rows, 76);
                            state.reset_conversation(engine, sampler);
                            rxinfer::tui::print_status_badge(stderr, theme, "info", "Conversation reset after benchmark");
                        } else {
                            std::fprintf(stderr,
                                         "[Benchmark | prefill(%d)=%.1f tok/s (%.1f ms) | decode(M=1)=%.2f tok/s (%.2f ms/tok) | verify(M=4)=%.2f tok/s]\n",
                                         pre_len, pre.tok_per_sec, pre.elapsed_ms,
                                         dec.tok_per_sec, dec.ms_per_tok, ver4.tok_per_sec);
                            state.reset_conversation(engine, sampler);
                            std::fprintf(stderr, "[Benchmark complete; conversation reset]\n");
                        }
                    } else {
                        const std::string sug = editor.suggest_command(cmd);
                        if (!sug.empty()) {
                            badge("warn",
                                  "Unknown command '" + cmd + "' — did you mean " + sug + "? (type /help)",
                                  "[Unknown command '" + cmd + "'. Did you mean '" + sug + "'? Type /help for commands.]");
                        } else {
                            badge("warn",
                                  "Unknown command '" + cmd + "' — type /help for commands",
                                  "[Unknown command '" + cmd + "'. Type /help for commands.]");
                        }
                    }
                } catch (const std::exception& ex) {
                    badge("err", std::string("Command error: ") + ex.what(), std::string("[Command error: ") + ex.what() + "]");
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
                    const double ms = std::chrono::duration<double, std::milli>(tsc1 - tsc0).count();
                    char mbuf[512], pbuf[512];
                    std::snprintf(mbuf, sizeof(mbuf), "Saved prompt cache (%d tok) to %s in %s",
                                  engine.position(), target.c_str(), rxinfer::tui::format_duration_compact(ms).c_str());
                    std::snprintf(pbuf, sizeof(pbuf), "[Saved prompt cache (%d tok) to %s in %.1f ms]",
                                  engine.position(), target.c_str(), ms);
                    badge("ok", mbuf, pbuf);
                }
            } else if (auto_prompt_cache && !auto_cache_dir.empty()) {
                const auto tsc0 = std::chrono::steady_clock::now();
                const std::string saved = save_to_auto_cache(auto_cache_dir, engine, quiet);
                const auto tsc1 = std::chrono::steady_clock::now();
                if (!saved.empty() && !quiet) {
                    const double ms = std::chrono::duration<double, std::milli>(tsc1 - tsc0).count();
                    char mbuf[512], pbuf[512];
                    std::snprintf(mbuf, sizeof(mbuf), "Auto-cached session (%d tok) in %s",
                                  engine.position(), rxinfer::tui::format_duration_compact(ms).c_str());
                    std::snprintf(pbuf, sizeof(pbuf), "[Auto-cached session (%d tok) to %s in %.1f ms]",
                                  engine.position(), saved.c_str(), ms);
                    badge("info", mbuf, pbuf);
                }
            } else if (!prompt_cache_path.empty()) {
                const auto tsc0 = std::chrono::steady_clock::now();
                engine.save_prompt_cache(prompt_cache_path);
                const auto tsc1 = std::chrono::steady_clock::now();
                if (!quiet) {
                    const double ms = std::chrono::duration<double, std::milli>(tsc1 - tsc0).count();
                    char mbuf[512], pbuf[512];
                    std::snprintf(mbuf, sizeof(mbuf), "Saved prompt cache (%d tok) to %s in %s",
                                  engine.position(), prompt_cache_path.c_str(),
                                  rxinfer::tui::format_duration_compact(ms).c_str());
                    std::snprintf(pbuf, sizeof(pbuf), "[Saved prompt cache (%d tok) to %s in %.1f ms]",
                                  engine.position(), prompt_cache_path.c_str(), ms);
                    badge("ok", mbuf, pbuf);
                }
            }
        }

        if (!quiet && interactive && state.turns > 0) {
            const double sess_decode_tps =
                (state.total_decode_ms > 0.0) ? (state.total_decode_calls * 1000.0 / state.total_decode_ms) : 0.0;
            if (theme.color) {
                char mbuf[256];
                std::snprintf(mbuf, sizeof(mbuf), "Session ended · %d turn%s · %lld prompt + %lld gen tok · avg %.2f tok/s",
                              state.turns, state.turns == 1 ? "" : "s",
                              static_cast<long long>(state.total_prompt_tokens),
                              static_cast<long long>(state.total_generated_tokens),
                              sess_decode_tps);
                rxinfer::tui::print_status_badge(stderr, theme, "info", mbuf);
            } else {
                std::fprintf(stderr,
                             "[Session: %d turns | prompt=%lld tok | gen=%lld tok | avg decode=%.2f tok/s]\n",
                             state.turns,
                             static_cast<long long>(state.total_prompt_tokens),
                             static_cast<long long>(state.total_generated_tokens),
                             sess_decode_tps);
            }
        }

        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "rxinfer error: %s\n", ex.what());
        return 1;
    }
}

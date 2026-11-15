// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/common.hpp"
#include "rxinfer/config.hpp"
#include "rxinfer/engine.hpp"
#include "rxinfer/loader.hpp"
#include "rxinfer/sampler.hpp"
#include "rxinfer/tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
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

std::string read_text_file(const std::string& path) {
    std::ifstream ifs(path);
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

rxinfer::DType parse_kv_dtype(const std::string& s) {
    if (s == "f16" || s == "fp16") return rxinfer::DType::F16;
    if (s == "fp8" || s == "fp8_e4m3") return rxinfer::DType::FP8_E4M3;
    if (s == "fp4" || s == "fp4_e2m1" || s == "fp4_e2m1_g16") return rxinfer::DType::FP4_E2M1_G16;
    throw std::runtime_error("Unsupported --kv-dtype '" + s + "' (expected f16, fp8, or fp4)");
}

// Buffers incomplete trailing UTF-8 byte sequences when streaming BPE tokens.
class Utf8StreamPrinter {
public:
    void push(const std::string& chunk) {
        buf_ += chunk;
        flush_complete(false);
    }
    void finish() {
        flush_complete(true);
        std::fflush(stdout);
    }

private:
    std::string buf_;

    void flush_complete(bool force_all) {
        if (buf_.empty()) return;
        if (force_all) {
            std::fwrite(buf_.data(), 1, buf_.size(), stdout);
            buf_.clear();
            return;
        }
        size_t safe = 0;
        while (safe < buf_.size()) {
            const uint8_t b = static_cast<uint8_t>(buf_[safe]);
            size_t need = 1;
            if ((b & 0x80u) == 0) need = 1;
            else if ((b & 0xE0u) == 0xC0u) need = 2;
            else if ((b & 0xF0u) == 0xE0u) need = 3;
            else if ((b & 0xF8u) == 0xF0u) need = 4;
            if (safe + need > buf_.size()) break;
            safe += need;
        }
        if (safe > 0) {
            std::fwrite(buf_.data(), 1, safe, stdout);
            std::fflush(stdout);
            buf_.erase(0, safe);
        }
    }
};

void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s --model <hf_dir|model.gguf|model.rxr4|synthetic> [options]\n"
                 "Options:\n"
                 "  --tokenizer <path>        Path to tokenizer.json (auto-detected if omitted)\n"
                 "  --synthetic-config <json> HF config.json used when --model synthetic\n"
                 "  -i, --interactive         Interactive multi-turn chat (keeps model hot in VRAM)\n"
                 "  --prompt <text>           Prompt text to generate from (or Turn 1 when paired with -i)\n"
                 "  --prompt-file <path>      Read prompt text from file\n"
                 "  --system <text>           Optional system prompt\n"
                 "  --chat                    Wrap prompt in chat template (default ON in -i)\n"
                 "  --no-chat                 Disable chat template (raw continuation mode)\n"
                 "  --think                   Enable <think> mode in chat template\n"
                 "  --max-tokens <N>          Max new tokens per turn (default: 256, or 1024 in -i)\n"
                 "  --max-seq-len <N>         Max context length (default: 4096)\n"
                 "  --max-batch-tokens <N>    Prefill chunk size (default: 256)\n"
                 "  --kv-dtype <f16|fp8|fp4>  KV cache dtype (default: f16)\n"
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
                 "  /reset, /clear          Clear KV + GDN state and start a new conversation\n"
                 "  /stats                  Show context usage, VRAM, and true TPS stats\n"
                 "  /think [on|off]         Toggle or set <think> mode\n"
                 "  /system <text>          Set system prompt (resets conversation)\n"
                 "  /temp <F>               Set sampling temperature (0 = greedy GPU argmax)\n"
                 "  /top-k <N>              Set top-k (0 = off)\n"
                 "  /top-p <F>              Set top-p (1.0 = off)\n"
                 "  /min-p <F>              Set min-p (0.0 = off)\n"
                 "  /repeat-penalty <F>     Set repetition penalty (1.0 = off)\n"
                 "  /max-tokens <N>         Set max tokens generated per turn\n"
                 "  /seed <N>               Set sampling seed (0 = random)\n"
                 "  /bench                  Run prefill/decode/verify benchmark (resets state)\n"
                 "  /quit, /exit, /q        Exit (or Ctrl-D; Ctrl-C interrupts generation)\n"
                 "Tip: End a line with '\\' to continue onto a multi-line message.\n\n");
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

struct ChatSessionState {
    bool is_first_turn = true;
    bool close_previous_turn = false;
    std::vector<int32_t> pending_tokens;

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
    }
};

struct TurnOptions {
    bool chat_mode = false;
    bool think_mode = false;
    bool ignore_eos = false;
    bool quiet = false;
    bool interactive = false;
    int32_t max_tokens = 256;
    int32_t max_seq_len = 4096;
    std::string system_prompt;
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
                 "        Sampler: temp=%.2f top_k=%d top_p=%.2f min_p=%.2f rep=%.2f path=%s\n"
                 "        Session: %d turns | prompt=%lld tok (%.1f tok/s) | gen=%lld tok | decode=%lld steps (%.2f tok/s, %.2f ms/tok)\n",
                 engine.position(), engine.options().max_seq_len,
                 100.0 * engine.position() / std::max(1, engine.options().max_seq_len),
                 engine.total_vram_used_bytes() / double(1ULL << 30),
                 engine.weights().weight_arena.used() / double(1ULL << 30),
                 engine.state_vram_bytes() / double(1ULL << 30),
                 engine.scratch_vram_bytes() / double(1ULL << 30),
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

bool run_turn(rxinfer::InferenceEngine& engine,
              const rxinfer::Tokenizer& tokenizer,
              rxinfer::Sampler& sampler,
              ChatSessionState& state,
              const std::string& input_text,
              const TurnOptions& opts,
              std::vector<uint16_t>& logits_f16_scratch) {
    std::string delta_text;
    if (opts.chat_mode) {
        delta_text = tokenizer.format_chat_turn(
            input_text, opts.system_prompt, state.is_first_turn, state.close_previous_turn, opts.think_mode);
    } else {
        delta_text = (state.is_first_turn || input_text.empty()) ? input_text : ("\n" + input_text);
    }

    const bool add_bos = state.is_first_turn && state.pending_tokens.empty() && tokenizer.should_add_bos();
    std::vector<int32_t> encoded = tokenizer.encode(delta_text, false);
    if (add_bos && tokenizer.bos_id() >= 0 &&
        (encoded.empty() || encoded[0] != tokenizer.bos_id())) {
        encoded.insert(encoded.begin(), tokenizer.bos_id());
    }

    std::vector<int32_t> turn_ids = state.pending_tokens;
    turn_ids.insert(turn_ids.end(), encoded.begin(), encoded.end());

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

    sampler.record_tokens(encoded);
    const int32_t vocab_size = engine.config().vocab_size;

    const auto tp0 = std::chrono::steady_clock::now();
    int32_t next_tok = engine.prefill(turn_ids);
    if (sampler.config().uses_host_logits()) {
        engine.copy_last_logits_f16(logits_f16_scratch);
        next_tok = sampler.sample_from_f16_bits(logits_f16_scratch.data(), vocab_size);
    }
    const auto tp1 = std::chrono::steady_clock::now();
    const double prefill_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    state.pending_tokens.clear();
    state.is_first_turn = false;
    state.close_previous_turn = opts.chat_mode;

    Utf8StreamPrinter printer;
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

    if (interrupted) {
        g_interrupted = 0;
        if (!opts.quiet) std::fprintf(stderr, "[interrupted]\n");
    } else if (hit_ctx_limit && !opts.quiet) {
        std::fprintf(stderr, "[context limit %d reached; use /reset for a new conversation]\n", opts.max_seq_len);
    }

    state.turns++;
    state.last_prompt_tokens = static_cast<int32_t>(turn_ids.size());
    state.last_generated_tokens = generated;
    state.last_decode_calls = decode_calls;
    state.last_prefill_ms = prefill_ms;
    state.last_decode_ms = decode_ms;

    state.total_prompt_tokens += turn_ids.size();
    state.total_generated_tokens += generated;
    state.total_decode_calls += decode_calls;
    state.total_prefill_ms += prefill_ms;
    state.total_decode_ms += decode_ms;

    if (!opts.quiet) {
        const double p_tps = (turn_ids.size() * 1000.0) / std::max(0.001, prefill_ms);
        const double total_ms = prefill_ms + decode_ms;
        const double e2e_tps = (generated > 0) ? (generated * 1000.0 / std::max(0.001, total_ms)) : 0.0;
        if (decode_calls > 0) {
            const double d_tps = (decode_calls * 1000.0) / std::max(0.001, decode_ms);
            const double ms_per_tok = decode_ms / decode_calls;
            std::fprintf(stderr,
                         "[Prompt: %zu tok in %.1f ms (%.1f tok/s, TTFT %.1f ms) | Decode: %d steps in %.1f ms (%.2f tok/s, %.2f ms/tok) | Out: %d tok (E2E %.2f tok/s) | Ctx: %d/%d]\n",
                         turn_ids.size(), prefill_ms, p_tps, prefill_ms,
                         decode_calls, decode_ms, d_tps, ms_per_tok,
                         generated, e2e_tps, engine.position(), opts.max_seq_len);
        } else {
            std::fprintf(stderr,
                         "[Prompt: %zu tok in %.1f ms (%.1f tok/s, TTFT %.1f ms) | Out: %d tok (0 decode steps) | Ctx: %d/%d]\n",
                         turn_ids.size(), prefill_ms, p_tps, prefill_ms,
                         generated, engine.position(), opts.max_seq_len);
        }
    }

    return true;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string model_path;
        std::string tokenizer_path;
        std::string synthetic_cfg_path;
        std::string prompt;
        std::string prompt_file;
        std::string system_prompt;
        std::string save_rxr4_path;
        bool has_prompt = false;
        bool interactive = false;
        bool chat_flag_set = false;
        bool chat_flag_val = false;
        bool think_mode = false;
        bool run_bench = false;
        bool no_graph = false;
        bool ignore_eos = false;
        bool quiet = false;
        bool has_max_tokens = false;
        int32_t max_tokens = 256;
        int32_t max_seq_len = 4096;
        int32_t max_batch_tokens = 256;
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
                prompt = need_val("--prompt");
                has_prompt = true;
            } else if (arg == "--prompt-file") {
                prompt_file = need_val("--prompt-file");
                has_prompt = true;
            } else if (arg == "--system") {
                system_prompt = need_val("--system");
            } else if (arg == "--chat") {
                chat_flag_set = true;
                chat_flag_val = true;
            } else if (arg == "--no-chat") {
                chat_flag_set = true;
                chat_flag_val = false;
            } else if (arg == "--think") {
                think_mode = true;
            } else if (arg == "--max-tokens") {
                max_tokens = std::stoi(need_val("--max-tokens"));
                has_max_tokens = true;
            } else if (arg == "--max-seq-len") {
                max_seq_len = std::stoi(need_val("--max-seq-len"));
            } else if (arg == "--max-batch-tokens") {
                max_batch_tokens = std::stoi(need_val("--max-batch-tokens"));
            } else if (arg == "--kv-dtype") {
                kv_dtype = parse_kv_dtype(need_val("--kv-dtype"));
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
        if (!prompt_file.empty()) {
            prompt = read_text_file(prompt_file);
        }
        if (interactive && !has_max_tokens) {
            max_tokens = 1024;
        }
        if (max_tokens < 1) {
            throw std::runtime_error("--max-tokens must be >= 1");
        }

        const bool chat_mode = chat_flag_set ? chat_flag_val : interactive;

        const auto dev = rxinfer::query_device_info(0);
        if (!quiet) {
            std::fprintf(stderr, "rxinfer | GPU: %s (%s) | VRAM: %.2f GiB | WGPs: %d\n",
                         dev.name.c_str(), dev.gcn_arch_name.c_str(),
                         dev.total_vram_bytes / double(1ULL << 30),
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
                const std::string cand = parent_dir(model_path) + "/tokenizer.json";
                if (path_exists(cand)) tokenizer_path = cand;
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

        rxinfer::EngineOptions eopts{};
        eopts.max_seq_len = max_seq_len;
        eopts.max_batch_tokens = max_batch_tokens;
        eopts.kv_dtype = kv_dtype;
        eopts.enable_hip_graph = !no_graph;

        rxinfer::InferenceEngine engine(std::move(weights), eopts);
        max_seq_len = engine.options().max_seq_len;

        if (!quiet) {
            std::fprintf(stderr, "Total allocated VRAM: %.2f GiB (kv=%s, hipGraph=%s, max_seq=%d)\n",
                         engine.total_vram_used_bytes() / double(1ULL << 30),
                         std::string(rxinfer::dtype_name(kv_dtype)).c_str(),
                         eopts.enable_hip_graph ? "on" : "off",
                         max_seq_len);
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
        topts.ignore_eos = ignore_eos;
        topts.quiet = quiet;
        topts.interactive = interactive;
        topts.max_tokens = max_tokens;
        topts.max_seq_len = max_seq_len;
        topts.system_prompt = system_prompt;
        std::vector<uint16_t> logits_f16_scratch;

        if (interactive) {
            install_interactive_sigint();
            if (!quiet) {
                const char* tpl_name =
                    chat_mode ? rxinfer::Tokenizer::chat_format_name(tokenizer.detect_chat_format()) : "raw";
                std::fprintf(stderr,
                             "Interactive chat ready | template=%s | think=%s | temp=%.2f top_k=%d top_p=%.2f rep=%.2f | max_tokens=%d\n"
                             "Type your message, /help for commands, /quit or Ctrl-D to exit.\n",
                             tpl_name, topts.think_mode ? "on" : "off",
                             sampler.config().temperature, sampler.config().top_k,
                             sampler.config().top_p, sampler.config().repeat_penalty,
                             topts.max_tokens);
            }
        }

        if (has_prompt) {
            run_turn(engine, tokenizer, sampler, state, prompt, topts, logits_f16_scratch);
            if (!interactive) return 0;
        }

        std::string pending_multiline;
        std::string line;

        while (true) {
            if (!quiet) {
                std::fprintf(stderr, "%s", pending_multiline.empty() ? "\n> " : ".. ");
                std::fflush(stderr);
            }

            g_interrupted = 0;
            if (!std::getline(std::cin, line)) {
                if (g_interrupted) {
                    g_interrupted = 0;
                    pending_multiline.clear();
                    std::cin.clear();
                    std::fprintf(stderr, "\n");
                    continue;
                }
                break;
            }

            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
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

            if (full_input.empty()) continue;

            // Escape leading slash with "\/..." to send literal slash text.
            if (full_input.size() >= 2 && full_input[0] == '\\' && full_input[1] == '/') {
                full_input.erase(0, 1);
            } else if (full_input[0] == '/') {
                const std::string cmd_trimmed = trim_ws(full_input);
                const size_t sp = cmd_trimmed.find_first_of(" \t");
                const std::string cmd = (sp == std::string::npos) ? cmd_trimmed : cmd_trimmed.substr(0, sp);
                const std::string arg = (sp == std::string::npos) ? "" : trim_ws(cmd_trimmed.substr(sp + 1));

                try {
                    if (cmd == "/quit" || cmd == "/exit" || cmd == "/q") {
                        break;
                    } else if (cmd == "/help" || cmd == "/?") {
                        print_interactive_help();
                    } else if (cmd == "/reset" || cmd == "/clear") {
                        state.reset_conversation(engine, sampler);
                        std::fprintf(stderr, "[Conversation reset. Context 0 / %d]\n", topts.max_seq_len);
                    } else if (cmd == "/stats") {
                        print_session_stats(state, engine, sampler);
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
                    std::fprintf(stderr, "[Invalid command argument: %s]\n", ex.what());
                }
                continue;
            }

            run_turn(engine, tokenizer, sampler, state, full_input, topts, logits_f16_scratch);
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

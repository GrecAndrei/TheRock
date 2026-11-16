// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rxinfer/engine.hpp"
#include "rxinfer/sampler.hpp"
#include "rxinfer/tokenizer.hpp"
#include "rxinfer/tools.hpp"
#include "rxinfer/tui.hpp"

#include <csignal>
#include <cstdint>
#include <string>
#include <vector>

namespace rxinfer {
namespace server {

struct ChatCompletionRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    std::vector<bool> message_had_explicit_reasoning;
    std::vector<ToolFunctionDef> tools;
    std::string tool_choice = "auto"; // "auto", "none", "required", "function"
    std::string forced_tool_name;
    bool stream = false;
    bool stream_include_usage = false;

    bool has_temperature = false;
    float temperature = 0.0f;
    bool has_top_k = false;
    int32_t top_k = 40;
    bool has_top_p = false;
    float top_p = 0.95f;
    bool has_min_p = false;
    float min_p = 0.0f;
    bool has_repeat_penalty = false;
    float repeat_penalty = 1.0f;
    bool has_presence_penalty = false;
    float presence_penalty = 0.0f;
    bool has_seed = false;
    uint64_t seed = 0;

    bool has_max_tokens = false;
    int32_t max_tokens = 0;

    bool has_enable_thinking = false;
    bool enable_thinking = true;
    bool has_preserve_thinking = false;
    bool preserve_thinking = true;
    std::string reasoning_effort;

    bool ignore_eos = false;
    std::vector<std::string> stop;
};

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;
    std::string model_id = "rxinfer";
    std::vector<ToolFunctionDef> default_tools;
    std::string default_system_prompt;
    bool default_enable_thinking = true;
    bool default_preserve_thinking = true;
    std::string default_reasoning_effort;
    int32_t default_max_tokens = 0;
    bool ignore_eos = false;
    bool quiet = false;
    const volatile std::sig_atomic_t* stop_flag = nullptr;
    rxinfer::tui::UiTheme theme{};
};

// Parse an OpenAI /v1/chat/completions JSON request body. Throws std::runtime_error on invalid JSON/schema.
ChatCompletionRequest parse_chat_completion_request(const std::string& json_body);

// Build an OpenAI chat.completion JSON response.
std::string build_chat_completion_json(const std::string& id,
                                       int64_t created,
                                       const std::string& model,
                                       const tools::ParsedAssistantOutput& parsed,
                                       const std::string& finish_reason,
                                       int32_t prompt_tokens,
                                       int32_t cached_tokens,
                                       int32_t completion_tokens);

// Build an OpenAI chat.completion.chunk JSON payload (without "data: " prefix).
std::string build_chat_completion_chunk_json(const std::string& id,
                                             int64_t created,
                                             const std::string& model,
                                             bool include_role,
                                             const std::string& reasoning_delta,
                                             const std::string& content_delta,
                                             const std::vector<ToolCall>& tool_call_deltas,
                                             int tool_call_base_index,
                                             const std::string& finish_reason);

// Run blocking OpenAI-compatible HTTP server until `stop_flag` becomes non-zero.
int run_openai_server(InferenceEngine& engine,
                      const Tokenizer& tokenizer,
                      const SamplerConfig& base_sampler_cfg,
                      const ServerOptions& opts);

} // namespace server
} // namespace rxinfer

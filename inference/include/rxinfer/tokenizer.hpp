// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rxinfer {

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments; // JSON object string
};

struct ToolFunctionDef {
    std::string name;
    std::string description;
    std::string parameters_json; // JSON schema object string
    std::string raw_json;        // Optional verbatim OpenAI tool JSON entry
};

struct ChatMessage {
    std::string role;              // "system", "developer", "user", "assistant", "tool"
    std::string content;           // Visible text content
    std::string reasoning_content; // Optional thinking content (served back in <think>...</think>)
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
    std::string name;
    std::string raw_assistant_body; // Optional exact server-captured assistant text for 100% KV-prefix reuse
};

struct ChatTemplateOptions {
    bool add_generation_prompt = true;
    bool enable_thinking = false;
    bool preserve_thinking = true; // Preserve <think> across turns AND after tool calls
    std::string reasoning_effort;  // Optional: "xhigh", "medium", "low"
};

class Tokenizer {
public:
    Tokenizer() = default;

    // Load from HuggingFace tokenizer.json (and optional sibling tokenizer_config.json / config.json).
    static Tokenizer from_hf_json_file(const std::string& tokenizer_json_path);
    static Tokenizer from_hf_json_string(const std::string& json_text);

    // Load embedded tokenizer metadata directly from a GGUF v2/v3 file.
    static Tokenizer from_gguf(const std::string& gguf_path);

    // Encode text to token IDs (matches added/special tokens verbatim).
    std::vector<int32_t> encode(const std::string& text, bool add_bos = false) const;

    // Decode token IDs back to UTF-8 text.
    std::string decode(const std::vector<int32_t>& ids, bool skip_special_tokens = true) const;
    std::string decode_token(int32_t id, bool skip_special_tokens = true) const;

    // Format a single-turn or multi-turn ChatML prompt (Qwen2/3/3.5 style).
    std::string format_chatml(const std::string& user_prompt,
                              const std::string& system_prompt = "",
                              bool add_generation_prompt = true,
                              bool enable_thinking = false) const;

    enum class ChatFormat : uint8_t {
        ChatML = 0, // <|im_start|>role\n...<|im_end|>\n
        Llama3 = 1, // <|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>
    };

    ChatFormat detect_chat_format() const;
    static const char* chat_format_name(ChatFormat fmt);

    // Format either Turn 0 (is_first_turn=true) or an incremental Turn N>0 delta
    // (closing the previous assistant turn when close_previous_turn=true).
    std::string format_system_prefix(const std::string& system_prompt) const;
    std::string format_system_prompt(const std::string& system_prompt,
                                     const std::vector<ToolFunctionDef>& tools = {},
                                     const ChatTemplateOptions& opts = {}) const;
    std::string format_chat_turn(const std::string& user_prompt,
                                 const std::string& system_prompt = "",
                                 bool is_first_turn = true,
                                 bool close_previous_turn = false,
                                 bool enable_thinking = false) const;

    // Format full multi-turn / agentic conversation with optional tools, tool calls,
    // tool responses, and preserved reasoning (<think>...</think>).
    std::string format_chat_messages(const std::vector<ChatMessage>& messages,
                                     const std::vector<ToolFunctionDef>& tools = {},
                                     const ChatTemplateOptions& opts = {}) const;

    // Incremental delta for appending tool response(s) onto an open assistant turn in KV cache.
    std::string format_tool_response_turn(const std::vector<ChatMessage>& tool_messages,
                                          bool close_previous_turn = true,
                                          bool enable_thinking = false) const;

    // Extract any <think>...</think> reasoning from `text` into `out_reasoning` and `out_visible`.
    static void split_thinking(const std::string& text,
                               std::string& out_reasoning,
                               std::string& out_visible);

    bool loaded() const { return !id_to_token_.empty(); }
    int32_t vocab_size() const { return static_cast<int32_t>(id_to_token_.size()); }
    int32_t bos_id() const { return bos_id_; }
    bool should_add_bos() const { return add_bos_token_ && bos_id_ >= 0; }
    int32_t eos_id() const { return eos_ids_.empty() ? -1 : eos_ids_[0]; }
    const std::vector<int32_t>& eos_ids() const { return eos_ids_; }
    bool is_eos(int32_t id) const;
    bool is_special(int32_t id) const { return special_ids_.count(id) > 0; }

    int32_t token_to_id(const std::string& tok) const;
    const std::string& id_to_token(int32_t id) const;

    void set_bos_id(int32_t id) { bos_id_ = id; }
    void set_add_bos_token(bool v) { add_bos_token_ = v; }
    void add_eos_id(int32_t id);

private:
    struct MergeEntry {
        int32_t rank = 0;
        int32_t merged_id = -1;
    };
    struct AddedToken {
        std::string content;
        int32_t id = -1;
        bool special = false;
    };

    bool byte_level_ = false;
    bool metaspace_ = false;
    bool byte_fallback_ = false;
    bool add_prefix_space_ = false;
    bool add_bos_token_ = false;
    bool use_scores_bpe_ = false;
    int max_digit_chunk_ = 1; // 1 for Qwen2/3/3.5, 3 for Llama3

    int32_t bos_id_ = -1;
    int32_t unk_id_ = -1;
    std::vector<int32_t> eos_ids_;

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<uint64_t, MergeEntry> merge_ranks_;
    std::vector<float> scores_;
    std::unordered_set<int32_t> special_ids_;
    std::unordered_set<int32_t> added_token_ids_;
    std::vector<AddedToken> added_tokens_;
    std::vector<std::vector<size_t>> added_by_first_byte_;

    std::string byte_to_utf8_[256];
    int16_t cp_to_byte_[512]{};

    void init_byte_tables();
    void finalize_added_tokens();
    std::vector<std::string> pretokenize_bytelevel(const std::string& text) const;
    void encode_bpe_segment(const std::string& segment, std::vector<int32_t>& out) const;
    void encode_sp_scores_segment(const std::string& segment, bool is_start_of_text, std::vector<int32_t>& out) const;
};

} // namespace rxinfer

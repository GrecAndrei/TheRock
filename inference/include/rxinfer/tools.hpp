// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rxinfer/tokenizer.hpp"

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace rxinfer {
namespace tools {

struct ParsedAssistantOutput {
    std::string reasoning_content;
    std::string content;
    std::vector<ToolCall> tool_calls;
};

// Parse complete raw model output into separated reasoning, visible text, and tool calls.
// Supports Qwen3.5 XML (<tool_call><function=...><parameter=...>...</parameter></function></tool_call>),
// JSON <tool_call>{...}</tool_call>, bare <function=...>, and Llama3 <|python_tag|>{...}.
// Never parses tool calls from inside <think>...</think> reasoning sections.
ParsedAssistantOutput parse_assistant_output(const std::string& raw_output,
                                             bool started_inside_think,
                                             const std::vector<ToolFunctionDef>& tools = {},
                                             int call_id_seed = 1);

// Streaming parser that separates <think>...</think> reasoning, visible content,
// and <tool_call>...</tool_call> across arbitrary token chunk boundaries without leaking tags.
class StreamingAssistantParser {
public:
    struct Delta {
        std::string reasoning_content;
        std::string content;
        std::vector<ToolCall> tool_calls;
    };

    explicit StreamingAssistantParser(bool started_inside_think,
                                      std::vector<ToolFunctionDef> tools = {},
                                      int call_id_seed = 1);

    Delta feed(std::string_view piece);
    Delta finish();

    const std::string& reasoning_content() const { return full_reasoning_; }
    const std::string& content() const { return full_content_; }
    const std::vector<ToolCall>& tool_calls() const { return tool_calls_; }
    int next_call_id() const { return next_call_id_; }

private:
    bool in_think_ = false;
    bool in_tool_call_ = false;
    bool in_python_tag_ = false;
    bool saw_think_content_ = false;
    bool eat_leading_think_nl_ = false;
    bool skip_leading_ws_after_think_ = false;
    bool tool_uses_bare_function_ = false;
    std::vector<ToolFunctionDef> tools_;
    int next_call_id_ = 1;

    std::string pending_;
    std::string tool_buf_;
    std::string full_reasoning_;
    std::string full_content_;
    std::vector<ToolCall> tool_calls_;

    void process(Delta& out);
    void flush_content(Delta& out, std::string_view text);
    void flush_reasoning(Delta& out, std::string_view text);
};

// Load OpenAI-format tools JSON array from string or file:
// [{"type":"function","function":{"name":...,"description":...,"parameters":{...}}}, ...]
std::vector<ToolFunctionDef> parse_tools_json(const std::string& json_text);
std::vector<ToolFunctionDef> load_tools_file(const std::string& path);

// Built-in local workspace tools (read_file, list_dir, search_text, write_file, run_command).
std::vector<ToolFunctionDef> builtin_workspace_tools();
bool is_builtin_tool(const std::string& name);
bool is_mutating_tool(const std::string& name);

struct ToolExecOptions {
    bool allow_mutating = false;
    std::function<bool(const ToolCall&)> confirm_cb; // Optional interactive approval prompt
    std::string external_exec_cmd;                   // Optional command to run custom JSON tools
    size_t max_output_bytes = 65536;
};

struct ToolExecResult {
    bool ok = false;
    std::string output;
};

ToolExecResult execute_tool_call(const ToolCall& tc, const ToolExecOptions& opts);

} // namespace tools
} // namespace rxinfer

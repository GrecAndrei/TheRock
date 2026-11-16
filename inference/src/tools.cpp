// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/tools.hpp"
#include "config_json.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace rxinfer {
namespace tools {
namespace {

std::string trim_ws(std::string_view sv) {
    size_t a = 0;
    while (a < sv.size() && (sv[a] == ' ' || sv[a] == '\t' || sv[a] == '\r' || sv[a] == '\n')) ++a;
    size_t b = sv.size();
    while (b > a && (sv[b - 1] == ' ' || sv[b - 1] == '\t' || sv[b - 1] == '\r' || sv[b - 1] == '\n')) --b;
    return std::string(sv.substr(a, b - a));
}

std::string trim_trailing_ws(const std::string& s) {
    size_t b = s.size();
    while (b > 0 && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) --b;
    return s.substr(0, b);
}

std::string strip_single_envelope_newlines(std::string_view sv) {
    if (!sv.empty() && sv.front() == '\r') sv.remove_prefix(1);
    if (!sv.empty() && sv.front() == '\n') sv.remove_prefix(1);
    if (!sv.empty() && sv.back() == '\n') sv.remove_suffix(1);
    if (!sv.empty() && sv.back() == '\r') sv.remove_suffix(1);
    return std::string(sv);
}

std::string make_call_id(int seq) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "call_rx_%d", seq);
    return std::string(buf);
}

std::string lookup_param_schema_type(const std::vector<ToolFunctionDef>& tools,
                                      const std::string& fn_name,
                                      const std::string& param_name) {
    for (const auto& t : tools) {
        if (t.name != fn_name) continue;
        std::string params_src = t.parameters_json;
        if (params_src.empty() && !t.raw_json.empty()) {
            try {
                const json::Value raw = json::parse(t.raw_json);
                const json::Value* fn = raw.get("function");
                if (fn && fn->get("parameters")) {
                    params_src = json::dump(*fn->get("parameters"));
                }
            } catch (...) {
            }
        }
        if (params_src.empty()) return {};
        try {
            const json::Value p = json::parse(params_src);
            const json::Value* props = p.get("properties");
            if (!props) return {};
            const json::Value* field = props->get(param_name);
            if (!field) return {};
            if (const auto* tv = field->get("type")) {
                if (tv->is_string()) return tv->as_str();
                if (tv->is_array() && !tv->as_array().empty()) {
                    for (const auto& el : tv->as_array()) {
                        if (el.is_string() && el.as_str() != "null") return el.as_str();
                    }
                }
            }
        } catch (...) {
        }
        return {};
    }
    return {};
}

bool is_strict_number_literal(const std::string& s) {
    if (s.empty()) return false;
    char* end = nullptr;
    (void)std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
}

json::Value coerce_param_value(const std::string& raw_val, const std::string& schema_type) {
    const std::string trimmed = trim_ws(raw_val);
    if (schema_type == "string") {
        return json::Value{raw_val};
    }
    if (schema_type == "integer" || schema_type == "number") {
        if (is_strict_number_literal(trimmed)) {
            try {
                const json::Value v = json::parse(trimmed);
                if (v.is_number()) return v;
            } catch (...) {
            }
        }
        return json::Value{raw_val};
    }
    if (schema_type == "boolean") {
        if (trimmed == "true" || trimmed == "True" || trimmed == "TRUE") return json::Value{true};
        if (trimmed == "false" || trimmed == "False" || trimmed == "FALSE") return json::Value{false};
        return json::Value{raw_val};
    }
    if (schema_type == "array" || schema_type == "object") {
        try {
            const json::Value v = json::parse(trimmed);
            if ((schema_type == "array" && v.is_array()) ||
                (schema_type == "object" && v.is_object())) {
                return v;
            }
        } catch (...) {
        }
        return json::Value{raw_val};
    }
    if (schema_type == "null" && trimmed == "null") {
        return json::Value{nullptr};
    }

    // No schema type available: infer cleanly only for unambiguous JSON literals
    if (trimmed == "true") return json::Value{true};
    if (trimmed == "false") return json::Value{false};
    if (trimmed == "null") return json::Value{nullptr};
    if (!trimmed.empty() && (trimmed.front() == '{' || trimmed.front() == '[')) {
        try {
            return json::parse(trimmed);
        } catch (...) {
        }
    }
    if (is_strict_number_literal(trimmed) &&
        !(trimmed.size() > 1 && trimmed[0] == '0' && trimmed[1] >= '0' && trimmed[1] <= '9')) {
        try {
            const json::Value v = json::parse(trimmed);
            if (v.is_number()) return v;
        } catch (...) {
        }
    }
    return json::Value{raw_val};
}

bool parse_xml_function_block(std::string_view fn_block,
                              const std::vector<ToolFunctionDef>& tools,
                              int& next_call_id,
                              std::vector<ToolCall>& out_calls) {
    const std::string_view prefix = "<function=";
    size_t pos = fn_block.find(prefix);
    if (pos == std::string_view::npos) return false;
    bool parsed_any = false;

    while (pos != std::string_view::npos) {
        const size_t name_start = pos + prefix.size();
        const size_t name_end = fn_block.find('>', name_start);
        if (name_end == std::string_view::npos) break;
        const std::string fn_name = trim_ws(fn_block.substr(name_start, name_end - name_start));

        const size_t body_start = name_end + 1;
        const size_t fn_close = fn_block.find("</function>", body_start);
        const std::string_view body =
            (fn_close == std::string_view::npos) ? fn_block.substr(body_start)
                                                 : fn_block.substr(body_start, fn_close - body_start);

        json::Object args_obj;
        size_t p_pos = 0;
        while (p_pos < body.size()) {
            const size_t p_open = body.find("<parameter=", p_pos);
            if (p_open == std::string_view::npos) break;
            const size_t k_start = p_open + std::string_view("<parameter=").size();
            const size_t k_end = body.find('>', k_start);
            if (k_end == std::string_view::npos) break;
            const std::string p_name = trim_ws(body.substr(k_start, k_end - k_start));
            const size_t v_start = k_end + 1;
            const size_t p_close = body.find("</parameter>", v_start);
            const std::string_view raw_slice =
                (p_close == std::string_view::npos) ? body.substr(v_start)
                                                    : body.substr(v_start, p_close - v_start);
            const std::string p_val = strip_single_envelope_newlines(raw_slice);
            const std::string schema_type = lookup_param_schema_type(tools, fn_name, p_name);
            args_obj[p_name] = coerce_param_value(p_val, schema_type);
            p_pos = (p_close == std::string_view::npos) ? body.size() : (p_close + std::string_view("</parameter>").size());
        }

        if (!fn_name.empty() && fn_name.find_first_of(" \t\r\n<>\"'") == std::string::npos) {
            ToolCall tc;
            tc.id = make_call_id(next_call_id++);
            tc.name = fn_name;
            tc.arguments = json::dump(json::Value{std::move(args_obj)});
            out_calls.push_back(std::move(tc));
            parsed_any = true;
        }

        if (fn_close == std::string_view::npos) break;
        pos = fn_block.find(prefix, fn_close + std::string_view("</function>").size());
    }
    return parsed_any;
}

bool parse_json_tool_call(std::string_view json_text,
                          int& next_call_id,
                          std::vector<ToolCall>& out_calls) {
    const std::string trimmed = trim_ws(json_text);
    if (trimmed.empty()) return false;
    try {
        const json::Value v = json::parse(trimmed);
        auto extract_one = [&](const json::Value& item) -> bool {
            if (!item.is_object()) return false;
            const json::Value* fn = item.get("function");
            const json::Value& src = (fn && fn->is_object()) ? *fn : item;
            const json::Value* nv = src.get("name");
            if (!nv || !nv->is_string() || nv->as_str().empty()) return false;
            const json::Value* av = src.get("arguments");
            if (!av) av = src.get("parameters");

            ToolCall tc;
            if (const auto* idv = item.get("id"); idv && idv->is_string() && !idv->as_str().empty()) {
                tc.id = idv->as_str();
            } else {
                tc.id = make_call_id(next_call_id++);
            }
            tc.name = nv->as_str();
            if (!av) {
                tc.arguments = "{}";
            } else if (av->is_string()) {
                tc.arguments = av->as_str();
            } else {
                tc.arguments = json::dump(*av);
            }
            out_calls.push_back(std::move(tc));
            return true;
        };

        if (v.is_array()) {
            bool any = false;
            for (const auto& el : v.as_array()) {
                if (extract_one(el)) any = true;
            }
            return any;
        }
        return extract_one(v);
    } catch (...) {
        return false;
    }
}

bool parse_tool_call_payload(std::string_view payload,
                             const std::vector<ToolFunctionDef>& tools,
                             int& next_call_id,
                             std::vector<ToolCall>& out_calls) {
    if (payload.find("<function=") != std::string_view::npos) {
        return parse_xml_function_block(payload, tools, next_call_id, out_calls);
    }
    return parse_json_tool_call(payload, next_call_id, out_calls);
}

void parse_visible_and_tools(std::string_view text,
                             const std::vector<ToolFunctionDef>& tools,
                             int& next_call_id,
                             std::string& out_visible,
                             std::vector<ToolCall>& out_calls) {
    size_t pos = 0;
    while (pos < text.size()) {
        const size_t p_tc = text.find("<tool_call>", pos);
        const size_t p_fn = text.find("<function=", pos);
        const size_t p_py = text.find("<|python_tag|>", pos);

        size_t next = std::min({p_tc, p_fn, p_py});
        if (next == std::string::npos) {
            out_visible.append(text.data() + pos, text.size() - pos);
            break;
        }

        out_visible.append(text.data() + pos, next - pos);
        if (next == p_tc) {
            const size_t start = next + std::string_view("<tool_call>").size();
            const size_t end = text.find("</tool_call>", start);
            if (end == std::string_view::npos) {
                const size_t fn_end = text.find("</function>", start);
                if (fn_end != std::string_view::npos) {
                    const size_t slice_end = fn_end + std::string_view("</function>").size();
                    if (!parse_tool_call_payload(text.substr(start, slice_end - start), tools, next_call_id, out_calls)) {
                        out_visible.append(text.data() + next, slice_end - next);
                    }
                    pos = slice_end;
                    continue;
                }
                out_visible.append(text.data() + next, text.size() - next);
                break;
            }
            if (!parse_tool_call_payload(text.substr(start, end - start), tools, next_call_id, out_calls)) {
                out_visible.append(text.data() + next, end + std::string_view("</tool_call>").size() - next);
            }
            pos = end + std::string_view("</tool_call>").size();
        } else if (next == p_fn) {
            const size_t end = text.find("</function>", next);
            if (end == std::string_view::npos) {
                out_visible.append(text.data() + next, text.size() - next);
                break;
            }
            const size_t slice_end = end + std::string_view("</function>").size();
            if (!parse_xml_function_block(text.substr(next, slice_end - next), tools, next_call_id, out_calls)) {
                out_visible.append(text.data() + next, slice_end - next);
            }
            pos = slice_end;
        } else {
            // Llama3 <|python_tag|>
            const size_t start = next + std::string_view("<|python_tag|>").size();
            const size_t end = text.find("<|eom_id|>", start);
            const size_t slice_end = (end == std::string_view::npos) ? text.size() : end;
            if (!parse_json_tool_call(text.substr(start, slice_end - start), next_call_id, out_calls)) {
                out_visible.append(text.data() + next, slice_end - next);
            }
            pos = (end == std::string_view::npos) ? text.size() : (end + std::string_view("<|eom_id|>").size());
        }
    }
}

size_t longest_prefix_match_len(std::string_view buf, std::initializer_list<std::string_view> needles) {
    size_t keep = 0;
    for (std::string_view needle : needles) {
        const size_t max_k = std::min(buf.size(), needle.size() - 1);
        for (size_t k = max_k; k > keep; --k) {
            if (buf.compare(buf.size() - k, k, needle.data(), k) == 0) {
                keep = k;
                break;
            }
        }
    }
    return keep;
}

bool ensure_parent_dir(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos || slash == 0) return true;
    const std::string dir = path.substr(0, slash);
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
            struct stat st{};
            if (::stat(cur.c_str(), &st) != 0) {
                if (::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
            } else if (!S_ISDIR(st.st_mode)) {
                return false;
            }
        }
        if (next == std::string::npos) break;
        i = next + 1;
    }
    return true;
}

bool is_probably_binary(const std::string& sample) {
    for (unsigned char c : sample) {
        if (c == '\0') return true;
    }
    return false;
}

} // namespace

ParsedAssistantOutput parse_assistant_output(const std::string& raw_output,
                                             bool started_inside_think,
                                             const std::vector<ToolFunctionDef>& tools,
                                             int call_id_seed) {
    ParsedAssistantOutput res;
    int next_call_id = std::max(1, call_id_seed);
    std::string after_think;

    if (started_inside_think) {
        std::string_view sv = raw_output;
        if (sv.rfind("<think>", 0) == 0) {
            sv.remove_prefix(std::string_view("<think>").size());
        }
        const size_t cl = sv.find("</think>");
        if (cl == std::string_view::npos) {
            res.reasoning_content = trim_ws(sv);
            return res;
        }
        res.reasoning_content = trim_ws(sv.substr(0, cl));
        after_think = std::string(sv.substr(cl + std::string_view("</think>").size()));
    } else {
        size_t pos = 0;
        while (pos < raw_output.size()) {
            const size_t op = raw_output.find("<think>", pos);
            if (op == std::string::npos) {
                after_think.append(raw_output, pos, raw_output.size() - pos);
                break;
            }
            after_think.append(raw_output, pos, op - pos);
            const size_t body_start = op + std::string_view("<think>").size();
            const size_t cl = raw_output.find("</think>", body_start);
            if (cl == std::string::npos) {
                if (!res.reasoning_content.empty()) res.reasoning_content.push_back('\n');
                res.reasoning_content += trim_ws(std::string_view(raw_output).substr(body_start));
                pos = raw_output.size();
                break;
            }
            if (!res.reasoning_content.empty()) res.reasoning_content.push_back('\n');
            res.reasoning_content += trim_ws(std::string_view(raw_output).substr(body_start, cl - body_start));
            pos = cl + std::string_view("</think>").size();
        }
    }

    std::string visible;
    parse_visible_and_tools(after_think, tools, next_call_id, visible, res.tool_calls);
    res.content = trim_ws(visible);
    res.reasoning_content = trim_ws(res.reasoning_content);
    return res;
}

StreamingAssistantParser::StreamingAssistantParser(bool started_inside_think,
                                                   std::vector<ToolFunctionDef> tools,
                                                   int call_id_seed)
    : in_think_(started_inside_think),
      tools_(std::move(tools)),
      next_call_id_(std::max(1, call_id_seed)) {}

void StreamingAssistantParser::flush_reasoning(Delta& out, std::string_view text) {
    if (text.empty()) return;
    if (eat_leading_think_nl_) {
        if (text.front() == '\r') text.remove_prefix(1);
        if (!text.empty() && text.front() == '\n') {
            text.remove_prefix(1);
            eat_leading_think_nl_ = false;
        } else if (!text.empty()) {
            eat_leading_think_nl_ = false;
        }
        if (text.empty()) return;
    }
    saw_think_content_ = true;
    out.reasoning_content.append(text.data(), text.size());
    full_reasoning_.append(text.data(), text.size());
}

void StreamingAssistantParser::flush_content(Delta& out, std::string_view text) {
    if (text.empty()) return;
    if (skip_leading_ws_after_think_) {
        size_t i = 0;
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t' || text[i] == '\r' || text[i] == '\n')) {
            ++i;
        }
        text.remove_prefix(i);
        if (text.empty()) return;
        skip_leading_ws_after_think_ = false;
    }
    out.content.append(text.data(), text.size());
    full_content_.append(text.data(), text.size());
}

void StreamingAssistantParser::process(Delta& out) {
    while (!pending_.empty()) {
        if (in_think_) {
            if (!saw_think_content_ && pending_.rfind("<think>", 0) == 0) {
                pending_.erase(0, std::string_view("<think>").size());
                eat_leading_think_nl_ = true;
                saw_think_content_ = true;
                continue;
            }
            const size_t cl = pending_.find("</think>");
            if (cl != std::string::npos) {
                flush_reasoning(out, std::string_view(pending_).substr(0, cl));
                pending_.erase(0, cl + std::string_view("</think>").size());
                in_think_ = false;
                eat_leading_think_nl_ = false;
                skip_leading_ws_after_think_ = true;
                continue;
            }
            const size_t keep = !saw_think_content_
                                    ? longest_prefix_match_len(pending_, {"</think>", "<think>"})
                                    : longest_prefix_match_len(pending_, {"</think>"});
            const size_t emit_len = pending_.size() - keep;
            if (emit_len > 0) {
                flush_reasoning(out, std::string_view(pending_).substr(0, emit_len));
                pending_.erase(0, emit_len);
            }
            break;
        }

        if (in_tool_call_) {
            if (!tool_uses_bare_function_) {
                const size_t cl = pending_.find("</tool_call>");
                if (cl == std::string::npos) {
                    const size_t keep = longest_prefix_match_len(pending_, {"</tool_call>"});
                    const size_t move_len = pending_.size() - keep;
                    if (move_len > 0) {
                        tool_buf_.append(pending_, 0, move_len);
                        pending_.erase(0, move_len);
                    }
                    break;
                }
                tool_buf_.append(pending_, 0, cl);
                pending_.erase(0, cl + std::string_view("</tool_call>").size());
                std::string payload = std::move(tool_buf_);
                tool_buf_.clear();
                in_tool_call_ = false;
                const size_t before = tool_calls_.size();
                if (!parse_tool_call_payload(payload, tools_, next_call_id_, tool_calls_)) {
                    flush_content(out, "<tool_call>" + payload + "</tool_call>");
                } else {
                    for (size_t ti = before; ti < tool_calls_.size(); ++ti) {
                        out.tool_calls.push_back(tool_calls_[ti]);
                    }
                }
                skip_leading_ws_after_think_ = true;
                continue;
            } else {
                const size_t cl = pending_.find("</function>");
                if (cl == std::string::npos) {
                    const size_t keep = longest_prefix_match_len(pending_, {"</function>"});
                    const size_t move_len = pending_.size() - keep;
                    if (move_len > 0) {
                        tool_buf_.append(pending_, 0, move_len);
                        pending_.erase(0, move_len);
                    }
                    break;
                }
                const size_t end_tag = cl + std::string_view("</function>").size();
                tool_buf_.append(pending_, 0, end_tag);
                pending_.erase(0, end_tag);
                std::string payload = std::move(tool_buf_);
                tool_buf_.clear();
                in_tool_call_ = false;
                tool_uses_bare_function_ = false;
                const size_t before = tool_calls_.size();
                if (!parse_xml_function_block(payload, tools_, next_call_id_, tool_calls_)) {
                    flush_content(out, payload);
                } else {
                    for (size_t ti = before; ti < tool_calls_.size(); ++ti) {
                        out.tool_calls.push_back(tool_calls_[ti]);
                    }
                }
                skip_leading_ws_after_think_ = true;
                continue;
            }
        }

        if (in_python_tag_) {
            const size_t cl = pending_.find("<|eom_id|>");
            if (cl == std::string::npos) {
                const size_t keep = longest_prefix_match_len(pending_, {"<|eom_id|>"});
                const size_t move_len = pending_.size() - keep;
                if (move_len > 0) {
                    tool_buf_.append(pending_, 0, move_len);
                    pending_.erase(0, move_len);
                }
                break;
            }
            tool_buf_.append(pending_, 0, cl);
            pending_.erase(0, cl + std::string_view("<|eom_id|>").size());
            std::string payload = std::move(tool_buf_);
            tool_buf_.clear();
            in_python_tag_ = false;
            const size_t before = tool_calls_.size();
            if (!parse_json_tool_call(payload, next_call_id_, tool_calls_)) {
                flush_content(out, "<|python_tag|>" + payload);
            } else {
                for (size_t ti = before; ti < tool_calls_.size(); ++ti) {
                    out.tool_calls.push_back(tool_calls_[ti]);
                }
            }
            continue;
        }

        const size_t p_th = pending_.find("<think>");
        const size_t p_tc = pending_.find("<tool_call>");
        const size_t p_fn = pending_.find("<function=");
        const size_t p_py = pending_.find("<|python_tag|>");
        const size_t next = std::min({p_th, p_tc, p_fn, p_py});

        if (next != std::string::npos) {
            if (next > 0) {
                flush_content(out, std::string_view(pending_).substr(0, next));
            }
            if (next == p_th) {
                pending_.erase(0, next + std::string_view("<think>").size());
                in_think_ = true;
                eat_leading_think_nl_ = true;
                saw_think_content_ = true;
            } else if (next == p_tc) {
                pending_.erase(0, next + std::string_view("<tool_call>").size());
                in_tool_call_ = true;
                tool_uses_bare_function_ = false;
                tool_buf_.clear();
            } else if (next == p_fn) {
                pending_.erase(0, next);
                in_tool_call_ = true;
                tool_uses_bare_function_ = true;
                tool_buf_.clear();
            } else {
                pending_.erase(0, next + std::string_view("<|python_tag|>").size());
                in_python_tag_ = true;
                tool_buf_.clear();
            }
            continue;
        }

        const size_t keep = longest_prefix_match_len(
            pending_, {"<think>", "<tool_call>", "<function=", "<|python_tag|>"});
        const size_t emit_len = pending_.size() - keep;
        if (emit_len > 0) {
            flush_content(out, std::string_view(pending_).substr(0, emit_len));
            pending_.erase(0, emit_len);
        }
        break;
    }
}

StreamingAssistantParser::Delta StreamingAssistantParser::feed(std::string_view piece) {
    Delta out;
    if (piece.empty()) return out;
    pending_.append(piece.data(), piece.size());
    process(out);
    return out;
}

StreamingAssistantParser::Delta StreamingAssistantParser::finish() {
    Delta out;
    process(out);
    if (in_think_) {
        if (!pending_.empty()) {
            flush_reasoning(out, pending_);
            pending_.clear();
        }
    } else if (in_tool_call_) {
        tool_buf_ += pending_;
        pending_.clear();
        const size_t before = tool_calls_.size();
        if (!tool_uses_bare_function_ &&
            tool_buf_.find("</function>") != std::string::npos &&
            parse_tool_call_payload(tool_buf_, tools_, next_call_id_, tool_calls_)) {
            for (size_t ti = before; ti < tool_calls_.size(); ++ti) {
                out.tool_calls.push_back(tool_calls_[ti]);
            }
            tool_buf_.clear();
        } else {
            flush_content(out, (tool_uses_bare_function_ ? "" : "<tool_call>") + tool_buf_);
            tool_buf_.clear();
        }
    } else if (in_python_tag_) {
        tool_buf_ += pending_;
        pending_.clear();
        const size_t before = tool_calls_.size();
        if (!parse_json_tool_call(tool_buf_, next_call_id_, tool_calls_)) {
            flush_content(out, "<|python_tag|>" + tool_buf_);
        } else {
            for (size_t ti = before; ti < tool_calls_.size(); ++ti) {
                out.tool_calls.push_back(tool_calls_[ti]);
            }
        }
        tool_buf_.clear();
    } else if (!pending_.empty()) {
        flush_content(out, pending_);
        pending_.clear();
    }
    full_reasoning_ = trim_ws(full_reasoning_);
    full_content_ = trim_trailing_ws(full_content_);
    return out;
}

std::vector<ToolFunctionDef> parse_tools_json(const std::string& json_text) {
    const json::Value v = json::parse(json_text);
    const json::Array* arr = nullptr;
    if (v.is_array()) {
        arr = &v.as_array();
    } else if (v.is_object() && v.get("tools") && v.get("tools")->is_array()) {
        arr = &v.get("tools")->as_array();
    } else {
        throw std::runtime_error("Expected JSON array of OpenAI tool definitions");
    }

    std::vector<ToolFunctionDef> out;
    for (const auto& item : *arr) {
        if (!item.is_object()) continue;
        const json::Value* fn = item.get("function");
        const json::Value& src = (fn && fn->is_object()) ? *fn : item;
        const json::Value* nv = src.get("name");
        if (!nv || !nv->is_string() || nv->as_str().empty()) continue;

        ToolFunctionDef def;
        def.name = nv->as_str();
        if (const auto* dv = src.get("description"); dv && dv->is_string()) {
            def.description = dv->as_str();
        }
        if (const auto* pv = src.get("parameters"); pv && pv->is_object()) {
            def.parameters_json = json::dump(*pv);
        } else {
            def.parameters_json = R"({"type":"object","properties":{}})";
        }
        if (fn && fn->is_object()) {
            def.raw_json = json::dump(item);
        }
        out.push_back(std::move(def));
    }
    return out;
}

std::vector<ToolFunctionDef> load_tools_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open tools file: " + path);
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return parse_tools_json(oss.str());
}

std::vector<ToolFunctionDef> builtin_workspace_tools() {
    std::vector<ToolFunctionDef> out;
    out.push_back({
        "read_file",
        "Read UTF-8 text lines from a file on disk. Use offset (1-based line) and limit to read sections cleanly.",
        R"json({"type":"object","properties":{"path":{"type":"string","description":"File path to read"},"offset":{"type":"integer","description":"1-based starting line number (default 1)"},"limit":{"type":"integer","description":"Maximum number of lines to read (default 200)"}},"required":["path"]})json",
        "",
    });
    out.push_back({
        "list_dir",
        "List files and subdirectories in a directory.",
        R"json({"type":"object","properties":{"path":{"type":"string","description":"Directory path (default '.')"},"max_entries":{"type":"integer","description":"Maximum entries to return (default 200)"}}})json",
        "",
    });
    out.push_back({
        "search_text",
        "Search for a regex or substring pattern recursively across text files.",
        R"json({"type":"object","properties":{"pattern":{"type":"string","description":"Text or regex pattern to match"},"path":{"type":"string","description":"File or directory path (default '.')"},"max_matches":{"type":"integer","description":"Maximum matching lines (default 50)"}},"required":["pattern"]})json",
        "",
    });
    out.push_back({
        "write_file",
        "Write or overwrite a text file on disk (creating parent directories if needed).",
        R"json({"type":"object","properties":{"path":{"type":"string","description":"Destination file path"},"content":{"type":"string","description":"Full file contents to write"}},"required":["path","content"]})json",
        "",
    });
    out.push_back({
        "run_command",
        "Execute a shell command via /bin/sh -c and capture combined stdout and stderr.",
        R"json({"type":"object","properties":{"command":{"type":"string","description":"Shell command to execute"},"workdir":{"type":"string","description":"Optional working directory"}},"required":["command"]})json",
        "",
    });
    return out;
}

bool is_builtin_tool(const std::string& name) {
    return name == "read_file" || name == "list_dir" || name == "search_text" ||
           name == "write_file" || name == "run_command";
}

bool is_mutating_tool(const std::string& name) {
    return name == "write_file" || name == "run_command";
}

namespace {

std::string truncate_tool_output(std::string out, size_t max_bytes) {
    if (out.size() <= max_bytes) return out;
    out.resize(max_bytes);
    out += "\n...[truncated]...";
    return out;
}

ToolExecResult exec_read_file(const json::Value& args, size_t max_bytes) {
    const json::Value* pv = args.get("path");
    if (!pv || !pv->is_string() || pv->as_str().empty()) {
        return {false, "Error: 'path' string argument is required."};
    }
    const std::string path = pv->as_str();
    const int64_t offset = std::max<int64_t>(1, args.get("offset") ? args.get("offset")->as_i64(1) : 1);
    const int64_t limit = std::clamp<int64_t>(args.get("limit") ? args.get("limit")->as_i64(200) : 200, 1, 2000);

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return {false, "Error: cannot open file '" + path + "'"};

    std::string line;
    int64_t ln = 0;
    int64_t shown = 0;
    std::ostringstream oss;
    while (std::getline(ifs, line)) {
        ++ln;
        if (ln < offset) continue;
        if (shown >= limit) continue;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        oss << ln << ": " << line << "\n";
        ++shown;
    }
    std::ostringstream hdr;
    hdr << "File: " << path << " (lines " << offset << "-" << (offset + std::max<int64_t>(0, shown - 1))
        << " of " << ln << ")\n" << oss.str();
    return {true, truncate_tool_output(hdr.str(), max_bytes)};
}

ToolExecResult exec_list_dir(const json::Value& args, size_t max_bytes) {
    const std::string path =
        (args.get("path") && args.get("path")->is_string() && !args.get("path")->as_str().empty())
            ? args.get("path")->as_str()
            : ".";
    const int64_t max_entries =
        std::clamp<int64_t>(args.get("max_entries") ? args.get("max_entries")->as_i64(200) : 200, 1, 2000);

    DIR* d = ::opendir(path.c_str());
    if (!d) return {false, "Error: cannot open directory '" + path + "'"};
    std::vector<std::string> items;
    struct dirent* ent = nullptr;
    while ((ent = ::readdir(d)) != nullptr) {
        const std::string name = ent->d_name;
        if (name == "." || name == "..") continue;
        const std::string full = (path == "/") ? ("/" + name) : (path + "/" + name);
        struct stat st{};
        if (::stat(full.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            items.push_back(name + "/");
        } else {
            items.push_back(name);
        }
    }
    ::closedir(d);
    std::sort(items.begin(), items.end());

    std::ostringstream oss;
    const size_t n = std::min<size_t>(items.size(), static_cast<size_t>(max_entries));
    for (size_t i = 0; i < n; ++i) {
        oss << items[i] << "\n";
    }
    if (items.size() > n) {
        oss << "...(" << (items.size() - n) << " more entries)\n";
    }
    return {true, truncate_tool_output(oss.str(), max_bytes)};
}

void search_file_lines(const std::string& fpath,
                       const std::string& pattern,
                       const std::regex* re,
                       int64_t max_matches,
                       int64_t& count,
                       std::ostringstream& oss) {
    if (count >= max_matches) return;
    std::ifstream ifs(fpath, std::ios::binary);
    if (!ifs) return;

    std::string head(512, '\0');
    ifs.read(head.data(), static_cast<std::streamsize>(head.size()));
    head.resize(static_cast<size_t>(ifs.gcount()));
    if (is_probably_binary(head)) return;
    ifs.clear();
    ifs.seekg(0);

    std::string line;
    int64_t ln = 0;
    while (std::getline(ifs, line)) {
        ++ln;
        const bool hit = re ? std::regex_search(line, *re) : (line.find(pattern) != std::string::npos);
        if (hit) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.size() > 240) line.resize(240);
            oss << fpath << ":" << ln << ": " << line << "\n";
            if (++count >= max_matches) break;
        }
    }
}

void search_recursive(const std::string& path,
                      const std::string& pattern,
                      const std::regex* re,
                      int64_t max_matches,
                      int64_t& count,
                      std::ostringstream& oss) {
    if (count >= max_matches) return;
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) return;
    if (S_ISREG(st.st_mode)) {
        if (st.st_size <= (8LL << 20)) {
            search_file_lines(path, pattern, re, max_matches, count, oss);
        }
        return;
    }
    if (!S_ISDIR(st.st_mode)) return;

    DIR* d = ::opendir(path.c_str());
    if (!d) return;
    std::vector<std::string> names;
    struct dirent* ent = nullptr;
    while ((ent = ::readdir(d)) != nullptr) {
        const std::string n = ent->d_name;
        if (n == "." || n == ".." || n == ".git" || n == ".venv") continue;
        names.push_back(n);
    }
    ::closedir(d);
    std::sort(names.begin(), names.end());
    for (const auto& n : names) {
        if (count >= max_matches) break;
        const std::string child = (path == "/") ? ("/" + n) : (path + "/" + n);
        search_recursive(child, pattern, re, max_matches, count, oss);
    }
}

ToolExecResult exec_search_text(const json::Value& args, size_t max_bytes) {
    const json::Value* pv = args.get("pattern");
    if (!pv || !pv->is_string() || pv->as_str().empty()) {
        return {false, "Error: 'pattern' string argument is required."};
    }
    const std::string pattern = pv->as_str();
    const std::string path =
        (args.get("path") && args.get("path")->is_string() && !args.get("path")->as_str().empty())
            ? args.get("path")->as_str()
            : ".";
    const int64_t max_matches =
        std::clamp<int64_t>(args.get("max_matches") ? args.get("max_matches")->as_i64(50) : 50, 1, 500);

    std::regex re_obj;
    const std::regex* re_ptr = nullptr;
    try {
        re_obj = std::regex(pattern);
        re_ptr = &re_obj;
    } catch (...) {
        re_ptr = nullptr;
    }

    int64_t count = 0;
    std::ostringstream oss;
    search_recursive(path, pattern, re_ptr, max_matches, count, oss);
    if (count == 0) {
        return {true, "(no matches)"};
    }
    return {true, truncate_tool_output(oss.str(), max_bytes)};
}

ToolExecResult exec_write_file(const json::Value& args) {
    const json::Value* pv = args.get("path");
    const json::Value* cv = args.get("content");
    if (!pv || !pv->is_string() || pv->as_str().empty() || !cv || !cv->is_string()) {
        return {false, "Error: 'path' and 'content' string arguments are required."};
    }
    const std::string path = pv->as_str();
    const std::string& content = cv->as_str();
    if (!ensure_parent_dir(path)) {
        return {false, "Error: failed to create parent directories for '" + path + "'"};
    }
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return {false, "Error: failed to open '" + path + "' for writing"};
    }
    ofs.write(content.data(), static_cast<std::streamsize>(content.size()));
    ofs.flush();
    if (!ofs) {
        return {false, "Error: failed writing '" + path + "'"};
    }
    return {true, "Wrote " + std::to_string(content.size()) + " bytes to " + path};
}

ToolExecResult exec_shell_pipe(const std::string& command,
                               const std::string& workdir,
                               const std::string& stdin_payload,
                               size_t max_bytes) {
    int out_pipe[2]{-1, -1};
    int in_pipe[2]{-1, -1};
    if (::pipe(out_pipe) != 0) {
        return {false, "Error: pipe() failed"};
    }
    if (!stdin_payload.empty() && ::pipe(in_pipe) != 0) {
        ::close(out_pipe[0]);
        ::close(out_pipe[1]);
        return {false, "Error: pipe() failed"};
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(out_pipe[0]);
        ::close(out_pipe[1]);
        if (in_pipe[0] >= 0) {
            ::close(in_pipe[0]);
            ::close(in_pipe[1]);
        }
        return {false, "Error: fork() failed"};
    }

    if (pid == 0) {
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::dup2(out_pipe[1], STDERR_FILENO);
        ::close(out_pipe[0]);
        ::close(out_pipe[1]);
        if (in_pipe[0] >= 0) {
            ::dup2(in_pipe[0], STDIN_FILENO);
            ::close(in_pipe[0]);
            ::close(in_pipe[1]);
        } else {
            const int devnull = ::open("/dev/null", O_RDONLY);
            if (devnull >= 0) {
                ::dup2(devnull, STDIN_FILENO);
                ::close(devnull);
            }
        }
        if (!workdir.empty()) {
            if (::chdir(workdir.c_str()) != 0) {
                std::fprintf(stderr, "chdir('%s') failed: %s\n", workdir.c_str(), std::strerror(errno));
                _exit(127);
            }
        }
        const char* argv[] = {"/bin/sh", "-c", command.c_str(), nullptr};
        ::execv("/bin/sh", const_cast<char* const*>(argv));
        _exit(127);
    }

    ::close(out_pipe[1]);
    if (in_pipe[0] >= 0) {
        ::close(in_pipe[0]);
        size_t off = 0;
        while (off < stdin_payload.size()) {
            const ssize_t w = ::write(in_pipe[1], stdin_payload.data() + off, stdin_payload.size() - off);
            if (w <= 0) break;
            off += static_cast<size_t>(w);
        }
        ::close(in_pipe[1]);
    }

    std::string output;
    char buf[4096];
    while (true) {
        const ssize_t n = ::read(out_pipe[0], buf, sizeof(buf));
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (n == 0) break;
        if (output.size() < max_bytes + 4096) {
            output.append(buf, static_cast<size_t>(n));
        }
    }
    ::close(out_pipe[0]);

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    const int code = WIFEXITED(status) ? WEXITSTATUS(status) : (WIFSIGNALED(status) ? (128 + WTERMSIG(status)) : -1);
    output = truncate_tool_output(std::move(output), max_bytes);
    if (code != 0) {
        output = "Exit code " + std::to_string(code) + "\n" + output;
        return {false, output};
    }
    return {true, output.empty() ? "(exit 0, no output)" : output};
}

} // namespace

ToolExecResult execute_tool_call(const ToolCall& tc, const ToolExecOptions& opts) {
    json::Value args;
    try {
        args = tc.arguments.empty() ? json::Value{json::Object{}} : json::parse(tc.arguments);
    } catch (const std::exception& ex) {
        return {false, std::string("Error: invalid tool arguments JSON: ") + ex.what()};
    }

    if (is_mutating_tool(tc.name) && !opts.allow_mutating) {
        if (!opts.confirm_cb || !opts.confirm_cb(tc)) {
            return {false, "Execution denied by user."};
        }
    }

    if (tc.name == "read_file") return exec_read_file(args, opts.max_output_bytes);
    if (tc.name == "list_dir") return exec_list_dir(args, opts.max_output_bytes);
    if (tc.name == "search_text") return exec_search_text(args, opts.max_output_bytes);
    if (tc.name == "write_file") return exec_write_file(args);
    if (tc.name == "run_command") {
        const json::Value* cv = args.get("command");
        if (!cv || !cv->is_string() || cv->as_str().empty()) {
            return {false, "Error: 'command' string argument is required."};
        }
        const std::string wd = (args.get("workdir") && args.get("workdir")->is_string())
                                   ? args.get("workdir")->as_str()
                                   : "";
        return exec_shell_pipe(cv->as_str(), wd, "", opts.max_output_bytes);
    }

    if (!opts.external_exec_cmd.empty()) {
        json::Object payload;
        payload.emplace("id", json::Value{tc.id});
        payload.emplace("name", json::Value{tc.name});
        payload.emplace("arguments", std::move(args));
        return exec_shell_pipe(opts.external_exec_cmd, "", json::dump(json::Value{std::move(payload)}),
                               opts.max_output_bytes);
    }

    return {false, "Error: no executor registered for tool '" + tc.name + "'"};
}

} // namespace tools
} // namespace rxinfer

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/server.hpp"
#include "config_json.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace rxinfer {
namespace server {
namespace {

std::string to_lower_ascii(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string trim_ascii(std::string_view sv) {
    size_t a = 0;
    while (a < sv.size() && (sv[a] == ' ' || sv[a] == '\t' || sv[a] == '\r' || sv[a] == '\n')) ++a;
    size_t b = sv.size();
    while (b > a && (sv[b - 1] == ' ' || sv[b - 1] == '\t' || sv[b - 1] == '\r' || sv[b - 1] == '\n')) --b;
    return std::string(sv.substr(a, b - a));
}

std::string extract_content_string(const json::Value* cv) {
    if (!cv || cv->is_null()) return "";
    if (cv->is_string()) return cv->as_str();
    if (cv->is_array()) {
        std::string out;
        for (const auto& part : cv->as_array()) {
            if (part.is_string()) {
                out += part.as_str();
                continue;
            }
            if (!part.is_object()) continue;
            const json::Value* tv = part.get("type");
            const std::string t = tv ? tv->as_str() : "";
            if (t.empty() || t == "text" || t == "input_text") {
                if (const auto* text_v = part.get("text")) {
                    if (text_v->is_string()) {
                        out += text_v->as_str();
                    } else if (text_v->is_object()) {
                        if (const auto* vv = text_v->get("value"); vv && vv->is_string()) {
                            out += vv->as_str();
                        }
                    }
                }
            }
        }
        return out;
    }
    if (cv->is_object()) {
        if (const auto* tv = cv->get("text"); tv && tv->is_string()) return tv->as_str();
        return json::dump(*cv);
    }
    return json::dump(*cv);
}

std::vector<ToolFunctionDef> parse_tools_array_value(const json::Array& arr) {
    std::vector<ToolFunctionDef> out;
    for (const auto& item : arr) {
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

bool send_all_fd(int fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        const ssize_t n = ::send(fd, data + sent, len - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool send_all_str(int fd, const std::string& s) {
    return send_all_fd(fd, s.data(), s.size());
}

void send_http_json(int fd, int status_code, const char* status_text, const std::string& json_body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status_code << " " << status_text << "\r\n"
        << "Content-Type: application/json; charset=utf-8\r\n"
        << "Content-Length: " << json_body.size() << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        << "Connection: close\r\n\r\n"
        << json_body;
    send_all_str(fd, oss.str());
}

void send_openai_error(int fd,
                       int status_code,
                       const char* status_text,
                       const std::string& message,
                       const std::string& err_type,
                       const std::string& err_code = "") {
    json::Object err;
    err.emplace("message", json::Value{message});
    err.emplace("type", json::Value{err_type});
    if (err_code.empty()) {
        err.emplace("code", json::Value{nullptr});
    } else {
        err.emplace("code", json::Value{err_code});
    }
    json::Object root;
    root.emplace("error", json::Value{std::move(err)});
    send_http_json(fd, status_code, status_text, json::dump(json::Value{std::move(root)}));
}

struct HttpRequest {
    std::string method;
    std::string path;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
};

bool decode_chunked_body(std::string& raw_buf, std::string& out_body, size_t max_body_bytes) {
    size_t pos = 0;
    out_body.clear();
    while (true) {
        const size_t line_end = raw_buf.find("\r\n", pos);
        if (line_end == std::string::npos) return false;
        std::string sz_str = raw_buf.substr(pos, line_end - pos);
        const size_t semi = sz_str.find(';');
        if (semi != std::string::npos) sz_str.resize(semi);
        sz_str = trim_ascii(sz_str);
        if (sz_str.empty()) throw std::runtime_error("Malformed chunked size");
        char* endp = nullptr;
        const unsigned long chunk_sz = std::strtoul(sz_str.c_str(), &endp, 16);
        if (endp == sz_str.c_str() || *endp != '\0') throw std::runtime_error("Invalid chunk hex size");
        pos = line_end + 2;
        if (chunk_sz == 0) {
            if (raw_buf.size() < pos + 2) return false;
            return true;
        }
        if (out_body.size() + chunk_sz > max_body_bytes) {
            throw std::runtime_error("HTTP request body exceeds limit");
        }
        if (raw_buf.size() < pos + chunk_sz + 2) return false;
        out_body.append(raw_buf, pos, chunk_sz);
        pos += chunk_sz + 2;
    }
}

bool read_http_request(int fd, HttpRequest& req, std::string& err_out) {
    constexpr size_t kMaxHeaderBytes = 64 * 1024;
    constexpr size_t kMaxBodyBytes = 32 * 1024 * 1024;

    std::string buf;
    char tmp[8192];
    size_t hdr_end = std::string::npos;

    while (hdr_end == std::string::npos) {
        struct pollfd pfd{fd, POLLIN, 0};
        const int pr = ::poll(&pfd, 1, 30000);
        if (pr <= 0) {
            err_out = (pr == 0) ? "Request timeout" : "Poll error";
            return false;
        }
        const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            err_out = "Connection closed";
            return false;
        }
        buf.append(tmp, static_cast<size_t>(n));
        hdr_end = buf.find("\r\n\r\n");
        if (hdr_end == std::string::npos && buf.size() > kMaxHeaderBytes) {
            err_out = "HTTP headers too large";
            return false;
        }
    }

    const std::string hdr_block = buf.substr(0, hdr_end);
    std::string body_part = buf.substr(hdr_end + 4);

    std::istringstream hss(hdr_block);
    std::string req_line;
    if (!std::getline(hss, req_line)) {
        err_out = "Missing HTTP request line";
        return false;
    }
    if (!req_line.empty() && req_line.back() == '\r') req_line.pop_back();
    {
        std::istringstream rls(req_line);
        std::string target;
        rls >> req.method >> target;
        if (req.method.empty() || target.empty()) {
            err_out = "Malformed HTTP request line";
            return false;
        }
        const size_t q = target.find('?');
        req.path = (q == std::string::npos) ? target : target.substr(0, q);
    }

    std::string line;
    while (std::getline(hss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t col = line.find(':');
        if (col == std::string::npos) continue;
        const std::string k = to_lower_ascii(trim_ascii(std::string_view(line).substr(0, col)));
        const std::string v = trim_ascii(std::string_view(line).substr(col + 1));
        req.headers[k] = v;
    }

    const auto te_it = req.headers.find("transfer-encoding");
    const bool is_chunked = (te_it != req.headers.end() && to_lower_ascii(te_it->second).find("chunked") != std::string::npos);

    if (is_chunked) {
        while (true) {
            try {
                if (decode_chunked_body(body_part, req.body, kMaxBodyBytes)) {
                    break;
                }
            } catch (const std::exception& ex) {
                err_out = ex.what();
                return false;
            }
            struct pollfd pfd{fd, POLLIN, 0};
            const int pr = ::poll(&pfd, 1, 30000);
            if (pr <= 0) {
                err_out = "Request body timeout";
                return false;
            }
            const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                err_out = "Unexpected EOF reading chunked body";
                return false;
            }
            body_part.append(tmp, static_cast<size_t>(n));
        }
        return true;
    }

    size_t content_length = 0;
    const auto cl_it = req.headers.find("content-length");
    if (cl_it != req.headers.end()) {
        content_length = static_cast<size_t>(std::strtoull(cl_it->second.c_str(), nullptr, 10));
        if (content_length > kMaxBodyBytes) {
            err_out = "Content-Length exceeds limit";
            return false;
        }
    }

    while (body_part.size() < content_length) {
        struct pollfd pfd{fd, POLLIN, 0};
        const int pr = ::poll(&pfd, 1, 30000);
        if (pr <= 0) {
            err_out = "Request body timeout";
            return false;
        }
        const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            err_out = "Unexpected EOF reading request body";
            return false;
        }
        body_part.append(tmp, static_cast<size_t>(n));
    }
    if (body_part.size() > content_length) {
        body_part.resize(content_length);
    }
    req.body = std::move(body_part);
    return true;
}

bool is_stop_token_srv(int32_t tok, const Tokenizer& tokenizer, const ModelConfig& cfg) {
    if (tokenizer.is_eos(tok)) return true;
    return std::find(cfg.eos_token_ids.begin(), cfg.eos_token_ids.end(), tok) != cfg.eos_token_ids.end();
}

std::vector<int32_t> encode_with_bos_srv(const Tokenizer& tokenizer, const std::string& text) {
    std::vector<int32_t> ids = tokenizer.encode(text, false);
    if (tokenizer.should_add_bos() && tokenizer.bos_id() >= 0 &&
        (ids.empty() || ids[0] != tokenizer.bos_id())) {
        ids.insert(ids.begin(), tokenizer.bos_id());
    }
    return ids;
}

std::vector<ToolCall> extract_tool_calls_from_msg(const json::Value& m) {
    std::vector<ToolCall> out;
    const json::Value* tcs = m.get("tool_calls");
    if (tcs && tcs->is_array()) {
        for (const auto& item : tcs->as_array()) {
            if (!item.is_object()) continue;
            const json::Value* fn = item.get("function");
            const json::Value& src = (fn && fn->is_object()) ? *fn : item;
            const json::Value* nv = src.get("name");
            if (!nv || !nv->is_string() || nv->as_str().empty()) continue;

            ToolCall tc;
            if (const auto* idv = item.get("id"); idv && idv->is_string()) {
                tc.id = idv->as_str();
            }
            tc.name = nv->as_str();
            const json::Value* av = src.get("arguments");
            if (!av) av = src.get("parameters");
            if (!av || av->is_null()) {
                tc.arguments = "{}";
            } else if (av->is_string()) {
                tc.arguments = av->as_str();
            } else {
                tc.arguments = json::dump(*av);
            }
            out.push_back(std::move(tc));
        }
    } else if (const auto* fc = m.get("function_call"); fc && fc->is_object()) {
        const json::Value* nv = fc->get("name");
        if (nv && nv->is_string() && !nv->as_str().empty()) {
            ToolCall tc;
            tc.id = "call_rx_fc";
            tc.name = nv->as_str();
            const json::Value* av = fc->get("arguments");
            if (!av || av->is_null()) tc.arguments = "{}";
            else if (av->is_string()) tc.arguments = av->as_str();
            else tc.arguments = json::dump(*av);
            out.push_back(std::move(tc));
        }
    }
    return out;
}

struct CachedAssistantTurn {
    std::string reasoning_content;
    std::string content;
    std::vector<ToolCall> tool_calls;
    std::string raw_assistant_body;
};

class RecentAssistantTurnCache {
public:
    void remember(const tools::ParsedAssistantOutput& parsed, const std::string& raw_assistant_body) {
        if (raw_assistant_body.empty()) return;
        CachedAssistantTurn entry;
        entry.reasoning_content = parsed.reasoning_content;
        entry.content = parsed.content;
        entry.tool_calls = parsed.tool_calls;
        entry.raw_assistant_body = raw_assistant_body;

        if (!parsed.tool_calls.empty()) {
            for (const auto& tc : parsed.tool_calls) {
                if (tc.id.empty()) continue;
                put_key("tc:" + tc.id, entry);
            }
        }
        const std::string c_trim = trim_ascii(parsed.content);
        if (!c_trim.empty()) {
            put_key("txt:" + c_trim + "\x1f" + trim_ascii(parsed.reasoning_content), entry);
        }
    }

    void hydrate_message(ChatMessage& msg, bool had_explicit_reasoning) const {
        if (!msg.tool_calls.empty()) {
            for (const auto& tc : msg.tool_calls) {
                if (tc.id.empty()) continue;
                const auto it = map_.find("tc:" + tc.id);
                if (it == map_.end()) continue;
                const auto& entry = it->second;
                if (msg.reasoning_content.empty() && !had_explicit_reasoning &&
                    !entry.reasoning_content.empty()) {
                    msg.reasoning_content = entry.reasoning_content;
                }
                if (matches_entry(entry, msg)) {
                    msg.raw_assistant_body = entry.raw_assistant_body;
                }
                return;
            }
        }
        const std::string c_trim = trim_ascii(msg.content);
        if (!c_trim.empty() && had_explicit_reasoning) {
            const auto it = map_.find("txt:" + c_trim + "\x1f" + trim_ascii(msg.reasoning_content));
            if (it != map_.end() && matches_entry(it->second, msg)) {
                msg.raw_assistant_body = it->second.raw_assistant_body;
            }
        }
    }

private:
    static bool matches_entry(const CachedAssistantTurn& entry, const ChatMessage& msg) {
        if (trim_ascii(msg.content) != trim_ascii(entry.content)) return false;
        if (trim_ascii(msg.reasoning_content) != trim_ascii(entry.reasoning_content)) return false;
        if (msg.tool_calls.size() != entry.tool_calls.size()) return false;
        for (size_t i = 0; i < msg.tool_calls.size(); ++i) {
            if (msg.tool_calls[i].id != entry.tool_calls[i].id) return false;
            if (msg.tool_calls[i].name != entry.tool_calls[i].name) return false;
            if (msg.tool_calls[i].arguments != entry.tool_calls[i].arguments) return false;
        }
        return true;
    }

    void put_key(const std::string& key, const CachedAssistantTurn& entry) {
        if (map_.find(key) == map_.end()) {
            order_.push_back(key);
        }
        map_[key] = entry;
        while (order_.size() > kMaxEntries) {
            map_.erase(order_.front());
            order_.pop_front();
        }
    }

    static constexpr size_t kMaxEntries = 512;
    std::unordered_map<std::string, CachedAssistantTurn> map_;
    std::deque<std::string> order_;
};

int create_listen_socket(const std::string& host, int port) {
    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    const std::string port_str = std::to_string(port);
    struct addrinfo* res = nullptr;
    const int gai = ::getaddrinfo(host.empty() ? nullptr : host.c_str(), port_str.c_str(), &hints, &res);
    if (gai != 0) {
        throw std::runtime_error(std::string("getaddrinfo failed: ") + ::gai_strerror(gai));
    }

    int listen_fd = -1;
    int last_err = 0;
    for (struct addrinfo* rp = res; rp != nullptr; rp = rp->ai_next) {
        const int fd = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd < 0) {
            last_err = errno;
            continue;
        }
        int yes = 1;
        ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        if (::bind(fd, rp->ai_addr, rp->ai_addrlen) == 0 && ::listen(fd, 64) == 0) {
            listen_fd = fd;
            break;
        }
        last_err = errno;
        ::close(fd);
    }
    ::freeaddrinfo(res);
    if (listen_fd < 0) {
        throw std::runtime_error("Failed to bind/listen on " + host + ":" + port_str + " (" +
                                 std::strerror(last_err) + ")");
    }
    return listen_fd;
}

} // namespace

ChatCompletionRequest parse_chat_completion_request(const std::string& json_body) {
    const json::Value root = json::parse(json_body);
    if (!root.is_object()) {
        throw std::runtime_error("Request body must be a JSON object");
    }

    ChatCompletionRequest req;
    if (const auto* mv = root.get("model"); mv && mv->is_string()) {
        req.model = mv->as_str();
    }

    const json::Value* msgs_v = root.get("messages");
    if (!msgs_v || !msgs_v->is_array() || msgs_v->as_array().empty()) {
        throw std::runtime_error("Missing or empty 'messages' array");
    }

    for (const auto& item : msgs_v->as_array()) {
        if (!item.is_object()) throw std::runtime_error("Each message must be a JSON object");
        const json::Value* rv = item.get("role");
        if (!rv || !rv->is_string() || rv->as_str().empty()) {
            throw std::runtime_error("Message missing string 'role'");
        }

        ChatMessage msg;
        msg.role = rv->as_str();
        msg.content = extract_content_string(item.get("content"));

        bool had_explicit_reasoning = false;
        const json::Value* rcv = item.get("reasoning_content");
        if (!rcv) rcv = item.get("reasoning");
        if (!rcv) rcv = item.get("thinking");
        if (rcv) {
            if (rcv->is_string()) {
                msg.reasoning_content = rcv->as_str();
                had_explicit_reasoning = true;
            } else if (rcv->is_null()) {
                // Treat null as omitted so fallback cache can help naive OpenAI SDK tool loops
                had_explicit_reasoning = false;
            }
        }
        if (msg.reasoning_content.empty() && msg.role == "assistant" &&
            msg.content.find("<think>") != std::string::npos) {
            std::string emb_r, vis_c;
            Tokenizer::split_thinking(msg.content, emb_r, vis_c);
            if (!emb_r.empty()) {
                msg.reasoning_content = std::move(emb_r);
                msg.content = std::move(vis_c);
                had_explicit_reasoning = true;
            }
        }

        msg.tool_calls = extract_tool_calls_from_msg(item);
        if (const auto* tcid = item.get("tool_call_id"); tcid && tcid->is_string()) {
            msg.tool_call_id = tcid->as_str();
        }
        if (const auto* nm = item.get("name"); nm && nm->is_string()) {
            msg.name = nm->as_str();
        }

        req.messages.push_back(std::move(msg));
        req.message_had_explicit_reasoning.push_back(had_explicit_reasoning);
    }

    if (const auto* tv = root.get("tools"); tv && tv->is_array()) {
        req.tools = parse_tools_array_value(tv->as_array());
    } else if (const auto* fv = root.get("functions"); fv && fv->is_array()) {
        req.tools = parse_tools_array_value(fv->as_array());
    }

    if (const auto* tcv = root.get("tool_choice")) {
        if (tcv->is_string()) {
            req.tool_choice = tcv->as_str();
        } else if (tcv->is_object()) {
            const json::Value* fn = tcv->get("function");
            const json::Value* nv = (fn && fn->is_object()) ? fn->get("name") : tcv->get("name");
            if (nv && nv->is_string() && !nv->as_str().empty()) {
                req.tool_choice = "function";
                req.forced_tool_name = nv->as_str();
            }
        }
    }

    if (const auto* sv = root.get("stream"); sv && sv->is_bool()) {
        req.stream = sv->as_bool(false);
    }
    if (const auto* sov = root.get("stream_options"); sov && sov->is_object()) {
        if (const auto* iuv = sov->get("include_usage"); iuv && iuv->is_bool()) {
            req.stream_include_usage = iuv->as_bool(false);
        }
    }

    if (const auto* v = root.get("temperature"); v && v->is_number()) {
        req.has_temperature = true;
        req.temperature = static_cast<float>(v->as_f64(0.0));
    }
    if (const auto* v = root.get("top_k"); v && v->is_number()) {
        req.has_top_k = true;
        req.top_k = static_cast<int32_t>(v->as_i64(40));
    }
    if (const auto* v = root.get("top_p"); v && v->is_number()) {
        req.has_top_p = true;
        req.top_p = static_cast<float>(v->as_f64(0.95));
    }
    if (const auto* v = root.get("min_p"); v && v->is_number()) {
        req.has_min_p = true;
        req.min_p = static_cast<float>(v->as_f64(0.0));
    }
    if (const auto* v = root.get("repetition_penalty")) {
        if (v->is_number()) {
            req.has_repeat_penalty = true;
            req.repeat_penalty = static_cast<float>(v->as_f64(1.0));
        }
    } else if (const auto* v2 = root.get("repeat_penalty"); v2 && v2->is_number()) {
        req.has_repeat_penalty = true;
        req.repeat_penalty = static_cast<float>(v2->as_f64(1.0));
    }
    if (const auto* v = root.get("presence_penalty"); v && v->is_number()) {
        req.has_presence_penalty = true;
        req.presence_penalty = static_cast<float>(v->as_f64(0.0));
    }
    if (const auto* v = root.get("seed"); v && v->is_number()) {
        req.has_seed = true;
        req.seed = static_cast<uint64_t>(std::max<int64_t>(0, v->as_i64(0)));
    }

    const json::Value* mtv = root.get("max_completion_tokens");
    if (!mtv || mtv->is_null()) mtv = root.get("max_tokens");
    if (mtv && mtv->is_number()) {
        req.has_max_tokens = true;
        const int64_t val = mtv->as_i64(0);
        req.max_tokens = (val <= 0) ? 0 : static_cast<int32_t>(val);
    }

    if (const auto* ctk = root.get("chat_template_kwargs"); ctk && ctk->is_object()) {
        if (const auto* et = ctk->get("enable_thinking"); et && et->is_bool()) {
            req.has_enable_thinking = true;
            req.enable_thinking = et->as_bool(true);
        }
        if (const auto* pt = ctk->get("preserve_thinking"); pt && pt->is_bool()) {
            req.has_preserve_thinking = true;
            req.preserve_thinking = pt->as_bool(true);
        }
        if (const auto* re = ctk->get("reasoning_effort"); re && re->is_string()) {
            req.reasoning_effort = re->as_str();
        }
    }
    if (const auto* et = root.get("enable_thinking"); et && et->is_bool()) {
        req.has_enable_thinking = true;
        req.enable_thinking = et->as_bool(true);
    }
    if (const auto* pt = root.get("preserve_thinking"); pt && pt->is_bool()) {
        req.has_preserve_thinking = true;
        req.preserve_thinking = pt->as_bool(true);
    }
    if (const auto* re = root.get("reasoning_effort"); re && re->is_string()) {
        req.reasoning_effort = re->as_str();
    }
    if (req.reasoning_effort == "none") {
        req.has_enable_thinking = true;
        req.enable_thinking = false;
    }

    if (const auto* ie = root.get("ignore_eos"); ie && ie->is_bool()) {
        req.ignore_eos = ie->as_bool(false);
    }
    if (const auto* stv = root.get("stop")) {
        if (stv->is_string() && !stv->as_str().empty()) {
            req.stop.push_back(stv->as_str());
        } else if (stv->is_array()) {
            for (const auto& el : stv->as_array()) {
                if (el.is_string() && !el.as_str().empty()) {
                    req.stop.push_back(el.as_str());
                }
            }
        }
    }

    return req;
}

namespace {

json::Array build_openai_tool_calls_json(const std::vector<ToolCall>& tool_calls, bool with_index) {
    json::Array arr;
    for (size_t i = 0; i < tool_calls.size(); ++i) {
        const auto& tc = tool_calls[i];
        json::Object fn;
        fn.emplace("name", json::Value{tc.name});
        fn.emplace("arguments", json::Value{tc.arguments.empty() ? std::string("{}") : tc.arguments});

        json::Object obj;
        if (with_index) {
            obj.emplace("index", json::Value{static_cast<double>(i)});
        }
        obj.emplace("id", json::Value{tc.id});
        obj.emplace("type", json::Value{std::string("function")});
        obj.emplace("function", json::Value{std::move(fn)});
        arr.push_back(json::Value{std::move(obj)});
    }
    return arr;
}

} // namespace

std::string build_chat_completion_json(const std::string& id,
                                       int64_t created,
                                       const std::string& model,
                                       const tools::ParsedAssistantOutput& parsed,
                                       const std::string& finish_reason,
                                       int32_t prompt_tokens,
                                       int32_t cached_tokens,
                                       int32_t completion_tokens) {
    json::Object msg;
    msg.emplace("role", json::Value{std::string("assistant")});
    if (parsed.content.empty() && !parsed.tool_calls.empty()) {
        msg.emplace("content", json::Value{nullptr});
    } else {
        msg.emplace("content", json::Value{parsed.content});
    }
    if (!parsed.reasoning_content.empty()) {
        msg.emplace("reasoning_content", json::Value{parsed.reasoning_content});
    }
    if (!parsed.tool_calls.empty()) {
        msg.emplace("tool_calls", json::Value{build_openai_tool_calls_json(parsed.tool_calls, false)});
    }

    json::Object choice;
    choice.emplace("index", json::Value{0.0});
    choice.emplace("message", json::Value{std::move(msg)});
    choice.emplace("finish_reason", json::Value{finish_reason});

    json::Array choices;
    choices.push_back(json::Value{std::move(choice)});

    json::Object ptd;
    ptd.emplace("cached_tokens", json::Value{static_cast<double>(cached_tokens)});

    json::Object usage;
    usage.emplace("prompt_tokens", json::Value{static_cast<double>(prompt_tokens)});
    usage.emplace("completion_tokens", json::Value{static_cast<double>(completion_tokens)});
    usage.emplace("total_tokens", json::Value{static_cast<double>(prompt_tokens + completion_tokens)});
    usage.emplace("prompt_tokens_details", json::Value{std::move(ptd)});

    json::Object root;
    root.emplace("id", json::Value{id});
    root.emplace("object", json::Value{std::string("chat.completion")});
    root.emplace("created", json::Value{static_cast<double>(created)});
    root.emplace("model", json::Value{model});
    root.emplace("choices", json::Value{std::move(choices)});
    root.emplace("usage", json::Value{std::move(usage)});
    return json::dump(json::Value{std::move(root)});
}

std::string build_chat_completion_chunk_json(const std::string& id,
                                             int64_t created,
                                             const std::string& model,
                                             bool include_role,
                                             const std::string& reasoning_delta,
                                             const std::string& content_delta,
                                             const std::vector<ToolCall>& tool_call_deltas,
                                             int tool_call_base_index,
                                             const std::string& finish_reason) {
    json::Object delta;
    if (include_role) {
        delta.emplace("role", json::Value{std::string("assistant")});
        delta.emplace("content", json::Value{std::string("")});
    }
    if (!reasoning_delta.empty()) {
        delta.emplace("reasoning_content", json::Value{reasoning_delta});
    }
    if (!content_delta.empty()) {
        delta.emplace("content", json::Value{content_delta});
    }
    if (!tool_call_deltas.empty()) {
        json::Array arr;
        for (size_t i = 0; i < tool_call_deltas.size(); ++i) {
            const auto& tc = tool_call_deltas[i];
            json::Object fn;
            fn.emplace("name", json::Value{tc.name});
            fn.emplace("arguments", json::Value{tc.arguments.empty() ? std::string("{}") : tc.arguments});
            json::Object obj;
            obj.emplace("index", json::Value{static_cast<double>(tool_call_base_index + static_cast<int>(i))});
            obj.emplace("id", json::Value{tc.id});
            obj.emplace("type", json::Value{std::string("function")});
            obj.emplace("function", json::Value{std::move(fn)});
            arr.push_back(json::Value{std::move(obj)});
        }
        delta.emplace("tool_calls", json::Value{std::move(arr)});
    }

    json::Object choice;
    choice.emplace("index", json::Value{0.0});
    choice.emplace("delta", json::Value{std::move(delta)});
    if (finish_reason.empty()) {
        choice.emplace("finish_reason", json::Value{nullptr});
    } else {
        choice.emplace("finish_reason", json::Value{finish_reason});
    }

    json::Array choices;
    choices.push_back(json::Value{std::move(choice)});

    json::Object root;
    root.emplace("id", json::Value{id});
    root.emplace("object", json::Value{std::string("chat.completion.chunk")});
    root.emplace("created", json::Value{static_cast<double>(created)});
    root.emplace("model", json::Value{model});
    root.emplace("choices", json::Value{std::move(choices)});
    return json::dump(json::Value{std::move(root)});
}

int run_openai_server(InferenceEngine& engine,
                      const Tokenizer& tokenizer,
                      const SamplerConfig& base_sampler_cfg,
                      const ServerOptions& opts) {
    const int listen_fd = create_listen_socket(opts.host, opts.port);
    const int32_t max_seq_len = engine.options().max_seq_len;
    const int32_t vocab_size = engine.config().vocab_size;
    std::vector<uint16_t> logits_f16_scratch;

    RecentAssistantTurnCache assistant_turn_cache;
    int64_t next_req_seq = 1;
    int next_call_id_seed = 1;

    if (!opts.quiet) {
        if (opts.theme.color) {
            std::vector<std::pair<std::string, std::string>> rows = {
                {"Endpoint", "http://" + opts.host + ":" + std::to_string(opts.port) + "/v1"},
                {"Model", opts.model_id},
                {"Context", rxinfer::tui::format_tokens_compact(max_seq_len) + " tok · KV " +
                                std::string(rxinfer::dtype_name(engine.options().kv_dtype))},
                {"Thinking", std::string(opts.default_enable_thinking ? "on" : "off") + " (preserve_thinking=" +
                                 (opts.default_preserve_thinking ? "on" : "off") + ")"},
                {"Default tools", opts.default_tools.empty()
                                      ? "none (client tools supported)"
                                      : (std::to_string(opts.default_tools.size()) + " preloaded")},
            };
            const std::string title = std::string(opts.theme.accent()) + "◆" + opts.theme.reset() + " " +
                                      opts.theme.bold() + "rxinfer OpenAI Server" + opts.theme.reset();
            rxinfer::tui::print_box_card(stderr, opts.theme, title, rows, 90);
            std::fprintf(stderr, "Listening on http://%s:%d (Ctrl-C to stop)\n", opts.host.c_str(), opts.port);
        } else {
            std::fprintf(stderr,
                         "rxinfer OpenAI server listening on http://%s:%d | model=%s | max_seq=%d | kv=%s | think=%s | preserve_thinking=%s\n",
                         opts.host.c_str(), opts.port, opts.model_id.c_str(), max_seq_len,
                         std::string(rxinfer::dtype_name(engine.options().kv_dtype)).c_str(),
                         opts.default_enable_thinking ? "on" : "off",
                         opts.default_preserve_thinking ? "on" : "off");
        }
    }

    while (!(opts.stop_flag && *opts.stop_flag)) {
        struct pollfd pfd{listen_fd, POLLIN, 0};
        const int pr = ::poll(&pfd, 1, 200);
        if (pr < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (pr == 0) continue;

        const int client_fd = ::accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            break;
        }

        HttpRequest http_req;
        std::string read_err;
        if (!read_http_request(client_fd, http_req, read_err)) {
            send_openai_error(client_fd, 400, "Bad Request", read_err, "invalid_request_error");
            ::close(client_fd);
            continue;
        }

        if (http_req.method == "OPTIONS") {
            const std::string resp =
                "HTTP/1.1 204 No Content\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                "Connection: close\r\n\r\n";
            send_all_str(client_fd, resp);
            ::close(client_fd);
            continue;
        }

        if ((http_req.path == "/health" || http_req.path == "/v1/health") && http_req.method == "GET") {
            json::Object h;
            h.emplace("status", json::Value{std::string("ok")});
            h.emplace("model", json::Value{opts.model_id});
            h.emplace("max_seq_len", json::Value{static_cast<double>(max_seq_len)});
            h.emplace("cached_tokens", json::Value{static_cast<double>(engine.cached_tokens().size())});
            h.emplace("kv_dtype", json::Value{std::string(rxinfer::dtype_name(engine.options().kv_dtype))});
            send_http_json(client_fd, 200, "OK", json::dump(json::Value{std::move(h)}));
            ::close(client_fd);
            continue;
        }

        if (!opts.api_key.empty()) {
            const auto ait = http_req.headers.find("authorization");
            const std::string expected = "Bearer " + opts.api_key;
            if (ait == http_req.headers.end() || ait->second != expected) {
                send_openai_error(client_fd, 401, "Unauthorized", "Invalid API key", "authentication_error",
                                  "invalid_api_key");
                ::close(client_fd);
                continue;
            }
        }

        if (http_req.path == "/v1/models" && http_req.method == "GET") {
            json::Object m;
            m.emplace("id", json::Value{opts.model_id});
            m.emplace("object", json::Value{std::string("model")});
            m.emplace("created", json::Value{static_cast<double>(std::time(nullptr))});
            m.emplace("owned_by", json::Value{std::string("rxinfer")});
            m.emplace("max_model_len", json::Value{static_cast<double>(max_seq_len)});
            json::Array data;
            data.push_back(json::Value{std::move(m)});
            json::Object root;
            root.emplace("object", json::Value{std::string("list")});
            root.emplace("data", json::Value{std::move(data)});
            send_http_json(client_fd, 200, "OK", json::dump(json::Value{std::move(root)}));
            ::close(client_fd);
            continue;
        }

        if (http_req.path != "/v1/chat/completions" || http_req.method != "POST") {
            send_openai_error(client_fd, 404, "Not Found", "Unknown endpoint: " + http_req.method + " " + http_req.path,
                              "invalid_request_error");
            ::close(client_fd);
            continue;
        }

        ChatCompletionRequest creq;
        try {
            creq = parse_chat_completion_request(http_req.body);
        } catch (const std::exception& ex) {
            send_openai_error(client_fd, 400, "Bad Request", ex.what(), "invalid_request_error");
            ::close(client_fd);
            continue;
        }

        const std::string resp_model = creq.model.empty() ? opts.model_id : creq.model;
        std::vector<ChatMessage> messages = std::move(creq.messages);

        // Recover missing reasoning_content on tool loops AND attach raw_assistant_body for 100% KV+GDN reuse.
        for (size_t mi = 0; mi < messages.size(); ++mi) {
            if (messages[mi].role == "assistant") {
                assistant_turn_cache.hydrate_message(messages[mi], creq.message_had_explicit_reasoning[mi]);
            }
        }

        if (!opts.default_system_prompt.empty()) {
            const bool has_sys = (!messages.empty() &&
                                  (messages[0].role == "system" || messages[0].role == "developer"));
            if (!has_sys) {
                ChatMessage sys_msg;
                sys_msg.role = "system";
                sys_msg.content = opts.default_system_prompt;
                messages.insert(messages.begin(), std::move(sys_msg));
            }
        }

        std::vector<ToolFunctionDef> effective_tools =
            !creq.tools.empty() ? creq.tools : opts.default_tools;
        if (creq.tool_choice == "none") {
            effective_tools.clear();
        } else if (creq.tool_choice == "function" && !creq.forced_tool_name.empty()) {
            std::vector<ToolFunctionDef> filtered;
            for (const auto& t : effective_tools) {
                if (t.name == creq.forced_tool_name) filtered.push_back(t);
            }
            if (!filtered.empty()) effective_tools = std::move(filtered);
            ChatMessage force_note;
            force_note.role = "system";
            force_note.content = "You MUST call the function '" + creq.forced_tool_name + "' in your response.";
            messages.insert(messages.begin(), std::move(force_note));
        } else if (creq.tool_choice == "required" && !effective_tools.empty()) {
            ChatMessage req_note;
            req_note.role = "system";
            req_note.content = "You MUST respond by calling one of the available functions.";
            messages.insert(messages.begin(), std::move(req_note));
        }

        const bool has_think_tokens =
            (tokenizer.token_to_id("<think>") >= 0 && tokenizer.token_to_id("</think>") >= 0);

        ChatTemplateOptions tpl_opts;
        tpl_opts.add_generation_prompt = true;
        tpl_opts.enable_thinking =
            creq.has_enable_thinking ? creq.enable_thinking : opts.default_enable_thinking;
        if (!has_think_tokens) tpl_opts.enable_thinking = false;
        tpl_opts.preserve_thinking =
            creq.has_preserve_thinking ? creq.preserve_thinking : opts.default_preserve_thinking;
        tpl_opts.reasoning_effort =
            !creq.reasoning_effort.empty() ? creq.reasoning_effort : opts.default_reasoning_effort;

        const std::string prompt_text = tokenizer.format_chat_messages(messages, effective_tools, tpl_opts);
        const std::vector<int32_t> prompt_ids = encode_with_bos_srv(tokenizer, prompt_text);

        if (prompt_ids.empty()) {
            send_openai_error(client_fd, 400, "Bad Request", "Prompt tokenized to 0 tokens", "invalid_request_error");
            ::close(client_fd);
            continue;
        }
        if (static_cast<int32_t>(prompt_ids.size()) >= max_seq_len) {
            send_openai_error(client_fd, 400, "Bad Request",
                              "Prompt tokens (" + std::to_string(prompt_ids.size()) +
                                  ") >= max_seq_len (" + std::to_string(max_seq_len) + ")",
                              "invalid_request_error", "context_length_exceeded");
            ::close(client_fd);
            continue;
        }

        SamplerConfig sc = base_sampler_cfg;
        if (creq.has_temperature) sc.temperature = creq.temperature;
        if (creq.has_top_k) sc.top_k = creq.top_k;
        if (creq.has_top_p) sc.top_p = creq.top_p;
        if (creq.has_min_p) sc.min_p = creq.min_p;
        if (creq.has_repeat_penalty) sc.repeat_penalty = creq.repeat_penalty;
        if (creq.has_presence_penalty) sc.presence_penalty = creq.presence_penalty;
        if (creq.has_seed) sc.seed = creq.seed;

        Sampler sampler(sc);
        sampler.record_tokens(prompt_ids);

        const int32_t max_tokens = creq.has_max_tokens ? creq.max_tokens : opts.default_max_tokens;
        const bool ignore_eos = creq.ignore_eos || opts.ignore_eos;

        const auto tp0 = std::chrono::steady_clock::now();
        const PrefillCacheResult cache_res = engine.prefill_cached(prompt_ids, true);
        int32_t next_tok = cache_res.next_token;
        if (sampler.config().uses_host_logits() && cache_res.evaluated_tokens > 0) {
            engine.copy_last_logits_f16(logits_f16_scratch);
            next_tok = sampler.sample_from_f16_bits(logits_f16_scratch.data(), vocab_size);
        }
        const auto tp1 = std::chrono::steady_clock::now();
        const double prefill_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

        const std::string comp_id = "chatcmpl-rx-" + std::to_string(next_req_seq++);
        const int64_t created_ts = static_cast<int64_t>(std::time(nullptr));
        const int32_t prompt_tok_count = static_cast<int32_t>(prompt_ids.size());
        const int32_t cached_tok_count = cache_res.matched_prefix_tokens;

        const bool started_inside_think = tpl_opts.enable_thinking && has_think_tokens;
        tools::StreamingAssistantParser stream_parser(started_inside_think, effective_tools, next_call_id_seed);
        std::string raw_generated;

        bool stream_ok = true;
        int streamed_tool_count = 0;

        size_t max_stop_len = 0;
        for (const auto& st : creq.stop) {
            max_stop_len = std::max(max_stop_len, st.size());
        }
        std::string content_holdback;

        auto emit_sse_chunk = [&](bool include_role,
                                  const std::string& r_delta,
                                  const std::string& c_delta,
                                  const std::vector<ToolCall>& tc_deltas,
                                  const std::string& finish_reason) {
            if (!creq.stream || !stream_ok) return;
            const std::string payload = build_chat_completion_chunk_json(
                comp_id, created_ts, resp_model, include_role, r_delta, c_delta, tc_deltas,
                streamed_tool_count, finish_reason);
            streamed_tool_count += static_cast<int>(tc_deltas.size());
            if (!send_all_str(client_fd, "data: " + payload + "\n\n")) {
                stream_ok = false;
            }
        };

        if (creq.stream) {
            const std::string hdr =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream; charset=utf-8\r\n"
                "Cache-Control: no-cache\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n\r\n";
            if (!send_all_str(client_fd, hdr)) {
                stream_ok = false;
            } else {
                emit_sse_chunk(true, "", "", {}, "");
            }
        }

        int32_t generated = 0;
        bool hit_length = false;
        bool hit_stop_str = false;
        const auto td_start = std::chrono::steady_clock::now();

        for (int32_t step = 0; max_tokens <= 0 || step < max_tokens; ++step) {
            if (!stream_ok || (opts.stop_flag && *opts.stop_flag)) break;
            if (!ignore_eos && is_stop_token_srv(next_tok, tokenizer, engine.config())) break;

            const std::string& raw_tok = tokenizer.id_to_token(next_tok);
            const std::string piece =
                (raw_tok == "<think>" || raw_tok == "</think>" || raw_tok == "<tool_call>" ||
                 raw_tok == "</tool_call>" || raw_tok == "<|python_tag|>" || raw_tok == "<|eom_id|>")
                    ? raw_tok
                    : tokenizer.decode_token(next_tok, true);
            raw_generated += piece;
            sampler.record_token(next_tok);
            ++generated;

            auto d = stream_parser.feed(piece);
            if (!creq.stop.empty() && !d.content.empty()) {
                content_holdback += d.content;
                d.content.clear();
                size_t earliest = std::string::npos;
                for (const auto& st : creq.stop) {
                    const size_t pos = content_holdback.find(st);
                    if (pos != std::string::npos) earliest = std::min(earliest, pos);
                }
                if (earliest != std::string::npos) {
                    d.content = content_holdback.substr(0, earliest);
                    content_holdback.clear();
                    hit_stop_str = true;
                } else if (content_holdback.size() >= max_stop_len) {
                    const size_t flush_n = content_holdback.size() - (max_stop_len - 1);
                    d.content = content_holdback.substr(0, flush_n);
                    content_holdback.erase(0, flush_n);
                }
            }
            if (!d.reasoning_content.empty() || !d.content.empty() || !d.tool_calls.empty()) {
                emit_sse_chunk(false, d.reasoning_content, d.content, d.tool_calls, "");
            }
            if (hit_stop_str) break;

            if (engine.position() >= max_seq_len) {
                hit_length = true;
                break;
            }
            if (max_tokens > 0 && step + 1 >= max_tokens) {
                hit_length = true;
                break;
            }

            int32_t cand_tok = engine.decode_step(next_tok);
            if (sampler.config().uses_host_logits()) {
                engine.copy_last_logits_f16(logits_f16_scratch);
                cand_tok = sampler.sample_from_f16_bits(logits_f16_scratch.data(), vocab_size);
            }
            next_tok = cand_tok;
        }

        auto tail = stream_parser.finish();
        if (!creq.stop.empty() && !hit_stop_str) {
            content_holdback += tail.content;
            tail.content.clear();
            size_t earliest = std::string::npos;
            for (const auto& st : creq.stop) {
                const size_t pos = content_holdback.find(st);
                if (pos != std::string::npos) earliest = std::min(earliest, pos);
            }
            if (earliest != std::string::npos) {
                tail.content = content_holdback.substr(0, earliest);
                hit_stop_str = true;
            } else {
                tail.content = std::move(content_holdback);
            }
            content_holdback.clear();
        }
        if (!tail.reasoning_content.empty() || !tail.content.empty() || !tail.tool_calls.empty()) {
            emit_sse_chunk(false, tail.reasoning_content, tail.content, tail.tool_calls, "");
        }

        // Save post-completion prefix checkpoint so subsequent tool/user turns reuse KV + GDN state.
        if (engine.position() >= 8) {
            engine.save_prefix_checkpoint(false);
        }

        tools::ParsedAssistantOutput parsed =
            tools::parse_assistant_output(raw_generated, started_inside_think, effective_tools, next_call_id_seed);
        if (!creq.stop.empty()) {
            size_t earliest = std::string::npos;
            for (const auto& st : creq.stop) {
                const size_t pos = parsed.content.find(st);
                if (pos != std::string::npos) earliest = std::min(earliest, pos);
            }
            if (earliest != std::string::npos) {
                parsed.content.resize(earliest);
                parsed.content = trim_ascii(parsed.content);
            }
        }
        next_call_id_seed += static_cast<int>(parsed.tool_calls.size());

        if (!hit_length && !hit_stop_str) {
            std::string raw_body;
            if (started_inside_think) {
                raw_body = "<think>\n" + raw_generated;
            } else if (has_think_tokens) {
                raw_body = "<think>\n\n</think>\n\n" + raw_generated;
            } else {
                raw_body = raw_generated;
            }
            assistant_turn_cache.remember(parsed, raw_body);
        }

        const std::string finish_reason =
            !parsed.tool_calls.empty() ? "tool_calls" : (hit_length ? "length" : "stop");

        if (creq.stream) {
            if (stream_ok) {
                emit_sse_chunk(false, "", "", {}, finish_reason);
                if (creq.stream_include_usage && stream_ok) {
                    json::Object ptd;
                    ptd.emplace("cached_tokens", json::Value{static_cast<double>(cached_tok_count)});
                    json::Object usage;
                    usage.emplace("prompt_tokens", json::Value{static_cast<double>(prompt_tok_count)});
                    usage.emplace("completion_tokens", json::Value{static_cast<double>(generated)});
                    usage.emplace("total_tokens", json::Value{static_cast<double>(prompt_tok_count + generated)});
                    usage.emplace("prompt_tokens_details", json::Value{std::move(ptd)});
                    json::Object uchunk;
                    uchunk.emplace("id", json::Value{comp_id});
                    uchunk.emplace("object", json::Value{std::string("chat.completion.chunk")});
                    uchunk.emplace("created", json::Value{static_cast<double>(created_ts)});
                    uchunk.emplace("model", json::Value{resp_model});
                    uchunk.emplace("choices", json::Value{json::Array{}});
                    uchunk.emplace("usage", json::Value{std::move(usage)});
                    send_all_str(client_fd, "data: " + json::dump(json::Value{std::move(uchunk)}) + "\n\n");
                }
                if (stream_ok) {
                    send_all_str(client_fd, "data: [DONE]\n\n");
                }
            }
        } else {
            const std::string resp_json = build_chat_completion_json(
                comp_id, created_ts, resp_model, parsed, finish_reason,
                prompt_tok_count, cached_tok_count, generated);
            send_http_json(client_fd, 200, "OK", resp_json);
        }

        ::close(client_fd);

        if (!opts.quiet) {
            const auto td_end = std::chrono::steady_clock::now();
            const double dec_ms = std::chrono::duration<double, std::milli>(td_end - td_start).count();
            const double dec_tps = (generated > 0 && dec_ms > 0.0) ? (generated * 1000.0 / dec_ms) : 0.0;
            std::fprintf(stderr,
                         "[%s] %s | prompt=%d (cached=%d, new=%d, %.1fms) | out=%d tok (%.1f tok/s) | tools=%zu | finish=%s\n",
                         comp_id.c_str(), creq.stream ? "SSE" : "JSON",
                         prompt_tok_count, cached_tok_count, cache_res.evaluated_tokens, prefill_ms,
                         generated, dec_tps, parsed.tool_calls.size(), finish_reason.c_str());
        }
    }

    ::close(listen_fd);
    return 0;
}

} // namespace server
} // namespace rxinfer

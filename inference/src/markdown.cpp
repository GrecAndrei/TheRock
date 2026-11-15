// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/markdown.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>

namespace rxinfer {

namespace {

bool is_ascii_digit(char c) {
    return c >= '0' && c <= '9';
}

bool is_ascii_alnum(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

bool is_horiz_ws(char c) {
    return c == ' ' || c == '\t';
}

std::string trim_ascii(std::string_view sv) {
    size_t a = 0;
    while (a < sv.size() && (sv[a] == ' ' || sv[a] == '\t' || sv[a] == '\r')) ++a;
    size_t b = sv.size();
    while (b > a && (sv[b - 1] == ' ' || sv[b - 1] == '\t' || sv[b - 1] == '\r')) --b;
    return std::string(sv.substr(a, b - a));
}

bool is_valid_hr_line(std::string_view trimmed_after_indent, char marker) {
    int count = 0;
    for (char c : trimmed_after_indent) {
        if (c == marker) {
            ++count;
        } else if (c != ' ' && c != '\t' && c != '\r') {
            return false;
        }
    }
    return count >= 3;
}

} // namespace

MarkdownStreamFormatter::MarkdownStreamFormatter(bool enabled, EmitFn emit)
    : enabled_(enabled), flush_stdout_(!emit), emit_(std::move(emit)) {
    if (!emit_) {
        emit_ = [](std::string_view sv) {
            if (!sv.empty()) {
                std::fwrite(sv.data(), 1, sv.size(), stdout);
            }
        };
    }
}

MarkdownStreamFormatter::~MarkdownStreamFormatter() {
    if (applied_style_.any()) {
        force_reset_style();
    }
}

void MarkdownStreamFormatter::set_enabled(bool enabled) {
    if (enabled_ == enabled) return;
    finish();
    enabled_ = enabled;
}

void MarkdownStreamFormatter::reset() {
    if (applied_style_.any()) {
        force_reset_style();
    }
    utf8_buf_.clear();
    in_think_ = false;
    in_code_fence_ = false;
    in_inline_code_ = false;
    in_blockquote_ = false;
    in_heading_ = 0;
    in_bold_ = false;
    in_italic_ = false;
    in_strike_ = false;
    bold_delim_ = 0;
    italic_delim_ = 0;
    inline_code_ticks_ = 0;
    at_line_start_ = true;
    eat_leading_ws_ = false;
    in_fence_info_ = false;
    fence_char_ = 0;
    fence_len_ = 0;
    line_prefix_.clear();
    fence_info_.clear();
    backslash_pending_ = false;
    inline_delim_.clear();
    tag_buf_.clear();
    consecutive_newlines_ = 0;
    prev_char_kind_ = CharKind::Whitespace;
}

void MarkdownStreamFormatter::emit_raw(std::string_view sv) {
    if (!sv.empty() && emit_) {
        emit_(sv);
    }
}

MarkdownStreamFormatter::StyleState MarkdownStreamFormatter::desired_style() const {
    StyleState s{};
    if (!enabled_) return s;

    if (in_think_) {
        s.dim = true;
        s.italic = true;
    }
    if (in_code_fence_) {
        s.dim = false;
        s.italic = false;
        s.fg = 36;
        return s;
    }
    if (in_inline_code_) {
        s.dim = false;
        s.italic = false;
        s.fg = 36;
        if (in_heading_ > 0) s.bold = true;
        return s;
    }
    if (in_heading_ > 0) {
        s.bold = true;
        if (in_heading_ == 1) {
            s.underline = true;
            s.fg = 36;
        } else if (in_heading_ == 2) {
            s.fg = 36;
        } else if (in_heading_ == 3) {
            s.fg = 33;
        } else {
            s.fg = 35;
        }
    }
    if (in_blockquote_) {
        s.dim = true;
        s.italic = true;
    }
    if (in_bold_) s.bold = true;
    if (in_italic_) s.italic = true;
    if (in_strike_) s.strike = true;
    return s;
}

void MarkdownStreamFormatter::sync_style() {
    if (!enabled_) return;
    const StyleState want = desired_style();
    if (want == applied_style_) return;

    if (!want.any()) {
        emit_raw("\x1b[0m");
        applied_style_ = want;
        return;
    }

    std::string seq = "\x1b[0";
    if (want.bold) seq += ";1";
    if (want.dim) seq += ";2";
    if (want.italic) seq += ";3";
    if (want.underline) seq += ";4";
    if (want.strike) seq += ";9";
    if (want.fg != 0) {
        seq.push_back(';');
        seq += std::to_string(want.fg);
    }
    seq.push_back('m');
    emit_raw(seq);
    applied_style_ = want;
}

void MarkdownStreamFormatter::force_reset_style() {
    if (applied_style_.any()) {
        emit_raw("\x1b[0m");
        applied_style_ = StyleState{};
    }
}

MarkdownStreamFormatter::CharKind MarkdownStreamFormatter::classify_cp(std::string_view cp) {
    if (cp.empty()) return CharKind::Whitespace;
    if (cp.size() > 1) return CharKind::Word; // Non-ASCII UTF-8 codepoints act as word chars
    const unsigned char c = static_cast<unsigned char>(cp[0]);
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f') {
        return CharKind::Whitespace;
    }
    if (is_ascii_alnum(static_cast<char>(c))) {
        return CharKind::Word;
    }
    return CharKind::Punct;
}

bool MarkdownStreamFormatter::is_escapable_punct(char c) {
    switch (c) {
        case '\\':
        case '`':
        case '*':
        case '_':
        case '{':
        case '}':
        case '[':
        case ']':
        case '(':
        case ')':
        case '#':
        case '+':
        case '-':
        case '.':
        case '!':
        case '|':
        case '~':
        case '>':
            return true;
        default:
            return false;
    }
}

void MarkdownStreamFormatter::push(std::string_view chunk) {
    if (chunk.empty()) return;
    utf8_buf_.append(chunk.data(), chunk.size());
    flush_utf8(false);
    if (flush_stdout_) {
        std::fflush(stdout);
    }
}

void MarkdownStreamFormatter::flush_utf8(bool force_all) {
    size_t pos = 0;
    while (pos < utf8_buf_.size()) {
        const uint8_t b = static_cast<uint8_t>(utf8_buf_[pos]);
        size_t need = 1;
        if ((b & 0x80u) == 0) {
            need = 1;
        } else if ((b & 0xE0u) == 0xC0u) {
            need = 2;
        } else if ((b & 0xF0u) == 0xE0u) {
            need = 3;
        } else if ((b & 0xF8u) == 0xF0u) {
            need = 4;
        } else {
            need = 1;
        }

        if (pos + need > utf8_buf_.size()) {
            if (!force_all) break;
            need = utf8_buf_.size() - pos;
        } else if (need > 1) {
            bool valid = true;
            for (size_t k = 1; k < need; ++k) {
                if ((static_cast<uint8_t>(utf8_buf_[pos + k]) & 0xC0u) != 0x80u) {
                    valid = false;
                    break;
                }
            }
            if (!valid) need = 1;
        }

        const std::string_view cp(utf8_buf_.data() + pos, need);
        if (!enabled_) {
            emit_raw(cp);
        } else {
            feed_cp(cp, true);
        }
        pos += need;
    }
    if (pos > 0) {
        utf8_buf_.erase(0, pos);
    }
}

void MarkdownStreamFormatter::feed_cp(std::string_view cp, bool allow_tag_match) {
    if (cp.size() == 1 && cp[0] == '\r') {
        return; // Normalize CRLF cleanly
    }

    if (in_code_fence_) {
        if (at_line_start_) {
            feed_line_start_fence_close(cp);
        } else {
            if (cp.size() == 1 && cp[0] == '\n') {
                emit_raw("\n");
                at_line_start_ = true;
            } else {
                sync_style();
                emit_raw(cp);
            }
        }
        return;
    }

    // Intercept <think> and </think> tags (outside code fences & inline code)
    if (allow_tag_match && !in_inline_code_) {
        if (!tag_buf_.empty()) {
            if (cp.size() == 1) {
                tag_buf_.push_back(cp[0]);
                constexpr std::string_view kOpen = "<think>";
                constexpr std::string_view kClose = "</think>";
                if (tag_buf_ == kOpen) {
                    tag_buf_.clear();
                    clear_inline_styles();
                    in_think_ = true;
                    force_reset_style();
                    emit_raw("\x1b[2;3m[thinking]\x1b[0m");
                    sync_style();
                    return;
                }
                if (tag_buf_ == kClose) {
                    tag_buf_.clear();
                    clear_inline_styles();
                    in_think_ = false;
                    force_reset_style();
                    if (!at_line_start_) {
                        emit_raw("\n");
                        at_line_start_ = true;
                    }
                    emit_raw("\x1b[2m[/thinking]\x1b[0m");
                    sync_style();
                    return;
                }
                const bool is_prefix =
                    (kOpen.rfind(tag_buf_, 0) == 0) || (kClose.rfind(tag_buf_, 0) == 0);
                if (is_prefix) {
                    return;
                }
            } else {
                tag_buf_.append(cp.data(), cp.size());
            }

            std::string replay = std::move(tag_buf_);
            tag_buf_.clear();
            for (size_t i = 0; i < replay.size();) {
                const uint8_t b = static_cast<uint8_t>(replay[i]);
                size_t len = 1;
                if ((b & 0xE0u) == 0xC0u) len = 2;
                else if ((b & 0xF0u) == 0xE0u) len = 3;
                else if ((b & 0xF8u) == 0xF0u) len = 4;
                len = std::min(len, replay.size() - i);
                feed_cp(std::string_view(replay.data() + i, len), false);
                i += len;
            }
            return;
        }

        if (cp.size() == 1 && cp[0] == '<') {
            if (at_line_start_ && !line_prefix_.empty()) {
                flush_line_prefix_as_inline();
            }
            if (backslash_pending_) {
                backslash_pending_ = false;
                emit_literal_text("\\<");
                return;
            }
            if (!inline_delim_.empty()) {
                resolve_inline_delim("<", CharKind::Punct);
            }
            tag_buf_ = "<";
            return;
        }
    }

    if (at_line_start_) {
        feed_line_start(cp);
    } else {
        feed_inline(cp, allow_tag_match);
    }
}

void MarkdownStreamFormatter::feed_line_start_fence_close(std::string_view cp) {
    if (cp.size() == 1 && cp[0] == '\n') {
        size_t i = 0;
        while (i < line_prefix_.size() && line_prefix_[i] == ' ') ++i;
        size_t fcount = 0;
        while (i < line_prefix_.size() && line_prefix_[i] == fence_char_) {
            ++fcount;
            ++i;
        }
        bool trailing_ws_only = true;
        while (i < line_prefix_.size()) {
            if (!is_horiz_ws(line_prefix_[i])) {
                trailing_ws_only = false;
                break;
            }
            ++i;
        }
        if (fcount >= static_cast<size_t>(fence_len_) && trailing_ws_only) {
            line_prefix_.clear();
            in_code_fence_ = false;
            fence_char_ = 0;
            fence_len_ = 0;
            force_reset_style();
            emit_raw("\x1b[2m────────────────────────────────\x1b[0m\n");
            sync_style();
            at_line_start_ = true;
            return;
        }
        sync_style();
        emit_raw(line_prefix_);
        emit_raw("\n");
        line_prefix_.clear();
        at_line_start_ = true;
        return;
    }

    if (cp.size() == 1 && line_prefix_.size() < 24) {
        const char ch = cp[0];
        size_t indent = 0;
        while (indent < line_prefix_.size() && line_prefix_[indent] == ' ') ++indent;
        size_t fcount = 0;
        while (indent + fcount < line_prefix_.size() && line_prefix_[indent + fcount] == fence_char_) {
            ++fcount;
        }
        const bool only_indent_so_far = (indent == line_prefix_.size());
        const bool in_fence_run = (indent <= 3) && (indent + fcount == line_prefix_.size());
        if ((only_indent_so_far && ch == ' ' && indent < 3) ||
            (in_fence_run && ch == fence_char_) ||
            (fcount >= static_cast<size_t>(fence_len_) && is_horiz_ws(ch))) {
            line_prefix_.push_back(ch);
            return;
        }
    }

    sync_style();
    emit_raw(line_prefix_);
    line_prefix_.clear();
    at_line_start_ = false;
    emit_raw(cp);
}

void MarkdownStreamFormatter::feed_line_start(std::string_view cp) {
    if (in_fence_info_) {
        if (cp.size() == 1 && cp[0] == '\n') {
            const std::string lang = trim_ascii(fence_info_);
            line_prefix_.clear();
            fence_info_.clear();
            in_fence_info_ = false;
            in_code_fence_ = true;
            clear_inline_styles();
            in_heading_ = 0;
            in_blockquote_ = false;
            force_reset_style();
            if (lang.empty()) {
                emit_raw("\x1b[2m────────────────────────────────\x1b[0m\n");
            } else {
                emit_raw("\x1b[2m─── ");
                emit_raw(lang);
                emit_raw(" ───\x1b[0m\n");
            }
            sync_style();
            at_line_start_ = true;
            return;
        }
        if (cp.size() == 1 && fence_char_ == '`' && cp[0] == '`') {
            // Backtick info string cannot contain backticks -> fallback to inline
            in_fence_info_ = false;
            std::string combined = line_prefix_ + fence_info_;
            line_prefix_.clear();
            fence_info_.clear();
            at_line_start_ = false;
            for (char c : combined) feed_inline(std::string_view(&c, 1), false);
            feed_inline(cp, false);
            return;
        }
        if (fence_info_.size() + cp.size() > 48) {
            in_fence_info_ = false;
            std::string combined = line_prefix_ + fence_info_;
            line_prefix_.clear();
            fence_info_.clear();
            at_line_start_ = false;
            for (char c : combined) feed_inline(std::string_view(&c, 1), false);
            feed_inline(cp, false);
            return;
        }
        fence_info_.append(cp.data(), cp.size());
        return;
    }

    if (cp.size() == 1 && cp[0] == '\n') {
        // Check if line_prefix_ is a horizontal rule (---, ***, ___)
        size_t indent = 0;
        while (indent < line_prefix_.size() && line_prefix_[indent] == ' ') ++indent;
        const std::string_view rest(line_prefix_.data() + indent, line_prefix_.size() - indent);
        if (indent <= 3 && !rest.empty() &&
            (rest[0] == '-' || rest[0] == '*' || rest[0] == '_') &&
            is_valid_hr_line(rest, rest[0])) {
            line_prefix_.clear();
            force_reset_style();
            emit_raw("\x1b[2m────────────────────────────────────────\x1b[0m\n");
            sync_style();
            at_line_start_ = true;
            consecutive_newlines_ = 1;
            prev_char_kind_ = CharKind::Whitespace;
            return;
        }
        if (indent <= 3 && rest == ">") {
            line_prefix_.clear();
            sync_style();
            emit_raw(std::string(indent, ' '));
            emit_raw("\x1b[36m▎\x1b[0m\n");
            sync_style();
            at_line_start_ = true;
            consecutive_newlines_ = 1;
            prev_char_kind_ = CharKind::Whitespace;
            return;
        }
        flush_line_prefix_as_inline();
        emit_newline();
        return;
    }

    if (cp.size() != 1) {
        flush_line_prefix_as_inline();
        feed_inline(cp, true);
        return;
    }

    const char ch = cp[0];
    size_t indent = 0;
    while (indent < line_prefix_.size() && line_prefix_[indent] == ' ') ++indent;
    const std::string_view rest(line_prefix_.data() + indent, line_prefix_.size() - indent);

    // 1. Leading spaces
    if (rest.empty()) {
        if (ch == ' ' && indent < 12) {
            line_prefix_.push_back(ch);
            return;
        }
        if (ch == '#' || ch == '*' || ch == '-' || ch == '+' || ch == '>' ||
            ch == '`' || ch == '~' || ch == '_' || is_ascii_digit(ch)) {
            line_prefix_.push_back(ch);
            return;
        }
        flush_line_prefix_as_inline();
        feed_inline(cp, true);
        return;
    }

    // 2. Headings: #{1,6} followed by space
    if (rest[0] == '#') {
        if (indent <= 3 && ch == '#' && rest.size() < 6) {
            line_prefix_.push_back(ch);
            return;
        }
        if (indent <= 3 && is_horiz_ws(ch)) {
            in_heading_ = static_cast<int>(rest.size());
            line_prefix_.clear();
            at_line_start_ = false;
            eat_leading_ws_ = true;
            consecutive_newlines_ = 0;
            prev_char_kind_ = CharKind::Whitespace;
            sync_style();
            return;
        }
        flush_line_prefix_as_inline();
        feed_inline(cp, true);
        return;
    }

    // 3. Blockquote: > followed by space
    if (rest == ">") {
        if (indent <= 3 && is_horiz_ws(ch)) {
            line_prefix_.clear();
            at_line_start_ = false;
            eat_leading_ws_ = true;
            in_blockquote_ = true;
            consecutive_newlines_ = 0;
            prev_char_kind_ = CharKind::Whitespace;
            force_reset_style();
            if (indent > 0) emit_raw(std::string(indent, ' '));
            emit_raw("\x1b[36m▎\x1b[0m ");
            sync_style();
            return;
        }
        flush_line_prefix_as_inline();
        feed_inline(cp, true);
        return;
    }

    // 4. Bullet + (plus only)
    if (rest == "+") {
        if (is_horiz_ws(ch)) {
            line_prefix_.clear();
            at_line_start_ = false;
            eat_leading_ws_ = true;
            consecutive_newlines_ = 0;
            prev_char_kind_ = CharKind::Whitespace;
            force_reset_style();
            const size_t level = indent / 2;
            const std::string pad(level * 2, ' ');
            emit_raw(pad);
            emit_raw(level == 0 ? "\x1b[36m•\x1b[0m " : (level == 1 ? "\x1b[36m◦\x1b[0m " : "\x1b[36m▪\x1b[0m "));
            sync_style();
            return;
        }
        flush_line_prefix_as_inline();
        feed_inline(cp, true);
        return;
    }

    // 5. '-' (bullet or --- HR)
    if (rest[0] == '-') {
        if (rest.size() == 1 && is_horiz_ws(ch)) {
            line_prefix_.clear();
            at_line_start_ = false;
            eat_leading_ws_ = true;
            consecutive_newlines_ = 0;
            prev_char_kind_ = CharKind::Whitespace;
            force_reset_style();
            const size_t level = indent / 2;
            emit_raw(std::string(level * 2, ' '));
            emit_raw(level == 0 ? "\x1b[36m•\x1b[0m " : (level == 1 ? "\x1b[36m◦\x1b[0m " : "\x1b[36m▪\x1b[0m "));
            sync_style();
            return;
        }
        if (indent <= 3 && (ch == '-' || (rest.size() >= 3 && is_horiz_ws(ch))) && line_prefix_.size() < 24) {
            line_prefix_.push_back(ch);
            return;
        }
        flush_line_prefix_as_inline();
        feed_inline(cp, true);
        return;
    }

    // 6. '_' (___ HR or _italic_ / __bold__)
    if (rest[0] == '_') {
        if (indent <= 3 && (ch == '_' || (rest.size() >= 3 && is_horiz_ws(ch))) && line_prefix_.size() < 24) {
            line_prefix_.push_back(ch);
            return;
        }
        flush_line_prefix_as_inline();
        feed_inline(cp, true);
        return;
    }

    // 7. '*' (bullet '* ', '***' HR, or *italic* / **bold** / ***bold italic***)
    if (rest[0] == '*') {
        if (rest.size() == 1 && is_horiz_ws(ch)) {
            line_prefix_.clear();
            at_line_start_ = false;
            eat_leading_ws_ = true;
            consecutive_newlines_ = 0;
            prev_char_kind_ = CharKind::Whitespace;
            force_reset_style();
            const size_t level = indent / 2;
            emit_raw(std::string(level * 2, ' '));
            emit_raw(level == 0 ? "\x1b[36m•\x1b[0m " : (level == 1 ? "\x1b[36m◦\x1b[0m " : "\x1b[36m▪\x1b[0m "));
            sync_style();
            return;
        }
        if (ch == '*' && rest.size() < 6) {
            line_prefix_.push_back(ch);
            return;
        }
        if (indent <= 3 && rest.size() >= 3 && is_horiz_ws(ch) && line_prefix_.size() < 24) {
            line_prefix_.push_back(ch);
            return;
        }
        flush_line_prefix_as_inline();
        feed_inline(cp, true);
        return;
    }

    // 8. Code fence opener (` or ~)
    if (rest[0] == '`' || rest[0] == '~') {
        const char m = rest[0];
        if (indent <= 3 && ch == m && rest.size() < 6) {
            line_prefix_.push_back(ch);
            return;
        }
        if (indent <= 3 && rest.size() >= 3) {
            fence_char_ = m;
            fence_len_ = static_cast<int>(rest.size());
            in_fence_info_ = true;
            fence_info_.clear();
            feed_line_start(cp);
            return;
        }
        flush_line_prefix_as_inline();
        feed_inline(cp, true);
        return;
    }

    // 9. Ordered list: 1. 2. ...
    if (is_ascii_digit(rest[0])) {
        const bool has_dot = (rest.back() == '.');
        if (!has_dot) {
            if (is_ascii_digit(ch) && rest.size() < 4) {
                line_prefix_.push_back(ch);
                return;
            }
            if (ch == '.') {
                line_prefix_.push_back(ch);
                return;
            }
        } else if (is_horiz_ws(ch)) {
            const std::string num_label(rest);
            line_prefix_.clear();
            at_line_start_ = false;
            eat_leading_ws_ = true;
            consecutive_newlines_ = 0;
            prev_char_kind_ = CharKind::Whitespace;
            force_reset_style();
            if (indent > 0) emit_raw(std::string(indent, ' '));
            emit_raw("\x1b[36m");
            emit_raw(num_label);
            emit_raw("\x1b[0m ");
            sync_style();
            return;
        }
        flush_line_prefix_as_inline();
        feed_inline(cp, true);
        return;
    }

    flush_line_prefix_as_inline();
    feed_inline(cp, true);
}

void MarkdownStreamFormatter::flush_line_prefix_as_inline() {
    at_line_start_ = false;
    if (line_prefix_.empty()) return;
    std::string prefix = std::move(line_prefix_);
    line_prefix_.clear();

    size_t indent = 0;
    while (indent < prefix.size() && prefix[indent] == ' ') ++indent;
    if (indent > 0) {
        emit_literal_text(std::string_view(prefix.data(), indent), true);
    }
    for (size_t i = indent; i < prefix.size(); ++i) {
        feed_inline(std::string_view(prefix.data() + i, 1), false);
    }
}

void MarkdownStreamFormatter::feed_inline(std::string_view cp, bool /*allow_tag_match*/) {
    if (eat_leading_ws_) {
        if (cp.size() == 1 && is_horiz_ws(cp[0])) {
            return;
        }
        eat_leading_ws_ = false;
    }

    if (cp.size() == 1 && cp[0] == '\n') {
        if (backslash_pending_) {
            backslash_pending_ = false;
            emit_literal_text("\\", true);
        }
        if (!inline_delim_.empty()) {
            resolve_inline_delim("\n", CharKind::Whitespace);
        }
        emit_newline();
        return;
    }

    consecutive_newlines_ = 0;

    if (in_inline_code_) {
        if (cp.size() == 1 && cp[0] == '`') {
            if (!inline_delim_.empty() && inline_delim_[0] != '`') {
                resolve_inline_delim("`", CharKind::Punct);
            }
            inline_delim_.push_back('`');
            return;
        }
        if (!inline_delim_.empty()) {
            resolve_inline_delim(cp, classify_cp(cp));
        }
        emit_literal_text(cp, true);
        return;
    }

    if (backslash_pending_) {
        backslash_pending_ = false;
        if (cp.size() == 1 && is_escapable_punct(cp[0])) {
            emit_literal_text(cp, true);
            return;
        }
        emit_literal_text("\\", true);
        // Fall through to process cp normally
    }

    if (cp.size() == 1) {
        const char ch = cp[0];
        if (ch == '\\') {
            if (!inline_delim_.empty()) {
                resolve_inline_delim("\\", CharKind::Punct);
            }
            backslash_pending_ = true;
            return;
        }
        if (ch == '*' || ch == '_' || ch == '~' || ch == '`') {
            if (!inline_delim_.empty() && inline_delim_[0] != ch) {
                resolve_inline_delim(cp, CharKind::Punct);
            }
            inline_delim_.push_back(ch);
            return;
        }
    }

    if (!inline_delim_.empty()) {
        resolve_inline_delim(cp, classify_cp(cp));
    }
    emit_literal_text(cp, true);
}

void MarkdownStreamFormatter::resolve_inline_delim(std::string_view /*next_cp*/, CharKind next_kind) {
    if (inline_delim_.empty()) return;
    std::string run = std::move(inline_delim_);
    inline_delim_.clear();

    const char d = run[0];
    const int n = static_cast<int>(run.size());

    if (d == '`') {
        if (in_inline_code_) {
            if (n == inline_code_ticks_) {
                in_inline_code_ = false;
                inline_code_ticks_ = 0;
                sync_style();
            } else {
                emit_literal_text(run, true);
            }
        } else {
            in_inline_code_ = true;
            inline_code_ticks_ = n;
            sync_style();
        }
        return;
    }

    if (d == '~') {
        if (n == 2) {
            const bool can_close = (prev_char_kind_ != CharKind::Whitespace);
            const bool can_open = (next_kind != CharKind::Whitespace);
            if (in_strike_ && can_close) {
                in_strike_ = false;
                sync_style();
                return;
            }
            if (!in_strike_ && can_open) {
                in_strike_ = true;
                sync_style();
                return;
            }
        }
        emit_literal_text(run, true);
        return;
    }

    if (d == '*' || d == '_') {
        if (n <= 3) {
            apply_emphasis_run(d, n, prev_char_kind_, next_kind);
        } else {
            // Very long run like **** -> literal
            emit_literal_text(run, true);
        }
    }
}

void MarkdownStreamFormatter::apply_emphasis_run(char delim, int count, CharKind prev_kind, CharKind next_kind) {
    bool can_close = (prev_kind != CharKind::Whitespace);
    bool can_open = (next_kind != CharKind::Whitespace);

    if (delim == '_') {
        if (prev_kind == CharKind::Word) can_open = false;
        if (next_kind == CharKind::Word) can_close = false;
    } else if (delim == '*' && count == 1) {
        // Keep math expressions like 2*x + 3*y literal when Word sits on both sides.
        if (prev_kind == CharKind::Word && next_kind == CharKind::Word) {
            can_open = false;
            can_close = false;
        }
    }

    if (count == 1) {
        if (in_italic_ && italic_delim_ == delim && can_close) {
            in_italic_ = false;
            italic_delim_ = 0;
            sync_style();
            return;
        }
        if (!in_italic_ && can_open) {
            in_italic_ = true;
            italic_delim_ = delim;
            sync_style();
            return;
        }
        emit_literal_text(std::string_view(&delim, 1), true);
        return;
    }

    if (count == 2) {
        if (in_bold_ && bold_delim_ == delim && can_close) {
            in_bold_ = false;
            bold_delim_ = 0;
            sync_style();
            return;
        }
        if (!in_bold_ && can_open) {
            in_bold_ = true;
            bold_delim_ = delim;
            sync_style();
            return;
        }
        const std::string two(2, delim);
        emit_literal_text(two, true);
        return;
    }

    // count == 3
    if (in_bold_ && bold_delim_ == delim && in_italic_ && italic_delim_ == delim && can_close) {
        in_bold_ = false;
        in_italic_ = false;
        bold_delim_ = 0;
        italic_delim_ = 0;
        sync_style();
        return;
    }
    if (!in_bold_ && !in_italic_ && can_open) {
        in_bold_ = true;
        in_italic_ = true;
        bold_delim_ = delim;
        italic_delim_ = delim;
        sync_style();
        return;
    }
    if (in_bold_ && bold_delim_ == delim && !in_italic_) {
        if (can_close && can_open) {
            in_bold_ = false;
            bold_delim_ = 0;
            in_italic_ = true;
            italic_delim_ = delim;
            sync_style();
            return;
        }
        if (can_close) {
            in_bold_ = false;
            bold_delim_ = 0;
            sync_style();
            emit_literal_text(std::string_view(&delim, 1), true);
            return;
        }
    }
    if (in_italic_ && italic_delim_ == delim && !in_bold_) {
        if (can_close && can_open) {
            in_italic_ = false;
            italic_delim_ = 0;
            in_bold_ = true;
            bold_delim_ = delim;
            sync_style();
            return;
        }
        if (can_close) {
            in_italic_ = false;
            italic_delim_ = 0;
            sync_style();
            const std::string two(2, delim);
            emit_literal_text(two, true);
            return;
        }
    }
    const std::string three(3, delim);
    emit_literal_text(three, true);
}

void MarkdownStreamFormatter::emit_literal_text(std::string_view sv, bool update_prev_kind) {
    if (sv.empty()) return;
    sync_style();
    emit_raw(sv);
    if (update_prev_kind) {
        if (sv.size() == 1) {
            prev_char_kind_ = classify_cp(sv);
        } else {
            const uint8_t last = static_cast<uint8_t>(sv.back());
            if ((last & 0x80u) != 0) {
                prev_char_kind_ = CharKind::Word;
            } else {
                prev_char_kind_ = classify_cp(sv.substr(sv.size() - 1, 1));
            }
        }
    }
}

void MarkdownStreamFormatter::clear_inline_styles() {
    in_bold_ = false;
    in_italic_ = false;
    in_strike_ = false;
    in_inline_code_ = false;
    bold_delim_ = 0;
    italic_delim_ = 0;
    inline_code_ticks_ = 0;
}

void MarkdownStreamFormatter::emit_newline() {
    in_heading_ = 0;
    in_blockquote_ = false;
    eat_leading_ws_ = false;
    at_line_start_ = true;
    ++consecutive_newlines_;
    if (consecutive_newlines_ >= 2) {
        clear_inline_styles();
    }
    sync_style();
    emit_raw("\n");
    prev_char_kind_ = CharKind::Whitespace;
}

void MarkdownStreamFormatter::finish() {
    flush_utf8(true);

    if (!enabled_) {
        std::fflush(stdout);
        return;
    }

    if (!tag_buf_.empty()) {
        std::string replay = std::move(tag_buf_);
        tag_buf_.clear();
        for (size_t i = 0; i < replay.size();) {
            const uint8_t b = static_cast<uint8_t>(replay[i]);
            size_t len = 1;
            if ((b & 0xE0u) == 0xC0u) len = 2;
            else if ((b & 0xF0u) == 0xE0u) len = 3;
            else if ((b & 0xF8u) == 0xF0u) len = 4;
            len = std::min(len, replay.size() - i);
            feed_cp(std::string_view(replay.data() + i, len), false);
            i += len;
        }
    }

    if (in_code_fence_) {
        if (at_line_start_ && !line_prefix_.empty()) {
            size_t indent = 0;
            while (indent < line_prefix_.size() && line_prefix_[indent] == ' ') ++indent;
            size_t fcount = 0;
            while (indent + fcount < line_prefix_.size() && line_prefix_[indent + fcount] == fence_char_) {
                ++fcount;
            }
            bool trailing_ws_only = true;
            for (size_t i = indent + fcount; i < line_prefix_.size(); ++i) {
                if (!is_horiz_ws(line_prefix_[i])) trailing_ws_only = false;
            }
            if (fcount >= static_cast<size_t>(fence_len_) && trailing_ws_only) {
                line_prefix_.clear();
                in_code_fence_ = false;
                force_reset_style();
                emit_raw("\x1b[2m────────────────────────────────\x1b[0m");
            } else {
                sync_style();
                emit_raw(line_prefix_);
                line_prefix_.clear();
            }
        }
        in_code_fence_ = false;
    } else if (at_line_start_) {
        if (in_fence_info_) {
            in_fence_info_ = false;
            std::string combined = line_prefix_ + fence_info_;
            line_prefix_.clear();
            fence_info_.clear();
            at_line_start_ = false;
            for (char c : combined) feed_inline(std::string_view(&c, 1), false);
        } else if (!line_prefix_.empty()) {
            size_t indent = 0;
            while (indent < line_prefix_.size() && line_prefix_[indent] == ' ') ++indent;
            const std::string_view rest(line_prefix_.data() + indent, line_prefix_.size() - indent);
            if (indent <= 3 && !rest.empty() &&
                (rest[0] == '-' || rest[0] == '*' || rest[0] == '_') &&
                is_valid_hr_line(rest, rest[0])) {
                line_prefix_.clear();
                force_reset_style();
                emit_raw("\x1b[2m────────────────────────────────────────\x1b[0m");
            } else {
                flush_line_prefix_as_inline();
            }
        }
    }

    if (backslash_pending_) {
        backslash_pending_ = false;
        emit_literal_text("\\", true);
    }
    if (!inline_delim_.empty()) {
        resolve_inline_delim("", CharKind::Whitespace);
    }

    clear_inline_styles();
    in_heading_ = 0;
    in_blockquote_ = false;
    in_think_ = false;
    in_code_fence_ = false;
    at_line_start_ = true;
    eat_leading_ws_ = false;
    force_reset_style();
    if (flush_stdout_) {
        std::fflush(stdout);
    }
}

} // namespace rxinfer

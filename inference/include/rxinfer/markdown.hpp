// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace rxinfer {

// Streaming UTF-8 + ANSI Markdown terminal formatter for LLM token output.
// - When enabled == false, acts as a clean UTF-8 boundary-buffered pass-through.
// - When enabled == true, renders headings, bold (** / __), italic (* / _),
//   bold-italic (*** / ___), strikethrough (~~), bullet & numbered lists,
//   blockquotes (>), horizontal rules (--- / *** / ___), inline code (`...`),
//   fenced code blocks (```lang ... ``` / ~~~), and <think>...</think> blocks
//   with safe ANSI SGR transitions and guaranteed reset on finish().
class MarkdownStreamFormatter {
public:
    using EmitFn = std::function<void(std::string_view)>;

    explicit MarkdownStreamFormatter(bool enabled = true, EmitFn emit = nullptr);
    ~MarkdownStreamFormatter();

    MarkdownStreamFormatter(const MarkdownStreamFormatter&) = delete;
    MarkdownStreamFormatter& operator=(const MarkdownStreamFormatter&) = delete;

    void set_enabled(bool enabled);
    bool enabled() const { return enabled_; }

    void push(std::string_view chunk);
    void finish();
    void reset();

private:
    enum class CharKind : uint8_t {
        Whitespace = 0,
        Punct = 1,
        Word = 2,
    };

    struct StyleState {
        bool bold = false;
        bool dim = false;
        bool italic = false;
        bool underline = false;
        bool strike = false;
        uint8_t fg = 0; // 0 = default, 32/33/35/36 = ANSI foreground code

        bool operator==(const StyleState& o) const {
            return bold == o.bold && dim == o.dim && italic == o.italic &&
                   underline == o.underline && strike == o.strike && fg == o.fg;
        }
        bool operator!=(const StyleState& o) const { return !(*this == o); }
        bool any() const { return bold || dim || italic || underline || strike || fg != 0; }
    };

    bool enabled_ = true;
    bool flush_stdout_ = false;
    EmitFn emit_;

    // UTF-8 byte reassembly buffer
    std::string utf8_buf_;

    // Style tracking
    StyleState applied_style_{};
    bool in_think_ = false;
    bool in_code_fence_ = false;
    bool in_inline_code_ = false;
    bool in_blockquote_ = false;
    int in_heading_ = 0;
    bool in_bold_ = false;
    bool in_italic_ = false;
    bool in_strike_ = false;
    char bold_delim_ = 0;
    char italic_delim_ = 0;
    int inline_code_ticks_ = 0;

    // Line-start disambiguation & fence header state
    bool at_line_start_ = true;
    bool eat_leading_ws_ = false;
    bool in_fence_info_ = false;
    char fence_char_ = 0;
    int fence_len_ = 0;
    std::string line_prefix_;
    std::string fence_info_;

    // Inline lookahead buffers
    bool backslash_pending_ = false;
    std::string inline_delim_;
    std::string tag_buf_;
    int consecutive_newlines_ = 0;
    CharKind prev_char_kind_ = CharKind::Whitespace;

    void emit_raw(std::string_view sv);
    void flush_utf8(bool force_all);
    void feed_cp(std::string_view cp, bool allow_tag_match);

    void feed_line_start_fence_close(std::string_view cp);
    void feed_line_start(std::string_view cp);
    void flush_line_prefix_as_inline();

    void feed_inline(std::string_view cp, bool allow_tag_match);
    void resolve_inline_delim(std::string_view next_cp, CharKind next_kind);
    void apply_emphasis_run(char delim, int count, CharKind prev_kind, CharKind next_kind);
    void emit_literal_text(std::string_view sv, bool update_prev_kind = true);
    void emit_newline();
    void clear_inline_styles();

    StyleState desired_style() const;
    void sync_style();
    void force_reset_style();

    static CharKind classify_cp(std::string_view cp);
    static bool is_escapable_punct(char c);
};

} // namespace rxinfer

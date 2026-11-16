// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/tui.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>

namespace rxinfer {
namespace tui {
namespace {

const char* kSpinnerFrames[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
constexpr int kNumSpinnerFrames = 10;

bool decode_utf8_cp(std::string_view sv, size_t pos, uint32_t& out_cp, size_t& out_len) {
    if (pos >= sv.size()) {
        out_cp = 0;
        out_len = 0;
        return false;
    }
    const uint8_t b0 = static_cast<uint8_t>(sv[pos]);
    if ((b0 & 0x80u) == 0) {
        out_cp = b0;
        out_len = 1;
        return true;
    }
    size_t need = 1;
    uint32_t cp = 0;
    if ((b0 & 0xE0u) == 0xC0u) {
        need = 2;
        cp = b0 & 0x1Fu;
    } else if ((b0 & 0xF0u) == 0xE0u) {
        need = 3;
        cp = b0 & 0x0Fu;
    } else if ((b0 & 0xF8u) == 0xF0u) {
        need = 4;
        cp = b0 & 0x07u;
    } else {
        out_cp = 0xFFFDu;
        out_len = 1;
        return true;
    }
    if (pos + need > sv.size()) {
        out_cp = 0xFFFDu;
        out_len = 1;
        return true;
    }
    for (size_t i = 1; i < need; ++i) {
        const uint8_t bi = static_cast<uint8_t>(sv[pos + i]);
        if ((bi & 0xC0u) != 0x80u) {
            out_cp = 0xFFFDu;
            out_len = 1;
            return true;
        }
        cp = (cp << 6) | (bi & 0x3Fu);
    }
    out_cp = cp;
    out_len = need;
    return true;
}

int codepoint_width(uint32_t cp) {
    if (cp < 0x20u || (cp >= 0x7Fu && cp < 0xA0u)) return 0;
    if ((cp >= 0x0300u && cp <= 0x036Fu) || (cp >= 0x1AB0u && cp <= 0x1AFFu) ||
        (cp >= 0x1DC0u && cp <= 0x1DFFu) || (cp >= 0x200Bu && cp <= 0x200Fu) ||
        (cp >= 0x2028u && cp <= 0x202Eu) || (cp >= 0xFE00u && cp <= 0xFE0Fu) ||
        (cp >= 0xFE20u && cp <= 0xFE2Fu)) {
        return 0;
    }
    if ((cp >= 0x1100u && cp <= 0x115Fu) || cp == 0x2329u || cp == 0x232Au ||
        (cp >= 0x2E80u && cp <= 0x303Eu) || (cp >= 0x3040u && cp <= 0xA4CFu) ||
        (cp >= 0xAC00u && cp <= 0xD7A3u) || (cp >= 0xF900u && cp <= 0xFAFFu) ||
        (cp >= 0xFE10u && cp <= 0xFE6Fu) || (cp >= 0xFF00u && cp <= 0xFF60u) ||
        (cp >= 0xFFE0u && cp <= 0xFFE6u) || (cp >= 0x1F300u && cp <= 0x1FAFFu) ||
        (cp >= 0x20000u && cp <= 0x3FFFDu)) {
        return 2;
    }
    return 1;
}

size_t utf8_prev_boundary(const std::string& s, size_t pos) {
    if (pos == 0) return 0;
    size_t i = pos - 1;
    while (i > 0 && (static_cast<uint8_t>(s[i]) & 0xC0u) == 0x80u) --i;
    return i;
}

size_t utf8_next_boundary(const std::string& s, size_t pos) {
    if (pos >= s.size()) return s.size();
    uint32_t cp = 0;
    size_t len = 1;
    decode_utf8_cp(s, pos, cp, len);
    return std::min(s.size(), pos + std::max<size_t>(1, len));
}

bool is_word_byte(unsigned char c) {
    return std::isalnum(c) || c == '_' || c == '-' || c == '/' || c == '.';
}

bool read_byte_timeout(int fd, uint8_t& out, int timeout_ms) {
    struct pollfd pfd{};
    pfd.fd = fd;
    pfd.events = POLLIN;
    while (true) {
        const int pr = ::poll(&pfd, 1, timeout_ms);
        if (pr < 0) {
            if (errno == EINTR) {
                if (timeout_ms < 0) continue;
                return false;
            }
            return false;
        }
        if (pr == 0) return false;
        const ssize_t r = ::read(fd, &out, 1);
        if (r == 1) return true;
        if (r < 0 && errno == EINTR) continue;
        return false;
    }
}

void write_all_fd(int fd, std::string_view sv) {
    size_t off = 0;
    while (off < sv.size()) {
        const ssize_t w = ::write(fd, sv.data() + off, sv.size() - off);
        if (w < 0) {
            if (errno == EINTR) continue;
            break;
        }
        off += static_cast<size_t>(w);
    }
}

std::string longest_common_prefix(const std::vector<std::string>& items) {
    if (items.empty()) return "";
    std::string p = items[0];
    for (size_t i = 1; i < items.size(); ++i) {
        size_t j = 0;
        const size_t n = std::min(p.size(), items[i].size());
        while (j < n && p[j] == items[i][j]) ++j;
        p.resize(j);
    }
    return p;
}

int edit_distance(std::string_view a, std::string_view b) {
    const size_t n = a.size();
    const size_t m = b.size();
    std::vector<int> prev(m + 1), cur(m + 1);
    for (size_t j = 0; j <= m; ++j) prev[j] = static_cast<int>(j);
    for (size_t i = 1; i <= n; ++i) {
        cur[0] = static_cast<int>(i);
        for (size_t j = 1; j <= m; ++j) {
            const int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost});
        }
        std::swap(prev, cur);
    }
    return prev[m];
}

std::string expand_tilde_prefix(const std::string& p) {
    if (!p.empty() && p[0] == '~' && (p.size() == 1 || p[1] == '/')) {
        const char* home = std::getenv("HOME");
        if (home && home[0] != '\0') {
            return std::string(home) + p.substr(1);
        }
    }
    return p;
}

struct RawTermGuard {
    int fd = -1;
    struct termios orig{};
    bool active = false;

    explicit RawTermGuard(int tty_fd) : fd(tty_fd) {
        if (::tcgetattr(fd, &orig) == 0) {
            struct termios raw = orig;
            raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
            raw.c_oflag |= OPOST;
            raw.c_cflag |= CS8;
            raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
            raw.c_cc[VMIN] = 1;
            raw.c_cc[VTIME] = 0;
            if (::tcsetattr(fd, TCSAFLUSH, &raw) == 0) {
                active = true;
            }
        }
    }
    ~RawTermGuard() {
        restore();
    }
    void restore() {
        if (active) {
            (void)::tcsetattr(fd, TCSAFLUSH, &orig);
            active = false;
        }
    }
    RawTermGuard(const RawTermGuard&) = delete;
    RawTermGuard& operator=(const RawTermGuard&) = delete;
};

struct VisualLineSlice {
    std::string prefix_styled;
    size_t prefix_width = 0;
    size_t byte_start = 0;
    size_t byte_end = 0;
    std::string ghost;
};

} // namespace

UiTheme UiTheme::detect(bool allow_color) {
    UiTheme t{};
    t.tty_stderr = (::isatty(STDERR_FILENO) != 0);
    t.tty_stdout = (::isatty(STDOUT_FILENO) != 0);

    const char* no_color = std::getenv("NO_COLOR");
    const bool has_no_color = (no_color != nullptr && no_color[0] != '\0');
    const char* term = std::getenv("TERM");
    const bool dumb_term = (term != nullptr && std::strcmp(term, "dumb") == 0);
    const char* colorterm = std::getenv("COLORTERM");

    t.color = allow_color && t.tty_stderr && !has_no_color && !dumb_term;
    if (t.color && colorterm != nullptr) {
        const std::string ct(colorterm);
        if (ct.find("truecolor") != std::string::npos || ct.find("24bit") != std::string::npos) {
            t.truecolor = true;
        }
    }
    return t;
}

void query_terminal_size(int fd, int& out_cols, int& out_rows) {
    struct winsize ws{};
    if (::ioctl(fd, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0) {
        out_cols = static_cast<int>(ws.ws_col);
        out_rows = static_cast<int>(ws.ws_row);
        return;
    }
    out_cols = 80;
    out_rows = 24;
}

size_t display_width(std::string_view sv) {
    size_t w = 0;
    for (size_t i = 0; i < sv.size();) {
        if (sv[i] == '\x1b' && i + 1 < sv.size() && sv[i + 1] == '[') {
            size_t j = i + 2;
            while (j < sv.size() && !(sv[j] >= '@' && sv[j] <= '~')) ++j;
            i = (j < sv.size()) ? (j + 1) : sv.size();
            continue;
        }
        uint32_t cp = 0;
        size_t len = 1;
        decode_utf8_cp(sv, i, cp, len);
        w += static_cast<size_t>(codepoint_width(cp));
        i += std::max<size_t>(1, len);
    }
    return w;
}

std::string truncate_display_width(std::string_view sv, size_t max_cols, const UiTheme& t) {
    if (display_width(sv) <= max_cols) return std::string(sv);
    if (max_cols == 0) return "";
    const size_t target = (max_cols > 1) ? (max_cols - 1) : 0;

    std::string out;
    out.reserve(sv.size() + 8);
    size_t w = 0;
    bool saw_ansi = false;
    for (size_t i = 0; i < sv.size();) {
        if (sv[i] == '\x1b' && i + 1 < sv.size() && sv[i + 1] == '[') {
            size_t j = i + 2;
            while (j < sv.size() && !(sv[j] >= '@' && sv[j] <= '~')) ++j;
            const size_t end = (j < sv.size()) ? (j + 1) : sv.size();
            out.append(sv.data() + i, end - i);
            saw_ansi = true;
            i = end;
            continue;
        }
        uint32_t cp = 0;
        size_t len = 1;
        decode_utf8_cp(sv, i, cp, len);
        const size_t cw = static_cast<size_t>(codepoint_width(cp));
        if (w + cw > target) break;
        out.append(sv.data() + i, len);
        w += cw;
        i += std::max<size_t>(1, len);
    }
    if (saw_ansi && t.color) out += t.reset();
    if (max_cols > 1) out += "…";
    return out;
}

std::string format_tokens_compact(int64_t n) {
    char buf[32];
    if (n >= 1000000) {
        std::snprintf(buf, sizeof(buf), "%.1fM", n / 1000000.0);
    } else if (n >= 100000) {
        std::snprintf(buf, sizeof(buf), "%.0fk", n / 1000.0);
    } else if (n >= 1000) {
        std::snprintf(buf, sizeof(buf), "%.1fk", n / 1000.0);
    } else {
        std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(n));
    }
    return std::string(buf);
}

std::string format_duration_compact(double ms) {
    char buf[32];
    if (ms < 1000.0) {
        std::snprintf(buf, sizeof(buf), "%.0f ms", ms);
    } else if (ms < 60000.0) {
        std::snprintf(buf, sizeof(buf), "%.1f s", ms / 1000.0);
    } else {
        const int total_s = static_cast<int>(ms / 1000.0 + 0.5);
        std::snprintf(buf, sizeof(buf), "%dm %02ds", total_s / 60, total_s % 60);
    }
    return std::string(buf);
}

std::string clean_model_display_name(const std::string& model_path, const std::string& fallback_type) {
    if (model_path == "synthetic" || model_path.empty()) {
        return fallback_type.empty() ? "model" : fallback_type;
    }
    std::string p = model_path;
    while (p.size() > 1 && (p.back() == '/' || p.back() == '\\')) p.pop_back();
    const size_t slash = p.find_last_of("/\\");
    std::string base = (slash == std::string::npos) ? p : p.substr(slash + 1);
    auto strip_ext = [&](const char* ext) {
        const size_t elen = std::strlen(ext);
        if (base.size() > elen && base.compare(base.size() - elen, elen, ext) == 0) {
            base.resize(base.size() - elen);
            return true;
        }
        return false;
    };
    if (!strip_ext(".rxr4")) {
        strip_ext(".gguf");
    }
    if (base.size() > 42) {
        base = base.substr(0, 41) + "…";
    }
    return base.empty() ? fallback_type : base;
}

std::string render_progress_bar(double frac, int width, const UiTheme& t) {
    frac = std::clamp(frac, 0.0, 1.0);
    width = std::max(4, width);
    const int filled = static_cast<int>(frac * width + 0.5);

    std::string bar;
    if (filled > 0) {
        bar += t.accent();
        for (int i = 0; i < filled; ++i) {
            bar += (i == filled - 1 && filled < width) ? "╸" : "━";
        }
        bar += t.reset();
    }
    if (filled < width) {
        bar += t.subtle();
        for (int i = filled; i < width; ++i) bar += "─";
        bar += t.reset();
    }
    return bar;
}

void print_box_card(FILE* out,
                    const UiTheme& t,
                    const std::string& title,
                    const std::vector<std::pair<std::string, std::string>>& rows,
                    int max_width) {
    int cols = 80, trows = 24;
    query_terminal_size(::fileno(out), cols, trows);

    if (cols < 54) {
        std::fprintf(out, "\n%s%s%s\n", t.bold(), title.c_str(), t.reset());
        for (const auto& [k, v] : rows) {
            if (k.empty()) {
                std::fprintf(out, "  %s\n", v.c_str());
            } else {
                std::fprintf(out, "  %s%-9s%s %s\n", t.muted(), k.c_str(), t.reset(), v.c_str());
            }
        }
        std::fprintf(out, "\n");
        return;
    }

    const int box_w = std::clamp(cols - 2, 54, max_width);
    const int inner_w = box_w - 2;

    size_t key_w = 0;
    for (const auto& [k, _] : rows) {
        key_w = std::max(key_w, display_width(k));
    }
    key_w = std::min<size_t>(key_w, 12);

    // Top border: ╭─ <title> ────...╮
    const std::string clean_title = truncate_display_width(title, static_cast<size_t>(std::max(8, inner_w - 6)), t);
    const size_t title_dw = display_width(clean_title);
    const int trail_dashes = std::max(1, inner_w - 3 - static_cast<int>(title_dw));

    std::fprintf(out, "%s╭─%s %s %s", t.subtle(), t.reset(), clean_title.c_str(), t.subtle());
    for (int i = 0; i < trail_dashes; ++i) std::fputs("─", out);
    std::fprintf(out, "╮%s\n", t.reset());

    for (const auto& [k, v] : rows) {
        std::string content = "  ";
        if (!k.empty()) {
            content += t.muted();
            content += k;
            const size_t kw = display_width(k);
            for (size_t i = kw; i < key_w; ++i) content.push_back(' ');
            content += t.reset();
            content += "  ";
        }
        content += v;

        const int content_max = std::max(1, inner_w - 1);
        const std::string clipped = truncate_display_width(content, static_cast<size_t>(content_max), t);
        const size_t cw = display_width(clipped);
        const int pad = std::max(0, inner_w - static_cast<int>(cw));

        std::fprintf(out, "%s│%s%s", t.subtle(), t.reset(), clipped.c_str());
        for (int i = 0; i < pad; ++i) std::fputc(' ', out);
        std::fprintf(out, "%s│%s\n", t.subtle(), t.reset());
    }

    std::fprintf(out, "%s╰", t.subtle());
    for (int i = 0; i < inner_w; ++i) std::fputs("─", out);
    std::fprintf(out, "╯%s\n", t.reset());
}

void print_status_badge(FILE* out, const UiTheme& t, const char* kind, const std::string& msg) {
    const char* glyph = "•";
    const char* col = t.muted();
    if (std::strcmp(kind, "ok") == 0) {
        glyph = "✓";
        col = t.ok();
    } else if (std::strcmp(kind, "warn") == 0) {
        glyph = "!";
        col = t.warn();
    } else if (std::strcmp(kind, "err") == 0) {
        glyph = "✗";
        col = t.err();
    } else if (std::strcmp(kind, "info") == 0) {
        glyph = "›";
        col = t.accent();
    }
    std::fprintf(out, "  %s%s%s %s%s%s\n", col, glyph, t.reset(), t.muted(), msg.c_str(), t.reset());
}

LoadProgressReporter::LoadProgressReporter(const UiTheme& theme, bool enabled, std::string label)
    : theme_(theme),
      enabled_(enabled && theme.tty_stderr),
      label_(std::move(label)),
      start_(std::chrono::steady_clock::now()),
      last_draw_(start_) {}

LoadProgressReporter::~LoadProgressReporter() {
    finish();
}

void LoadProgressReporter::update(size_t done_bytes, size_t total_bytes) {
    if (!enabled_ || total_bytes == 0) return;

    std::lock_guard<std::mutex> lk(mu_);
    const auto now = std::chrono::steady_clock::now();
    const double since_draw_ms = std::chrono::duration<double, std::milli>(now - last_draw_).count();
    if (drew_ && done_bytes < total_bytes && since_draw_ms < 85.0) return;
    last_draw_ = now;

    const double frac = std::clamp(static_cast<double>(done_bytes) / static_cast<double>(total_bytes), 0.0, 1.0);
    const double elapsed_s = std::max(0.001, std::chrono::duration<double>(now - start_).count());
    const double mb_s = (static_cast<double>(done_bytes) / (1024.0 * 1024.0)) / elapsed_s;
    const double done_gib = static_cast<double>(done_bytes) / double(1ULL << 30);
    const double total_gib = static_cast<double>(total_bytes) / double(1ULL << 30);

    int cols = 80, rows = 24;
    query_terminal_size(STDERR_FILENO, cols, rows);
    const int bar_w = std::clamp(cols - 56, 10, 26);

    char eta_buf[24] = "";
    if (done_bytes > (32ULL << 20) && frac > 0.02 && frac < 0.995 && mb_s > 1.0) {
        const double rem_s = (elapsed_s / frac) * (1.0 - frac);
        std::snprintf(eta_buf, sizeof(eta_buf), " · %s", format_duration_compact(rem_s * 1000.0).c_str());
    }

    const char* spin = kSpinnerFrames[frame_ % kNumSpinnerFrames];
    frame_++;

    std::string bar = render_progress_bar(frac, bar_w, theme_);
    std::fprintf(stderr,
                 "\r\x1b[2K  %s%s%s %sLoading %s%s  %s  %s%3.0f%%%s  %s%.1f/%.1f GiB · %.0f MB/s%s%s",
                 theme_.accent(), spin, theme_.reset(),
                 theme_.muted(), label_.c_str(), theme_.reset(),
                 bar.c_str(),
                 theme_.bold(), frac * 100.0, theme_.reset(),
                 theme_.muted(), done_gib, total_gib, mb_s, eta_buf, theme_.reset());
    std::fflush(stderr);
    drew_ = true;
}

void LoadProgressReporter::finish() {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lk(mu_);
    if (drew_) {
        std::fprintf(stderr, "\r\x1b[2K");
        std::fflush(stderr);
        drew_ = false;
    }
}

PrefillSpinner::PrefillSpinner(const UiTheme& theme, bool enabled, std::string label)
    : theme_(theme), enabled_(enabled && theme.tty_stderr), label_(std::move(label)) {
    if (!enabled_) return;
    worker_ = std::thread([this]() {
        const auto start = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lk(mu_);
        if (cv_.wait_for(lk, std::chrono::milliseconds(180), [&]() { return stop_requested_; })) {
            return;
        }
        int frame = 0;
        while (!stop_requested_) {
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            const char* spin = kSpinnerFrames[frame % kNumSpinnerFrames];
            frame++;
            std::fprintf(stderr, "\r\x1b[2K  %s%s%s %s%s · %s%s",
                         theme_.accent(), spin, theme_.reset(),
                         theme_.muted(), label_.c_str(),
                         format_duration_compact(elapsed_ms).c_str(),
                         theme_.reset());
            std::fflush(stderr);
            drew_ = true;
            if (cv_.wait_for(lk, std::chrono::milliseconds(80), [&]() { return stop_requested_; })) {
                break;
            }
        }
        if (drew_) {
            std::fprintf(stderr, "\r\x1b[2K");
            std::fflush(stderr);
            drew_ = false;
        }
    });
}

PrefillSpinner::~PrefillSpinner() {
    stop();
}

void PrefillSpinner::stop() {
    if (!enabled_) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stopped_) return;
        stop_requested_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    std::lock_guard<std::mutex> lk(mu_);
    stopped_ = true;
}

InteractiveLineEditor::InteractiveLineEditor(std::vector<CommandSpec> commands)
    : commands_(std::move(commands)) {}

void InteractiveLineEditor::add_history(const std::string& entry) {
    if (entry.empty()) return;
    if (!history_.empty() && history_.back() == entry) return;
    history_.push_back(entry);
    if (history_.size() > 500) {
        history_.erase(history_.begin());
    }
}

std::string InteractiveLineEditor::suggest_command(const std::string& unknown_cmd) const {
    if (unknown_cmd.empty()) return "";
    std::string norm = unknown_cmd;
    while (norm.size() > 1 && norm[0] == '/' && norm[1] == '/') {
        norm.erase(0, 1);
    }
    for (const auto& c : commands_) {
        if (c.name == norm) return c.name;
    }
    if (norm.size() >= 2) {
        for (const auto& c : commands_) {
            if (c.name.rfind(norm, 0) == 0) return c.name;
        }
    }
    int best_d = 3;
    std::string best;
    for (const auto& c : commands_) {
        const int d = edit_distance(norm, c.name);
        if (d < best_d) {
            best_d = d;
            best = c.name;
        }
    }
    return best;
}

InteractiveLineEditor::Status InteractiveLineEditor::read_line(const UiTheme& theme,
                                                               int attachments_count,
                                                               std::string& out_line) {
    out_line.clear();

    if (!::isatty(STDIN_FILENO) || !::isatty(STDERR_FILENO)) {
        if (!std::getline(std::cin, out_line)) {
            return Status::Eof;
        }
        if (!out_line.empty() && out_line.back() == '\r') out_line.pop_back();
        return Status::Line;
    }

    RawTermGuard raw_guard(STDIN_FILENO);
    if (!raw_guard.active) {
        if (!std::getline(std::cin, out_line)) return Status::Eof;
        if (!out_line.empty() && out_line.back() == '\r') out_line.pop_back();
        return Status::Line;
    }

    std::string buf;
    size_t cursor = 0;
    size_t hist_idx = history_.size();
    std::string draft;
    std::string below_msg;

    int rendered_rows = 0;
    int rendered_cursor_row = 0;

    std::string prompt0_plain;
    std::string prompt0_styled;
    if (attachments_count > 0) {
        char p_plain[64];
        std::snprintf(p_plain, sizeof(p_plain), "[+%d file%s] ❯ ",
                      attachments_count, attachments_count == 1 ? "" : "s");
        prompt0_plain = p_plain;
        char p_styled[192];
        std::snprintf(p_styled, sizeof(p_styled), "%s[+%d file%s]%s %s%s❯%s ",
                      theme.brand(), attachments_count, attachments_count == 1 ? "" : "s", theme.reset(),
                      theme.bold(), theme.accent(), theme.reset());
        prompt0_styled = p_styled;
    } else {
        prompt0_plain = "❯ ";
        prompt0_styled = std::string(theme.bold()) + theme.accent() + "❯" + theme.reset() + " ";
    }
    const std::string cont_plain = "│ ";
    const std::string cont_styled = std::string(theme.subtle()) + "│" + theme.reset() + " ";

    auto compute_ghost = [&]() -> std::string {
        if (cursor != buf.size() || buf.size() < 2 || buf[0] != '/') return "";
        if (buf.find('\n') != std::string::npos || buf.find(' ') != std::string::npos) return "";
        for (const auto& c : commands_) {
            if (c.name.size() > buf.size() && c.name.rfind(buf, 0) == 0) {
                return c.name.substr(buf.size());
            }
        }
        return "";
    };

    auto refresh = [&](bool show_hints) {
        int cols = 80, rows = 24;
        query_terminal_size(STDERR_FILENO, cols, rows);
        cols = std::max(20, cols);
        rows = std::max(8, rows);

        std::string frame;
        frame.reserve(buf.size() + 256);
        if (rendered_cursor_row > 0) {
            frame += "\x1b[" + std::to_string(rendered_cursor_row) + "A";
        }
        frame += "\r\x1b[J";

        // Split buf into logical lines
        struct LogicalLine {
            size_t start;
            size_t end;
        };
        std::vector<LogicalLine> llines;
        {
            size_t ls = 0;
            for (size_t i = 0; i < buf.size(); ++i) {
                if (buf[i] == '\n') {
                    llines.push_back({ls, i});
                    ls = i + 1;
                }
            }
            llines.push_back({ls, buf.size()});
        }

        // Build wrapped visual lines so sliding viewport keeps rendered height < terminal rows
        struct VisualPiece {
            std::string text;
            bool is_first_of_logical = false;
            size_t logical_idx = 0;
            size_t byte_begin = 0;
            size_t byte_end = 0;
            int cursor_col = -1; // >=0 if cursor sits on this visual row
        };
        std::vector<VisualPiece> vlines;
        const std::string ghost = show_hints ? compute_ghost() : "";

        for (size_t li = 0; li < llines.size(); ++li) {
            const size_t ls = llines[li].start;
            const size_t le = llines[li].end;
            const size_t pref_w = (li == 0) ? display_width(prompt0_plain) : display_width(cont_plain);
            const std::string& pref_s = (li == 0) ? prompt0_styled : cont_styled;

            VisualPiece cur_v;
            cur_v.is_first_of_logical = true;
            cur_v.logical_idx = li;
            cur_v.byte_begin = ls;
            cur_v.text = pref_s;
            int col = static_cast<int>(pref_w);

            for (size_t p = ls; p < le;) {
                if (cursor == p && cur_v.cursor_col < 0) {
                    cur_v.cursor_col = col;
                }
                uint32_t cp = 0;
                size_t clen = 1;
                decode_utf8_cp(buf, p, cp, clen);
                int cw = std::max(1, codepoint_width(cp));
                if (cp < 0x20u) cw = 1;

                if (col + cw >= cols && col > 0) {
                    cur_v.byte_end = p;
                    vlines.push_back(std::move(cur_v));
                    cur_v = VisualPiece{};
                    cur_v.is_first_of_logical = false;
                    cur_v.logical_idx = li;
                    cur_v.byte_begin = p;
                    col = 0;
                    if (cursor == p) cur_v.cursor_col = 0;
                }
                if (cp >= 0x20u) {
                    cur_v.text.append(buf.data() + p, clen);
                } else {
                    cur_v.text.push_back('?');
                }
                col += cw;
                p += std::max<size_t>(1, clen);
            }
            if (cursor == le && cur_v.cursor_col < 0) {
                cur_v.cursor_col = col;
            }
            if (li + 1 == llines.size() && !ghost.empty() && col < cols - 1) {
                const int avail = (cols - 1) - col;
                const std::string clipped_ghost =
                    truncate_display_width(ghost, static_cast<size_t>(avail), theme);
                cur_v.text += theme.subtle();
                cur_v.text += clipped_ghost;
                cur_v.text += theme.reset();
            }
            cur_v.byte_end = le;
            vlines.push_back(std::move(cur_v));
        }

        int cursor_vidx = 0;
        int cursor_col = static_cast<int>(display_width(prompt0_plain));
        for (int vi = 0; vi < static_cast<int>(vlines.size()); ++vi) {
            if (vlines[vi].cursor_col >= 0) {
                cursor_vidx = vi;
                cursor_col = vlines[vi].cursor_col;
                break;
            }
        }

        // Keep viewport within terminal height
        const int below_lines_cnt =
            below_msg.empty() ? 0 : (1 + static_cast<int>(std::count(below_msg.begin(), below_msg.end(), '\n')));
        const int max_edit_rows = std::max(3, rows - 3 - below_lines_cnt);
        int win_start = 0;
        int win_end = static_cast<int>(vlines.size());
        if (win_end > max_edit_rows) {
            win_start = std::clamp(cursor_vidx - max_edit_rows / 2, 0, win_end - max_edit_rows);
            win_end = win_start + max_edit_rows;
        }

        int cur_row = 0;
        int row_cursor = 0;
        if (win_start > 0) {
            char hdr[96];
            std::snprintf(hdr, sizeof(hdr), "%s  ↑ %d earlier visual line(s)%s\r\n",
                          theme.subtle(), win_start, theme.reset());
            frame += hdr;
            cur_row++;
        }
        for (int vi = win_start; vi < win_end; ++vi) {
            if (vi == cursor_vidx) row_cursor = cur_row;
            frame += vlines[vi].text;
            if (vi + 1 < win_end) {
                frame += "\r\n";
                cur_row++;
            }
        }
        if (win_end < static_cast<int>(vlines.size())) {
            char ftr[96];
            std::snprintf(ftr, sizeof(ftr), "\r\n%s  ↓ %d more visual line(s)%s",
                          theme.subtle(), static_cast<int>(vlines.size()) - win_end, theme.reset());
            frame += ftr;
            cur_row++;
        }
        if (!below_msg.empty()) {
            frame += "\r\n";
            for (char c : below_msg) {
                if (c == '\n') {
                    frame += "\r\n";
                    cur_row++;
                } else {
                    frame.push_back(c);
                }
            }
            cur_row++;
        }

        const int total_rows = cur_row + 1;
        const int move_up = cur_row - row_cursor;
        if (move_up > 0) {
            frame += "\x1b[" + std::to_string(move_up) + "A";
        }
        frame += "\r";
        if (cursor_col > 0) {
            frame += "\x1b[" + std::to_string(cursor_col) + "C";
        }

        write_all_fd(STDERR_FILENO, frame);
        rendered_rows = total_rows;
        rendered_cursor_row = row_cursor;
    };

    auto move_vertical = [&](int dir) -> bool {
        // Check if buf has multiple lines and cursor can move within buf
        size_t ls = buf.rfind('\n', cursor > 0 ? cursor - 1 : 0);
        if (cursor == 0) ls = std::string::npos;
        const size_t cur_line_start = (ls == std::string::npos) ? 0 : (ls + 1);
        const size_t col_bytes = cursor - cur_line_start;
        size_t cur_line_end = buf.find('\n', cursor);
        if (cur_line_end == std::string::npos) cur_line_end = buf.size();

        if (dir < 0 && cur_line_start > 0) {
            const size_t prev_end = cur_line_start - 1;
            const size_t prev_nl = (prev_end > 0) ? buf.rfind('\n', prev_end - 1) : std::string::npos;
            const size_t prev_start = (prev_nl == std::string::npos) ? 0 : (prev_nl + 1);
            cursor = std::min(prev_end, prev_start + col_bytes);
            while (cursor > prev_start && (static_cast<uint8_t>(buf[cursor]) & 0xC0u) == 0x80u) --cursor;
            return true;
        }
        if (dir > 0 && cur_line_end < buf.size()) {
            const size_t next_start = cur_line_end + 1;
            size_t next_end = buf.find('\n', next_start);
            if (next_end == std::string::npos) next_end = buf.size();
            cursor = std::min(next_end, next_start + col_bytes);
            while (cursor > next_start && (static_cast<uint8_t>(buf[cursor]) & 0xC0u) == 0x80u) --cursor;
            return true;
        }
        return false;
    };

    auto handle_tab = [&]() {
        below_msg.clear();
        if (buf.find('\n') != std::string::npos || cursor != buf.size()) return;

        if (!buf.empty() && buf[0] == '/' && buf.find(' ') == std::string::npos) {
            std::vector<const CommandSpec*> matches;
            std::vector<std::string> names;
            for (const auto& c : commands_) {
                if (c.name.rfind(buf, 0) == 0) {
                    matches.push_back(&c);
                    names.push_back(c.name);
                }
            }
            if (matches.size() == 1) {
                buf = matches[0]->name;
                if (!matches[0]->args_hint.empty()) buf.push_back(' ');
                cursor = buf.size();
            } else if (matches.size() > 1) {
                const std::string lcp = longest_common_prefix(names);
                if (lcp.size() > buf.size()) {
                    buf = lcp;
                    cursor = buf.size();
                } else {
                    const size_t limit = std::min<size_t>(matches.size(), 10);
                    for (size_t i = 0; i < limit; ++i) {
                        if (i > 0) below_msg.push_back('\n');
                        below_msg += "  ";
                        below_msg += theme.accent();
                        below_msg += matches[i]->name;
                        below_msg += theme.reset();
                        for (size_t p = matches[i]->name.size(); p < 18; ++p) below_msg.push_back(' ');
                        below_msg += theme.muted();
                        below_msg += matches[i]->desc;
                        below_msg += theme.reset();
                    }
                    if (matches.size() > limit) {
                        below_msg += "\n  ";
                        below_msg += theme.subtle();
                        below_msg += "... +" + std::to_string(matches.size() - limit) + " more";
                        below_msg += theme.reset();
                    }
                }
            }
            return;
        }

        // File path completion for commands that accept paths (/file, /attach, /save-cache, /load-cache)
        const size_t sp = buf.find(' ');
        if (sp != std::string::npos && buf[0] == '/') {
            const std::string cmd_name = buf.substr(0, sp);
            bool takes_path = false;
            for (const auto& c : commands_) {
                if (c.name == cmd_name && c.takes_path) {
                    takes_path = true;
                    break;
                }
            }
            if (!takes_path) return;

            size_t arg_start = sp + 1;
            while (arg_start < buf.size() && buf[arg_start] == ' ') ++arg_start;
            char quote = 0;
            if (arg_start < buf.size() && (buf[arg_start] == '"' || buf[arg_start] == '\'')) {
                quote = buf[arg_start++];
            }
            const std::string raw_tok = buf.substr(arg_start);
            if (!quote && raw_tok.find(' ') != std::string::npos) return;
            if (raw_tok == "~") {
                buf.push_back('/');
                cursor = buf.size();
                return;
            }

            const size_t slash = raw_tok.find_last_of('/');
            const std::string user_dir = (slash == std::string::npos) ? "" : raw_tok.substr(0, slash + 1);
            const std::string base_prefix = (slash == std::string::npos) ? raw_tok : raw_tok.substr(slash + 1);
            const std::string fs_dir = user_dir.empty() ? "." : expand_tilde_prefix(user_dir);

            DIR* d = ::opendir(fs_dir.c_str());
            if (!d) return;
            std::vector<std::string> cands;
            while (struct dirent* ent = ::readdir(d)) {
                const std::string name = ent->d_name;
                if (name == "." || name == "..") continue;
                if (base_prefix.empty() && !name.empty() && name[0] == '.') continue;
                if (name.rfind(base_prefix, 0) == 0) {
                    std::string full_fs = (fs_dir == "." ? "" : fs_dir) + name;
                    struct stat st{};
                    const bool is_dir = (::stat(full_fs.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
                    cands.push_back(user_dir + name + (is_dir ? "/" : ""));
                }
            }
            ::closedir(d);
            std::sort(cands.begin(), cands.end());

            if (cands.size() == 1) {
                buf.replace(arg_start, raw_tok.size(), cands[0]);
                cursor = buf.size();
            } else if (cands.size() > 1) {
                const std::string lcp = longest_common_prefix(cands);
                if (lcp.size() > raw_tok.size()) {
                    buf.replace(arg_start, raw_tok.size(), lcp);
                    cursor = buf.size();
                } else {
                    const size_t limit = std::min<size_t>(cands.size(), 10);
                    for (size_t i = 0; i < limit; ++i) {
                        if (i > 0) below_msg.push_back('\n');
                        below_msg += "  ";
                        below_msg += theme.brand();
                        below_msg += cands[i];
                        below_msg += theme.reset();
                    }
                    if (cands.size() > limit) {
                        below_msg += "\n  ";
                        below_msg += theme.subtle();
                        below_msg += "... +" + std::to_string(cands.size() - limit) + " more";
                        below_msg += theme.reset();
                    }
                }
            }
        }
    };

    refresh(true);

    while (true) {
        uint8_t ch = 0;
        if (!read_byte_timeout(STDIN_FILENO, ch, -1)) {
            raw_guard.restore();
            return Status::Eof;
        }

        if (ch == 0x03) { // Ctrl-C
            below_msg.clear();
            refresh(false);
            const int move_down = std::max(0, rendered_rows - 1 - rendered_cursor_row);
            if (move_down > 0) write_all_fd(STDERR_FILENO, "\x1b[" + std::to_string(move_down) + "B");
            write_all_fd(STDERR_FILENO, std::string(" ") + theme.muted() + "^C" + theme.reset() + "\r\n");
            raw_guard.restore();
            return Status::Interrupted;
        }

        if (ch == 0x04) { // Ctrl-D
            if (buf.empty()) {
                below_msg.clear();
                refresh(false);
                write_all_fd(STDERR_FILENO, "\r\n");
                raw_guard.restore();
                return Status::Eof;
            }
            below_msg.clear();
            if (cursor < buf.size()) {
                const size_t next = utf8_next_boundary(buf, cursor);
                buf.erase(cursor, next - cursor);
            }
            refresh(true);
            continue;
        }

        if (ch == '\r') { // Enter
            if (cursor == buf.size() && !buf.empty() && buf.back() == '\\') {
                buf.back() = '\n';
                below_msg.clear();
                refresh(true);
                continue;
            }
            below_msg.clear();
            refresh(false);
            const int move_down = std::max(0, rendered_rows - 1 - rendered_cursor_row);
            if (move_down > 0) write_all_fd(STDERR_FILENO, "\x1b[" + std::to_string(move_down) + "B");
            write_all_fd(STDERR_FILENO, "\r\n");
            out_line = std::move(buf);
            raw_guard.restore();
            return Status::Line;
        }

        if (ch == '\n') { // Ctrl-J -> insert newline
            below_msg.clear();
            buf.insert(cursor, 1, '\n');
            cursor += 1;
            refresh(true);
            continue;
        }

        if (ch == '\t') { // Tab
            handle_tab();
            refresh(true);
            continue;
        }

        below_msg.clear();

        if (ch == 0x7f || ch == 0x08) { // Backspace
            if (cursor > 0) {
                const size_t prev = utf8_prev_boundary(buf, cursor);
                buf.erase(prev, cursor - prev);
                cursor = prev;
            }
            refresh(true);
            continue;
        }

        if (ch == 0x01) { // Ctrl-A Home
            cursor = 0;
            refresh(true);
            continue;
        }
        if (ch == 0x05) { // Ctrl-E End
            cursor = buf.size();
            refresh(true);
            continue;
        }
        if (ch == 0x0b) { // Ctrl-K kill to end of line
            size_t eol = buf.find('\n', cursor);
            if (eol == std::string::npos) eol = buf.size();
            if (eol == cursor && eol < buf.size()) eol++;
            buf.erase(cursor, eol - cursor);
            refresh(true);
            continue;
        }
        if (ch == 0x15) { // Ctrl-U kill to start of line
            size_t sol = (cursor > 0) ? buf.rfind('\n', cursor - 1) : std::string::npos;
            sol = (sol == std::string::npos) ? 0 : (sol + 1);
            if (sol == cursor && cursor > 0) sol--;
            buf.erase(sol, cursor - sol);
            cursor = sol;
            refresh(true);
            continue;
        }
        if (ch == 0x17) { // Ctrl-W kill word backward
            size_t p = cursor;
            while (p > 0 && (buf[p - 1] == ' ' || buf[p - 1] == '\t')) --p;
            while (p > 0 && !(buf[p - 1] == ' ' || buf[p - 1] == '\t' || buf[p - 1] == '\n')) --p;
            buf.erase(p, cursor - p);
            cursor = p;
            refresh(true);
            continue;
        }
        if (ch == 0x0c) { // Ctrl-L clear screen & redraw
            write_all_fd(STDERR_FILENO, "\x1b[2J\x1b[H");
            rendered_rows = 0;
            rendered_cursor_row = 0;
            refresh(true);
            continue;
        }

        if (ch == 0x1b) { // Escape sequence
            uint8_t b1 = 0;
            if (!read_byte_timeout(STDIN_FILENO, b1, 25)) {
                refresh(true);
                continue;
            }
            if (b1 == '\r' || b1 == '\n') { // Alt+Enter -> newline
                buf.insert(cursor, 1, '\n');
                cursor += 1;
                refresh(true);
                continue;
            }
            if (b1 == 0x7f || b1 == 0x08) { // Alt+Backspace
                size_t p = cursor;
                while (p > 0 && !is_word_byte(static_cast<unsigned char>(buf[p - 1]))) --p;
                while (p > 0 && is_word_byte(static_cast<unsigned char>(buf[p - 1]))) --p;
                buf.erase(p, cursor - p);
                cursor = p;
                refresh(true);
                continue;
            }
            if (b1 == 'b' || b1 == 'B') { // Alt+B word left
                while (cursor > 0 && !is_word_byte(static_cast<unsigned char>(buf[cursor - 1]))) --cursor;
                while (cursor > 0 && is_word_byte(static_cast<unsigned char>(buf[cursor - 1]))) --cursor;
                refresh(true);
                continue;
            }
            if (b1 == 'f' || b1 == 'F') { // Alt+F word right
                while (cursor < buf.size() && !is_word_byte(static_cast<unsigned char>(buf[cursor]))) ++cursor;
                while (cursor < buf.size() && is_word_byte(static_cast<unsigned char>(buf[cursor]))) ++cursor;
                refresh(true);
                continue;
            }
            if (b1 == '[') {
                std::string params;
                uint8_t fin = 0;
                while (params.size() < 32) {
                    uint8_t b = 0;
                    if (!read_byte_timeout(STDIN_FILENO, b, 25)) break;
                    if (b >= '@' && b <= '~') {
                        fin = b;
                        break;
                    }
                    params.push_back(static_cast<char>(b));
                }
                if (fin == '~' && params == "200") {
                    // Bracketed paste until \x1b[201~
                    std::string pasted;
                    while (true) {
                        uint8_t pb = 0;
                        if (!read_byte_timeout(STDIN_FILENO, pb, 200)) break;
                        pasted.push_back(static_cast<char>(pb));
                        if (pasted.size() >= 6 &&
                            pasted.compare(pasted.size() - 6, 6, "\x1b[201~") == 0) {
                            pasted.resize(pasted.size() - 6);
                            break;
                        }
                    }
                    std::string norm;
                    norm.reserve(pasted.size());
                    for (size_t i = 0; i < pasted.size(); ++i) {
                        if (pasted[i] == '\r') {
                            norm.push_back('\n');
                            if (i + 1 < pasted.size() && pasted[i + 1] == '\n') ++i;
                        } else {
                            norm.push_back(pasted[i]);
                        }
                    }
                    // Trim a single accidental trailing newline on pasted snippets
                    if (!norm.empty() && norm.back() == '\n') norm.pop_back();
                    buf.insert(cursor, norm);
                    cursor += norm.size();
                    refresh(true);
                    continue;
                }
                if ((fin == 'u' && (params == "13;2" || params == "13;5")) ||
                    (fin == '~' && params == "27;2;13")) {
                    buf.insert(cursor, 1, '\n');
                    cursor += 1;
                    refresh(true);
                    continue;
                }
                if (fin == 'A') { // Up
                    if (!move_vertical(-1) && !history_.empty() && hist_idx > 0) {
                        if (hist_idx == history_.size()) draft = buf;
                        --hist_idx;
                        buf = history_[hist_idx];
                        cursor = buf.size();
                    }
                } else if (fin == 'B') { // Down
                    if (!move_vertical(1) && hist_idx < history_.size()) {
                        ++hist_idx;
                        buf = (hist_idx == history_.size()) ? draft : history_[hist_idx];
                        cursor = buf.size();
                    }
                } else if (fin == 'C') { // Right (or Ctrl+Right / accept ghost)
                    if (params == "1;5" || params == "5" || params == "1;3") {
                        while (cursor < buf.size() && !is_word_byte(static_cast<unsigned char>(buf[cursor]))) ++cursor;
                        while (cursor < buf.size() && is_word_byte(static_cast<unsigned char>(buf[cursor]))) ++cursor;
                    } else if (cursor == buf.size()) {
                        const std::string g = compute_ghost();
                        if (!g.empty()) {
                            buf += g;
                            cursor = buf.size();
                        }
                    } else {
                        cursor = utf8_next_boundary(buf, cursor);
                    }
                } else if (fin == 'D') { // Left (or Ctrl+Left)
                    if (params == "1;5" || params == "5" || params == "1;3") {
                        while (cursor > 0 && !is_word_byte(static_cast<unsigned char>(buf[cursor - 1]))) --cursor;
                        while (cursor > 0 && is_word_byte(static_cast<unsigned char>(buf[cursor - 1]))) --cursor;
                    } else {
                        cursor = utf8_prev_boundary(buf, cursor);
                    }
                } else if (fin == 'H' || (fin == '~' && (params == "1" || params == "7"))) {
                    cursor = 0;
                } else if (fin == 'F' || (fin == '~' && (params == "4" || params == "8"))) {
                    cursor = buf.size();
                } else if (fin == '~' && params == "3") {
                    if (cursor < buf.size()) {
                        const size_t next = utf8_next_boundary(buf, cursor);
                        buf.erase(cursor, next - cursor);
                    }
                }
                refresh(true);
                continue;
            }
            if (b1 == 'O') {
                uint8_t b2 = 0;
                (void)read_byte_timeout(STDIN_FILENO, b2, 25);
                if (b2 == 'H') cursor = 0;
                else if (b2 == 'F') cursor = buf.size();
                refresh(true);
                continue;
            }
            continue;
        }

        // Ignore other control bytes
        if (ch < 0x20u) continue;

        // Printable ASCII or multi-byte UTF-8 character
        std::string cp_bytes(1, static_cast<char>(ch));
        size_t need = 1;
        if ((ch & 0xE0u) == 0xC0u) need = 2;
        else if ((ch & 0xF0u) == 0xE0u) need = 3;
        else if ((ch & 0xF8u) == 0xF0u) need = 4;
        for (size_t i = 1; i < need; ++i) {
            uint8_t nb = 0;
            if (!read_byte_timeout(STDIN_FILENO, nb, 25)) break;
            cp_bytes.push_back(static_cast<char>(nb));
        }
        buf.insert(cursor, cp_bytes);
        cursor += cp_bytes.size();
        refresh(true);
    }
}

} // namespace tui
} // namespace rxinfer

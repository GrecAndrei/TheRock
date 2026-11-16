// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace rxinfer {
namespace tui {

struct UiTheme {
    bool color = false;
    bool truecolor = false;
    bool tty_stderr = false;
    bool tty_stdout = false;

    static UiTheme detect(bool allow_color);

    const char* reset() const { return color ? "\x1b[0m" : ""; }
    const char* bold() const { return color ? "\x1b[1m" : ""; }
    const char* dim() const { return color ? "\x1b[2m" : ""; }
    const char* italic() const { return color ? "\x1b[3m" : ""; }

    const char* accent() const {
        if (!color) return "";
        return truecolor ? "\x1b[38;2;86;212;221m" : "\x1b[38;5;80m";
    }
    const char* brand() const {
        if (!color) return "";
        return truecolor ? "\x1b[38;2;137;180;250m" : "\x1b[38;5;111m";
    }
    const char* ok() const {
        if (!color) return "";
        return truecolor ? "\x1b[38;2;166;227;161m" : "\x1b[38;5;114m";
    }
    const char* warn() const {
        if (!color) return "";
        return truecolor ? "\x1b[38;2;249;226;175m" : "\x1b[38;5;222m";
    }
    const char* err() const {
        if (!color) return "";
        return truecolor ? "\x1b[38;2;243;139;168m" : "\x1b[38;5;211m";
    }
    const char* muted() const {
        if (!color) return "";
        return truecolor ? "\x1b[38;2;147;153;178m" : "\x1b[38;5;245m";
    }
    const char* subtle() const {
        if (!color) return "";
        return truecolor ? "\x1b[38;2;88;91;112m" : "\x1b[38;5;240m";
    }
};

void query_terminal_size(int fd, int& out_cols, int& out_rows);
size_t display_width(std::string_view sv);
std::string truncate_display_width(std::string_view sv, size_t max_cols, const UiTheme& t);

std::string format_tokens_compact(int64_t n);
std::string format_duration_compact(double ms);
std::string clean_model_display_name(const std::string& model_path, const std::string& fallback_type);
std::string render_progress_bar(double frac, int width, const UiTheme& t);

void print_box_card(FILE* out,
                    const UiTheme& t,
                    const std::string& title,
                    const std::vector<std::pair<std::string, std::string>>& rows,
                    int max_width = 88);

void print_status_badge(FILE* out, const UiTheme& t, const char* kind, const std::string& msg);

class LoadProgressReporter {
public:
    LoadProgressReporter(const UiTheme& theme, bool enabled, std::string label);
    ~LoadProgressReporter();

    void update(size_t done_bytes, size_t total_bytes);
    void finish();

private:
    UiTheme theme_;
    bool enabled_ = false;
    std::string label_;
    std::mutex mu_;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point last_draw_;
    bool drew_ = false;
    int frame_ = 0;
};

class PrefillSpinner {
public:
    PrefillSpinner(const UiTheme& theme, bool enabled, std::string label);
    ~PrefillSpinner();

    PrefillSpinner(const PrefillSpinner&) = delete;
    PrefillSpinner& operator=(const PrefillSpinner&) = delete;

    void stop();

private:
    UiTheme theme_;
    bool enabled_ = false;
    std::string label_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_requested_ = false;
    bool stopped_ = false;
    bool drew_ = false;
    std::thread worker_;
};

class InteractiveLineEditor {
public:
    enum class Status { Line, Eof, Interrupted };

    struct CommandSpec {
        std::string name;
        std::string args_hint;
        std::string desc;
        bool takes_path = false;
    };

    explicit InteractiveLineEditor(std::vector<CommandSpec> commands);

    void add_history(const std::string& entry);
    std::string suggest_command(const std::string& unknown_cmd) const;

    Status read_line(const UiTheme& theme,
                     int attachments_count,
                     std::string& out_line);

private:
    std::vector<CommandSpec> commands_;
    std::vector<std::string> history_;
};

} // namespace tui
} // namespace rxinfer

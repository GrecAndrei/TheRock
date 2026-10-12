#pragma once

#include <cstdlib>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace qwen_native {

// Opt-in diagnostics must not interpret an explicit "0" as enabled. Keep this
// uncached so tests and scoped callers can restore the environment reliably.
inline bool environment_flag_enabled(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}

inline float configured_selector_weight() {
    const char *value = std::getenv("QWEN_NATIVE_SELECTOR_WEIGHT");
    if (!value) return 1.0F;
    std::size_t consumed = 0;
    float weight;
    try {
        weight = std::stof(value, &consumed);
    } catch (const std::exception &) {
        throw std::invalid_argument("selector weight must be finite and nonnegative");
    }
    if (consumed != std::strlen(value) || !std::isfinite(weight) || weight < 0.0F) {
        throw std::invalid_argument("selector weight must be finite and nonnegative");
    }
    return weight;
}

inline unsigned configured_ffn_down_cpu_percentage() {
    const char *value = std::getenv("QWEN_NATIVE_FFN_DOWN_CPU_PERCENT");
    if (!value) return 50;
    std::size_t consumed = 0;
    int percentage;
    try {
        percentage = std::stoi(value, &consumed);
    } catch (const std::exception &) {
        throw std::invalid_argument("FFN down CPU percentage must be an integer in [1, 99]");
    }
    if (consumed != std::strlen(value) || percentage < 1 || percentage > 99) {
        throw std::invalid_argument("FFN down CPU percentage must be an integer in [1, 99]");
    }
    return static_cast<unsigned>(percentage);
}

inline unsigned configured_extra_down_mebibytes() {
    const char *value = std::getenv("QWEN_NATIVE_EXTRA_DOWN_MB");
    if (!value) return 0;
    std::size_t consumed = 0;
    int amount;
    try {
        amount = std::stoi(value, &consumed);
    } catch (const std::exception &) {
        throw std::invalid_argument("extra down cache must be integer MiB in [0, 1024]");
    }
    if (consumed != std::strlen(value) || amount < 0 || amount > 1024) {
        throw std::invalid_argument("extra down cache must be integer MiB in [0, 1024]");
    }
    return static_cast<unsigned>(amount);
}

inline std::size_t bounded_residency_allowance(std::size_t requested_bytes,
                                              std::size_t free_bytes,
                                              std::size_t reserve_bytes) {
    if (free_bytes <= reserve_bytes) return 0;
    const auto available = free_bytes - reserve_bytes;
    return requested_bytes < available ? requested_bytes : available;
}

// Serial native fixture helper only: restore after the owning session drains.
// A null replacement leaves the variable untouched. Process-wide environment
// mutation must not be used while another session is executing in this process.
class ScopedEnvironment {
public:
    ScopedEnvironment(const char *name, const char *replacement) : name_(name) {
        if (!replacement) return;
        if (const char *value = std::getenv(name_)) {
            present_ = true;
            previous_ = value;
        }
        if (setenv(name_, replacement, 1)) {
            throw std::runtime_error("cannot set fixture environment");
        }
        changed_ = true;
    }
    ScopedEnvironment(const ScopedEnvironment &) = delete;
    ScopedEnvironment &operator=(const ScopedEnvironment &) = delete;
    ~ScopedEnvironment() {
        if (!changed_) return;
        if (present_) (void)setenv(name_, previous_.c_str(), 1);
        else (void)unsetenv(name_);
    }

private:
    const char *name_;
    std::string previous_;
    bool present_ = false;
    bool changed_ = false;
};

} // namespace qwen_native

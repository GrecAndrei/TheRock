#pragma once

#include <cstdlib>
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

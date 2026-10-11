#include "runtime_options.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr const char *kProfile = "QWEN_NATIVE_PREFILL_PROFILE";

class RestoreProfile {
public:
    RestoreProfile() {
        if (const char *value = std::getenv(kProfile)) {
            present_ = true;
            previous_ = value;
        }
    }
    ~RestoreProfile() {
        if (present_) (void)setenv(kProfile, previous_.c_str(), 1);
        else (void)unsetenv(kProfile);
    }

private:
    bool present_ = false;
    std::string previous_;
};

void require_mode(bool expected) {
    if (qwen_native::environment_flag_enabled(kProfile) != expected) {
        throw std::runtime_error("prefill profiling flag parsed incorrectly");
    }
}

} // namespace

int main() {
    try {
        RestoreProfile restore;
        if (unsetenv(kProfile)) throw std::runtime_error("unsetenv failed");
        require_mode(false);
        {
            qwen_native::ScopedEnvironment enabled(kProfile, "1");
            require_mode(true);
        }
        if (std::getenv(kProfile)) throw std::runtime_error("absent environment not restored");
        constexpr std::array<const char *, 7> disabled = {
            "", "0", "false", "true", "01", "yes", "1suffix"};
        for (const char *value : disabled) {
            if (setenv(kProfile, value, 1)) throw std::runtime_error("setenv failed");
            require_mode(false);
        }
        if (setenv(kProfile, "1", 1)) throw std::runtime_error("setenv failed");
        require_mode(true);
        {
            qwen_native::ScopedEnvironment disabled(kProfile, "0");
            require_mode(false);
        }
        require_mode(true);
        {
            qwen_native::ScopedEnvironment unchanged(kProfile, nullptr);
            require_mode(true);
        }
        try {
            qwen_native::ScopedEnvironment disabled(kProfile, "0");
            throw std::runtime_error("fixture unwind");
        } catch (const std::runtime_error &error) {
            if (std::string(error.what()) != "fixture unwind") throw;
            require_mode(true);
        }
        std::cout << "9 runtime-flag cases and scoped restore/unwind passed; CPU-only, not inference timing\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

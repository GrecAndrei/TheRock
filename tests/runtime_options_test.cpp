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
        constexpr const char *selector_name = "QWEN_NATIVE_SELECTOR_WEIGHT";
        qwen_native::ScopedEnvironment selector_restore(selector_name, "1");
        if (unsetenv(selector_name)) throw std::runtime_error("unsetenv failed");
        if (qwen_native::configured_selector_weight() != 1.0F) {
            throw std::runtime_error("selector default changed");
        }
        constexpr std::array<const char *, 4> weights = {"0", "0.5", "1", "2"};
        constexpr std::array<float, 4> expected = {0.0F, 0.5F, 1.0F, 2.0F};
        for (std::size_t index = 0; index < weights.size(); ++index) {
            qwen_native::ScopedEnvironment weight(selector_name, weights[index]);
            if (qwen_native::configured_selector_weight() != expected[index]) {
                throw std::runtime_error("selector weight parsed incorrectly");
            }
        }
        constexpr std::array<const char *, 8> invalid_weights = {
            "", "-1", "nan", "inf", "1suffix", "1.0 ", "1e999", "false"};
        for (const char *value : invalid_weights) {
            qwen_native::ScopedEnvironment weight(selector_name, value);
            bool rejected = false;
            try {
                (void)qwen_native::configured_selector_weight();
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            if (!rejected) throw std::runtime_error("invalid selector weight accepted");
        }
        constexpr const char *down_name = "QWEN_NATIVE_FFN_DOWN_CPU_PERCENT";
        qwen_native::ScopedEnvironment down_restore(down_name, "50");
        if (unsetenv(down_name)) throw std::runtime_error("unsetenv failed");
        if (qwen_native::configured_ffn_down_cpu_percentage() != 50) {
            throw std::runtime_error("FFN down default changed");
        }
        for (const char *value : {"1", "50", "99"}) {
            qwen_native::ScopedEnvironment option(down_name, value);
            if (qwen_native::configured_ffn_down_cpu_percentage() != std::stoul(value)) {
                throw std::runtime_error("FFN down percentage parsed incorrectly");
            }
        }
        for (const char *value : {"0", "100", "-1", "", "nan", "50.5", "50suffix", "9999999999"}) {
            qwen_native::ScopedEnvironment option(down_name, value);
            bool rejected = false;
            try {
                (void)qwen_native::configured_ffn_down_cpu_percentage();
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            if (!rejected) throw std::runtime_error("invalid FFN down percentage accepted");
        }
        constexpr const char *extra_name = "QWEN_NATIVE_EXTRA_DOWN_MB";
        qwen_native::ScopedEnvironment extra_restore(extra_name, "0");
        if (unsetenv(extra_name)) throw std::runtime_error("unsetenv failed");
        if (qwen_native::configured_extra_down_mebibytes() != 0) {
            throw std::runtime_error("extra cache default changed");
        }
        for (const char *value : {"0", "128", "1024"}) {
            qwen_native::ScopedEnvironment option(extra_name, value);
            if (qwen_native::configured_extra_down_mebibytes() != std::stoul(value)) {
                throw std::runtime_error("extra cache amount parsed incorrectly");
            }
        }
        for (const char *value : {"-1", "1025", "", "128.5", "128suffix", "nan", "9999999999"}) {
            qwen_native::ScopedEnvironment option(extra_name, value);
            bool rejected = false;
            try {
                (void)qwen_native::configured_extra_down_mebibytes();
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            if (!rejected) throw std::runtime_error("invalid extra cache amount accepted");
        }
        constexpr std::size_t mib = 1024 * 1024;
        for (const auto values : {
                 std::array<std::size_t, 4>{128, 700, 512, 128},
                 {128, 544, 512, 32}, {128, 512, 512, 0},
                 {128, 256, 512, 0}, {0, 1024, 512, 0}}) {
            if (qwen_native::bounded_residency_allowance(values[0] * mib, values[1] * mib,
                                                        values[2] * mib) != values[3] * mib) {
                throw std::runtime_error("free-memory guard allowance mismatch");
            }
        }
        std::cout << "runtime flags, restore/unwind, selector, FFN shard and guarded cache cases passed; CPU-only, not inference timing\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

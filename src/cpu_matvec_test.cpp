#include "cpu_matvec.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char *kMode = "QWEN_NATIVE_CPU_Q6_PREDECODE";
constexpr const char *kIq4Mode = "QWEN_NATIVE_CPU_IQ4_TILE_GEMM";
constexpr const char *kIq4Predecode = "QWEN_NATIVE_CPU_IQ4_PREDECODE";
constexpr const char *kKMode = "QWEN_NATIVE_CPU_K_TILE_GEMM";
constexpr float kCanary = 998877.0F;

class RestoreEnvironment {
public:
    explicit RestoreEnvironment(const char *name = kMode) : name_(name) {
        if (const char *value = std::getenv(name_)) {
            present_ = true;
            previous_ = value;
        }
    }
    ~RestoreEnvironment() {
        if (present_) (void)setenv(name_, previous_.c_str(), 1);
        else (void)unsetenv(name_);
    }
private:
    const char *name_;
    bool present_ = false;
    std::string previous_;
};

void set_mode(bool enabled) {
    if (setenv(kMode, enabled ? "1" : "0", 1)) {
        throw std::runtime_error("failed to set CPU predecode mode");
    }
}

struct Random {
    std::uint32_t state = 0x73827541U;
    std::uint32_t next() {
        state = state * 1664525U + 1013904223U;
        return state;
    }
};

std::vector<std::uint8_t> make_weights(std::size_t columns, std::size_t rows,
                                     std::uint32_t type, Random &random) {
    const std::size_t block_bytes = type == 14 ? 210 : type == 23 ? 136 : type == 13 ? 176 : 144;
    std::vector<std::uint8_t> weights(columns / 256 * rows * block_bytes);
    for (auto &value : weights) value = static_cast<std::uint8_t>(random.next() >> 24);
    for (std::size_t begin = 0; begin < weights.size(); begin += block_bytes) {
        // Include every finite FP16 exponent and signs, subnormals and zero.
        // Nonfinite scales are deliberately excluded from an equality oracle.
        for (std::size_t field = 0; field < (type == 12 || type == 13 ? 2U : 1U); ++field) {
            std::uint16_t bits = static_cast<std::uint16_t>(random.next() >> 16);
            if ((bits & 0x7c00U) == 0x7c00U) bits ^= 0x0400U;
            const auto offset = type == 14 ? 208 : field * 2;
            std::memcpy(weights.data() + begin + offset, &bits, sizeof(bits));
        }
    }
    return weights;
}

std::vector<float> make_input(std::size_t columns, Random &random) {
    std::vector<float> input(columns * 8);
    for (auto &value : input) value = (static_cast<int>(random.next() >> 20) - 2048) / 256.0F;
    return input;
}

void require_equal(const std::vector<float> &expected, const std::vector<float> &actual) {
    if (expected.size() != actual.size() ||
        std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float))) {
        throw std::runtime_error("CPU experiment changed output bits or canary guards");
    }
    for (std::size_t index = 8; index + 8 < expected.size(); ++index) {
        if (!std::isfinite(actual[index])) throw std::runtime_error("nonfinite CPU test output");
    }
}

void require_guards(const std::vector<float> &values) {
    for (std::size_t index = 0; index < 8; ++index) {
        if (values[index] != kCanary || values[values.size() - 1 - index] != kCanary) {
            throw std::runtime_error("CPU matvec overwrote output guards");
        }
    }
}

void self_test() {
    RestoreEnvironment restore;
    Random random;
    int fixtures = 0;
    for (const auto columns : {256U, 512U, 5120U, 17408U}) {
        for (const auto rows : {1U, 7U, 8U, 9U, 17U, 24U}) {
            const auto weights = make_weights(columns, rows, 14, random);
            const auto input = make_input(columns, random);
            for (const int threads : {1, 4}) {
                qwen_native::clear_cpu_packed_matrix_cache();
                std::vector<float> expected(rows * 8 + 16, kCanary);
                auto actual = expected;
                auto scalar = expected;
                set_mode(false);
                qwen_native::cpu_quant_matvec_batch8_rows_packed(weights.data(), 14,
                    input.data(), columns, expected.data() + 8, rows, threads);
                const auto old_bytes = qwen_native::cpu_packed_matrix_cache_bytes();
                const std::size_t blocks = ((rows + 7) / 8) * (columns / 256);
                if (old_bytes != blocks * 1680) throw std::runtime_error("incorrect original cache size");
                set_mode(true);
                qwen_native::cpu_quant_matvec_batch8_rows_packed(weights.data(), 14,
                    input.data(), columns, actual.data() + 8, rows, threads);
                if (qwen_native::cpu_packed_matrix_cache_bytes() != old_bytes + blocks * 2560) {
                    throw std::runtime_error("predecode cache key aliased original format");
                }
                require_equal(expected, actual);
                qwen_native::cpu_quant_matvec_batch8_rows(weights.data(), 14,
                    input.data(), columns, scalar.data() + 8, rows, threads);
                require_equal(scalar, actual);
                set_mode(false);
                qwen_native::cpu_quant_matvec_batch8_rows_packed(weights.data(), 14,
                    input.data(), columns, actual.data() + 8, rows, threads);
                require_equal(expected, actual);
                require_guards(actual);
                ++fixtures;
            }
        }
    }
    // Exercise both new dual-dispatch branches, and mixed unchanged Q4/Q6.
    for (const auto types : {std::array<std::uint32_t, 2>{14, 14}, {12, 14}, {14, 12}}) {
        constexpr std::size_t columns = 512, gate_rows = 17, up_rows = 9;
        qwen_native::clear_cpu_packed_matrix_cache();
        const auto gate = make_weights(columns, gate_rows, types[0], random);
        const auto up = make_weights(columns, up_rows, types[1], random);
        const auto input = make_input(columns, random);
        std::vector<float> expected_gate(gate_rows * 8 + 16, kCanary);
        std::vector<float> expected_up(up_rows * 8 + 16, kCanary);
        auto actual_gate = expected_gate;
        auto actual_up = expected_up;
        set_mode(false);
        qwen_native::cpu_quant_matvec_batch8_rows_packed_dual(gate.data(), types[0],
            expected_gate.data() + 8, gate_rows, up.data(), types[1], expected_up.data() + 8,
            up_rows, input.data(), columns, 4);
        set_mode(true);
        qwen_native::cpu_quant_matvec_batch8_rows_packed_dual(gate.data(), types[0],
            actual_gate.data() + 8, gate_rows, up.data(), types[1], actual_up.data() + 8,
            up_rows, input.data(), columns, 4);
        require_equal(expected_gate, actual_gate);
        require_equal(expected_up, actual_up);
        require_guards(actual_gate);
        require_guards(actual_up);
        ++fixtures;
    }
    std::array<float, 8> dummy{};
    for (const auto columns : {0U, 255U}) {
        bool rejected = false;
        try {
            qwen_native::cpu_quant_matvec_batch8_rows_packed(dummy.data(), 14,
                dummy.data(), columns, dummy.data(), 1, 1);
        } catch (const std::invalid_argument &) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid packed dimensions were accepted");
    }
    qwen_native::clear_cpu_packed_matrix_cache();
    if (qwen_native::cpu_packed_matrix_cache_bytes()) throw std::runtime_error("cache clear leaked entries");
    std::cout << "CPU Q6 predecode: " << fixtures
              << " bit-exact fixtures passed; scalar oracle, guards, cache keys, dual paths passed\n";
}

void iq4_gemm_self_test(bool predecoded = false) {
    RestoreEnvironment restore(kIq4Mode);
    RestoreEnvironment restore_predecode(kIq4Predecode);
    const char *mode_name = predecoded ? kIq4Predecode : kIq4Mode;
    if (setenv(predecoded ? kIq4Mode : kIq4Predecode, "0", 1)) {
        throw std::runtime_error("failed to isolate IQ4 experiment mode");
    }
    const auto mode = [=](bool enabled) {
        if (setenv(mode_name, enabled ? "1" : "0", 1)) {
            throw std::runtime_error("failed to set IQ4 microkernel mode");
        }
    };
    Random random;
    int fixtures = 0;
    for (const auto columns : {256U, 512U, 5120U, 17408U}) {
        for (const auto rows : {1U, 7U, 8U, 9U, 17U, 24U}) {
            const auto weights = make_weights(columns, rows, 23, random);
            const auto input = make_input(columns, random);
            for (const int threads : {1, 4}) {
                qwen_native::clear_cpu_packed_matrix_cache();
                std::vector<float> expected(rows * 8 + 16, kCanary);
                auto actual = expected;
                auto scalar = expected;
                mode(false);
                qwen_native::cpu_quant_matvec_batch8_rows_packed(weights.data(), 23,
                    input.data(), columns, expected.data() + 8, rows, threads);
                const auto bytes = qwen_native::cpu_packed_matrix_cache_bytes();
                if (bytes != ((rows + 7) / 8) * (columns / 256) * 1088) {
                    throw std::runtime_error("incorrect IQ4 cache size");
                }
                mode(true);
                qwen_native::cpu_quant_matvec_batch8_rows_packed(weights.data(), 23,
                    input.data(), columns, actual.data() + 8, rows, threads);
                require_equal(expected, actual);
                qwen_native::cpu_quant_matvec_batch8_rows(weights.data(), 23,
                    input.data(), columns, scalar.data() + 8, rows, threads);
                require_equal(scalar, actual);
                const auto decoded_bytes = predecoded ? ((rows + 7) / 8) * (columns / 256) * 2304 : 0;
                if (qwen_native::cpu_packed_matrix_cache_bytes() != bytes + decoded_bytes) {
                    throw std::runtime_error("incorrect IQ4 experiment cache bytes or aliased cache key");
                }
                if (predecoded) {
                    // The expanded layout takes precedence over tile-GEMM.
                    if (setenv(kIq4Mode, "1", 1)) throw std::runtime_error("failed to set combined IQ4 modes");
                    qwen_native::cpu_quant_matvec_batch8_rows_packed(weights.data(), 23,
                        input.data(), columns, actual.data() + 8, rows, threads);
                    require_equal(expected, actual);
                    if (setenv(kIq4Mode, "0", 1)) throw std::runtime_error("failed to restore isolated IQ4 mode");
                }
                mode(false);
                qwen_native::cpu_quant_matvec_batch8_rows_packed(weights.data(), 23,
                    input.data(), columns, actual.data() + 8, rows, threads);
                require_equal(expected, actual);
                require_guards(actual);
                ++fixtures;
            }
        }
    }
    for (const auto types : {std::array<std::uint32_t, 2>{23, 23}, {23, 12},
                            {12, 23}, {23, 14}, {14, 23}}) {
        constexpr std::size_t columns = 512, gate_rows = 17, up_rows = 9;
        qwen_native::clear_cpu_packed_matrix_cache();
        const auto gate = make_weights(columns, gate_rows, types[0], random);
        const auto up = make_weights(columns, up_rows, types[1], random);
        const auto input = make_input(columns, random);
        std::vector<float> expected_gate(gate_rows * 8 + 16, kCanary);
        std::vector<float> expected_up(up_rows * 8 + 16, kCanary);
        auto actual_gate = expected_gate;
        auto actual_up = expected_up;
        mode(false);
        qwen_native::cpu_quant_matvec_batch8_rows_packed_dual(gate.data(), types[0],
            expected_gate.data() + 8, gate_rows, up.data(), types[1], expected_up.data() + 8,
            up_rows, input.data(), columns, 4);
        mode(true);
        qwen_native::cpu_quant_matvec_batch8_rows_packed_dual(gate.data(), types[0],
            actual_gate.data() + 8, gate_rows, up.data(), types[1], actual_up.data() + 8,
            up_rows, input.data(), columns, 4);
        require_equal(expected_gate, actual_gate);
        require_equal(expected_up, actual_up);
        require_guards(actual_gate);
        require_guards(actual_up);
        ++fixtures;
    }
    qwen_native::clear_cpu_packed_matrix_cache();
    std::cout << "CPU IQ4_XS " << (predecoded ? "predecode" : "tile GEMM") << ": " << fixtures
              << " bit-exact fixtures passed; scalar oracle, guards, cache layout, dual paths passed\n";
}

void k_gemm_self_test() {
    RestoreEnvironment restore(kKMode);
    RestoreEnvironment restore_iq4(kIq4Mode);
    RestoreEnvironment restore_iq4_predecode(kIq4Predecode);
    RestoreEnvironment restore_q6;
    const auto mode = [](const char *name, bool enabled) {
        if (setenv(name, enabled ? "1" : "0", 1)) throw std::runtime_error("failed to set CPU test mode");
    };
    mode(kIq4Mode, false);
    mode(kIq4Predecode, false);
    Random random;
    int fixtures = 0;
    for (const auto type : {12U, 13U}) {
        for (const auto columns : {256U, 512U, 5120U, 17408U}) {
            for (const auto rows : {1U, 7U, 8U, 9U, 17U, 24U}) {
                const auto weights = make_weights(columns, rows, type, random);
                const auto input = make_input(columns, random);
                for (const int threads : {1, 4}) {
                    qwen_native::clear_cpu_packed_matrix_cache();
                    std::vector<float> expected(rows * 8 + 16, kCanary);
                    auto actual = expected;
                    auto scalar = expected;
                    mode(kKMode, false);
                    qwen_native::cpu_quant_matvec_batch8_rows_packed(weights.data(), type,
                        input.data(), columns, expected.data() + 8, rows, threads);
                    const auto bytes = qwen_native::cpu_packed_matrix_cache_bytes();
                    mode(kKMode, true);
                    qwen_native::cpu_quant_matvec_batch8_rows_packed(weights.data(), type,
                        input.data(), columns, actual.data() + 8, rows, threads);
                    require_equal(expected, actual);
                    qwen_native::cpu_quant_matvec_batch8_rows(weights.data(), type,
                        input.data(), columns, scalar.data() + 8, rows, threads);
                    require_equal(scalar, actual);
                    if (qwen_native::cpu_packed_matrix_cache_bytes() != bytes) {
                        throw std::runtime_error("K microkernel unexpectedly changed matrix cache bytes");
                    }
                    mode(kKMode, false);
                    qwen_native::cpu_quant_matvec_batch8_rows_packed(weights.data(), type,
                        input.data(), columns, actual.data() + 8, rows, threads);
                    require_equal(expected, actual);
                    require_guards(actual);
                    ++fixtures;
                }
            }
        }
    }
    mode(kMode, true);
    for (const auto types : {std::array<std::uint32_t, 2>{12, 12}, {13, 13}, {12, 13},
                            {13, 12}, {12, 23}, {23, 13}, {13, 14}, {14, 12}}) {
        for (const bool iq4_gemm : {false, true}) {
            for (const bool iq4_predecoded : {false, true}) {
                constexpr std::size_t columns = 512, gate_rows = 17, up_rows = 9;
                qwen_native::clear_cpu_packed_matrix_cache();
                mode(kIq4Mode, iq4_gemm);
                mode(kIq4Predecode, iq4_predecoded);
                const auto gate = make_weights(columns, gate_rows, types[0], random);
                const auto up = make_weights(columns, up_rows, types[1], random);
                const auto input = make_input(columns, random);
                std::vector<float> expected_gate(gate_rows * 8 + 16, kCanary);
                std::vector<float> expected_up(up_rows * 8 + 16, kCanary);
                auto actual_gate = expected_gate;
                auto actual_up = expected_up;
                mode(kKMode, false);
                qwen_native::cpu_quant_matvec_batch8_rows_packed_dual(gate.data(), types[0],
                    expected_gate.data() + 8, gate_rows, up.data(), types[1], expected_up.data() + 8,
                    up_rows, input.data(), columns, 4);
                mode(kKMode, true);
                qwen_native::cpu_quant_matvec_batch8_rows_packed_dual(gate.data(), types[0],
                    actual_gate.data() + 8, gate_rows, up.data(), types[1], actual_up.data() + 8,
                    up_rows, input.data(), columns, 4);
                require_equal(expected_gate, actual_gate);
                require_equal(expected_up, actual_up);
                require_guards(actual_gate);
                require_guards(actual_up);
                ++fixtures;
            }
        }
    }
    qwen_native::clear_cpu_packed_matrix_cache();
    std::cout << "CPU Q4/Q5_K tile GEMM: " << fixtures
              << " bit-exact fixtures passed; scalar oracle, interleaved FMA order, guards, dual flags passed\n";
}

void benchmark(const std::string &path, const std::string &name, int iterations, int threads,
               bool iq4_predecoded = false) {
    if (iterations < 1 || iterations > 1000 || threads < 1 || threads > 24) {
        throw std::invalid_argument("benchmark requires ITERATIONS in [1,1000], THREADS in [1,24]");
    }
    RestoreEnvironment restore;
    RestoreEnvironment restore_iq4(kIq4Mode);
    RestoreEnvironment restore_iq4_predecode(kIq4Predecode);
    const qwen_native::GgufFile model(path);
    const auto *tensor = model.find_tensor(name);
    if (!tensor || (tensor->type != 14 && tensor->type != 23) || tensor->dimensions.size() != 2 ||
        tensor->dimensions[0] == 0 || tensor->dimensions[0] % 256 || tensor->dimensions[1] == 0) {
        throw std::invalid_argument("benchmark needs a Q6_K or IQ4_XS matrix tensor");
    }
    const std::size_t columns = tensor->dimensions[0], rows = tensor->dimensions[1];
    const std::size_t bytes = columns / 256 * rows * (tensor->type == 14 ? 210 : 136);
    if (iq4_predecoded && tensor->type != 23) {
        throw std::invalid_argument("IQ4 predecode benchmark requires an IQ4_XS tensor");
    }
    const char *mode_name = tensor->type == 14 ? kMode : iq4_predecoded ? kIq4Predecode : kIq4Mode;
    const char *experiment = tensor->type == 14 ? "predecoded" : iq4_predecoded ? "predecoded-IQ4" : "tile-GEMM";
    if (tensor->type == 23 && setenv(iq4_predecoded ? kIq4Mode : kIq4Predecode, "0", 1)) {
        throw std::runtime_error("failed to isolate IQ4 benchmark mode");
    }
    if (bytes > 512ULL * 1024 * 1024) throw std::invalid_argument("benchmark tensor exceeds 512 MiB");
    std::vector<std::uint8_t> weights(bytes);
    std::ifstream file(path, std::ios::binary);
    file.seekg(model.data_offset() + tensor->offset);
    if (!file.read(reinterpret_cast<char *>(weights.data()), bytes)) {
        throw std::runtime_error("could not read benchmark tensor");
    }
    Random random;
    const auto input = make_input(columns, random);
    std::vector<float> expected(8 * rows), actual(8 * rows);
    qwen_native::clear_cpu_packed_matrix_cache();
    const auto measure = [&](bool decoded, std::vector<float> &output) {
        if (setenv(mode_name, decoded ? "1" : "0", 1)) {
            throw std::runtime_error("failed to set CPU benchmark experiment mode");
        }
        // Packing and two warm-up calls are explicitly outside matrix timing.
        for (int warmup = 0; warmup < 3; ++warmup) {
            qwen_native::cpu_quant_matvec_batch8_rows_packed(weights.data(), tensor->type,
                input.data(), columns, output.data(), rows, threads);
        }
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            qwen_native::cpu_quant_matvec_batch8_rows_packed(weights.data(), tensor->type,
                input.data(), columns, output.data(), rows, threads);
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()
             / iterations;
    };
    std::cout << "CPU component benchmark (not generated-token throughput): " << name
              << ", columns " << columns << ", rows " << rows << ", threads " << threads
              << ", iterations " << iterations << '\n';
    // Preserve all paired measurements; alternating order avoids one-sided
    // warming/order bias. Repeat on an idle host before selecting a default.
    for (int pair = 0; pair < 3; ++pair) {
        double old_ms, decoded_ms;
        if (pair % 2 == 0) {
            old_ms = measure(false, expected);
            decoded_ms = measure(true, actual);
        } else {
            decoded_ms = measure(true, actual);
            old_ms = measure(false, expected);
        }
        if (std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float))) {
            throw std::runtime_error("real CPU tensor lost bit-exactness");
        }
        std::cout << "  pair " << pair + 1 << ": original " << old_ms
                  << " ms, " << experiment << ' ' << decoded_ms << " ms, bit-exact yes\n";
    }
    qwen_native::clear_cpu_packed_matrix_cache();
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 1) {
            self_test();
            iq4_gemm_self_test();
            iq4_gemm_self_test(true);
            k_gemm_self_test();
        }
        else if (argc == 6 && std::string(argv[1]) == "--benchmark-iq4-predecode") {
            benchmark(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]), true);
        }
        else if (argc == 6 && std::string(argv[1]) == "--benchmark") {
            benchmark(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]));
        } else {
            std::cerr << "usage: " << argv[0] << " [--benchmark|--benchmark-iq4-predecode MODEL.gguf TENSOR ITERATIONS THREADS]\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

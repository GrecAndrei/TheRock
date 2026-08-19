#pragma once

#include <cstddef>
#include <cstdint>

namespace qwen_native {

struct Q4KBenchmarkResult {
    float milliseconds = 0.0F;
    float bandwidth_gbps = 0.0F;
    float max_absolute_error = 0.0F;
    float relative_rmse = 0.0F;
    float kernel_max_absolute_error = 0.0F;
};

struct FfnBenchmarkResult {
    float milliseconds = 0.0F;
    float weight_bandwidth_gbps = 0.0F;
    float verifier_tokens_per_second = 0.0F;
    float max_absolute_error = 0.0F;
};

Q4KBenchmarkResult benchmark_q4k(const void *weights, std::size_t input_size,
                                 std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_q4k_batch8(const void *weights, std::size_t input_size,
                                        std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_q5k(const void *weights, std::size_t input_size,
                                 std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_q5k_batch8(const void *weights, std::size_t input_size,
                                        std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_q6k(const void *weights, std::size_t input_size,
                                 std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_q6k_batch8(const void *weights, std::size_t input_size,
                                        std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_n23q4_from_q4k(const void *weights, std::size_t input_size,
                                             std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_n23q4_batch8_from_q4k(const void *weights, std::size_t input_size,
                                                    std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_n23kv4(std::size_t context_size, int iterations);
Q4KBenchmarkResult benchmark_n23kv4_batch8(std::size_t context_size, int iterations);
Q4KBenchmarkResult benchmark_n23kv4_flash_batch8(std::size_t context_size,
                                                  int partitions, int iterations);
Q4KBenchmarkResult benchmark_gated_delta_step(int iterations);
FfnBenchmarkResult benchmark_ffn_batch8(const void *gate_weights, std::uint32_t gate_type,
                                         const void *up_weights, std::uint32_t up_type,
                                         const void *down_weights, std::uint32_t down_type,
                                         std::size_t input_size, std::size_t hidden_size,
                                         std::size_t output_size, int iterations);
FfnBenchmarkResult benchmark_ffn_batch8_n23q4(const void *gate_weights,
                                               const void *up_weights,
                                               const void *down_weights,
                                               std::uint32_t down_type,
                                               std::size_t input_size,
                                               std::size_t hidden_size,
                                               std::size_t output_size, int iterations);

} // namespace qwen_native

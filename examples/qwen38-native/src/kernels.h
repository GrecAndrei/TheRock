#pragma once

#include <cstddef>

namespace qwen_native {

struct Q4KBenchmarkResult {
    float milliseconds = 0.0F;
    float bandwidth_gbps = 0.0F;
    float max_absolute_error = 0.0F;
    float relative_rmse = 0.0F;
    float kernel_max_absolute_error = 0.0F;
};

Q4KBenchmarkResult benchmark_q4k(const void *weights, std::size_t input_size,
                                 std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_q4k_batch8(const void *weights, std::size_t input_size,
                                        std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_n23q4_from_q4k(const void *weights, std::size_t input_size,
                                             std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_n23q4_batch8_from_q4k(const void *weights, std::size_t input_size,
                                                    std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_n23kv4(std::size_t context_size, int iterations);
Q4KBenchmarkResult benchmark_n23kv4_batch8(std::size_t context_size, int iterations);
Q4KBenchmarkResult benchmark_n23kv4_flash_batch8(std::size_t context_size,
                                                  int partitions, int iterations);

} // namespace qwen_native

#pragma once

#include <cstddef>
#include <cstdint>

namespace qwen_native {

// CPU half of a row-split batch-8 quantized matrix multiplication. Input and
// output are batch-major; weights point at row_begin in native GGUF packing.
void cpu_quant_matvec_batch8_rows(const void *weights, std::uint32_t type,
                                  const float *input, std::size_t input_size,
                                  float *output, std::size_t row_count,
                                  int thread_count);
void cpu_quant_matvec_batch8_rows_packed(const void *weights, std::uint32_t type,
                                         const float *input, std::size_t input_size,
                                         float *output, std::size_t row_count,
                                         int thread_count);
void clear_cpu_packed_matrix_cache();
std::size_t cpu_packed_matrix_cache_bytes();

struct CpuMatvecBenchmarkResult {
    double milliseconds = 0.0;
    double bandwidth_gbps = 0.0;
    double packing_milliseconds = 0.0;
    float max_absolute_error = 0.0F;
};

CpuMatvecBenchmarkResult benchmark_cpu_q4k_soa_batch8(
    const void *weights, const float *input, std::size_t input_size,
    std::size_t row_count, int iterations, int thread_count,
    const float *reference_output);
CpuMatvecBenchmarkResult benchmark_cpu_q5k_soa_batch8(
    const void *weights, const float *input, std::size_t input_size,
    std::size_t row_count, int iterations, int thread_count,
    const float *reference_output);
CpuMatvecBenchmarkResult benchmark_cpu_q6k_soa_batch8(
    const void *weights, const float *input, std::size_t input_size,
    std::size_t row_count, int iterations, int thread_count,
    const float *reference_output);
CpuMatvecBenchmarkResult benchmark_cpu_iq4xs_soa_batch8(
    const void *weights, const float *input, std::size_t input_size,
    std::size_t row_count, int iterations, int thread_count,
    const float *reference_output);

void compute_dflash_noncausal_attention(
    const float *queries, const float *keys, const float *values,
    const std::uint16_t *cache_keys, const std::uint16_t *cache_values,
    std::size_t cached, float *output);

} // namespace qwen_native

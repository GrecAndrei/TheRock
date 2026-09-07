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
// Fused Gate+Up: computes both matrices from the same batch-8 input in a single
// thread-pool barrier (saves 64 barriers per 64-layer decode step).
void cpu_quant_matvec_batch8_rows_packed_dual(
    const void *gate_weights, std::uint32_t gate_type, float *gate_output,
    std::size_t gate_rows, const void *up_weights, std::uint32_t up_type,
    float *up_output, std::size_t up_rows, const float *input,
    std::size_t input_size, int thread_count);
// Fused QKV triple for attention (saves 2 barriers per attention layer, 32 per decode)
void cpu_quant_matvec_batch8_rows_packed_triple(
    const void *q_weights, std::uint32_t q_type, float *q_output, std::size_t q_rows,
    const void *k_weights, std::uint32_t k_type, float *k_output, std::size_t k_rows,
    const void *v_weights, std::uint32_t v_type, float *v_output, std::size_t v_rows,
    const float *input, std::size_t input_size, int thread_count);
// Large batch (up to 512) for prefill
void cpu_quant_matvec_batch_large(const void *weights, std::uint32_t type,
                                   const float *input, std::size_t input_size,
                                   float *output, std::size_t row_count, std::size_t batch_size,
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

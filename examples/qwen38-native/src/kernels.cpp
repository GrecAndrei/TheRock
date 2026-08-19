#include "kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen_native {
namespace {

constexpr int kQuantBlock = 256;
constexpr int kWarpSize = 32;

struct Q4KBlock {
    std::uint16_t d;
    std::uint16_t dmin;
    std::uint8_t scales[12];
    std::uint8_t quants[128];
};
static_assert(sizeof(Q4KBlock) == 144);

// N23Q4 is laid out for one gfx1032 wave32 to produce four output rows.
// Each eight-lane subgroup consumes one row using v_dot4_i32_i8. Four FP16
// scales, biases, and four contiguous signed-int4 groups make each weight load
// by the wave a single 64-byte span. It preserves Q4_K's asymmetric group
// values at 5 bits/weight, trading 11% more bytes for no scale-bit unpacking.
struct N23Q4TileGroup {
    std::uint16_t scales[4];
    std::uint16_t biases[4];
    std::uint8_t quants[4][16];
};
static_assert(sizeof(N23Q4TileGroup) == 80);

void hip_check(hipError_t status, const char *operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

__device__ __forceinline__ void scale_min(const std::uint8_t *packed, int index,
                                           std::uint8_t &scale, std::uint8_t &minimum) {
    if (index < 4) {
        scale = packed[index] & 63;
        minimum = packed[index + 4] & 63;
    } else {
        scale = (packed[index + 4] & 0x0F) | ((packed[index - 4] >> 6) << 4);
        minimum = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
    }
}

__global__ void q4k_matvec_kernel(const Q4KBlock *__restrict__ weights,
                                  const float *__restrict__ input,
                                  float *__restrict__ output,
                                  int blocks_per_row, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kWarpSize) {
        return;
    }

    float sum = 0.0F;
    const Q4KBlock *row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const Q4KBlock &block = row_weights[block_index];
        const float scale_all = __half2float(*reinterpret_cast<const __half *>(&block.d));
        const float min_all = __half2float(*reinterpret_cast<const __half *>(&block.dmin));
        const float *block_input = input + static_cast<std::size_t>(block_index) * kQuantBlock;

#pragma unroll
        for (int group = 0; group < 4; ++group) {
            std::uint8_t scale0 = 0;
            std::uint8_t min0 = 0;
            std::uint8_t scale1 = 0;
            std::uint8_t min1 = 0;
            scale_min(block.scales, group * 2, scale0, min0);
            scale_min(block.scales, group * 2 + 1, scale1, min1);
            const std::uint8_t quant = block.quants[group * kWarpSize + lane];
            const int base = group * 64 + lane;
            const float weight0 = scale_all * static_cast<float>(scale0) * static_cast<float>(quant & 0x0F)
                                - min_all * static_cast<float>(min0);
            const float weight1 = scale_all * static_cast<float>(scale1) * static_cast<float>(quant >> 4)
                                - min_all * static_cast<float>(min1);
            sum = fmaf(weight0, block_input[base], sum);
            sum = fmaf(weight1, block_input[base + 32], sum);
        }
    }

#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, kWarpSize);
    }
    if (lane == 0) {
        output[row] = sum;
    }
}

__global__ void q4k_matvec_batch8_kernel(const Q4KBlock *__restrict__ weights,
                                         const float *__restrict__ input,
                                         float *__restrict__ output,
                                         int input_size, int blocks_per_row, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kWarpSize) {
        return;
    }

    float sums[8] = {};
    const Q4KBlock *row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const Q4KBlock &block = row_weights[block_index];
        const float scale_all = __half2float(*reinterpret_cast<const __half *>(&block.d));
        const float min_all = __half2float(*reinterpret_cast<const __half *>(&block.dmin));

#pragma unroll
        for (int group = 0; group < 4; ++group) {
            std::uint8_t scale0 = 0;
            std::uint8_t min0 = 0;
            std::uint8_t scale1 = 0;
            std::uint8_t min1 = 0;
            scale_min(block.scales, group * 2, scale0, min0);
            scale_min(block.scales, group * 2 + 1, scale1, min1);
            const std::uint8_t quant = block.quants[group * kWarpSize + lane];
            const int index0 = block_index * kQuantBlock + group * 64 + lane;
            const int index1 = index0 + 32;
            const float weight0 = scale_all * static_cast<float>(scale0) * static_cast<float>(quant & 0x0F)
                                - min_all * static_cast<float>(min0);
            const float weight1 = scale_all * static_cast<float>(scale1) * static_cast<float>(quant >> 4)
                                - min_all * static_cast<float>(min1);
#pragma unroll
            for (int batch = 0; batch < 8; ++batch) {
                const float *batch_input = input + static_cast<std::size_t>(batch) * input_size;
                sums[batch] = fmaf(weight0, batch_input[index0], sums[batch]);
                sums[batch] = fmaf(weight1, batch_input[index1], sums[batch]);
            }
        }
    }

#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
        for (int batch = 0; batch < 8; ++batch) {
            sums[batch] += __shfl_down(sums[batch], offset, kWarpSize);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int batch = 0; batch < 8; ++batch) {
            output[static_cast<std::size_t>(batch) * rows + row] = sums[batch];
        }
    }
}

__global__ void n23q4_matvec_kernel(const N23Q4TileGroup *__restrict__ weights,
                                    const std::int8_t *__restrict__ input,
                                    const float *__restrict__ input_scales,
                                    const int *__restrict__ input_sums,
                                    float *__restrict__ output,
                                    int groups_per_row, int rows) {
    const int tile = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int row_in_tile = lane / 8;
    const int lane_in_row = lane & 7;
    const int row = tile * 4 + row_in_tile;
    if (lane >= kWarpSize) {
        return;
    }

    float sum = 0.0F;
    const auto *tile_weights = weights + static_cast<std::size_t>(tile) * groups_per_row;
    for (int group = 0; group < groups_per_row; ++group) {
        const auto &packed = tile_weights[group];
        const auto *nibbles = packed.quants[row_in_tile] + 2 * lane_in_row;
        const std::uint8_t low = nibbles[0];
        const std::uint8_t high = nibbles[1];
        const int w0 = ((static_cast<int>(low & 0x0F) ^ 8) - 8);
        const int w1 = ((static_cast<int>(low >> 4) ^ 8) - 8);
        const int w2 = ((static_cast<int>(high & 0x0F) ^ 8) - 8);
        const int w3 = ((static_cast<int>(high >> 4) ^ 8) - 8);
        const std::uint32_t weight_word = static_cast<std::uint8_t>(w0)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(w1)) << 8)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(w2)) << 16)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(w3)) << 24);
        const auto *input_word = reinterpret_cast<const std::uint32_t *>(
            input + static_cast<std::size_t>(group) * 32 + lane_in_row * 4);
        const int dot = __builtin_amdgcn_sdot4(static_cast<int>(weight_word),
                                                static_cast<int>(*input_word), 0, false);
        const float weight_scale = __half2float(
            *reinterpret_cast<const __half *>(&packed.scales[row_in_tile]));
        const float weight_bias = __half2float(
            *reinterpret_cast<const __half *>(&packed.biases[row_in_tile]));
        sum += (static_cast<float>(dot) * weight_scale
                + 0.125F * static_cast<float>(input_sums[group]) * weight_bias)
             * input_scales[group];
    }

#pragma unroll
    for (int offset = 4; offset > 0; offset /= 2) {
        const float other = __shfl_down(sum, offset, kWarpSize);
        if (lane_in_row + offset < 8) {
            sum += other;
        }
    }
    if (lane_in_row == 0 && row < rows) {
        output[row] = sum;
    }
}

__global__ void n23q4_batch8_kernel(const N23Q4TileGroup *__restrict__ weights,
                                    const std::int8_t *__restrict__ input,
                                    const float *__restrict__ input_scales,
                                    const int *__restrict__ input_sums,
                                    float *__restrict__ output,
                                    int input_size, int groups_per_row, int rows) {
    const int tile = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int row_in_tile = lane / 8;
    const int lane_in_row = lane & 7;
    const int row = tile * 4 + row_in_tile;
    if (lane >= kWarpSize) {
        return;
    }

    float sums[8] = {};
    const auto *tile_weights = weights + static_cast<std::size_t>(tile) * groups_per_row;
    for (int group = 0; group < groups_per_row; ++group) {
        const auto &packed = tile_weights[group];
        const auto *nibbles = packed.quants[row_in_tile] + 2 * lane_in_row;
        const std::uint8_t low = nibbles[0];
        const std::uint8_t high = nibbles[1];
        const int w0 = ((static_cast<int>(low & 0x0F) ^ 8) - 8);
        const int w1 = ((static_cast<int>(low >> 4) ^ 8) - 8);
        const int w2 = ((static_cast<int>(high & 0x0F) ^ 8) - 8);
        const int w3 = ((static_cast<int>(high >> 4) ^ 8) - 8);
        const std::uint32_t weight_word = static_cast<std::uint8_t>(w0)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(w1)) << 8)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(w2)) << 16)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(w3)) << 24);
        const float weight_scale = __half2float(
            *reinterpret_cast<const __half *>(&packed.scales[row_in_tile]));
        const float weight_bias = __half2float(
            *reinterpret_cast<const __half *>(&packed.biases[row_in_tile]));

#pragma unroll
        for (int batch = 0; batch < 8; ++batch) {
            const auto *input_word = reinterpret_cast<const std::uint32_t *>(
                input + static_cast<std::size_t>(batch) * input_size
                + static_cast<std::size_t>(group) * 32 + lane_in_row * 4);
            const int dot = __builtin_amdgcn_sdot4(static_cast<int>(weight_word),
                                                    static_cast<int>(*input_word), 0, false);
            const std::size_t group_index = static_cast<std::size_t>(batch) * groups_per_row + group;
            sums[batch] += (static_cast<float>(dot) * weight_scale
                            + 0.125F * static_cast<float>(input_sums[group_index]) * weight_bias)
                         * input_scales[group_index];
        }
    }

#pragma unroll
    for (int offset = 4; offset > 0; offset /= 2) {
#pragma unroll
        for (int batch = 0; batch < 8; ++batch) {
            const float other = __shfl_down(sums[batch], offset, kWarpSize);
            if (lane_in_row + offset < 8) {
                sums[batch] += other;
            }
        }
    }
    if (lane_in_row == 0 && row < rows) {
#pragma unroll
        for (int batch = 0; batch < 8; ++batch) {
            output[static_cast<std::size_t>(batch) * rows + row] = sums[batch];
        }
    }
}

float fp16_to_float(std::uint16_t bits) {
    __half value;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return __half2float(value);
}

void scale_min_host(const std::uint8_t *packed, int index,
                    std::uint8_t &scale, std::uint8_t &minimum) {
    if (index < 4) {
        scale = packed[index] & 63;
        minimum = packed[index + 4] & 63;
    } else {
        scale = (packed[index + 4] & 0x0F) | ((packed[index - 4] >> 6) << 4);
        minimum = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
    }
}

float q4k_reference_row(const Q4KBlock *weights, const float *input, std::size_t blocks) {
    float sum = 0.0F;
    for (std::size_t block_index = 0; block_index < blocks; ++block_index) {
        const auto &block = weights[block_index];
        const float scale_all = fp16_to_float(block.d);
        const float min_all = fp16_to_float(block.dmin);
        for (int group = 0; group < 4; ++group) {
            std::uint8_t scale0 = 0;
            std::uint8_t min0 = 0;
            std::uint8_t scale1 = 0;
            std::uint8_t min1 = 0;
            scale_min_host(block.scales, group * 2, scale0, min0);
            scale_min_host(block.scales, group * 2 + 1, scale1, min1);
            for (int lane = 0; lane < kWarpSize; ++lane) {
                const auto quant = block.quants[group * kWarpSize + lane];
                const auto base = block_index * kQuantBlock + group * 64 + lane;
                const float weight0 = scale_all * static_cast<float>(scale0) * static_cast<float>(quant & 0x0F)
                                    - min_all * static_cast<float>(min0);
                const float weight1 = scale_all * static_cast<float>(scale1) * static_cast<float>(quant >> 4)
                                    - min_all * static_cast<float>(min1);
                sum = std::fma(weight0, input[base], sum);
                sum = std::fma(weight1, input[base + 32], sum);
            }
        }
    }
    return sum;
}

std::uint16_t float_to_fp16(float value) {
    const __half converted = __float2half(value);
    std::uint16_t bits = 0;
    std::memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

std::vector<N23Q4TileGroup> convert_to_n23q4(const Q4KBlock *weights,
                                             std::size_t input_size,
                                             std::size_t output_size) {
    const std::size_t q4k_blocks_per_row = input_size / kQuantBlock;
    const std::size_t groups_per_row = input_size / 32;
    const std::size_t tiles = (output_size + 3) / 4;
    std::vector<N23Q4TileGroup> packed(tiles * groups_per_row);
    for (std::size_t tile = 0; tile < tiles; ++tile) {
        for (int row_in_tile = 0; row_in_tile < 4; ++row_in_tile) {
            const std::size_t row = tile * 4 + static_cast<std::size_t>(row_in_tile);
            if (row >= output_size) {
                continue;
            }
            const auto *row_weights = weights + row * q4k_blocks_per_row;
            for (std::size_t block = 0; block < q4k_blocks_per_row; ++block) {
                const auto &source = row_weights[block];
                const float super_scale = fp16_to_float(source.d);
                const float super_min = fp16_to_float(source.dmin);
                for (int group_in_block = 0; group_in_block < 8; ++group_in_block) {
                    const std::size_t group = block * 8 + static_cast<std::size_t>(group_in_block);
                    auto &destination = packed[tile * groups_per_row + group];
                    std::uint8_t quant_scale = 0;
                    std::uint8_t quant_min = 0;
                    scale_min_host(source.scales, group_in_block, quant_scale, quant_min);
                    const float scale = super_scale * static_cast<float>(quant_scale);
                    const float minimum = super_min * static_cast<float>(quant_min);
                    destination.scales[row_in_tile] = float_to_fp16(scale);
                    destination.biases[row_in_tile] = float_to_fp16(8.0F * scale - minimum);
                    const int chunk = group_in_block / 2;
                    const int shift = (group_in_block & 1) * 4;
                    for (int i = 0; i < 16; ++i) {
                        const int first_source = (source.quants[chunk * 32 + 2 * i] >> shift) & 0x0F;
                        const int second_source = (source.quants[chunk * 32 + 2 * i + 1] >> shift) & 0x0F;
                        const int first = (first_source - 8) & 0x0F;
                        const int second = (second_source - 8) & 0x0F;
                        destination.quants[row_in_tile][i] = static_cast<std::uint8_t>(first | (second << 4));
                    }
                }
            }
        }
    }
    return packed;
}

void quantize_input_q8(const std::vector<float> &input, std::vector<std::int8_t> &quantized,
                       std::vector<float> &scales, std::vector<int> &sums) {
    const std::size_t groups = input.size() / 32;
    quantized.resize(input.size());
    scales.resize(groups);
    sums.resize(groups);
    for (std::size_t group = 0; group < groups; ++group) {
        float maximum = 0.0F;
        for (int i = 0; i < 32; ++i) {
            maximum = std::max(maximum, std::abs(input[group * 32 + i]));
        }
        const float scale = maximum / 127.0F;
        scales[group] = scale;
        const float inverse = scale == 0.0F ? 0.0F : 1.0F / scale;
        for (int i = 0; i < 32; ++i) {
            quantized[group * 32 + i] = static_cast<std::int8_t>(std::clamp(
                static_cast<int>(std::nearbyint(input[group * 32 + i] * inverse)), -127, 127));
            sums[group] += quantized[group * 32 + i];
        }
    }
}

Q4KBenchmarkResult compare_results(const std::vector<float> &actual,
                                   const Q4KBlock *weights,
                                   const std::vector<float> &input,
                                   std::size_t blocks_per_row) {
    Q4KBenchmarkResult result;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (std::size_t row = 0; row < actual.size(); ++row) {
        const float expected = q4k_reference_row(weights + row * blocks_per_row,
                                                 input.data(), blocks_per_row);
        const float difference = expected - actual[row];
        result.max_absolute_error = std::max(result.max_absolute_error, std::abs(difference));
        squared_error += static_cast<double>(difference) * difference;
        squared_reference += static_cast<double>(expected) * expected;
    }
    result.relative_rmse = squared_reference == 0.0
        ? 0.0F
        : static_cast<float>(std::sqrt(squared_error / squared_reference));
    return result;
}

std::vector<float> n23q4_reference(const std::vector<N23Q4TileGroup> &weights,
                                   const std::vector<std::int8_t> &input,
                                   const std::vector<float> &input_scales,
                                   const std::vector<int> &input_sums,
                                   std::size_t groups_per_row,
                                   std::size_t rows) {
    std::vector<float> output(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t tile = row / 4;
        const int row_in_tile = static_cast<int>(row % 4);
        float sum = 0.0F;
        for (std::size_t group = 0; group < groups_per_row; ++group) {
            const auto &packed = weights[tile * groups_per_row + group];
            int dot = 0;
            for (int i = 0; i < 32; ++i) {
                const std::uint8_t byte = packed.quants[row_in_tile][i / 2];
                const int nibble = (byte >> ((i & 1) * 4)) & 0x0F;
                const int weight = nibble >= 8 ? nibble - 16 : nibble;
                dot += weight * input[group * 32 + static_cast<std::size_t>(i)];
            }
            const float scale = fp16_to_float(packed.scales[row_in_tile]);
            const float bias = fp16_to_float(packed.biases[row_in_tile]);
            sum += (static_cast<float>(dot) * scale
                    + static_cast<float>(input_sums[group]) * bias) * input_scales[group];
        }
        output[row] = sum;
    }
    return output;
}

} // namespace

Q4KBenchmarkResult benchmark_q4k(const void *weights, std::size_t input_size,
                                 std::size_t output_size, int iterations) {
    if (input_size == 0 || input_size % kQuantBlock != 0 || output_size == 0 || iterations <= 0) {
        throw std::invalid_argument("invalid Q4_K benchmark dimensions or iteration count");
    }
    const auto blocks_per_row = input_size / kQuantBlock;
    const auto weight_bytes = blocks_per_row * output_size * sizeof(Q4KBlock);

    std::vector<float> input(input_size);
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::generate(input.begin(), input.end(), [&] { return distribution(generator); });
    std::vector<float> output(output_size);

    void *device_weights = nullptr;
    float *device_input = nullptr;
    float *device_output = nullptr;
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    hip_check(hipMalloc(&device_weights, weight_bytes), "hipMalloc weights");
    try {
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_input), input.size() * sizeof(float)), "hipMalloc input");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_output), output.size() * sizeof(float)), "hipMalloc output");
        hip_check(hipMemcpy(device_weights, weights, weight_bytes, hipMemcpyHostToDevice), "hipMemcpy weights");
        hip_check(hipMemcpy(device_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy input");
        hip_check(hipEventCreate(&begin), "hipEventCreate begin");
        hip_check(hipEventCreate(&end), "hipEventCreate end");

        const dim3 grid(static_cast<unsigned int>(output_size));
        const dim3 block(kWarpSize);
        for (int i = 0; i < 5; ++i) {
            hipLaunchKernelGGL(q4k_matvec_kernel, grid, block, 0, nullptr,
                               static_cast<const Q4KBlock *>(device_weights), device_input,
                               device_output, static_cast<int>(blocks_per_row), static_cast<int>(output_size));
        }
        hip_check(hipDeviceSynchronize(), "Q4_K warmup");
        hip_check(hipEventRecord(begin), "hipEventRecord begin");
        for (int i = 0; i < iterations; ++i) {
            hipLaunchKernelGGL(q4k_matvec_kernel, grid, block, 0, nullptr,
                               static_cast<const Q4KBlock *>(device_weights), device_input,
                               device_output, static_cast<int>(blocks_per_row), static_cast<int>(output_size));
        }
        hip_check(hipEventRecord(end), "hipEventRecord end");
        hip_check(hipEventSynchronize(end), "hipEventSynchronize");
        float elapsed = 0.0F;
        hip_check(hipEventElapsedTime(&elapsed, begin, end), "hipEventElapsedTime");
        hip_check(hipMemcpy(output.data(), device_output, output.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy output");

        Q4KBenchmarkResult result;
        result.milliseconds = elapsed / static_cast<float>(iterations);
        result.bandwidth_gbps = static_cast<float>(weight_bytes) / (result.milliseconds * 1.0e6F);
        const auto *host_weights = static_cast<const Q4KBlock *>(weights);
        const std::size_t rows_to_check = std::min<std::size_t>(output_size, 8);
        for (std::size_t row = 0; row < rows_to_check; ++row) {
            const float expected = q4k_reference_row(host_weights + row * blocks_per_row,
                                                     input.data(), blocks_per_row);
            result.max_absolute_error = std::max(result.max_absolute_error, std::abs(expected - output[row]));
        }

        hip_check(hipEventDestroy(end), "hipEventDestroy end");
        hip_check(hipEventDestroy(begin), "hipEventDestroy begin");
        hip_check(hipFree(device_output), "hipFree output");
        hip_check(hipFree(device_input), "hipFree input");
        hip_check(hipFree(device_weights), "hipFree weights");
        return result;
    } catch (...) {
        if (end) (void) hipEventDestroy(end);
        if (begin) (void) hipEventDestroy(begin);
        if (device_output) (void) hipFree(device_output);
        if (device_input) (void) hipFree(device_input);
        (void) hipFree(device_weights);
        throw;
    }
}

Q4KBenchmarkResult benchmark_q4k_batch8(const void *weights, std::size_t input_size,
                                        std::size_t output_size, int iterations) {
    if (input_size == 0 || input_size % kQuantBlock != 0 || output_size == 0 || iterations <= 0) {
        throw std::invalid_argument("invalid Q4_K batch-8 benchmark dimensions or iteration count");
    }
    constexpr std::size_t batch_size = 8;
    const auto blocks_per_row = input_size / kQuantBlock;
    const auto weight_bytes = blocks_per_row * output_size * sizeof(Q4KBlock);
    std::vector<float> input(input_size * batch_size);
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::generate(input.begin(), input.end(), [&] { return distribution(generator); });
    std::vector<float> output(output_size * batch_size);

    void *device_weights = nullptr;
    float *device_input = nullptr;
    float *device_output = nullptr;
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    hip_check(hipMalloc(&device_weights, weight_bytes), "hipMalloc batch8 weights");
    try {
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_input), input.size() * sizeof(float)), "hipMalloc batch8 input");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_output), output.size() * sizeof(float)), "hipMalloc batch8 output");
        hip_check(hipMemcpy(device_weights, weights, weight_bytes, hipMemcpyHostToDevice), "hipMemcpy batch8 weights");
        hip_check(hipMemcpy(device_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy batch8 input");
        hip_check(hipEventCreate(&begin), "hipEventCreate batch8 begin");
        hip_check(hipEventCreate(&end), "hipEventCreate batch8 end");

        const dim3 grid(static_cast<unsigned int>(output_size));
        const dim3 block(kWarpSize);
        for (int i = 0; i < 5; ++i) {
            hipLaunchKernelGGL(q4k_matvec_batch8_kernel, grid, block, 0, nullptr,
                               static_cast<const Q4KBlock *>(device_weights), device_input,
                               device_output, static_cast<int>(input_size),
                               static_cast<int>(blocks_per_row), static_cast<int>(output_size));
        }
        hip_check(hipDeviceSynchronize(), "Q4_K batch8 warmup");
        hip_check(hipEventRecord(begin), "hipEventRecord batch8 begin");
        for (int i = 0; i < iterations; ++i) {
            hipLaunchKernelGGL(q4k_matvec_batch8_kernel, grid, block, 0, nullptr,
                               static_cast<const Q4KBlock *>(device_weights), device_input,
                               device_output, static_cast<int>(input_size),
                               static_cast<int>(blocks_per_row), static_cast<int>(output_size));
        }
        hip_check(hipEventRecord(end), "hipEventRecord batch8 end");
        hip_check(hipEventSynchronize(end), "hipEventSynchronize batch8");
        float elapsed = 0.0F;
        hip_check(hipEventElapsedTime(&elapsed, begin, end), "hipEventElapsedTime batch8");
        hip_check(hipMemcpy(output.data(), device_output, output.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy batch8 output");

        Q4KBenchmarkResult result;
        result.milliseconds = elapsed / static_cast<float>(iterations);
        result.bandwidth_gbps = static_cast<float>(weight_bytes) / (result.milliseconds * 1.0e6F);
        const auto *host_weights = static_cast<const Q4KBlock *>(weights);
        const std::size_t rows_to_check = std::min<std::size_t>(output_size, 8);
        for (std::size_t batch = 0; batch < batch_size; ++batch) {
            for (std::size_t row = 0; row < rows_to_check; ++row) {
                const float expected = q4k_reference_row(host_weights + row * blocks_per_row,
                                                         input.data() + batch * input_size,
                                                         blocks_per_row);
                result.max_absolute_error = std::max(
                    result.max_absolute_error, std::abs(expected - output[batch * output_size + row]));
            }
        }
        hip_check(hipEventDestroy(end), "hipEventDestroy batch8 end");
        hip_check(hipEventDestroy(begin), "hipEventDestroy batch8 begin");
        hip_check(hipFree(device_output), "hipFree batch8 output");
        hip_check(hipFree(device_input), "hipFree batch8 input");
        hip_check(hipFree(device_weights), "hipFree batch8 weights");
        return result;
    } catch (...) {
        if (end) (void) hipEventDestroy(end);
        if (begin) (void) hipEventDestroy(begin);
        if (device_output) (void) hipFree(device_output);
        if (device_input) (void) hipFree(device_input);
        (void) hipFree(device_weights);
        throw;
    }
}

Q4KBenchmarkResult benchmark_n23q4_from_q4k(const void *weights, std::size_t input_size,
                                             std::size_t output_size, int iterations) {
    if (input_size == 0 || input_size % kQuantBlock != 0 || output_size == 0 || iterations <= 0) {
        throw std::invalid_argument("invalid N23Q4 benchmark dimensions or iteration count");
    }
    const auto blocks_per_row = input_size / kQuantBlock;
    const auto groups_per_row = input_size / 32;
    const auto tiles = (output_size + 3) / 4;
    const auto *q4k_weights = static_cast<const Q4KBlock *>(weights);
    auto packed_weights = convert_to_n23q4(q4k_weights, input_size, output_size);

    std::vector<float> input(input_size);
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::generate(input.begin(), input.end(), [&] { return distribution(generator); });
    std::vector<std::int8_t> quantized_input;
    std::vector<float> input_scales;
    std::vector<int> input_sums;
    quantize_input_q8(input, quantized_input, input_scales, input_sums);
    std::vector<float> output(output_size);

    N23Q4TileGroup *device_weights = nullptr;
    std::int8_t *device_input = nullptr;
    float *device_scales = nullptr;
    int *device_sums = nullptr;
    float *device_output = nullptr;
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    hip_check(hipMalloc(reinterpret_cast<void **>(&device_weights), packed_weights.size() * sizeof(N23Q4TileGroup)), "hipMalloc N23Q4 weights");
    try {
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_input), quantized_input.size()), "hipMalloc N23Q4 input");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_scales), input_scales.size() * sizeof(float)), "hipMalloc N23Q4 scales");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_sums), input_sums.size() * sizeof(int)), "hipMalloc N23Q4 sums");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_output), output.size() * sizeof(float)), "hipMalloc N23Q4 output");
        hip_check(hipMemcpy(device_weights, packed_weights.data(), packed_weights.size() * sizeof(N23Q4TileGroup), hipMemcpyHostToDevice), "hipMemcpy N23Q4 weights");
        hip_check(hipMemcpy(device_input, quantized_input.data(), quantized_input.size(), hipMemcpyHostToDevice), "hipMemcpy N23Q4 input");
        hip_check(hipMemcpy(device_scales, input_scales.data(), input_scales.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy N23Q4 scales");
        hip_check(hipMemcpy(device_sums, input_sums.data(), input_sums.size() * sizeof(int), hipMemcpyHostToDevice), "hipMemcpy N23Q4 sums");
        hip_check(hipEventCreate(&begin), "hipEventCreate N23Q4 begin");
        hip_check(hipEventCreate(&end), "hipEventCreate N23Q4 end");

        const dim3 grid(static_cast<unsigned int>(tiles));
        const dim3 block(kWarpSize);
        for (int i = 0; i < 5; ++i) {
            hipLaunchKernelGGL(n23q4_matvec_kernel, grid, block, 0, nullptr,
                               device_weights, device_input, device_scales, device_sums, device_output,
                               static_cast<int>(groups_per_row), static_cast<int>(output_size));
        }
        hip_check(hipDeviceSynchronize(), "N23Q4 warmup");
        hip_check(hipEventRecord(begin), "hipEventRecord N23Q4 begin");
        for (int i = 0; i < iterations; ++i) {
            hipLaunchKernelGGL(n23q4_matvec_kernel, grid, block, 0, nullptr,
                               device_weights, device_input, device_scales, device_sums, device_output,
                               static_cast<int>(groups_per_row), static_cast<int>(output_size));
        }
        hip_check(hipEventRecord(end), "hipEventRecord N23Q4 end");
        hip_check(hipEventSynchronize(end), "hipEventSynchronize N23Q4");
        float elapsed = 0.0F;
        hip_check(hipEventElapsedTime(&elapsed, begin, end), "hipEventElapsedTime N23Q4");
        hip_check(hipMemcpy(output.data(), device_output, output.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy N23Q4 output");

        const auto reference = n23q4_reference(packed_weights, quantized_input, input_scales,
                                              input_sums, groups_per_row, output_size);
        auto result = compare_results(reference, q4k_weights, input, blocks_per_row);
        for (std::size_t row = 0; row < output_size; ++row) {
            result.kernel_max_absolute_error = std::max(
                result.kernel_max_absolute_error, std::abs(reference[row] - output[row]));
        }
        result.milliseconds = elapsed / static_cast<float>(iterations);
        result.bandwidth_gbps = static_cast<float>(packed_weights.size() * sizeof(N23Q4TileGroup))
                              / (result.milliseconds * 1.0e6F);
        hip_check(hipEventDestroy(end), "hipEventDestroy N23Q4 end");
        hip_check(hipEventDestroy(begin), "hipEventDestroy N23Q4 begin");
        hip_check(hipFree(device_output), "hipFree N23Q4 output");
        hip_check(hipFree(device_scales), "hipFree N23Q4 scales");
        hip_check(hipFree(device_sums), "hipFree N23Q4 sums");
        hip_check(hipFree(device_input), "hipFree N23Q4 input");
        hip_check(hipFree(device_weights), "hipFree N23Q4 weights");
        return result;
    } catch (...) {
        if (end) (void) hipEventDestroy(end);
        if (begin) (void) hipEventDestroy(begin);
        if (device_output) (void) hipFree(device_output);
        if (device_scales) (void) hipFree(device_scales);
        if (device_sums) (void) hipFree(device_sums);
        if (device_input) (void) hipFree(device_input);
        (void) hipFree(device_weights);
        throw;
    }
}

Q4KBenchmarkResult benchmark_n23q4_batch8_from_q4k(const void *weights, std::size_t input_size,
                                                    std::size_t output_size, int iterations) {
    if (input_size == 0 || input_size % kQuantBlock != 0 || output_size == 0 || iterations <= 0) {
        throw std::invalid_argument("invalid N23Q4 batch-8 benchmark dimensions or iteration count");
    }
    constexpr std::size_t batch_size = 8;
    const auto groups_per_row = input_size / 32;
    const auto tiles = (output_size + 3) / 4;
    const auto *q4k_weights = static_cast<const Q4KBlock *>(weights);
    auto packed_weights = convert_to_n23q4(q4k_weights, input_size, output_size);
    std::vector<float> input(input_size * batch_size);
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::generate(input.begin(), input.end(), [&] { return distribution(generator); });
    std::vector<std::int8_t> quantized_input;
    std::vector<float> input_scales;
    std::vector<int> input_sums;
    quantize_input_q8(input, quantized_input, input_scales, input_sums);
    std::vector<float> output(output_size * batch_size);

    N23Q4TileGroup *device_weights = nullptr;
    std::int8_t *device_input = nullptr;
    float *device_scales = nullptr;
    int *device_sums = nullptr;
    float *device_output = nullptr;
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    hip_check(hipMalloc(reinterpret_cast<void **>(&device_weights), packed_weights.size() * sizeof(N23Q4TileGroup)), "hipMalloc N23Q4 batch8 weights");
    try {
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_input), quantized_input.size()), "hipMalloc N23Q4 batch8 input");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_scales), input_scales.size() * sizeof(float)), "hipMalloc N23Q4 batch8 scales");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_sums), input_sums.size() * sizeof(int)), "hipMalloc N23Q4 batch8 sums");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_output), output.size() * sizeof(float)), "hipMalloc N23Q4 batch8 output");
        hip_check(hipMemcpy(device_weights, packed_weights.data(), packed_weights.size() * sizeof(N23Q4TileGroup), hipMemcpyHostToDevice), "hipMemcpy N23Q4 batch8 weights");
        hip_check(hipMemcpy(device_input, quantized_input.data(), quantized_input.size(), hipMemcpyHostToDevice), "hipMemcpy N23Q4 batch8 input");
        hip_check(hipMemcpy(device_scales, input_scales.data(), input_scales.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy N23Q4 batch8 scales");
        hip_check(hipMemcpy(device_sums, input_sums.data(), input_sums.size() * sizeof(int), hipMemcpyHostToDevice), "hipMemcpy N23Q4 batch8 sums");
        hip_check(hipEventCreate(&begin), "hipEventCreate N23Q4 batch8 begin");
        hip_check(hipEventCreate(&end), "hipEventCreate N23Q4 batch8 end");

        const dim3 grid(static_cast<unsigned int>(tiles));
        const dim3 block(kWarpSize);
        for (int i = 0; i < 5; ++i) {
            hipLaunchKernelGGL(n23q4_batch8_kernel, grid, block, 0, nullptr,
                               device_weights, device_input, device_scales, device_sums,
                               device_output, static_cast<int>(input_size),
                               static_cast<int>(groups_per_row), static_cast<int>(output_size));
        }
        hip_check(hipDeviceSynchronize(), "N23Q4 batch8 warmup");
        hip_check(hipEventRecord(begin), "hipEventRecord N23Q4 batch8 begin");
        for (int i = 0; i < iterations; ++i) {
            hipLaunchKernelGGL(n23q4_batch8_kernel, grid, block, 0, nullptr,
                               device_weights, device_input, device_scales, device_sums,
                               device_output, static_cast<int>(input_size),
                               static_cast<int>(groups_per_row), static_cast<int>(output_size));
        }
        hip_check(hipEventRecord(end), "hipEventRecord N23Q4 batch8 end");
        hip_check(hipEventSynchronize(end), "hipEventSynchronize N23Q4 batch8");
        float elapsed = 0.0F;
        hip_check(hipEventElapsedTime(&elapsed, begin, end), "hipEventElapsedTime N23Q4 batch8");
        hip_check(hipMemcpy(output.data(), device_output, output.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy N23Q4 batch8 output");

        Q4KBenchmarkResult result;
        result.milliseconds = elapsed / static_cast<float>(iterations);
        result.bandwidth_gbps = static_cast<float>(packed_weights.size() * sizeof(N23Q4TileGroup))
                              / (result.milliseconds * 1.0e6F);
        const auto blocks_per_row = input_size / kQuantBlock;
        const std::size_t rows_to_check = std::min<std::size_t>(output_size, 8);
        double squared_error = 0.0;
        double squared_reference = 0.0;
        for (std::size_t batch = 0; batch < batch_size; ++batch) {
            for (std::size_t row = 0; row < rows_to_check; ++row) {
                const float expected = q4k_reference_row(q4k_weights + row * blocks_per_row,
                                                         input.data() + batch * input_size,
                                                         blocks_per_row);
                const float actual = output[batch * output_size + row];
                const float difference = expected - actual;
                result.max_absolute_error = std::max(result.max_absolute_error, std::abs(difference));
                squared_error += static_cast<double>(difference) * difference;
                squared_reference += static_cast<double>(expected) * expected;
            }
        }
        result.relative_rmse = static_cast<float>(std::sqrt(squared_error / squared_reference));
        hip_check(hipEventDestroy(end), "hipEventDestroy N23Q4 batch8 end");
        hip_check(hipEventDestroy(begin), "hipEventDestroy N23Q4 batch8 begin");
        hip_check(hipFree(device_output), "hipFree N23Q4 batch8 output");
        hip_check(hipFree(device_sums), "hipFree N23Q4 batch8 sums");
        hip_check(hipFree(device_scales), "hipFree N23Q4 batch8 scales");
        hip_check(hipFree(device_input), "hipFree N23Q4 batch8 input");
        hip_check(hipFree(device_weights), "hipFree N23Q4 batch8 weights");
        return result;
    } catch (...) {
        if (end) (void) hipEventDestroy(end);
        if (begin) (void) hipEventDestroy(begin);
        if (device_output) (void) hipFree(device_output);
        if (device_sums) (void) hipFree(device_sums);
        if (device_scales) (void) hipFree(device_scales);
        if (device_input) (void) hipFree(device_input);
        (void) hipFree(device_weights);
        throw;
    }
}

} // namespace qwen_native

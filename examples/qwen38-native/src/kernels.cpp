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
#include <type_traits>
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

// The target uses all three K-quants in its hot path.  Keeping their native
// GGUF packing here means a verifier can stream each tensor directly from the
// mapped file; it never has to materialise FP16 weights in host or device RAM.
struct Q5KBlock {
    std::uint16_t d;
    std::uint16_t dmin;
    std::uint8_t scales[12];
    std::uint8_t high_bits[32];
    std::uint8_t quants[128];
};
static_assert(sizeof(Q5KBlock) == 176);

struct Q6KBlock {
    std::uint8_t low_quants[128];
    std::uint8_t high_quants[64];
    std::int8_t scales[16];
    std::uint16_t d;
};
static_assert(sizeof(Q6KBlock) == 210);

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

struct N23KV4Group {
    std::uint16_t scale;
    std::uint8_t quants[16];
};
static_assert(sizeof(N23KV4Group) == 18);

void hip_check(hipError_t status, const char *operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

// The semantic executor streams one quantized matrix at a time because the
// target weights do not fit beside a 128K cache.  Keep the staging allocations
// resident across projections: hipMalloc/hipFree for every projection was
// dominating the otherwise useful first-token correctness pass.
struct MatvecWorkspace {
    void *weights = nullptr;
    std::size_t weight_capacity = 0;
    float *input = nullptr;
    std::size_t input_capacity = 0;
    float *output = nullptr;
    std::size_t output_capacity = 0;

    ~MatvecWorkspace() {
        if (output) (void) hipFree(output);
        if (input) (void) hipFree(input);
        if (weights) (void) hipFree(weights);
    }

    void reserve_weights(std::size_t bytes) {
        if (bytes <= weight_capacity) {
            return;
        }
        if (weights) {
            hip_check(hipFree(weights), "hipFree native matvec weight staging");
            weights = nullptr;
            weight_capacity = 0;
        }
        hip_check(hipMalloc(&weights, bytes), "hipMalloc native matvec weight staging");
        weight_capacity = bytes;
    }

    void reserve_input(std::size_t values) {
        if (values <= input_capacity) {
            return;
        }
        if (input) {
            hip_check(hipFree(input), "hipFree native matvec input staging");
            input = nullptr;
            input_capacity = 0;
        }
        hip_check(hipMalloc(reinterpret_cast<void **>(&input), values * sizeof(float)), "hipMalloc native matvec input staging");
        input_capacity = values;
    }

    void reserve_output(std::size_t values) {
        if (values <= output_capacity) {
            return;
        }
        if (output) {
            hip_check(hipFree(output), "hipFree native matvec output staging");
            output = nullptr;
            output_capacity = 0;
        }
        hip_check(hipMalloc(reinterpret_cast<void **>(&output), values * sizeof(float)), "hipMalloc native matvec output staging");
        output_capacity = values;
    }
};

MatvecWorkspace &matvec_workspace() {
    static MatvecWorkspace workspace;
    return workspace;
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

__global__ void q5k_matvec_kernel(const Q5KBlock *__restrict__ weights,
                                  const float *__restrict__ input,
                                  float *__restrict__ output,
                                  int blocks_per_row, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kWarpSize) {
        return;
    }

    float sum = 0.0F;
    const Q5KBlock *row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const Q5KBlock &block = row_weights[block_index];
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
            const std::uint8_t high = block.high_bits[lane];
            const int base = group * 64 + lane;
            const float code0 = static_cast<float>((quant & 0x0F) + ((high & (1U << (group * 2))) ? 16 : 0));
            const float code1 = static_cast<float>((quant >> 4) + ((high & (2U << (group * 2))) ? 16 : 0));
            const float weight0 = scale_all * static_cast<float>(scale0) * code0
                                - min_all * static_cast<float>(min0);
            const float weight1 = scale_all * static_cast<float>(scale1) * code1
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

__global__ void q5k_matvec_batch8_kernel(const Q5KBlock *__restrict__ weights,
                                         const float *__restrict__ input,
                                         float *__restrict__ output,
                                         int input_size, int blocks_per_row, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kWarpSize) {
        return;
    }

    float sums[8] = {};
    const Q5KBlock *row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const Q5KBlock &block = row_weights[block_index];
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
            const std::uint8_t high = block.high_bits[lane];
            const int index0 = block_index * kQuantBlock + group * 64 + lane;
            const int index1 = index0 + 32;
            const float code0 = static_cast<float>((quant & 0x0F) + ((high & (1U << (group * 2))) ? 16 : 0));
            const float code1 = static_cast<float>((quant >> 4) + ((high & (2U << (group * 2))) ? 16 : 0));
            const float weight0 = scale_all * static_cast<float>(scale0) * code0
                                - min_all * static_cast<float>(min0);
            const float weight1 = scale_all * static_cast<float>(scale1) * code1
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

__global__ void q6k_matvec_kernel(const Q6KBlock *__restrict__ weights,
                                  const float *__restrict__ input,
                                  float *__restrict__ output,
                                  int blocks_per_row, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kWarpSize) {
        return;
    }

    float sum = 0.0F;
    const Q6KBlock *row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const Q6KBlock &block = row_weights[block_index];
        const float scale_all = __half2float(*reinterpret_cast<const __half *>(&block.d));
        const float *block_input = input + static_cast<std::size_t>(block_index) * kQuantBlock;

#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int offset = half * 64;
            const std::uint8_t low0 = block.low_quants[offset + lane];
            const std::uint8_t low1 = block.low_quants[offset + lane + 32];
            const std::uint8_t high = block.high_quants[half * 32 + lane];
            const int scale_offset = half * 8;
            const int code0 = static_cast<int>(low0 & 0x0F) + ((high & 0x03) << 4) - 32;
            const int code1 = static_cast<int>(low1 & 0x0F) + (((high >> 2) & 0x03) << 4) - 32;
            const int code2 = static_cast<int>(low0 >> 4) + (((high >> 4) & 0x03) << 4) - 32;
            const int code3 = static_cast<int>(low1 >> 4) + (((high >> 6) & 0x03) << 4) - 32;
            sum = fmaf(scale_all * static_cast<float>(block.scales[scale_offset]) * static_cast<float>(code0),
                       block_input[offset + lane], sum);
            sum = fmaf(scale_all * static_cast<float>(block.scales[scale_offset + 2]) * static_cast<float>(code1),
                       block_input[offset + lane + 32], sum);
            sum = fmaf(scale_all * static_cast<float>(block.scales[scale_offset + 4]) * static_cast<float>(code2),
                       block_input[offset + lane + 64], sum);
            sum = fmaf(scale_all * static_cast<float>(block.scales[scale_offset + 6]) * static_cast<float>(code3),
                       block_input[offset + lane + 96], sum);
        }
    }

#pragma unroll
    for (int reduction = kWarpSize / 2; reduction > 0; reduction /= 2) {
        sum += __shfl_down(sum, reduction, kWarpSize);
    }
    if (lane == 0) {
        output[row] = sum;
    }
}

__global__ void q6k_matvec_batch8_kernel(const Q6KBlock *__restrict__ weights,
                                         const float *__restrict__ input,
                                         float *__restrict__ output,
                                         int input_size, int blocks_per_row, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kWarpSize) {
        return;
    }

    float sums[8] = {};
    const Q6KBlock *row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const Q6KBlock &block = row_weights[block_index];
        const float scale_all = __half2float(*reinterpret_cast<const __half *>(&block.d));

#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int offset = half * 64;
            const std::uint8_t low0 = block.low_quants[offset + lane];
            const std::uint8_t low1 = block.low_quants[offset + lane + 32];
            const std::uint8_t high = block.high_quants[half * 32 + lane];
            const int scale_offset = half * 8;
            const float weight0 = scale_all * static_cast<float>(block.scales[scale_offset])
                                * static_cast<float>(static_cast<int>(low0 & 0x0F) + ((high & 0x03) << 4) - 32);
            const float weight1 = scale_all * static_cast<float>(block.scales[scale_offset + 2])
                                * static_cast<float>(static_cast<int>(low1 & 0x0F) + (((high >> 2) & 0x03) << 4) - 32);
            const float weight2 = scale_all * static_cast<float>(block.scales[scale_offset + 4])
                                * static_cast<float>(static_cast<int>(low0 >> 4) + (((high >> 4) & 0x03) << 4) - 32);
            const float weight3 = scale_all * static_cast<float>(block.scales[scale_offset + 6])
                                * static_cast<float>(static_cast<int>(low1 >> 4) + (((high >> 6) & 0x03) << 4) - 32);
#pragma unroll
            for (int batch = 0; batch < 8; ++batch) {
                const float *batch_input = input + static_cast<std::size_t>(batch) * input_size;
                sums[batch] = fmaf(weight0, batch_input[block_index * kQuantBlock + offset + lane], sums[batch]);
                sums[batch] = fmaf(weight1, batch_input[block_index * kQuantBlock + offset + lane + 32], sums[batch]);
                sums[batch] = fmaf(weight2, batch_input[block_index * kQuantBlock + offset + lane + 64], sums[batch]);
                sums[batch] = fmaf(weight3, batch_input[block_index * kQuantBlock + offset + lane + 96], sums[batch]);
            }
        }
    }

#pragma unroll
    for (int reduction = kWarpSize / 2; reduction > 0; reduction /= 2) {
#pragma unroll
        for (int batch = 0; batch < 8; ++batch) {
            sums[batch] += __shfl_down(sums[batch], reduction, kWarpSize);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int batch = 0; batch < 8; ++batch) {
            output[static_cast<std::size_t>(batch) * rows + row] = sums[batch];
        }
    }
}

__global__ void silu_multiply_kernel(const float *__restrict__ gate,
                                     const float *__restrict__ up,
                                     float *__restrict__ output, int elements) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                    + static_cast<int>(threadIdx.x);
    if (index < elements) {
        const float value = gate[index];
        output[index] = (value / (1.0F + expf(-value))) * up[index];
    }
}

// One wave quantizes a contiguous 32-element activation group.  N23Q4 keeps
// the resulting scale and signed sum so asymmetric Q4_K rows can be evaluated
// with RDNA2's four-way int8 dot-product instruction.
__global__ void quantize_q8_groups_kernel(const float *__restrict__ input,
                                          std::int8_t *__restrict__ quantized,
                                          float *__restrict__ scales,
                                          int *__restrict__ sums,
                                          int groups_per_input) {
    const int lane = static_cast<int>(threadIdx.x);
    const int group = static_cast<int>(blockIdx.x);
    const int input_group = group % groups_per_input;
    const int batch = group / groups_per_input;
    const std::size_t index = static_cast<std::size_t>(batch) * groups_per_input * 32
                            + static_cast<std::size_t>(input_group) * 32 + lane;
    const float value = input[index];
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        maximum = fmaxf(maximum, __shfl_down(maximum, offset, kWarpSize));
    }
    __shared__ float group_scale;
    if (lane == 0) {
        group_scale = maximum / 127.0F;
        scales[group] = group_scale;
    }
    __syncthreads();
    const float inverse = group_scale == 0.0F ? 0.0F : 1.0F / group_scale;
    const int quant = max(-127, min(127, static_cast<int>(nearbyintf(value * inverse))));
    quantized[index] = static_cast<std::int8_t>(quant);
    int sum = quant;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, kWarpSize);
    }
    if (lane == 0) {
        sums[group] = sum;
    }
}

// Autoregressive gated-delta recurrence for one target recurrent layer.  A
// block owns one 128x128 value-head state, updates it in FP16, and produces
// the 128-element state@q result in the same launch.  The target has 48 value
// heads, so this avoids a full-state materialisation between the delta update
// and the output projection.
__global__ void gated_delta_step_kernel(const float *__restrict__ queries,
                                        const float *__restrict__ keys,
                                        const float *__restrict__ values,
                                        const float *__restrict__ decays,
                                        const float *__restrict__ betas,
                                        __half *__restrict__ states,
                                        float *__restrict__ output,
                                        int head_size) {
    const int head = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    const std::size_t vector_offset = static_cast<std::size_t>(head) * head_size;
    const std::size_t state_offset = vector_offset * head_size;
    __shared__ float state_key[128];
    __shared__ float delta[128];
    if (thread < head_size) {
        float sum = 0.0F;
        for (int column = 0; column < head_size; ++column) {
            sum = fmaf(__half2float(states[state_offset + static_cast<std::size_t>(thread) * head_size + column]),
                       keys[vector_offset + column], sum);
        }
        state_key[thread] = sum;
        delta[thread] = (values[vector_offset + thread] - sum) * betas[head];
    }
    __syncthreads();

    const float decay = decays[head];
    for (int element = thread; element < head_size * head_size; element += static_cast<int>(blockDim.x)) {
        const int row = element / head_size;
        const int column = element % head_size;
        const float updated = decay * __half2float(states[state_offset + element])
                            + delta[row] * keys[vector_offset + column];
        states[state_offset + element] = __float2half(updated);
    }
    __syncthreads();

    if (thread < head_size) {
        float sum = 0.0F;
        for (int column = 0; column < head_size; ++column) {
            sum = fmaf(__half2float(states[state_offset + static_cast<std::size_t>(thread) * head_size + column]),
                       queries[vector_offset + column], sum);
        }
        output[vector_offset + thread] = sum;
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

__global__ void n23kv4_score_kernel(const N23KV4Group *__restrict__ keys,
                                    const std::int8_t *__restrict__ queries,
                                    const float *__restrict__ query_scales,
                                    float *__restrict__ scores,
                                    int context_size) {
    constexpr int warps_per_block = 4;
    constexpr int groups_per_head = 8;
    constexpr int query_heads = 24;
    constexpr int kv_heads = 4;
    constexpr int queries_per_kv = query_heads / kv_heads;
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int token = static_cast<int>(blockIdx.x) * warps_per_block + warp;
    const int query_head = static_cast<int>(blockIdx.y);
    if (token >= context_size || query_head >= query_heads) {
        return;
    }
    const int kv_head = query_head / queries_per_kv;
    const int lane_group = lane / 8;
    const int lane_in_group = lane & 7;
    const auto *token_keys = keys
        + (static_cast<std::size_t>(kv_head) * context_size + token) * groups_per_head;
    float partial = 0.0F;

#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int group = half * 4 + lane_group;
        const auto &key_group = token_keys[group];
        const auto *nibbles = key_group.quants + 2 * lane_in_group;
        const std::uint8_t low = nibbles[0];
        const std::uint8_t high = nibbles[1];
        const int w0 = ((static_cast<int>(low & 0x0F) ^ 8) - 8);
        const int w1 = ((static_cast<int>(low >> 4) ^ 8) - 8);
        const int w2 = ((static_cast<int>(high & 0x0F) ^ 8) - 8);
        const int w3 = ((static_cast<int>(high >> 4) ^ 8) - 8);
        const std::uint32_t key_word = static_cast<std::uint8_t>(w0)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(w1)) << 8)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(w2)) << 16)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(w3)) << 24);
        const auto *query_word = reinterpret_cast<const std::uint32_t *>(
            queries + (static_cast<std::size_t>(query_head) * groups_per_head + group) * 32
            + lane_in_group * 4);
        const int dot = __builtin_amdgcn_sdot4(static_cast<int>(key_word),
                                                static_cast<int>(*query_word), 0, false);
        const float key_scale = __half2float(*reinterpret_cast<const __half *>(&key_group.scale));
        partial += static_cast<float>(dot) * key_scale
                 * query_scales[query_head * groups_per_head + group];
    }

#pragma unroll
    for (int offset = 4; offset > 0; offset /= 2) {
        const float other = __shfl_down(partial, offset, kWarpSize);
        if (lane_in_group + offset < 8) {
            partial += other;
        }
    }
    const float group0 = __shfl(partial, 0, kWarpSize);
    const float group1 = __shfl(partial, 8, kWarpSize);
    const float group2 = __shfl(partial, 16, kWarpSize);
    const float group3 = __shfl(partial, 24, kWarpSize);
    if (lane == 0) {
        scores[static_cast<std::size_t>(query_head) * context_size + token]
            = group0 + group1 + group2 + group3;
    }
}

__global__ void n23kv4_score_batch8_kernel(const N23KV4Group *__restrict__ keys,
                                           const std::int8_t *__restrict__ queries,
                                           const float *__restrict__ query_scales,
                                           float *__restrict__ scores,
                                           int context_size) {
    constexpr int warps_per_block = 4;
    constexpr int groups_per_head = 8;
    constexpr int query_heads = 24;
    constexpr int kv_heads = 4;
    constexpr int queries_per_kv = query_heads / kv_heads;
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int token = static_cast<int>(blockIdx.x) * warps_per_block + warp;
    const int query_head = static_cast<int>(blockIdx.y);
    if (token >= context_size || query_head >= query_heads) {
        return;
    }
    const int kv_head = query_head / queries_per_kv;
    const int lane_group = lane / 8;
    const int lane_in_group = lane & 7;
    const auto *token_keys = keys
        + (static_cast<std::size_t>(kv_head) * context_size + token) * groups_per_head;
    float partial[8] = {};

#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int group = half * 4 + lane_group;
        const auto &key_group = token_keys[group];
        const auto *nibbles = key_group.quants + 2 * lane_in_group;
        const std::uint8_t low = nibbles[0];
        const std::uint8_t high = nibbles[1];
        const int w0 = ((static_cast<int>(low & 0x0F) ^ 8) - 8);
        const int w1 = ((static_cast<int>(low >> 4) ^ 8) - 8);
        const int w2 = ((static_cast<int>(high & 0x0F) ^ 8) - 8);
        const int w3 = ((static_cast<int>(high >> 4) ^ 8) - 8);
        const std::uint32_t key_word = static_cast<std::uint8_t>(w0)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(w1)) << 8)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(w2)) << 16)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(w3)) << 24);
        const float key_scale = __half2float(*reinterpret_cast<const __half *>(&key_group.scale));
#pragma unroll
        for (int batch = 0; batch < 8; ++batch) {
            const std::size_t query_group = (static_cast<std::size_t>(batch) * query_heads
                                             + query_head) * groups_per_head + group;
            const auto *query_word = reinterpret_cast<const std::uint32_t *>(
                queries + query_group * 32 + lane_in_group * 4);
            const int dot = __builtin_amdgcn_sdot4(static_cast<int>(key_word),
                                                    static_cast<int>(*query_word), 0, false);
            partial[batch] += static_cast<float>(dot) * key_scale * query_scales[query_group];
        }
    }

#pragma unroll
    for (int offset = 4; offset > 0; offset /= 2) {
#pragma unroll
        for (int batch = 0; batch < 8; ++batch) {
            const float other = __shfl_down(partial[batch], offset, kWarpSize);
            if (lane_in_group + offset < 8) {
                partial[batch] += other;
            }
        }
    }
#pragma unroll
    for (int batch = 0; batch < 8; ++batch) {
        const float total = __shfl(partial[batch], 0, kWarpSize)
                          + __shfl(partial[batch], 8, kWarpSize)
                          + __shfl(partial[batch], 16, kWarpSize)
                          + __shfl(partial[batch], 24, kWarpSize);
        if (lane == 0) {
            scores[(static_cast<std::size_t>(batch) * query_heads + query_head)
                   * context_size + token] = total;
        }
    }
}

__global__ void n23kv4_flash_partial_kernel(const N23KV4Group *__restrict__ keys,
                                            const N23KV4Group *__restrict__ values,
                                            const std::int8_t *__restrict__ queries,
                                            const float *__restrict__ query_scales,
                                            float *__restrict__ partials,
                                            int context_size, int partitions) {
    constexpr int groups_per_head = 8;
    constexpr int query_heads = 24;
    constexpr int query_groups_per_kv = 6;
    constexpr int partial_stride = 258;
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / kWarpSize;
    const int lane = thread & (kWarpSize - 1);
    const int query_in_group = warp;
    const int kv_head = static_cast<int>(blockIdx.y) / query_groups_per_kv;
    const int local_query_group = static_cast<int>(blockIdx.y) % query_groups_per_kv;
    const int local_query = local_query_group * 8 + query_in_group;
    const int batch = local_query / 6;
    const int query_head = kv_head * 6 + local_query % 6;
    const int partition = static_cast<int>(blockIdx.x);
    const int chunk = (context_size + partitions - 1) / partitions;
    const int begin = partition * chunk;
    const int end = min(context_size, begin + chunk);
    const int lane_group = lane / 8;
    const int lane_in_group = lane & 7;

    __shared__ float tile_scores[8];
    __shared__ float state_max[8];
    __shared__ float state_sum[8];
    __shared__ float rescale[8];
    __shared__ float probabilities[8];
    if (thread < 8) {
        state_max[thread] = -INFINITY;
        state_sum[thread] = 0.0F;
    }
    __syncthreads();

    float accumulator[8] = {};
    for (int token = begin; token < end; ++token) {
        float score_partial = 0.0F;
        const auto *token_keys = keys
            + (static_cast<std::size_t>(kv_head) * context_size + token) * groups_per_head;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int group = half * 4 + lane_group;
            const auto &key_group = token_keys[group];
            const auto *nibbles = key_group.quants + 2 * lane_in_group;
            const std::uint8_t low = nibbles[0];
            const std::uint8_t high = nibbles[1];
            const int w0 = ((static_cast<int>(low & 0x0F) ^ 8) - 8);
            const int w1 = ((static_cast<int>(low >> 4) ^ 8) - 8);
            const int w2 = ((static_cast<int>(high & 0x0F) ^ 8) - 8);
            const int w3 = ((static_cast<int>(high >> 4) ^ 8) - 8);
            const std::uint32_t key_word = static_cast<std::uint8_t>(w0)
                | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(w1)) << 8)
                | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(w2)) << 16)
                | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(w3)) << 24);
            const std::size_t query_group = (static_cast<std::size_t>(batch) * query_heads
                                             + query_head) * groups_per_head + group;
            const auto *query_word = reinterpret_cast<const std::uint32_t *>(
                queries + query_group * 32 + lane_in_group * 4);
            const int dot = __builtin_amdgcn_sdot4(static_cast<int>(key_word),
                                                    static_cast<int>(*query_word), 0, false);
            const float key_scale = __half2float(
                *reinterpret_cast<const __half *>(&key_group.scale));
            score_partial += static_cast<float>(dot) * key_scale
                           * query_scales[query_group] * 0.0625F;
        }
#pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            const float other = __shfl_down(score_partial, offset, kWarpSize);
            if (lane_in_group + offset < 8) {
                score_partial += other;
            }
        }
        const float score = __shfl(score_partial, 0, kWarpSize)
                          + __shfl(score_partial, 8, kWarpSize)
                          + __shfl(score_partial, 16, kWarpSize)
                          + __shfl(score_partial, 24, kWarpSize);
        if (lane == 0) {
            tile_scores[query_in_group] = score;
        }
        __syncthreads();

        if (thread < 8) {
            const float old_max = state_max[thread];
            const float new_max = fmaxf(old_max, tile_scores[thread]);
            const float scale_old = isinf(old_max) ? 0.0F : expf(old_max - new_max);
            const float probability = expf(tile_scores[thread] - new_max);
            rescale[thread] = scale_old;
            probabilities[thread] = probability;
            state_sum[thread] = state_sum[thread] * scale_old + probability;
            state_max[thread] = new_max;
        }
        __syncthreads();

        const int value_group_index = thread / 32;
        const int value_index = thread & 31;
        const auto &value_group = values[(static_cast<std::size_t>(kv_head) * context_size + token)
                                         * groups_per_head + value_group_index];
        const auto value_byte = value_group.quants[value_index / 2];
        const int value_nibble = (value_byte >> ((value_index & 1) * 4)) & 0x0F;
        const float value = static_cast<float>(value_nibble >= 8 ? value_nibble - 16 : value_nibble)
                          * __half2float(*reinterpret_cast<const __half *>(&value_group.scale));
#pragma unroll
        for (int query = 0; query < 8; ++query) {
            accumulator[query] = accumulator[query] * rescale[query]
                + probabilities[query] * value;
        }
        __syncthreads();
    }

#pragma unroll
    for (int query = 0; query < 8; ++query) {
        const int local = local_query_group * 8 + query;
        const int query_batch = local / 6;
        const int query_head_out = kv_head * 6 + local % 6;
        const int global_query_out = query_batch * query_heads + query_head_out;
        float *destination = partials
            + (static_cast<std::size_t>(global_query_out) * partitions + partition) * partial_stride;
        destination[thread] = accumulator[query];
        if (thread == 0) {
            destination[256] = state_max[query];
            destination[257] = state_sum[query];
        }
    }
}

__global__ void n23kv4_flash_reduce_kernel(const float *__restrict__ partials,
                                           float *__restrict__ output,
                                           int partitions) {
    constexpr int partial_stride = 258;
    const int query = static_cast<int>(blockIdx.x);
    const int dimension = static_cast<int>(threadIdx.x);
    __shared__ float global_max;
    __shared__ float global_sum;
    if (dimension == 0) {
        float maximum = -INFINITY;
        for (int partition = 0; partition < partitions; ++partition) {
            maximum = fmaxf(maximum,
                partials[(static_cast<std::size_t>(query) * partitions + partition)
                         * partial_stride + 256]);
        }
        float sum = 0.0F;
        for (int partition = 0; partition < partitions; ++partition) {
            const auto *source = partials
                + (static_cast<std::size_t>(query) * partitions + partition) * partial_stride;
            sum += source[257] * expf(source[256] - maximum);
        }
        global_max = maximum;
        global_sum = sum;
    }
    __syncthreads();
    float value = 0.0F;
    for (int partition = 0; partition < partitions; ++partition) {
        const auto *source = partials
            + (static_cast<std::size_t>(query) * partitions + partition) * partial_stride;
        value += source[dimension] * expf(source[256] - global_max);
    }
    output[static_cast<std::size_t>(query) * 256 + dimension] = value / global_sum;
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

float q5k_reference_row(const Q5KBlock *weights, const float *input, std::size_t blocks) {
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
                const std::uint8_t quant = block.quants[group * kWarpSize + lane];
                const std::uint8_t high = block.high_bits[lane];
                const std::size_t base = block_index * kQuantBlock + group * 64 + lane;
                const float code0 = static_cast<float>((quant & 0x0F) + ((high & (1U << (group * 2))) ? 16 : 0));
                const float code1 = static_cast<float>((quant >> 4) + ((high & (2U << (group * 2))) ? 16 : 0));
                const float weight0 = scale_all * static_cast<float>(scale0) * code0
                                    - min_all * static_cast<float>(min0);
                const float weight1 = scale_all * static_cast<float>(scale1) * code1
                                    - min_all * static_cast<float>(min1);
                sum = std::fma(weight0, input[base], sum);
                sum = std::fma(weight1, input[base + 32], sum);
            }
        }
    }
    return sum;
}

float q6k_reference_row(const Q6KBlock *weights, const float *input, std::size_t blocks) {
    float sum = 0.0F;
    for (std::size_t block_index = 0; block_index < blocks; ++block_index) {
        const auto &block = weights[block_index];
        const float scale_all = fp16_to_float(block.d);
        for (int half = 0; half < 2; ++half) {
            const int offset = half * 64;
            const int scale_offset = half * 8;
            for (int lane = 0; lane < kWarpSize; ++lane) {
                const std::uint8_t low0 = block.low_quants[offset + lane];
                const std::uint8_t low1 = block.low_quants[offset + lane + 32];
                const std::uint8_t high = block.high_quants[half * 32 + lane];
                const int code0 = static_cast<int>(low0 & 0x0F) + ((high & 0x03) << 4) - 32;
                const int code1 = static_cast<int>(low1 & 0x0F) + (((high >> 2) & 0x03) << 4) - 32;
                const int code2 = static_cast<int>(low0 >> 4) + (((high >> 4) & 0x03) << 4) - 32;
                const int code3 = static_cast<int>(low1 >> 4) + (((high >> 6) & 0x03) << 4) - 32;
                const std::size_t base = block_index * kQuantBlock + offset + lane;
                sum = std::fma(scale_all * static_cast<float>(block.scales[scale_offset]) * static_cast<float>(code0), input[base], sum);
                sum = std::fma(scale_all * static_cast<float>(block.scales[scale_offset + 2]) * static_cast<float>(code1), input[base + 32], sum);
                sum = std::fma(scale_all * static_cast<float>(block.scales[scale_offset + 4]) * static_cast<float>(code2), input[base + 64], sum);
                sum = std::fma(scale_all * static_cast<float>(block.scales[scale_offset + 6]) * static_cast<float>(code3), input[base + 96], sum);
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

std::size_t k_quant_weight_bytes(std::uint32_t type, std::size_t input_size,
                                 std::size_t output_size) {
    if (input_size == 0 || input_size % kQuantBlock != 0 || output_size == 0) {
        throw std::invalid_argument("K-quant matrix dimensions must be non-zero multiples of 256");
    }
    const std::size_t blocks = input_size / kQuantBlock;
    switch (type) {
    case 12: return blocks * output_size * sizeof(Q4KBlock);
    case 13: return blocks * output_size * sizeof(Q5KBlock);
    case 14: return blocks * output_size * sizeof(Q6KBlock);
    default: throw std::invalid_argument("matrix is not a supported K-quant type");
    }
}

void launch_k_matvec_batch8(std::uint32_t type, const void *weights, const float *input,
                            float *output, std::size_t input_size, std::size_t output_size) {
    const dim3 grid(static_cast<unsigned int>(output_size));
    const dim3 block(kWarpSize);
    const int blocks_per_row = static_cast<int>(input_size / kQuantBlock);
    switch (type) {
    case 12:
        hipLaunchKernelGGL(q4k_matvec_batch8_kernel, grid, block, 0, nullptr,
                           static_cast<const Q4KBlock *>(weights), input, output,
                           static_cast<int>(input_size), blocks_per_row,
                           static_cast<int>(output_size));
        return;
    case 13:
        hipLaunchKernelGGL(q5k_matvec_batch8_kernel, grid, block, 0, nullptr,
                           static_cast<const Q5KBlock *>(weights), input, output,
                           static_cast<int>(input_size), blocks_per_row,
                           static_cast<int>(output_size));
        return;
    case 14:
        hipLaunchKernelGGL(q6k_matvec_batch8_kernel, grid, block, 0, nullptr,
                           static_cast<const Q6KBlock *>(weights), input, output,
                           static_cast<int>(input_size), blocks_per_row,
                           static_cast<int>(output_size));
        return;
    default:
        throw std::invalid_argument("matrix is not a supported K-quant type");
    }
}

void launch_k_matvec(std::uint32_t type, const void *weights, const float *input,
                     float *output, std::size_t input_size, std::size_t output_size) {
    const dim3 grid(static_cast<unsigned int>(output_size));
    const dim3 block(kWarpSize);
    const int blocks_per_row = static_cast<int>(input_size / kQuantBlock);
    switch (type) {
    case 12:
        hipLaunchKernelGGL(q4k_matvec_kernel, grid, block, 0, nullptr,
                           static_cast<const Q4KBlock *>(weights), input, output,
                           blocks_per_row, static_cast<int>(output_size));
        return;
    case 13:
        hipLaunchKernelGGL(q5k_matvec_kernel, grid, block, 0, nullptr,
                           static_cast<const Q5KBlock *>(weights), input, output,
                           blocks_per_row, static_cast<int>(output_size));
        return;
    case 14:
        hipLaunchKernelGGL(q6k_matvec_kernel, grid, block, 0, nullptr,
                           static_cast<const Q6KBlock *>(weights), input, output,
                           blocks_per_row, static_cast<int>(output_size));
        return;
    default:
        throw std::invalid_argument("matrix is not a supported K-quant type");
    }
}

void launch_n23q4_matvec_batch8(const N23Q4TileGroup *weights,
                                const std::int8_t *input, const float *input_scales,
                                const int *input_sums, float *output,
                                std::size_t input_size, std::size_t output_size) {
    const dim3 grid(static_cast<unsigned int>((output_size + 3) / 4));
    const dim3 block(kWarpSize);
    hipLaunchKernelGGL(n23q4_batch8_kernel, grid, block, 0, nullptr,
                       weights, input, input_scales, input_sums, output,
                       static_cast<int>(input_size), static_cast<int>(input_size / 32),
                       static_cast<int>(output_size));
}

float k_quant_reference_row(std::uint32_t type, const void *weights, const float *input,
                            std::size_t blocks_per_row) {
    switch (type) {
    case 12:
        return q4k_reference_row(static_cast<const Q4KBlock *>(weights), input, blocks_per_row);
    case 13:
        return q5k_reference_row(static_cast<const Q5KBlock *>(weights), input, blocks_per_row);
    case 14:
        return q6k_reference_row(static_cast<const Q6KBlock *>(weights), input, blocks_per_row);
    default:
        throw std::invalid_argument("matrix is not a supported K-quant type");
    }
}

template <typename Block, typename Reference>
Q4KBenchmarkResult benchmark_k_matvec(const void *weights, std::size_t input_size,
                                      std::size_t output_size, int iterations,
                                      bool batch_eight, Reference reference) {
    if (input_size == 0 || input_size % kQuantBlock != 0 || output_size == 0 || iterations <= 0) {
        throw std::invalid_argument("invalid K-quant benchmark dimensions or iteration count");
    }
    constexpr std::size_t batch_size = 8;
    const std::size_t batch = batch_eight ? batch_size : 1;
    const std::size_t blocks_per_row = input_size / kQuantBlock;
    const std::size_t weight_bytes = blocks_per_row * output_size * sizeof(Block);
    std::vector<float> input(input_size * batch);
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::generate(input.begin(), input.end(), [&] { return distribution(generator); });
    std::vector<float> output(output_size * batch);

    Block *device_weights = nullptr;
    float *device_input = nullptr;
    float *device_output = nullptr;
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    hip_check(hipMalloc(reinterpret_cast<void **>(&device_weights), weight_bytes), "hipMalloc K-quant weights");
    try {
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_input), input.size() * sizeof(float)), "hipMalloc K-quant input");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_output), output.size() * sizeof(float)), "hipMalloc K-quant output");
        hip_check(hipMemcpy(device_weights, weights, weight_bytes, hipMemcpyHostToDevice), "hipMemcpy K-quant weights");
        hip_check(hipMemcpy(device_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy K-quant input");
        hip_check(hipEventCreate(&begin), "hipEventCreate K-quant begin");
        hip_check(hipEventCreate(&end), "hipEventCreate K-quant end");

        const dim3 grid(static_cast<unsigned int>(output_size));
        const dim3 block(kWarpSize);
        const auto launch = [&] {
            if constexpr (std::is_same_v<Block, Q5KBlock>) {
                if (batch_eight) {
                    hipLaunchKernelGGL(q5k_matvec_batch8_kernel, grid, block, 0, nullptr,
                                       device_weights, device_input, device_output,
                                       static_cast<int>(input_size), static_cast<int>(blocks_per_row),
                                       static_cast<int>(output_size));
                } else {
                    hipLaunchKernelGGL(q5k_matvec_kernel, grid, block, 0, nullptr,
                                       device_weights, device_input, device_output,
                                       static_cast<int>(blocks_per_row), static_cast<int>(output_size));
                }
            } else {
                static_assert(std::is_same_v<Block, Q6KBlock>);
                if (batch_eight) {
                    hipLaunchKernelGGL(q6k_matvec_batch8_kernel, grid, block, 0, nullptr,
                                       device_weights, device_input, device_output,
                                       static_cast<int>(input_size), static_cast<int>(blocks_per_row),
                                       static_cast<int>(output_size));
                } else {
                    hipLaunchKernelGGL(q6k_matvec_kernel, grid, block, 0, nullptr,
                                       device_weights, device_input, device_output,
                                       static_cast<int>(blocks_per_row), static_cast<int>(output_size));
                }
            }
        };
        for (int i = 0; i < 5; ++i) {
            launch();
        }
        hip_check(hipDeviceSynchronize(), "K-quant warmup");
        hip_check(hipEventRecord(begin), "hipEventRecord K-quant begin");
        for (int i = 0; i < iterations; ++i) {
            launch();
        }
        hip_check(hipEventRecord(end), "hipEventRecord K-quant end");
        hip_check(hipEventSynchronize(end), "hipEventSynchronize K-quant");
        float elapsed = 0.0F;
        hip_check(hipEventElapsedTime(&elapsed, begin, end), "hipEventElapsedTime K-quant");
        hip_check(hipMemcpy(output.data(), device_output, output.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy K-quant output");

        Q4KBenchmarkResult result;
        result.milliseconds = elapsed / static_cast<float>(iterations);
        result.bandwidth_gbps = static_cast<float>(weight_bytes) / (result.milliseconds * 1.0e6F);
        const auto *host_weights = static_cast<const Block *>(weights);
        const std::size_t rows_to_check = std::min<std::size_t>(output_size, 8);
        for (std::size_t input_index = 0; input_index < batch; ++input_index) {
            for (std::size_t row = 0; row < rows_to_check; ++row) {
                const float expected = reference(host_weights + row * blocks_per_row,
                                                 input.data() + input_index * input_size,
                                                 blocks_per_row);
                result.max_absolute_error = std::max(
                    result.max_absolute_error,
                    std::abs(expected - output[input_index * output_size + row]));
            }
        }
        hip_check(hipEventDestroy(end), "hipEventDestroy K-quant end");
        hip_check(hipEventDestroy(begin), "hipEventDestroy K-quant begin");
        hip_check(hipFree(device_output), "hipFree K-quant output");
        hip_check(hipFree(device_input), "hipFree K-quant input");
        hip_check(hipFree(device_weights), "hipFree K-quant weights");
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

Q4KBenchmarkResult benchmark_q5k(const void *weights, std::size_t input_size,
                                 std::size_t output_size, int iterations) {
    return benchmark_k_matvec<Q5KBlock>(weights, input_size, output_size, iterations, false,
                                        q5k_reference_row);
}

Q4KBenchmarkResult benchmark_q5k_batch8(const void *weights, std::size_t input_size,
                                        std::size_t output_size, int iterations) {
    return benchmark_k_matvec<Q5KBlock>(weights, input_size, output_size, iterations, true,
                                        q5k_reference_row);
}

Q4KBenchmarkResult benchmark_q6k(const void *weights, std::size_t input_size,
                                 std::size_t output_size, int iterations) {
    return benchmark_k_matvec<Q6KBlock>(weights, input_size, output_size, iterations, false,
                                        q6k_reference_row);
}

Q4KBenchmarkResult benchmark_q6k_batch8(const void *weights, std::size_t input_size,
                                        std::size_t output_size, int iterations) {
    return benchmark_k_matvec<Q6KBlock>(weights, input_size, output_size, iterations, true,
                                        q6k_reference_row);
}

void execute_k_matvec(const void *weights, std::uint32_t weight_type,
                      const float *input, std::size_t input_size,
                      float *output, std::size_t output_size) {
    const std::size_t weight_bytes = k_quant_weight_bytes(weight_type, input_size, output_size);
    auto &workspace = matvec_workspace();
    workspace.reserve_weights(weight_bytes);
    workspace.reserve_input(input_size);
    workspace.reserve_output(output_size);
    hip_check(hipMemcpy(workspace.weights, weights, weight_bytes, hipMemcpyHostToDevice), "hipMemcpy native matvec weights");
    hip_check(hipMemcpy(workspace.input, input, input_size * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy native matvec input");
    launch_k_matvec(weight_type, workspace.weights, workspace.input, workspace.output, input_size, output_size);
    hip_check(hipGetLastError(), "native matvec launch");
    hip_check(hipMemcpy(output, workspace.output, output_size * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy native matvec output");
}

void execute_k_matvec_batch8(const void *weights, std::uint32_t weight_type,
                             const float *input, std::size_t input_size,
                             float *output, std::size_t output_size) {
    constexpr std::size_t batch_size = 8;
    const std::size_t weight_bytes = k_quant_weight_bytes(weight_type, input_size, output_size);
    auto &workspace = matvec_workspace();
    workspace.reserve_weights(weight_bytes);
    workspace.reserve_input(batch_size * input_size);
    workspace.reserve_output(batch_size * output_size);
    hip_check(hipMemcpy(workspace.weights, weights, weight_bytes, hipMemcpyHostToDevice), "hipMemcpy native batch8 matvec weights");
    hip_check(hipMemcpy(workspace.input, input, batch_size * input_size * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy native batch8 matvec input");
    launch_k_matvec_batch8(weight_type, workspace.weights, workspace.input, workspace.output,
                           input_size, output_size);
    hip_check(hipGetLastError(), "native batch8 matvec launch");
    hip_check(hipMemcpy(output, workspace.output, batch_size * output_size * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy native batch8 matvec output");
}

FfnBenchmarkResult benchmark_ffn_batch8(const void *gate_weights, std::uint32_t gate_type,
                                         const void *up_weights, std::uint32_t up_type,
                                         const void *down_weights, std::uint32_t down_type,
                                         std::size_t input_size, std::size_t hidden_size,
                                         std::size_t output_size, int iterations) {
    constexpr std::size_t batch_size = 8;
    if (input_size % kQuantBlock != 0 || hidden_size % kQuantBlock != 0 ||
        input_size == 0 || hidden_size == 0 || output_size == 0 || iterations <= 0) {
        throw std::invalid_argument("invalid FFN batch-8 dimensions or iteration count");
    }
    const std::size_t gate_bytes = k_quant_weight_bytes(gate_type, input_size, hidden_size);
    const std::size_t up_bytes = k_quant_weight_bytes(up_type, input_size, hidden_size);
    const std::size_t down_bytes = k_quant_weight_bytes(down_type, hidden_size, output_size);
    std::vector<float> input(batch_size * input_size);
    std::vector<float> hidden(batch_size * hidden_size);
    std::vector<float> output(batch_size * output_size);
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::generate(input.begin(), input.end(), [&] { return distribution(generator); });

    void *device_gate_weights = nullptr;
    void *device_up_weights = nullptr;
    void *device_down_weights = nullptr;
    float *device_input = nullptr;
    float *device_gate = nullptr;
    float *device_up = nullptr;
    float *device_hidden = nullptr;
    float *device_output = nullptr;
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    hip_check(hipMalloc(&device_gate_weights, gate_bytes), "hipMalloc FFN gate weights");
    try {
        hip_check(hipMalloc(&device_up_weights, up_bytes), "hipMalloc FFN up weights");
        hip_check(hipMalloc(&device_down_weights, down_bytes), "hipMalloc FFN down weights");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_input), input.size() * sizeof(float)), "hipMalloc FFN input");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_gate), hidden.size() * sizeof(float)), "hipMalloc FFN gate");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_up), hidden.size() * sizeof(float)), "hipMalloc FFN up");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_hidden), hidden.size() * sizeof(float)), "hipMalloc FFN hidden");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_output), output.size() * sizeof(float)), "hipMalloc FFN output");
        hip_check(hipMemcpy(device_gate_weights, gate_weights, gate_bytes, hipMemcpyHostToDevice), "hipMemcpy FFN gate weights");
        hip_check(hipMemcpy(device_up_weights, up_weights, up_bytes, hipMemcpyHostToDevice), "hipMemcpy FFN up weights");
        hip_check(hipMemcpy(device_down_weights, down_weights, down_bytes, hipMemcpyHostToDevice), "hipMemcpy FFN down weights");
        hip_check(hipMemcpy(device_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy FFN input");
        hip_check(hipEventCreate(&begin), "hipEventCreate FFN begin");
        hip_check(hipEventCreate(&end), "hipEventCreate FFN end");

        const dim3 pointwise_block(256);
        const dim3 pointwise_grid(static_cast<unsigned int>((hidden.size() + pointwise_block.x - 1) / pointwise_block.x));
        const auto launch = [&] {
            launch_k_matvec_batch8(gate_type, device_gate_weights, device_input, device_gate,
                                   input_size, hidden_size);
            launch_k_matvec_batch8(up_type, device_up_weights, device_input, device_up,
                                   input_size, hidden_size);
            hipLaunchKernelGGL(silu_multiply_kernel, pointwise_grid, pointwise_block, 0, nullptr,
                               device_gate, device_up, device_hidden, static_cast<int>(hidden.size()));
            launch_k_matvec_batch8(down_type, device_down_weights, device_hidden, device_output,
                                   hidden_size, output_size);
        };
        for (int i = 0; i < 3; ++i) {
            launch();
        }
        hip_check(hipDeviceSynchronize(), "FFN batch-8 warmup");
        hip_check(hipEventRecord(begin), "hipEventRecord FFN begin");
        for (int i = 0; i < iterations; ++i) {
            launch();
        }
        hip_check(hipEventRecord(end), "hipEventRecord FFN end");
        hip_check(hipEventSynchronize(end), "hipEventSynchronize FFN");
        float elapsed = 0.0F;
        hip_check(hipEventElapsedTime(&elapsed, begin, end), "hipEventElapsedTime FFN");
        hip_check(hipMemcpy(hidden.data(), device_hidden, hidden.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy FFN hidden");
        hip_check(hipMemcpy(output.data(), device_output, output.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy FFN output");

        FfnBenchmarkResult result;
        result.milliseconds = elapsed / static_cast<float>(iterations);
        result.weight_bandwidth_gbps = static_cast<float>(gate_bytes + up_bytes + down_bytes)
                                     / (result.milliseconds * 1.0e6F);
        result.verifier_tokens_per_second = static_cast<float>(batch_size) * 1000.0F / result.milliseconds;
        const std::size_t rows_to_check = std::min<std::size_t>(output_size, 4);
        const std::size_t down_blocks = hidden_size / kQuantBlock;
        const std::size_t down_row_bytes = down_bytes / output_size;
        for (std::size_t batch = 0; batch < batch_size; ++batch) {
            for (std::size_t row = 0; row < rows_to_check; ++row) {
                const auto *row_weights = static_cast<const char *>(down_weights) + row * down_row_bytes;
                const float expected = k_quant_reference_row(down_type, row_weights,
                                                             hidden.data() + batch * hidden_size,
                                                             down_blocks);
                result.max_absolute_error = std::max(
                    result.max_absolute_error,
                    std::abs(expected - output[batch * output_size + row]));
            }
        }

        hip_check(hipEventDestroy(end), "hipEventDestroy FFN end");
        hip_check(hipEventDestroy(begin), "hipEventDestroy FFN begin");
        hip_check(hipFree(device_output), "hipFree FFN output");
        hip_check(hipFree(device_hidden), "hipFree FFN hidden");
        hip_check(hipFree(device_up), "hipFree FFN up");
        hip_check(hipFree(device_gate), "hipFree FFN gate");
        hip_check(hipFree(device_input), "hipFree FFN input");
        hip_check(hipFree(device_down_weights), "hipFree FFN down weights");
        hip_check(hipFree(device_up_weights), "hipFree FFN up weights");
        hip_check(hipFree(device_gate_weights), "hipFree FFN gate weights");
        return result;
    } catch (...) {
        if (end) (void) hipEventDestroy(end);
        if (begin) (void) hipEventDestroy(begin);
        if (device_output) (void) hipFree(device_output);
        if (device_hidden) (void) hipFree(device_hidden);
        if (device_up) (void) hipFree(device_up);
        if (device_gate) (void) hipFree(device_gate);
        if (device_input) (void) hipFree(device_input);
        if (device_down_weights) (void) hipFree(device_down_weights);
        if (device_up_weights) (void) hipFree(device_up_weights);
        (void) hipFree(device_gate_weights);
        throw;
    }
}

FfnBenchmarkResult benchmark_ffn_batch8_n23q4(const void *gate_weights,
                                               const void *up_weights,
                                               const void *down_weights,
                                               std::uint32_t down_type,
                                               std::size_t input_size,
                                               std::size_t hidden_size,
                                               std::size_t output_size, int iterations) {
    constexpr std::size_t batch_size = 8;
    if (input_size % kQuantBlock != 0 || hidden_size % kQuantBlock != 0 ||
        input_size == 0 || hidden_size == 0 || output_size == 0 || iterations <= 0) {
        throw std::invalid_argument("invalid N23Q4 FFN batch-8 dimensions or iteration count");
    }
    const auto packed_gate = convert_to_n23q4(static_cast<const Q4KBlock *>(gate_weights), input_size, hidden_size);
    const auto packed_up = convert_to_n23q4(static_cast<const Q4KBlock *>(up_weights), input_size, hidden_size);
    const std::size_t down_bytes = k_quant_weight_bytes(down_type, hidden_size, output_size);
    std::vector<float> input(batch_size * input_size);
    std::vector<float> hidden(batch_size * hidden_size);
    std::vector<float> output(batch_size * output_size);
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::generate(input.begin(), input.end(), [&] { return distribution(generator); });

    N23Q4TileGroup *device_gate_weights = nullptr;
    N23Q4TileGroup *device_up_weights = nullptr;
    void *device_down_weights = nullptr;
    float *device_input = nullptr;
    std::int8_t *device_quantized_input = nullptr;
    float *device_input_scales = nullptr;
    int *device_input_sums = nullptr;
    float *device_gate = nullptr;
    float *device_up = nullptr;
    float *device_hidden = nullptr;
    float *device_output = nullptr;
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    const std::size_t quant_groups = batch_size * input_size / 32;
    hip_check(hipMalloc(reinterpret_cast<void **>(&device_gate_weights),
                        packed_gate.size() * sizeof(N23Q4TileGroup)), "hipMalloc N23Q4 FFN gate weights");
    try {
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_up_weights),
                            packed_up.size() * sizeof(N23Q4TileGroup)), "hipMalloc N23Q4 FFN up weights");
        hip_check(hipMalloc(&device_down_weights, down_bytes), "hipMalloc N23Q4 FFN down weights");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_input), input.size() * sizeof(float)), "hipMalloc N23Q4 FFN input");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_quantized_input), input.size()), "hipMalloc N23Q4 FFN input quant");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_input_scales), quant_groups * sizeof(float)), "hipMalloc N23Q4 FFN input scales");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_input_sums), quant_groups * sizeof(int)), "hipMalloc N23Q4 FFN input sums");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_gate), hidden.size() * sizeof(float)), "hipMalloc N23Q4 FFN gate");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_up), hidden.size() * sizeof(float)), "hipMalloc N23Q4 FFN up");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_hidden), hidden.size() * sizeof(float)), "hipMalloc N23Q4 FFN hidden");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_output), output.size() * sizeof(float)), "hipMalloc N23Q4 FFN output");
        hip_check(hipMemcpy(device_gate_weights, packed_gate.data(), packed_gate.size() * sizeof(N23Q4TileGroup), hipMemcpyHostToDevice), "hipMemcpy N23Q4 FFN gate weights");
        hip_check(hipMemcpy(device_up_weights, packed_up.data(), packed_up.size() * sizeof(N23Q4TileGroup), hipMemcpyHostToDevice), "hipMemcpy N23Q4 FFN up weights");
        hip_check(hipMemcpy(device_down_weights, down_weights, down_bytes, hipMemcpyHostToDevice), "hipMemcpy N23Q4 FFN down weights");
        hip_check(hipMemcpy(device_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy N23Q4 FFN input");
        hip_check(hipEventCreate(&begin), "hipEventCreate N23Q4 FFN begin");
        hip_check(hipEventCreate(&end), "hipEventCreate N23Q4 FFN end");

        const dim3 quant_grid(static_cast<unsigned int>(quant_groups));
        const dim3 quant_block(kWarpSize);
        const dim3 pointwise_block(256);
        const dim3 pointwise_grid(static_cast<unsigned int>((hidden.size() + pointwise_block.x - 1) / pointwise_block.x));
        const auto launch = [&] {
            hipLaunchKernelGGL(quantize_q8_groups_kernel, quant_grid, quant_block, 0, nullptr,
                               device_input, device_quantized_input, device_input_scales,
                               device_input_sums, static_cast<int>(input_size / 32));
            launch_n23q4_matvec_batch8(device_gate_weights, device_quantized_input,
                                       device_input_scales, device_input_sums, device_gate,
                                       input_size, hidden_size);
            launch_n23q4_matvec_batch8(device_up_weights, device_quantized_input,
                                       device_input_scales, device_input_sums, device_up,
                                       input_size, hidden_size);
            hipLaunchKernelGGL(silu_multiply_kernel, pointwise_grid, pointwise_block, 0, nullptr,
                               device_gate, device_up, device_hidden, static_cast<int>(hidden.size()));
            launch_k_matvec_batch8(down_type, device_down_weights, device_hidden, device_output,
                                   hidden_size, output_size);
        };
        for (int i = 0; i < 3; ++i) {
            launch();
        }
        hip_check(hipDeviceSynchronize(), "N23Q4 FFN batch-8 warmup");
        hip_check(hipEventRecord(begin), "hipEventRecord N23Q4 FFN begin");
        for (int i = 0; i < iterations; ++i) {
            launch();
        }
        hip_check(hipEventRecord(end), "hipEventRecord N23Q4 FFN end");
        hip_check(hipEventSynchronize(end), "hipEventSynchronize N23Q4 FFN");
        float elapsed = 0.0F;
        hip_check(hipEventElapsedTime(&elapsed, begin, end), "hipEventElapsedTime N23Q4 FFN");
        hip_check(hipMemcpy(hidden.data(), device_hidden, hidden.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy N23Q4 FFN hidden");
        hip_check(hipMemcpy(output.data(), device_output, output.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy N23Q4 FFN output");

        FfnBenchmarkResult result;
        result.milliseconds = elapsed / static_cast<float>(iterations);
        result.weight_bandwidth_gbps = static_cast<float>((packed_gate.size() + packed_up.size()) * sizeof(N23Q4TileGroup) + down_bytes)
                                     / (result.milliseconds * 1.0e6F);
        result.verifier_tokens_per_second = static_cast<float>(batch_size) * 1000.0F / result.milliseconds;
        const std::size_t rows_to_check = std::min<std::size_t>(output_size, 4);
        const std::size_t down_blocks = hidden_size / kQuantBlock;
        const std::size_t down_row_bytes = down_bytes / output_size;
        for (std::size_t batch = 0; batch < batch_size; ++batch) {
            for (std::size_t row = 0; row < rows_to_check; ++row) {
                const auto *row_weights = static_cast<const char *>(down_weights) + row * down_row_bytes;
                const float expected = k_quant_reference_row(down_type, row_weights,
                                                             hidden.data() + batch * hidden_size,
                                                             down_blocks);
                result.max_absolute_error = std::max(
                    result.max_absolute_error,
                    std::abs(expected - output[batch * output_size + row]));
            }
        }

        hip_check(hipEventDestroy(end), "hipEventDestroy N23Q4 FFN end");
        hip_check(hipEventDestroy(begin), "hipEventDestroy N23Q4 FFN begin");
        hip_check(hipFree(device_output), "hipFree N23Q4 FFN output");
        hip_check(hipFree(device_hidden), "hipFree N23Q4 FFN hidden");
        hip_check(hipFree(device_up), "hipFree N23Q4 FFN up");
        hip_check(hipFree(device_gate), "hipFree N23Q4 FFN gate");
        hip_check(hipFree(device_input_sums), "hipFree N23Q4 FFN input sums");
        hip_check(hipFree(device_input_scales), "hipFree N23Q4 FFN input scales");
        hip_check(hipFree(device_quantized_input), "hipFree N23Q4 FFN input quant");
        hip_check(hipFree(device_input), "hipFree N23Q4 FFN input");
        hip_check(hipFree(device_down_weights), "hipFree N23Q4 FFN down weights");
        hip_check(hipFree(device_up_weights), "hipFree N23Q4 FFN up weights");
        hip_check(hipFree(device_gate_weights), "hipFree N23Q4 FFN gate weights");
        return result;
    } catch (...) {
        if (end) (void) hipEventDestroy(end);
        if (begin) (void) hipEventDestroy(begin);
        if (device_output) (void) hipFree(device_output);
        if (device_hidden) (void) hipFree(device_hidden);
        if (device_up) (void) hipFree(device_up);
        if (device_gate) (void) hipFree(device_gate);
        if (device_input_sums) (void) hipFree(device_input_sums);
        if (device_input_scales) (void) hipFree(device_input_scales);
        if (device_quantized_input) (void) hipFree(device_quantized_input);
        if (device_input) (void) hipFree(device_input);
        if (device_down_weights) (void) hipFree(device_down_weights);
        if (device_up_weights) (void) hipFree(device_up_weights);
        (void) hipFree(device_gate_weights);
        throw;
    }
}

Q4KBenchmarkResult benchmark_gated_delta_step(int iterations) {
    constexpr int value_heads = 48;
    constexpr int head_size = 128;
    constexpr int warmup_iterations = 3;
    if (iterations <= 0) {
        throw std::invalid_argument("gated-delta benchmark needs a positive iteration count");
    }
    const std::size_t vector_values = static_cast<std::size_t>(value_heads) * head_size;
    const std::size_t state_values = vector_values * head_size;
    std::vector<float> queries(vector_values);
    std::vector<float> keys(vector_values);
    std::vector<float> values(vector_values);
    std::vector<float> decays(value_heads);
    std::vector<float> betas(value_heads);
    std::vector<std::uint16_t> reference_state(state_values, 0);
    std::vector<float> reference_output(vector_values);
    std::vector<float> output(vector_values);
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> activation(-0.2F, 0.2F);
    std::uniform_real_distribution<float> decay(0.90F, 0.99F);
    std::uniform_real_distribution<float> beta(0.01F, 0.10F);
    for (auto *vector : {&queries, &keys, &values}) {
        std::generate(vector->begin(), vector->end(), [&] { return activation(generator); });
    }
    std::generate(decays.begin(), decays.end(), [&] { return decay(generator); });
    std::generate(betas.begin(), betas.end(), [&] { return beta(generator); });

    const auto reference_step = [&] {
        std::array<float, head_size> state_key{};
        for (int head = 0; head < value_heads; ++head) {
            const std::size_t vector_offset = static_cast<std::size_t>(head) * head_size;
            const std::size_t state_offset = vector_offset * head_size;
            for (int row = 0; row < head_size; ++row) {
                float sum = 0.0F;
                for (int column = 0; column < head_size; ++column) {
                    sum = std::fma(fp16_to_float(reference_state[state_offset + static_cast<std::size_t>(row) * head_size + column]),
                                   keys[vector_offset + column], sum);
                }
                state_key[row] = sum;
            }
            for (int row = 0; row < head_size; ++row) {
                const float update = (values[vector_offset + row] - state_key[row]) * betas[head];
                for (int column = 0; column < head_size; ++column) {
                    const std::size_t offset = state_offset + static_cast<std::size_t>(row) * head_size + column;
                    const float next = decays[head] * fp16_to_float(reference_state[offset])
                                     + update * keys[vector_offset + column];
                    reference_state[offset] = float_to_fp16(next);
                }
            }
            for (int row = 0; row < head_size; ++row) {
                float sum = 0.0F;
                for (int column = 0; column < head_size; ++column) {
                    sum = std::fma(fp16_to_float(reference_state[state_offset + static_cast<std::size_t>(row) * head_size + column]),
                                   queries[vector_offset + column], sum);
                }
                reference_output[vector_offset + row] = sum;
            }
        }
    };

    float *device_queries = nullptr;
    float *device_keys = nullptr;
    float *device_values = nullptr;
    float *device_decays = nullptr;
    float *device_betas = nullptr;
    __half *device_states = nullptr;
    float *device_output = nullptr;
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    hip_check(hipMalloc(reinterpret_cast<void **>(&device_queries), queries.size() * sizeof(float)), "hipMalloc GDN queries");
    try {
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_keys), keys.size() * sizeof(float)), "hipMalloc GDN keys");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_values), values.size() * sizeof(float)), "hipMalloc GDN values");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_decays), decays.size() * sizeof(float)), "hipMalloc GDN decays");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_betas), betas.size() * sizeof(float)), "hipMalloc GDN betas");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_states), state_values * sizeof(__half)), "hipMalloc GDN states");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_output), output.size() * sizeof(float)), "hipMalloc GDN output");
        hip_check(hipMemcpy(device_queries, queries.data(), queries.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy GDN queries");
        hip_check(hipMemcpy(device_keys, keys.data(), keys.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy GDN keys");
        hip_check(hipMemcpy(device_values, values.data(), values.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy GDN values");
        hip_check(hipMemcpy(device_decays, decays.data(), decays.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy GDN decays");
        hip_check(hipMemcpy(device_betas, betas.data(), betas.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy GDN betas");
        hip_check(hipMemset(device_states, 0, state_values * sizeof(__half)), "hipMemset GDN states");
        hip_check(hipEventCreate(&begin), "hipEventCreate GDN begin");
        hip_check(hipEventCreate(&end), "hipEventCreate GDN end");

        const dim3 grid(value_heads);
        const dim3 block(256);
        const auto launch = [&] {
            hipLaunchKernelGGL(gated_delta_step_kernel, grid, block, 0, nullptr,
                               device_queries, device_keys, device_values, device_decays,
                               device_betas, device_states, device_output, head_size);
        };
        for (int i = 0; i < warmup_iterations; ++i) {
            launch();
        }
        hip_check(hipDeviceSynchronize(), "GDN warmup");
        hip_check(hipEventRecord(begin), "hipEventRecord GDN begin");
        for (int i = 0; i < iterations; ++i) {
            launch();
        }
        hip_check(hipEventRecord(end), "hipEventRecord GDN end");
        hip_check(hipEventSynchronize(end), "hipEventSynchronize GDN");
        float elapsed = 0.0F;
        hip_check(hipEventElapsedTime(&elapsed, begin, end), "hipEventElapsedTime GDN");
        hip_check(hipMemcpy(output.data(), device_output, output.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy GDN output");
        for (int i = 0; i < warmup_iterations + iterations; ++i) {
            reference_step();
        }

        Q4KBenchmarkResult result;
        result.milliseconds = elapsed / static_cast<float>(iterations);
        const std::size_t state_bytes = state_values * sizeof(__half);
        result.bandwidth_gbps = static_cast<float>(3 * state_bytes)
                              / (result.milliseconds * 1.0e6F);
        for (std::size_t index = 0; index < output.size(); ++index) {
            result.max_absolute_error = std::max(result.max_absolute_error,
                                                 std::abs(reference_output[index] - output[index]));
        }
        hip_check(hipEventDestroy(end), "hipEventDestroy GDN end");
        hip_check(hipEventDestroy(begin), "hipEventDestroy GDN begin");
        hip_check(hipFree(device_output), "hipFree GDN output");
        hip_check(hipFree(device_states), "hipFree GDN states");
        hip_check(hipFree(device_betas), "hipFree GDN betas");
        hip_check(hipFree(device_decays), "hipFree GDN decays");
        hip_check(hipFree(device_values), "hipFree GDN values");
        hip_check(hipFree(device_keys), "hipFree GDN keys");
        hip_check(hipFree(device_queries), "hipFree GDN queries");
        return result;
    } catch (...) {
        if (end) (void) hipEventDestroy(end);
        if (begin) (void) hipEventDestroy(begin);
        if (device_output) (void) hipFree(device_output);
        if (device_states) (void) hipFree(device_states);
        if (device_betas) (void) hipFree(device_betas);
        if (device_decays) (void) hipFree(device_decays);
        if (device_values) (void) hipFree(device_values);
        if (device_keys) (void) hipFree(device_keys);
        (void) hipFree(device_queries);
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

Q4KBenchmarkResult benchmark_n23kv4(std::size_t context_size, int iterations) {
    if (context_size == 0 || iterations <= 0) {
        throw std::invalid_argument("invalid N23KV4 benchmark context or iteration count");
    }
    constexpr std::size_t head_size = 256;
    constexpr std::size_t groups_per_head = head_size / 32;
    constexpr std::size_t query_heads = 24;
    constexpr std::size_t kv_heads = 4;
    const std::size_t key_groups = context_size * kv_heads * groups_per_head;
    std::vector<N23KV4Group> keys(key_groups);
    std::vector<std::int8_t> queries(query_heads * head_size);
    std::vector<float> query_scales(query_heads * groups_per_head);
    std::vector<float> scores(query_heads * context_size);
    std::mt19937 generator(42);
    std::uniform_int_distribution<int> q4_distribution(-8, 7);
    std::uniform_int_distribution<int> q8_distribution(-127, 127);
    std::uniform_real_distribution<float> scale_distribution(0.001F, 0.02F);
    for (auto &group : keys) {
        group.scale = float_to_fp16(scale_distribution(generator));
        for (auto &byte : group.quants) {
            const int first = q4_distribution(generator);
            const int second = q4_distribution(generator);
            byte = static_cast<std::uint8_t>((first & 0x0F) | ((second & 0x0F) << 4));
        }
    }
    for (auto &value : queries) {
        value = static_cast<std::int8_t>(q8_distribution(generator));
    }
    for (auto &scale : query_scales) {
        scale = scale_distribution(generator);
    }

    N23KV4Group *device_keys = nullptr;
    std::int8_t *device_queries = nullptr;
    float *device_query_scales = nullptr;
    float *device_scores = nullptr;
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    const std::size_t key_bytes = keys.size() * sizeof(N23KV4Group);
    hip_check(hipMalloc(reinterpret_cast<void **>(&device_keys), key_bytes), "hipMalloc N23KV4 keys");
    try {
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_queries), queries.size()), "hipMalloc N23KV4 queries");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_query_scales), query_scales.size() * sizeof(float)), "hipMalloc N23KV4 query scales");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_scores), scores.size() * sizeof(float)), "hipMalloc N23KV4 scores");
        hip_check(hipMemcpy(device_keys, keys.data(), key_bytes, hipMemcpyHostToDevice), "hipMemcpy N23KV4 keys");
        hip_check(hipMemcpy(device_queries, queries.data(), queries.size(), hipMemcpyHostToDevice), "hipMemcpy N23KV4 queries");
        hip_check(hipMemcpy(device_query_scales, query_scales.data(), query_scales.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy N23KV4 query scales");
        hip_check(hipEventCreate(&begin), "hipEventCreate N23KV4 begin");
        hip_check(hipEventCreate(&end), "hipEventCreate N23KV4 end");

        const dim3 block(128);
        const dim3 grid(static_cast<unsigned int>((context_size + 3) / 4), query_heads);
        for (int i = 0; i < 3; ++i) {
            hipLaunchKernelGGL(n23kv4_score_kernel, grid, block, 0, nullptr,
                               device_keys, device_queries, device_query_scales,
                               device_scores, static_cast<int>(context_size));
        }
        hip_check(hipDeviceSynchronize(), "N23KV4 warmup");
        hip_check(hipEventRecord(begin), "hipEventRecord N23KV4 begin");
        for (int i = 0; i < iterations; ++i) {
            hipLaunchKernelGGL(n23kv4_score_kernel, grid, block, 0, nullptr,
                               device_keys, device_queries, device_query_scales,
                               device_scores, static_cast<int>(context_size));
        }
        hip_check(hipEventRecord(end), "hipEventRecord N23KV4 end");
        hip_check(hipEventSynchronize(end), "hipEventSynchronize N23KV4");
        float elapsed = 0.0F;
        hip_check(hipEventElapsedTime(&elapsed, begin, end), "hipEventElapsedTime N23KV4");
        hip_check(hipMemcpy(scores.data(), device_scores, scores.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy N23KV4 scores");

        Q4KBenchmarkResult result;
        result.milliseconds = elapsed / static_cast<float>(iterations);
        const std::size_t effective_key_bytes = key_bytes * (query_heads / kv_heads);
        result.bandwidth_gbps = static_cast<float>(effective_key_bytes)
                              / (result.milliseconds * 1.0e6F);
        const std::size_t tokens_to_check = std::min<std::size_t>(context_size, 16);
        for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
            const std::size_t kv_head = query_head / (query_heads / kv_heads);
            for (std::size_t token = 0; token < tokens_to_check; ++token) {
                float expected = 0.0F;
                const auto *token_keys = keys.data()
                    + (kv_head * context_size + token) * groups_per_head;
                for (std::size_t group = 0; group < groups_per_head; ++group) {
                    const float key_scale = fp16_to_float(token_keys[group].scale);
                    const float query_scale = query_scales[query_head * groups_per_head + group];
                    int dot = 0;
                    for (int i = 0; i < 32; ++i) {
                        const auto byte = token_keys[group].quants[i / 2];
                        const int nibble = (byte >> ((i & 1) * 4)) & 0x0F;
                        const int key = nibble >= 8 ? nibble - 16 : nibble;
                        dot += key * queries[(query_head * groups_per_head + group) * 32 + i];
                    }
                    expected += static_cast<float>(dot) * key_scale * query_scale;
                }
                const float actual = scores[query_head * context_size + token];
                result.max_absolute_error = std::max(result.max_absolute_error, std::abs(expected - actual));
            }
        }
        hip_check(hipEventDestroy(end), "hipEventDestroy N23KV4 end");
        hip_check(hipEventDestroy(begin), "hipEventDestroy N23KV4 begin");
        hip_check(hipFree(device_scores), "hipFree N23KV4 scores");
        hip_check(hipFree(device_query_scales), "hipFree N23KV4 query scales");
        hip_check(hipFree(device_queries), "hipFree N23KV4 queries");
        hip_check(hipFree(device_keys), "hipFree N23KV4 keys");
        return result;
    } catch (...) {
        if (end) (void) hipEventDestroy(end);
        if (begin) (void) hipEventDestroy(begin);
        if (device_scores) (void) hipFree(device_scores);
        if (device_query_scales) (void) hipFree(device_query_scales);
        if (device_queries) (void) hipFree(device_queries);
        (void) hipFree(device_keys);
        throw;
    }
}

Q4KBenchmarkResult benchmark_n23kv4_batch8(std::size_t context_size, int iterations) {
    if (context_size == 0 || iterations <= 0) {
        throw std::invalid_argument("invalid N23KV4 batch-8 benchmark context or iteration count");
    }
    constexpr std::size_t batch_size = 8;
    constexpr std::size_t head_size = 256;
    constexpr std::size_t groups_per_head = head_size / 32;
    constexpr std::size_t query_heads = 24;
    constexpr std::size_t kv_heads = 4;
    const std::size_t key_groups = context_size * kv_heads * groups_per_head;
    std::vector<N23KV4Group> keys(key_groups);
    std::vector<std::int8_t> queries(batch_size * query_heads * head_size);
    std::vector<float> query_scales(batch_size * query_heads * groups_per_head);
    std::vector<float> scores(batch_size * query_heads * context_size);
    std::mt19937 generator(42);
    std::uniform_int_distribution<int> q4_distribution(-8, 7);
    std::uniform_int_distribution<int> q8_distribution(-127, 127);
    std::uniform_real_distribution<float> scale_distribution(0.001F, 0.02F);
    for (auto &group : keys) {
        group.scale = float_to_fp16(scale_distribution(generator));
        for (auto &byte : group.quants) {
            const int first = q4_distribution(generator);
            const int second = q4_distribution(generator);
            byte = static_cast<std::uint8_t>((first & 0x0F) | ((second & 0x0F) << 4));
        }
    }
    for (auto &value : queries) {
        value = static_cast<std::int8_t>(q8_distribution(generator));
    }
    for (auto &scale : query_scales) {
        scale = scale_distribution(generator);
    }

    N23KV4Group *device_keys = nullptr;
    std::int8_t *device_queries = nullptr;
    float *device_query_scales = nullptr;
    float *device_scores = nullptr;
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    const std::size_t key_bytes = keys.size() * sizeof(N23KV4Group);
    hip_check(hipMalloc(reinterpret_cast<void **>(&device_keys), key_bytes), "hipMalloc N23KV4 batch8 keys");
    try {
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_queries), queries.size()), "hipMalloc N23KV4 batch8 queries");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_query_scales), query_scales.size() * sizeof(float)), "hipMalloc N23KV4 batch8 query scales");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_scores), scores.size() * sizeof(float)), "hipMalloc N23KV4 batch8 scores");
        hip_check(hipMemcpy(device_keys, keys.data(), key_bytes, hipMemcpyHostToDevice), "hipMemcpy N23KV4 batch8 keys");
        hip_check(hipMemcpy(device_queries, queries.data(), queries.size(), hipMemcpyHostToDevice), "hipMemcpy N23KV4 batch8 queries");
        hip_check(hipMemcpy(device_query_scales, query_scales.data(), query_scales.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy N23KV4 batch8 query scales");
        hip_check(hipEventCreate(&begin), "hipEventCreate N23KV4 batch8 begin");
        hip_check(hipEventCreate(&end), "hipEventCreate N23KV4 batch8 end");

        const dim3 block(128);
        const dim3 grid(static_cast<unsigned int>((context_size + 3) / 4), query_heads);
        for (int i = 0; i < 3; ++i) {
            hipLaunchKernelGGL(n23kv4_score_batch8_kernel, grid, block, 0, nullptr,
                               device_keys, device_queries, device_query_scales,
                               device_scores, static_cast<int>(context_size));
        }
        hip_check(hipDeviceSynchronize(), "N23KV4 batch8 warmup");
        hip_check(hipEventRecord(begin), "hipEventRecord N23KV4 batch8 begin");
        for (int i = 0; i < iterations; ++i) {
            hipLaunchKernelGGL(n23kv4_score_batch8_kernel, grid, block, 0, nullptr,
                               device_keys, device_queries, device_query_scales,
                               device_scores, static_cast<int>(context_size));
        }
        hip_check(hipEventRecord(end), "hipEventRecord N23KV4 batch8 end");
        hip_check(hipEventSynchronize(end), "hipEventSynchronize N23KV4 batch8");
        float elapsed = 0.0F;
        hip_check(hipEventElapsedTime(&elapsed, begin, end), "hipEventElapsedTime N23KV4 batch8");
        hip_check(hipMemcpy(scores.data(), device_scores, scores.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy N23KV4 batch8 scores");

        Q4KBenchmarkResult result;
        result.milliseconds = elapsed / static_cast<float>(iterations);
        const std::size_t effective_key_bytes = key_bytes * (query_heads / kv_heads);
        result.bandwidth_gbps = static_cast<float>(effective_key_bytes)
                              / (result.milliseconds * 1.0e6F);
        const std::size_t tokens_to_check = std::min<std::size_t>(context_size, 8);
        for (std::size_t batch = 0; batch < batch_size; ++batch) {
            for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
                const std::size_t kv_head = query_head / (query_heads / kv_heads);
                for (std::size_t token = 0; token < tokens_to_check; ++token) {
                    float expected = 0.0F;
                    const auto *token_keys = keys.data()
                        + (kv_head * context_size + token) * groups_per_head;
                    for (std::size_t group = 0; group < groups_per_head; ++group) {
                        const std::size_t query_group = (batch * query_heads + query_head)
                                                      * groups_per_head + group;
                        int dot = 0;
                        for (int i = 0; i < 32; ++i) {
                            const auto byte = token_keys[group].quants[i / 2];
                            const int nibble = (byte >> ((i & 1) * 4)) & 0x0F;
                            const int key = nibble >= 8 ? nibble - 16 : nibble;
                            dot += key * queries[query_group * 32 + i];
                        }
                        expected += static_cast<float>(dot) * fp16_to_float(token_keys[group].scale)
                                  * query_scales[query_group];
                    }
                    const float actual = scores[(batch * query_heads + query_head) * context_size + token];
                    result.max_absolute_error = std::max(result.max_absolute_error, std::abs(expected - actual));
                }
            }
        }
        hip_check(hipEventDestroy(end), "hipEventDestroy N23KV4 batch8 end");
        hip_check(hipEventDestroy(begin), "hipEventDestroy N23KV4 batch8 begin");
        hip_check(hipFree(device_scores), "hipFree N23KV4 batch8 scores");
        hip_check(hipFree(device_query_scales), "hipFree N23KV4 batch8 query scales");
        hip_check(hipFree(device_queries), "hipFree N23KV4 batch8 queries");
        hip_check(hipFree(device_keys), "hipFree N23KV4 batch8 keys");
        return result;
    } catch (...) {
        if (end) (void) hipEventDestroy(end);
        if (begin) (void) hipEventDestroy(begin);
        if (device_scores) (void) hipFree(device_scores);
        if (device_query_scales) (void) hipFree(device_query_scales);
        if (device_queries) (void) hipFree(device_queries);
        (void) hipFree(device_keys);
        throw;
    }
}

Q4KBenchmarkResult benchmark_n23kv4_flash_batch8(std::size_t context_size,
                                                  int partitions, int iterations) {
    if (context_size == 0 || partitions <= 0 || iterations <= 0) {
        throw std::invalid_argument("invalid N23KV4 flash benchmark parameters");
    }
    constexpr std::size_t batch_size = 8;
    constexpr std::size_t head_size = 256;
    constexpr std::size_t groups_per_head = head_size / 32;
    constexpr std::size_t query_heads = 24;
    constexpr std::size_t kv_heads = 4;
    constexpr std::size_t query_groups_per_kv = 6;
    constexpr std::size_t partial_stride = 258;
    const std::size_t key_groups = context_size * kv_heads * groups_per_head;
    std::vector<N23KV4Group> keys(key_groups);
    std::vector<N23KV4Group> values(key_groups);
    std::vector<std::int8_t> queries(batch_size * query_heads * head_size);
    std::vector<float> query_scales(batch_size * query_heads * groups_per_head);
    std::vector<float> output(batch_size * query_heads * head_size);
    std::mt19937 generator(42);
    std::uniform_int_distribution<int> q4_distribution(-8, 7);
    std::uniform_int_distribution<int> q8_distribution(-127, 127);
    std::uniform_real_distribution<float> scale_distribution(0.001F, 0.02F);
    auto fill_kv = [&](std::vector<N23KV4Group> &cache) {
        for (auto &group : cache) {
            group.scale = float_to_fp16(scale_distribution(generator));
            for (auto &byte : group.quants) {
                const int first = q4_distribution(generator);
                const int second = q4_distribution(generator);
                byte = static_cast<std::uint8_t>((first & 0x0F) | ((second & 0x0F) << 4));
            }
        }
    };
    fill_kv(keys);
    fill_kv(values);
    for (auto &value : queries) {
        value = static_cast<std::int8_t>(q8_distribution(generator));
    }
    for (auto &scale : query_scales) {
        scale = scale_distribution(generator);
    }

    N23KV4Group *device_keys = nullptr;
    N23KV4Group *device_values = nullptr;
    std::int8_t *device_queries = nullptr;
    float *device_query_scales = nullptr;
    float *device_partials = nullptr;
    float *device_output = nullptr;
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    const std::size_t cache_bytes = keys.size() * sizeof(N23KV4Group);
    const std::size_t partial_count = batch_size * query_heads
                                    * static_cast<std::size_t>(partitions) * partial_stride;
    hip_check(hipMalloc(reinterpret_cast<void **>(&device_keys), cache_bytes), "hipMalloc flash keys");
    try {
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_values), cache_bytes), "hipMalloc flash values");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_queries), queries.size()), "hipMalloc flash queries");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_query_scales), query_scales.size() * sizeof(float)), "hipMalloc flash query scales");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_partials), partial_count * sizeof(float)), "hipMalloc flash partials");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_output), output.size() * sizeof(float)), "hipMalloc flash output");
        hip_check(hipMemcpy(device_keys, keys.data(), cache_bytes, hipMemcpyHostToDevice), "hipMemcpy flash keys");
        hip_check(hipMemcpy(device_values, values.data(), cache_bytes, hipMemcpyHostToDevice), "hipMemcpy flash values");
        hip_check(hipMemcpy(device_queries, queries.data(), queries.size(), hipMemcpyHostToDevice), "hipMemcpy flash queries");
        hip_check(hipMemcpy(device_query_scales, query_scales.data(), query_scales.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy flash query scales");
        hip_check(hipEventCreate(&begin), "hipEventCreate flash begin");
        hip_check(hipEventCreate(&end), "hipEventCreate flash end");

        const dim3 partial_grid(static_cast<unsigned int>(partitions), kv_heads * query_groups_per_kv);
        const dim3 partial_block(256);
        const dim3 reduce_grid(batch_size * query_heads);
        const dim3 reduce_block(256);
        auto launch = [&] {
            hipLaunchKernelGGL(n23kv4_flash_partial_kernel, partial_grid, partial_block, 0, nullptr,
                               device_keys, device_values, device_queries, device_query_scales,
                               device_partials, static_cast<int>(context_size), partitions);
            hipLaunchKernelGGL(n23kv4_flash_reduce_kernel, reduce_grid, reduce_block, 0, nullptr,
                               device_partials, device_output, partitions);
        };
        for (int i = 0; i < 2; ++i) {
            launch();
        }
        hip_check(hipDeviceSynchronize(), "N23KV4 flash warmup");
        hip_check(hipEventRecord(begin), "hipEventRecord flash begin");
        for (int i = 0; i < iterations; ++i) {
            launch();
        }
        hip_check(hipEventRecord(end), "hipEventRecord flash end");
        hip_check(hipEventSynchronize(end), "hipEventSynchronize flash");
        float elapsed = 0.0F;
        hip_check(hipEventElapsedTime(&elapsed, begin, end), "hipEventElapsedTime flash");
        hip_check(hipMemcpy(output.data(), device_output, output.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy flash output");

        Q4KBenchmarkResult result;
        result.milliseconds = elapsed / static_cast<float>(iterations);
        const std::size_t effective_cache_bytes = 2 * cache_bytes * query_groups_per_kv;
        result.bandwidth_gbps = static_cast<float>(effective_cache_bytes)
                              / (result.milliseconds * 1.0e6F);

        std::vector<float> scores(context_size);
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t token = 0; token < context_size; ++token) {
            float score = 0.0F;
            const auto *token_keys = keys.data() + token * groups_per_head;
            for (std::size_t group = 0; group < groups_per_head; ++group) {
                int dot = 0;
                for (int i = 0; i < 32; ++i) {
                    const auto byte = token_keys[group].quants[i / 2];
                    const int nibble = (byte >> ((i & 1) * 4)) & 0x0F;
                    const int key = nibble >= 8 ? nibble - 16 : nibble;
                    dot += key * queries[group * 32 + i];
                }
                score += static_cast<float>(dot) * fp16_to_float(token_keys[group].scale)
                       * query_scales[group] * 0.0625F;
            }
            scores[token] = score;
            maximum = std::max(maximum, score);
        }
        double denominator = 0.0;
        for (const float score : scores) {
            denominator += std::exp(static_cast<double>(score - maximum));
        }
        for (int dimension = 0; dimension < 8; ++dimension) {
            double expected = 0.0;
            const int group = dimension / 32;
            const int index = dimension & 31;
            for (std::size_t token = 0; token < context_size; ++token) {
                const auto &value_group = values[token * groups_per_head + group];
                const auto byte = value_group.quants[index / 2];
                const int nibble = (byte >> ((index & 1) * 4)) & 0x0F;
                const int quant = nibble >= 8 ? nibble - 16 : nibble;
                const float value = static_cast<float>(quant) * fp16_to_float(value_group.scale);
                expected += std::exp(static_cast<double>(scores[token] - maximum)) * value;
            }
            expected /= denominator;
            result.max_absolute_error = std::max(result.max_absolute_error,
                std::abs(static_cast<float>(expected) - output[dimension]));
        }
        hip_check(hipEventDestroy(end), "hipEventDestroy flash end");
        hip_check(hipEventDestroy(begin), "hipEventDestroy flash begin");
        hip_check(hipFree(device_output), "hipFree flash output");
        hip_check(hipFree(device_partials), "hipFree flash partials");
        hip_check(hipFree(device_query_scales), "hipFree flash query scales");
        hip_check(hipFree(device_queries), "hipFree flash queries");
        hip_check(hipFree(device_values), "hipFree flash values");
        hip_check(hipFree(device_keys), "hipFree flash keys");
        return result;
    } catch (...) {
        if (end) (void) hipEventDestroy(end);
        if (begin) (void) hipEventDestroy(begin);
        if (device_output) (void) hipFree(device_output);
        if (device_partials) (void) hipFree(device_partials);
        if (device_query_scales) (void) hipFree(device_query_scales);
        if (device_queries) (void) hipFree(device_queries);
        if (device_values) (void) hipFree(device_values);
        (void) hipFree(device_keys);
        throw;
    }
}

} // namespace qwen_native

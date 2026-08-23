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
#include <unordered_map>
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

struct Q8_0Block {
    std::uint16_t d;
    std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34);

struct IQ4XSBlock {
    std::uint16_t d;
    std::uint16_t scales_h;
    std::uint8_t scales_l[4];
    std::uint8_t qs[128];
};
static_assert(sizeof(IQ4XSBlock) == 136);

struct IQ4NLBlock {
    std::uint16_t d;
    std::uint8_t qs[16];
};
static_assert(sizeof(IQ4NLBlock) == 18);

struct Q3KBlock {
    std::uint8_t hmask[32];
    std::uint8_t qs[64];
    std::uint8_t scales[12];
    std::uint16_t d;
};
static_assert(sizeof(Q3KBlock) == 110);

constexpr int kQ8Block = 32;
constexpr int kIQ4NLBlock = 32;

__device__ constexpr int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

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

// The 16 GiB target cannot reside entirely on this 8 GiB card, but a stable
// subset can.  Holding the output head and FFN down projections avoids copying
// their multi-gigabyte working set on every decoded token.  Cache entries are
// keyed by the mapped GGUF address and deliberately live for the process.
struct ResidentWeightCache {
    struct Entry {
        void *device = nullptr;
        std::size_t bytes = 0;
    };

    ~ResidentWeightCache() {
        for (const auto &[host, entry] : entries) {
            (void) host;
            if (entry.device) (void) hipFree(entry.device);
        }
    }

    void set_budget(std::size_t bytes) {
        if (bytes < resident_bytes) {
            throw std::runtime_error("resident weight-cache budget cannot shrink after population");
        }
        budget_bytes = bytes;
    }

    const void *find_or_add(const void *host, std::size_t bytes, bool keep) {
        const auto existing = entries.find(host);
        if (existing != entries.end()) {
            if (existing->second.bytes != bytes) {
                throw std::runtime_error("resident weight-cache pointer has inconsistent size");
            }
            return existing->second.device;
        }
        if (!keep || bytes > budget_bytes - resident_bytes) {
            return nullptr;
        }
        void *device = nullptr;
        if (hipMalloc(&device, bytes) != hipSuccess) {
            return nullptr;
        }
        if (hipMemcpy(device, host, bytes, hipMemcpyHostToDevice) != hipSuccess) {
            (void) hipFree(device);
            return nullptr;
        }
        entries.emplace(host, Entry{device, bytes});
        resident_bytes += bytes;
        return device;
    }

    ResidentWeightCacheStats stats() const {
        return {budget_bytes, resident_bytes, entries.size()};
    }

    std::size_t budget_bytes = 0;
    std::size_t resident_bytes = 0;
    std::unordered_map<const void *, Entry> entries;
};

ResidentWeightCache &resident_weight_cache() {
    static ResidentWeightCache cache;
    return cache;
}

struct N23DraftCache {
    struct Entry {
        void *device = nullptr;
        std::size_t input_size = 0;
        std::size_t output_size = 0;
    };
    ~N23DraftCache() {
        for (const auto &[host, entry] : entries) {
            (void) host;
            if (entry.device) (void) hipFree(entry.device);
        }
    }
    std::unordered_map<const void *, Entry> entries;
};

N23DraftCache &n23_draft_cache() {
    static N23DraftCache cache;
    return cache;
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

__global__ void q8_0_matvec_batch8_kernel(const Q8_0Block *__restrict__ weights,
                                          const float *__restrict__ input,
                                          float *__restrict__ output,
                                          int input_size, int blocks_per_row, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kWarpSize) return;

    float sums[8] = {};
    const Q8_0Block *row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const Q8_0Block &block = row_weights[bi];
        const float d = __half2float(*reinterpret_cast<const __half *>(&block.d));
        const float weight = d * static_cast<float>(block.qs[lane]);
        const int idx = bi * kQ8Block + lane;
#pragma unroll
        for (int batch = 0; batch < 8; ++batch) {
            sums[batch] = fmaf(weight, input[static_cast<std::size_t>(batch) * input_size + idx], sums[batch]);
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

__global__ void iq4xs_matvec_batch8_kernel(const IQ4XSBlock *__restrict__ weights,
                                           const float *__restrict__ input,
                                           float *__restrict__ output,
                                           int input_size, int blocks_per_row, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kWarpSize) return;

    float sums[8] = {};
    const IQ4XSBlock *row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const IQ4XSBlock &block = row_weights[bi];
        const float d = __half2float(*reinterpret_cast<const __half *>(&block.d));
#pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((block.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
                         | (((block.scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d * static_cast<float>(ls - 32);
            int nibble;
            if (lane < 16) {
                nibble = block.qs[ib * 16 + lane] & 0xf;
            } else {
                nibble = block.qs[ib * 16 + lane - 16] >> 4;
            }
            const float weight = dl * static_cast<float>(kvalues_iq4nl[nibble]);
            const int idx = bi * kQuantBlock + ib * 32 + lane;
#pragma unroll
            for (int batch = 0; batch < 8; ++batch) {
                sums[batch] = fmaf(weight, input[static_cast<std::size_t>(batch) * input_size + idx], sums[batch]);
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

__global__ void iq4nl_matvec_batch8_kernel(const IQ4NLBlock *__restrict__ weights,
                                           const float *__restrict__ input,
                                           float *__restrict__ output,
                                           int input_size, int blocks_per_row, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kWarpSize) return;

    float sums[8] = {};
    const IQ4NLBlock *row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const IQ4NLBlock &block = row_weights[bi];
        const float d = __half2float(*reinterpret_cast<const __half *>(&block.d));
        int nibble;
        if (lane < 16) {
            nibble = block.qs[lane] & 0xf;
        } else {
            nibble = block.qs[lane - 16] >> 4;
        }
        const float weight = d * static_cast<float>(kvalues_iq4nl[nibble]);
        const int idx = bi * kIQ4NLBlock + lane;
#pragma unroll
        for (int batch = 0; batch < 8; ++batch) {
            sums[batch] = fmaf(weight, input[static_cast<std::size_t>(batch) * input_size + idx], sums[batch]);
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

__device__ int q3k_decode_scale(const std::uint8_t *scales, int sb) {
    int low4, high2;
    const int j = sb & 3;
    if (sb < 4) {
        low4 = scales[j] & 0xf;
        high2 = (scales[8 + j] >> 0) & 3;
    } else if (sb < 8) {
        low4 = scales[j] >> 4;
        high2 = (scales[8 + j] >> 2) & 3;
    } else if (sb < 12) {
        low4 = scales[4 + j] & 0xf;
        high2 = (scales[8 + j] >> 4) & 3;
    } else {
        low4 = scales[4 + j] >> 4;
        high2 = (scales[8 + j] >> 6) & 3;
    }
    return (low4 | (high2 << 4)) - 32;
}

__global__ void q3k_matvec_batch8_kernel(const Q3KBlock *__restrict__ weights,
                                         const float *__restrict__ input,
                                         float *__restrict__ output,
                                         int input_size, int blocks_per_row, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kWarpSize) return;

    float sums[8] = {};
    const Q3KBlock *row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const Q3KBlock &block = row_weights[bi];
        const float d = __half2float(*reinterpret_cast<const __half *>(&block.d));
#pragma unroll
        for (int iter = 0; iter < 8; ++iter) {
            const int p = iter * 32 + lane;
            const int sb = p / 16;
            const int pos_in_sub = p % 16;
            const int q_group = sb / 4;
            const int shift = (sb % 4) * 2;
            const int low2 = (block.qs[q_group * 16 + pos_in_sub] >> shift) & 3;
            const int high1 = (block.hmask[p / 8] >> (p % 8)) & 1;
            const int q3 = low2 | (high1 << 2);
            const float dl = d * static_cast<float>(q3k_decode_scale(block.scales, sb));
            const float weight = dl * static_cast<float>(q3 - 4);
            const int idx = bi * kQuantBlock + p;
#pragma unroll
            for (int batch = 0; batch < 8; ++batch) {
                sums[batch] = fmaf(weight, input[static_cast<std::size_t>(batch) * input_size + idx], sums[batch]);
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

__global__ void q8_0_matvec_kernel(const Q8_0Block *__restrict__ weights,
                                   const float *__restrict__ input,
                                   float *__restrict__ output,
                                   int blocks_per_row, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kWarpSize) return;

    float sum = 0.0F;
    const Q8_0Block *row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const Q8_0Block &block = row_weights[bi];
        const float d = __half2float(*reinterpret_cast<const __half *>(&block.d));
        const float weight = d * static_cast<float>(block.qs[lane]);
        sum = fmaf(weight, input[bi * kQ8Block + lane], sum);
    }
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, kWarpSize);
    }
    if (lane == 0) {
        output[row] = sum;
    }
}

__global__ void iq4xs_matvec_kernel(const IQ4XSBlock *__restrict__ weights,
                                    const float *__restrict__ input,
                                    float *__restrict__ output,
                                    int blocks_per_row, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kWarpSize) return;

    float sum = 0.0F;
    const IQ4XSBlock *row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const IQ4XSBlock &block = row_weights[bi];
        const float d = __half2float(*reinterpret_cast<const __half *>(&block.d));
#pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((block.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
                         | (((block.scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d * static_cast<float>(ls - 32);
            int nibble;
            if (lane < 16) {
                nibble = block.qs[ib * 16 + lane] & 0xf;
            } else {
                nibble = block.qs[ib * 16 + lane - 16] >> 4;
            }
            const float weight = dl * static_cast<float>(kvalues_iq4nl[nibble]);
            sum = fmaf(weight, input[bi * kQuantBlock + ib * 32 + lane], sum);
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

__global__ void iq4nl_matvec_kernel(const IQ4NLBlock *__restrict__ weights,
                                    const float *__restrict__ input,
                                    float *__restrict__ output,
                                    int blocks_per_row, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kWarpSize) return;

    float sum = 0.0F;
    const IQ4NLBlock *row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const IQ4NLBlock &block = row_weights[bi];
        const float d = __half2float(*reinterpret_cast<const __half *>(&block.d));
        int nibble;
        if (lane < 16) {
            nibble = block.qs[lane] & 0xf;
        } else {
            nibble = block.qs[lane - 16] >> 4;
        }
        const float weight = d * static_cast<float>(kvalues_iq4nl[nibble]);
        sum = fmaf(weight, input[bi * kIQ4NLBlock + lane], sum);
    }
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, kWarpSize);
    }
    if (lane == 0) {
        output[row] = sum;
    }
}

__global__ void q3k_matvec_kernel(const Q3KBlock *__restrict__ weights,
                                  const float *__restrict__ input,
                                  float *__restrict__ output,
                                  int blocks_per_row, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kWarpSize) return;

    float sum = 0.0F;
    const Q3KBlock *row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (int bi = 0; bi < blocks_per_row; ++bi) {
        const Q3KBlock &block = row_weights[bi];
        const float d = __half2float(*reinterpret_cast<const __half *>(&block.d));
#pragma unroll
        for (int iter = 0; iter < 8; ++iter) {
            const int p = iter * 32 + lane;
            const int sb = p / 16;
            const int pos_in_sub = p % 16;
            const int q_group = sb / 4;
            const int shift = (sb % 4) * 2;
            const int low2 = (block.qs[q_group * 16 + pos_in_sub] >> shift) & 3;
            const int high1 = (block.hmask[p / 8] >> (p % 8)) & 1;
            const int q3 = low2 | (high1 << 2);
            const float dl = d * static_cast<float>(q3k_decode_scale(block.scales, sb));
            const float weight = dl * static_cast<float>(q3 - 4);
            sum = fmaf(weight, input[bi * kQuantBlock + p], sum);
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
    __shared__ float delta[128];
    const float decay = decays[head];
    if (thread < head_size) {
        float predicted = 0.0F;
        for (int row = 0; row < head_size; ++row) {
            predicted = fmaf(__half2float(states[state_offset + static_cast<std::size_t>(row) * head_size + thread]),
                             keys[vector_offset + row], predicted);
        }
        delta[thread] = (values[vector_offset + thread] - decay * predicted) * betas[head];
    }
    __syncthreads();

    for (int element = thread; element < head_size * head_size; element += static_cast<int>(blockDim.x)) {
        const int row = element / head_size;
        const int column = element % head_size;
        const float updated = decay * __half2float(states[state_offset + element])
                            + keys[vector_offset + row] * delta[column];
        states[state_offset + element] = __float2half(updated);
    }
    __syncthreads();

    if (thread < head_size) {
        float sum = 0.0F;
        for (int row = 0; row < head_size; ++row) {
            sum = fmaf(__half2float(states[state_offset + static_cast<std::size_t>(row) * head_size + thread]),
                       queries[vector_offset + row], sum);
        }
        output[vector_offset + thread] = sum * 0.08838834764831843F;
    }
}

__global__ void gated_delta_step_batch8_kernel(const float *__restrict__ queries,
                                               const float *__restrict__ keys,
                                               const float *__restrict__ values,
                                               const float *__restrict__ decays,
                                               const float *__restrict__ betas,
                                               __half *__restrict__ states,
                                               float *__restrict__ output,
                                               int head_size, int batch_count) {
    const int head = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    const std::size_t vector_offset = static_cast<std::size_t>(head) * head_size;
    const std::size_t state_offset = vector_offset * head_size;
    const std::size_t vectors_per_batch = 48ULL * head_size;
    __shared__ float delta[128];

    for (int token = 0; token < batch_count; ++token) {
        const float *t_keys = keys + token * vectors_per_batch;
        const float *t_values = values + token * vectors_per_batch;
        const float *t_queries = queries + token * vectors_per_batch;
        const float t_decay = decays[token * 48 + head];
        const float t_beta = betas[token * 48 + head];

        if (thread < head_size) {
            float predicted = 0.0F;
            for (int row = 0; row < head_size; ++row) {
                predicted = fmaf(__half2float(states[state_offset + static_cast<std::size_t>(row) * head_size + thread]),
                                 t_keys[vector_offset + row], predicted);
            }
            delta[thread] = (t_values[vector_offset + thread] - t_decay * predicted) * t_beta;
        }
        __syncthreads();

        for (int element = thread; element < head_size * head_size; element += static_cast<int>(blockDim.x)) {
            const int row = element / head_size;
            const int column = element % head_size;
            const float updated = t_decay * __half2float(states[state_offset + element])
                                + t_keys[vector_offset + row] * delta[column];
            states[state_offset + element] = __float2half(updated);
        }
        __syncthreads();

        if (thread < head_size) {
            float sum = 0.0F;
            for (int row = 0; row < head_size; ++row) {
                sum = fmaf(__half2float(states[state_offset + static_cast<std::size_t>(row) * head_size + thread]),
                           t_queries[vector_offset + row], sum);
            }
            output[token * vectors_per_batch + vector_offset + thread] = sum * 0.08838834764831843F;
        }
        __syncthreads();
    }
}

// GPU-native recurrent: conv1d + SiLU + split QKV + L2 norm.
// One block per channel group, iterates over 8 tokens sequentially (state is serial).
// qkv_width = 2*key_heads*head_size + value_heads*head_size = 7680 for this model.
// Output: queries[8*recurrent_width], keys[8*recurrent_width], values[8*recurrent_width]
// where recurrent_width = value_heads * head_size = 6144.
__global__ void recurrent_conv1d_batch8_kernel(
        const float *__restrict__ qkv_input,   // slot[0]: 8 × qkv_width
        float *__restrict__ conv_state,         // 3 × qkv_width (persistent per layer)
        const float *__restrict__ conv_weights, // 4 × qkv_width
        float *__restrict__ queries,            // 8 × recurrent_width output
        float *__restrict__ keys,               // 8 × recurrent_width output
        float *__restrict__ values,             // 8 × recurrent_width output
        int qkv_width, int key_heads, int value_heads, int head_size) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= qkv_width) return;

    const int recurrent_width = value_heads * head_size;
    const float w0 = conv_weights[channel * 4 + 0];
    const float w1 = conv_weights[channel * 4 + 1];
    const float w2 = conv_weights[channel * 4 + 2];
    const float w3 = conv_weights[channel * 4 + 3];

    float s0 = conv_state[channel * 3 + 0];
    float s1 = conv_state[channel * 3 + 1];
    float s2 = conv_state[channel * 3 + 2];

    for (int token = 0; token < 8; ++token) {
        const float input_val = qkv_input[token * qkv_width + channel];
        float sum = fmaf(s0, w0, fmaf(s1, w1, fmaf(s2, w2, input_val * w3)));
        float activated = sum / (1.0f + expf(-sum)); // SiLU

        s0 = s1;
        s1 = s2;
        s2 = input_val;

        // Split into Q/K/V based on channel position
        const int key_region = key_heads * head_size;
        if (channel < key_region) {
            // Q region: replicate each key head to value_heads/key_heads copies
            const int key_head = channel / head_size;
            const int dim = channel % head_size;
            const int heads_per_key = value_heads / key_heads;
            for (int rep = 0; rep < heads_per_key; ++rep) {
                const int out_head = key_head * heads_per_key + rep;
                queries[token * recurrent_width + out_head * head_size + dim] = activated;
            }
        } else if (channel < 2 * key_region) {
            // K region: replicate each key head to value_heads/key_heads copies
            const int key_head = (channel - key_region) / head_size;
            const int dim = (channel - key_region) % head_size;
            const int heads_per_key = value_heads / key_heads;
            for (int rep = 0; rep < heads_per_key; ++rep) {
                const int out_head = key_head * heads_per_key + rep;
                keys[token * recurrent_width + out_head * head_size + dim] = activated;
            }
        } else {
            // V region: direct mapping
            const int v_offset = channel - 2 * key_region;
            values[token * recurrent_width + v_offset] = activated;
        }
    }

    conv_state[channel * 3 + 0] = s0;
    conv_state[channel * 3 + 1] = s1;
    conv_state[channel * 3 + 2] = s2;
}

// L2 normalize queries and keys per head (in-place).
// Grid: value_heads blocks, 128 threads (= head_size).
__global__ void recurrent_l2_norm_batch8_kernel(
        float *__restrict__ queries,  // 8 × recurrent_width
        float *__restrict__ keys,     // 8 × recurrent_width
        int recurrent_width, int head_size, float epsilon) {
    const int head = blockIdx.x;
    const int dim = threadIdx.x;
    if (dim >= head_size) return;

    for (int token = 0; token < 8; ++token) {
        const int base = token * recurrent_width + head * head_size;

        // L2 norm queries
        __shared__ float q_sum_sq[128];
        q_sum_sq[dim] = queries[base + dim] * queries[base + dim];
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (dim < stride) q_sum_sq[dim] += q_sum_sq[dim + stride];
            __syncthreads();
        }
        float q_scale = rsqrtf(q_sum_sq[0] / head_size + epsilon);
        queries[base + dim] *= q_scale;

        // L2 norm keys
        __shared__ float k_sum_sq[128];
        k_sum_sq[dim] = keys[base + dim] * keys[base + dim];
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (dim < stride) k_sum_sq[dim] += k_sum_sq[dim + stride];
            __syncthreads();
        }
        float k_scale = rsqrtf(k_sum_sq[0] / head_size + epsilon);
        keys[base + dim] *= k_scale;
        __syncthreads();
    }
}

// Compute decay and beta from hidden state via F32 matmul (5120→48).
// Grid: value_heads(48) blocks × 1, 256 threads per block.
// Each block computes beta[head] and alpha[head] for all 8 tokens.
__global__ void recurrent_decay_beta_batch8_kernel(
        const float *__restrict__ hidden,        // 8 × embedding_size (pipeline hidden)
        const float *__restrict__ beta_weights,  // 48 × embedding_size (row-major)
        const float *__restrict__ alpha_weights, // 48 × embedding_size (row-major)
        const float *__restrict__ dt_bias,       // 48
        const float *__restrict__ a_weights,     // 48
        float *__restrict__ decays,              // 8 × 48 output
        float *__restrict__ betas,               // 8 × 48 output
        int embedding_size) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float partial_beta[256];
    __shared__ float partial_alpha[256];

    const float *beta_row = beta_weights + head * embedding_size;
    const float *alpha_row = alpha_weights + head * embedding_size;
    const float bias = dt_bias[head];
    const float a_val = a_weights[head];

    for (int token = 0; token < 8; ++token) {
        const float *input = hidden + token * embedding_size;
        float sum_beta = 0.0f;
        float sum_alpha = 0.0f;
        for (int i = tid; i < embedding_size; i += 256) {
            sum_beta = fmaf(beta_row[i], input[i], sum_beta);
            sum_alpha = fmaf(alpha_row[i], input[i], sum_alpha);
        }
        partial_beta[tid] = sum_beta;
        partial_alpha[tid] = sum_alpha;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial_beta[tid] += partial_beta[tid + stride];
                partial_alpha[tid] += partial_alpha[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            float alpha_val = partial_alpha[0];
            float sp = logf(1.0f + expf(alpha_val + bias)); // softplus
            decays[token * 48 + head] = expf(sp * a_val);
            betas[token * 48 + head] = 1.0f / (1.0f + expf(-partial_beta[0])); // sigmoid
        }
        __syncthreads();
    }
}

// SSM norm + SiLU gate multiply.
// Input: gdn_output (8 × recurrent_width), gate (slot[1], 8 × recurrent_width)
// norm_weights: head_size floats (applied per head)
// Output: written to pipeline hidden buffer (8 × recurrent_width)
__global__ void recurrent_ssm_norm_gate_batch8_kernel(
        const float *__restrict__ gdn_output,  // 8 × recurrent_width
        const float *__restrict__ gate,        // slot[1]: 8 × recurrent_width
        const float *__restrict__ norm_weights,// head_size
        float *__restrict__ output,            // 8 × recurrent_width → pipeline hidden
        int recurrent_width, int /*value_heads*/, int head_size, float epsilon) {
    const int head = blockIdx.x;
    const int dim = threadIdx.x;
    if (dim >= head_size) return;

    __shared__ float sum_sq_shared[128];

    for (int token = 0; token < 8; ++token) {
        const int base = token * recurrent_width + head * head_size;
        float val = gdn_output[base + dim];

        sum_sq_shared[dim] = val * val;
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (dim < stride) sum_sq_shared[dim] += sum_sq_shared[dim + stride];
            __syncthreads();
        }
        float scale = rsqrtf(sum_sq_shared[0] / head_size + epsilon);

        float normed = val * scale * norm_weights[dim];
        float g = gate[base + dim];
        float silu_g = g / (1.0f + expf(-g));
        output[base + dim] = normed * silu_g;
        __syncthreads();
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
    if (input_size == 0 || output_size == 0) {
        throw std::invalid_argument("matrix dimensions must be non-zero");
    }
    switch (type) {
    case 8:  return (input_size / kQ8Block) * output_size * sizeof(Q8_0Block);
    case 11: return (input_size / kQuantBlock) * output_size * sizeof(Q3KBlock);
    case 12: return (input_size / kQuantBlock) * output_size * sizeof(Q4KBlock);
    case 13: return (input_size / kQuantBlock) * output_size * sizeof(Q5KBlock);
    case 14: return (input_size / kQuantBlock) * output_size * sizeof(Q6KBlock);
    case 20: return (input_size / kIQ4NLBlock) * output_size * sizeof(IQ4NLBlock);
    case 21: return (input_size / kQuantBlock) * output_size * sizeof(Q3KBlock);
    case 23: return (input_size / kQuantBlock) * output_size * sizeof(IQ4XSBlock);
    default: throw std::invalid_argument("matrix is not a supported quant type");
    }
}

void launch_k_matvec_batch8(std::uint32_t type, const void *weights, const float *input,
                            float *output, std::size_t input_size, std::size_t output_size) {
    const dim3 grid(static_cast<unsigned int>(output_size));
    const dim3 block(kWarpSize);
    switch (type) {
    case 8:
        hipLaunchKernelGGL(q8_0_matvec_batch8_kernel, grid, block, 0, nullptr,
                           static_cast<const Q8_0Block *>(weights), input, output,
                           static_cast<int>(input_size),
                           static_cast<int>(input_size / kQ8Block),
                           static_cast<int>(output_size));
        return;
    case 11:
        hipLaunchKernelGGL(q3k_matvec_batch8_kernel, grid, block, 0, nullptr,
                           static_cast<const Q3KBlock *>(weights), input, output,
                           static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        return;
    case 12:
        hipLaunchKernelGGL(q4k_matvec_batch8_kernel, grid, block, 0, nullptr,
                           static_cast<const Q4KBlock *>(weights), input, output,
                           static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        return;
    case 13:
        hipLaunchKernelGGL(q5k_matvec_batch8_kernel, grid, block, 0, nullptr,
                           static_cast<const Q5KBlock *>(weights), input, output,
                           static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        return;
    case 14:
        hipLaunchKernelGGL(q6k_matvec_batch8_kernel, grid, block, 0, nullptr,
                           static_cast<const Q6KBlock *>(weights), input, output,
                           static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        return;
    case 20:
        hipLaunchKernelGGL(iq4nl_matvec_batch8_kernel, grid, block, 0, nullptr,
                           static_cast<const IQ4NLBlock *>(weights), input, output,
                           static_cast<int>(input_size),
                           static_cast<int>(input_size / kIQ4NLBlock),
                           static_cast<int>(output_size));
        return;
    case 21:
        hipLaunchKernelGGL(q3k_matvec_batch8_kernel, grid, block, 0, nullptr,
                           static_cast<const Q3KBlock *>(weights), input, output,
                           static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        return;
    case 23:
        hipLaunchKernelGGL(iq4xs_matvec_batch8_kernel, grid, block, 0, nullptr,
                           static_cast<const IQ4XSBlock *>(weights), input, output,
                           static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        return;
    default:
        throw std::invalid_argument("matrix is not a supported quant type");
    }
}

void launch_k_matvec(std::uint32_t type, const void *weights, const float *input,
                     float *output, std::size_t input_size, std::size_t output_size) {
    const dim3 grid(static_cast<unsigned int>(output_size));
    const dim3 block(kWarpSize);
    switch (type) {
    case 12:
        hipLaunchKernelGGL(q4k_matvec_kernel, grid, block, 0, nullptr,
                           static_cast<const Q4KBlock *>(weights), input, output,
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        return;
    case 13:
        hipLaunchKernelGGL(q5k_matvec_kernel, grid, block, 0, nullptr,
                           static_cast<const Q5KBlock *>(weights), input, output,
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        return;
    case 14:
        hipLaunchKernelGGL(q6k_matvec_kernel, grid, block, 0, nullptr,
                           static_cast<const Q6KBlock *>(weights), input, output,
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        return;
    case 8:
        hipLaunchKernelGGL(q8_0_matvec_kernel, grid, block, 0, nullptr,
                           static_cast<const Q8_0Block *>(weights), input, output,
                           static_cast<int>(input_size / kQ8Block),
                           static_cast<int>(output_size));
        return;
    case 11:
    case 21:
        hipLaunchKernelGGL(q3k_matvec_kernel, grid, block, 0, nullptr,
                           static_cast<const Q3KBlock *>(weights), input, output,
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        return;
    case 20:
        hipLaunchKernelGGL(iq4nl_matvec_kernel, grid, block, 0, nullptr,
                           static_cast<const IQ4NLBlock *>(weights), input, output,
                           static_cast<int>(input_size / kIQ4NLBlock),
                           static_cast<int>(output_size));
        return;
    case 23:
        hipLaunchKernelGGL(iq4xs_matvec_kernel, grid, block, 0, nullptr,
                           static_cast<const IQ4XSBlock *>(weights), input, output,
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        return;
    default:
        throw std::invalid_argument("matrix is not a supported quant type");
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

void set_resident_weight_cache_budget(std::size_t bytes) {
    resident_weight_cache().set_budget(bytes);
}

ResidentWeightCacheStats resident_weight_cache_stats() {
    return resident_weight_cache().stats();
}

bool preload_resident_k_matrix(const void *weights, std::uint32_t weight_type,
                               std::size_t input_size, std::size_t output_size) {
    const std::size_t weight_bytes = k_quant_weight_bytes(weight_type, input_size, output_size);
    return resident_weight_cache().find_or_add(weights, weight_bytes, true) != nullptr;
}

void execute_k_matvec_cached(const void *weights, std::uint32_t weight_type,
                             const float *input, std::size_t input_size,
                             float *output, std::size_t output_size,
                             bool keep_resident) {
    const std::size_t weight_bytes = k_quant_weight_bytes(weight_type, input_size, output_size);
    auto &workspace = matvec_workspace();
    const void *device_weights = resident_weight_cache().find_or_add(weights, weight_bytes, keep_resident);
    if (!device_weights) {
        workspace.reserve_weights(weight_bytes);
        hip_check(hipMemcpy(workspace.weights, weights, weight_bytes, hipMemcpyHostToDevice),
                  "hipMemcpy native cached matvec weights");
        device_weights = workspace.weights;
    }
    workspace.reserve_input(input_size);
    workspace.reserve_output(output_size);
    hip_check(hipMemcpy(workspace.input, input, input_size * sizeof(float), hipMemcpyHostToDevice),
              "hipMemcpy native cached matvec input");
    launch_k_matvec(weight_type, device_weights, workspace.input, workspace.output, input_size, output_size);
    hip_check(hipGetLastError(), "native cached matvec launch");
    hip_check(hipMemcpy(output, workspace.output, output_size * sizeof(float), hipMemcpyDeviceToHost),
              "hipMemcpy native cached matvec output");
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

void execute_k_matvec_cached_batch8(const void *weights, std::uint32_t weight_type,
                                    const float *input, std::size_t input_size,
                                    float *output, std::size_t output_size,
                                    bool keep_resident) {
    constexpr std::size_t batch_size = 8;
    const std::size_t weight_bytes = k_quant_weight_bytes(weight_type, input_size, output_size);
    auto &workspace = matvec_workspace();
    const void *device_weights = resident_weight_cache().find_or_add(weights, weight_bytes, keep_resident);
    if (!device_weights) {
        workspace.reserve_weights(weight_bytes);
        hip_check(hipMemcpy(workspace.weights, weights, weight_bytes, hipMemcpyHostToDevice),
                  "hipMemcpy native cached batch8 matvec weights");
        device_weights = workspace.weights;
    }
    workspace.reserve_input(batch_size * input_size);
    workspace.reserve_output(batch_size * output_size);
    hip_check(hipMemcpy(workspace.input, input, batch_size * input_size * sizeof(float), hipMemcpyHostToDevice),
              "hipMemcpy native cached batch8 matvec input");
    launch_k_matvec_batch8(weight_type, device_weights, workspace.input, workspace.output,
                           input_size, output_size);
    hip_check(hipGetLastError(), "native cached batch8 matvec launch");
    hip_check(hipMemcpy(output, workspace.output, batch_size * output_size * sizeof(float), hipMemcpyDeviceToHost),
              "hipMemcpy native cached batch8 matvec output");
}

void *create_gated_delta_state() {
    constexpr std::size_t state_values = 48ULL * 128ULL * 128ULL;
    void *state = nullptr;
    hip_check(hipMalloc(&state, state_values * sizeof(__half)), "hipMalloc gated-delta state");
    hip_check(hipMemset(state, 0, state_values * sizeof(__half)), "hipMemset gated-delta state");
    return state;
}

void destroy_gated_delta_state(void *state) {
    if (state) {
        (void) hipFree(state);
    }
}

void upload_gated_delta_state(void *state, const float *host_state) {
    if (!state || !host_state) {
        throw std::invalid_argument("gated-delta upload requires state and host data");
    }
    constexpr std::size_t state_values = 48ULL * 128ULL * 128ULL;
    std::vector<__half> packed(state_values);
    for (std::size_t index = 0; index < state_values; ++index) {
        packed[index] = __float2half(host_state[index]);
    }
    hip_check(hipMemcpy(state, packed.data(), packed.size() * sizeof(__half), hipMemcpyHostToDevice),
              "hipMemcpy gated-delta state upload");
}

void download_gated_delta_state(const void *state, float *host_state) {
    if (!state || !host_state) {
        throw std::invalid_argument("gated-delta download requires state and host data");
    }
    constexpr std::size_t state_values = 48ULL * 128ULL * 128ULL;
    std::vector<__half> packed(state_values);
    hip_check(hipMemcpy(packed.data(), state, packed.size() * sizeof(__half), hipMemcpyDeviceToHost),
              "hipMemcpy gated-delta state download");
    for (std::size_t index = 0; index < state_values; ++index) {
        host_state[index] = __half2float(packed[index]);
    }
}

void execute_gated_delta_step(const float *queries, const float *keys,
                              const float *values, const float *decays,
                              const float *betas, void *state, float *output) {
    if (!queries || !keys || !values || !decays || !betas || !state || !output) {
        throw std::invalid_argument("gated-delta step requires non-null buffers");
    }
    constexpr std::size_t vector_values = 48ULL * 128ULL;
    struct Workspace {
        float *queries = nullptr;
        float *keys = nullptr;
        float *values = nullptr;
        float *decays = nullptr;
        float *betas = nullptr;
        float *output = nullptr;
        ~Workspace() {
            if (output) (void) hipFree(output);
            if (betas) (void) hipFree(betas);
            if (decays) (void) hipFree(decays);
            if (values) (void) hipFree(values);
            if (keys) (void) hipFree(keys);
            if (queries) (void) hipFree(queries);
        }
    };
    static Workspace workspace;
    if (!workspace.queries) {
        hip_check(hipMalloc(reinterpret_cast<void **>(&workspace.queries), vector_values * sizeof(float)),
                  "hipMalloc gated-delta queries");
        hip_check(hipMalloc(reinterpret_cast<void **>(&workspace.keys), vector_values * sizeof(float)),
                  "hipMalloc gated-delta keys");
        hip_check(hipMalloc(reinterpret_cast<void **>(&workspace.values), vector_values * sizeof(float)),
                  "hipMalloc gated-delta values");
        hip_check(hipMalloc(reinterpret_cast<void **>(&workspace.decays), 48 * sizeof(float)),
                  "hipMalloc gated-delta decays");
        hip_check(hipMalloc(reinterpret_cast<void **>(&workspace.betas), 48 * sizeof(float)),
                  "hipMalloc gated-delta betas");
        hip_check(hipMalloc(reinterpret_cast<void **>(&workspace.output), vector_values * sizeof(float)),
                  "hipMalloc gated-delta output");
    }
    hip_check(hipMemcpy(workspace.queries, queries, vector_values * sizeof(float), hipMemcpyHostToDevice),
              "hipMemcpy gated-delta queries");
    hip_check(hipMemcpy(workspace.keys, keys, vector_values * sizeof(float), hipMemcpyHostToDevice),
              "hipMemcpy gated-delta keys");
    hip_check(hipMemcpy(workspace.values, values, vector_values * sizeof(float), hipMemcpyHostToDevice),
              "hipMemcpy gated-delta values");
    hip_check(hipMemcpy(workspace.decays, decays, 48 * sizeof(float), hipMemcpyHostToDevice),
              "hipMemcpy gated-delta decays");
    hip_check(hipMemcpy(workspace.betas, betas, 48 * sizeof(float), hipMemcpyHostToDevice),
              "hipMemcpy gated-delta betas");
    hipLaunchKernelGGL(gated_delta_step_kernel, dim3(48), dim3(256), 0, nullptr,
                       workspace.queries, workspace.keys, workspace.values,
                       workspace.decays, workspace.betas, static_cast<__half *>(state),
                       workspace.output, 128);
    hip_check(hipGetLastError(), "gated-delta step launch");
    hip_check(hipMemcpy(output, workspace.output, vector_values * sizeof(float), hipMemcpyDeviceToHost),
              "hipMemcpy gated-delta output");
}

void execute_gated_delta_step_batch8(const float *queries, const float *keys,
                                     const float *values, const float *decays,
                                     const float *betas, void *state, float *output,
                                     int batch_count) {
    constexpr std::size_t vector_values = 48ULL * 128ULL;
    const std::size_t total_vectors = static_cast<std::size_t>(batch_count) * vector_values;
    struct Workspace {
        float *queries = nullptr;
        float *keys = nullptr;
        float *values = nullptr;
        float *decays = nullptr;
        float *betas = nullptr;
        float *output = nullptr;
        std::size_t capacity = 0;
        ~Workspace() {
            if (output) (void) hipFree(output);
            if (betas) (void) hipFree(betas);
            if (decays) (void) hipFree(decays);
            if (values) (void) hipFree(values);
            if (keys) (void) hipFree(keys);
            if (queries) (void) hipFree(queries);
        }
    };
    static Workspace workspace;
    if (workspace.capacity < total_vectors) {
        if (workspace.queries) {
            (void) hipFree(workspace.queries);
            (void) hipFree(workspace.keys);
            (void) hipFree(workspace.values);
            (void) hipFree(workspace.decays);
            (void) hipFree(workspace.betas);
            (void) hipFree(workspace.output);
        }
        hip_check(hipMalloc(reinterpret_cast<void **>(&workspace.queries), total_vectors * sizeof(float)),
                  "hipMalloc gated-delta-b8 queries");
        hip_check(hipMalloc(reinterpret_cast<void **>(&workspace.keys), total_vectors * sizeof(float)),
                  "hipMalloc gated-delta-b8 keys");
        hip_check(hipMalloc(reinterpret_cast<void **>(&workspace.values), total_vectors * sizeof(float)),
                  "hipMalloc gated-delta-b8 values");
        hip_check(hipMalloc(reinterpret_cast<void **>(&workspace.decays), static_cast<std::size_t>(batch_count) * 48 * sizeof(float)),
                  "hipMalloc gated-delta-b8 decays");
        hip_check(hipMalloc(reinterpret_cast<void **>(&workspace.betas), static_cast<std::size_t>(batch_count) * 48 * sizeof(float)),
                  "hipMalloc gated-delta-b8 betas");
        hip_check(hipMalloc(reinterpret_cast<void **>(&workspace.output), total_vectors * sizeof(float)),
                  "hipMalloc gated-delta-b8 output");
        workspace.capacity = total_vectors;
    }
    hip_check(hipMemcpy(workspace.queries, queries, total_vectors * sizeof(float), hipMemcpyHostToDevice),
              "hipMemcpy gated-delta-b8 queries");
    hip_check(hipMemcpy(workspace.keys, keys, total_vectors * sizeof(float), hipMemcpyHostToDevice),
              "hipMemcpy gated-delta-b8 keys");
    hip_check(hipMemcpy(workspace.values, values, total_vectors * sizeof(float), hipMemcpyHostToDevice),
              "hipMemcpy gated-delta-b8 values");
    hip_check(hipMemcpy(workspace.decays, decays, static_cast<std::size_t>(batch_count) * 48 * sizeof(float), hipMemcpyHostToDevice),
              "hipMemcpy gated-delta-b8 decays");
    hip_check(hipMemcpy(workspace.betas, betas, static_cast<std::size_t>(batch_count) * 48 * sizeof(float), hipMemcpyHostToDevice),
              "hipMemcpy gated-delta-b8 betas");
    hipLaunchKernelGGL(gated_delta_step_batch8_kernel, dim3(48), dim3(256), 0, nullptr,
                       workspace.queries, workspace.keys, workspace.values,
                       workspace.decays, workspace.betas, static_cast<__half *>(state),
                       workspace.output, 128, batch_count);
    hip_check(hipGetLastError(), "gated-delta-b8 step launch");
    hip_check(hipMemcpy(output, workspace.output, total_vectors * sizeof(float), hipMemcpyDeviceToHost),
              "hipMemcpy gated-delta-b8 output");
}

void preload_n23q4_matrix(const void *weights, std::size_t input_size, std::size_t output_size) {
    if (!weights || input_size == 0 || input_size % kQuantBlock != 0 || output_size == 0) {
        throw std::invalid_argument("invalid N23Q4 preload matrix");
    }
    auto &cache = n23_draft_cache();
    const auto found = cache.entries.find(weights);
    if (found != cache.entries.end()) {
        if (found->second.input_size != input_size || found->second.output_size != output_size) {
            throw std::runtime_error("N23Q4 cache matrix dimensions are inconsistent");
        }
        return;
    }
    const auto packed = convert_to_n23q4(static_cast<const Q4KBlock *>(weights), input_size, output_size);
    void *device = nullptr;
    if (hipMalloc(&device, packed.size() * sizeof(N23Q4TileGroup)) != hipSuccess) {
        return;
    }
    if (hipMemcpy(device, packed.data(), packed.size() * sizeof(N23Q4TileGroup), hipMemcpyHostToDevice) != hipSuccess) {
        (void) hipFree(device);
        return;
    }
    cache.entries.emplace(weights, N23DraftCache::Entry{device, input_size, output_size});
}

void execute_n23q4_batch8_cached(const void *weights, const float *input,
                                  std::size_t input_size, std::size_t output_size,
                                  float *output) {
    constexpr std::size_t batch_size = 8;
    if (!weights || !input || !output) {
        throw std::invalid_argument("N23Q4 batch execution requires non-null buffers");
    }
    preload_n23q4_matrix(weights, input_size, output_size);
    const auto &entry = n23_draft_cache().entries.at(weights);
    struct Workspace {
        float *input = nullptr;
        std::int8_t *quantized = nullptr;
        float *scales = nullptr;
        int *sums = nullptr;
        float *output = nullptr;
        std::size_t input_values = 0;
        std::size_t output_values = 0;
        ~Workspace() {
            if (output) (void) hipFree(output);
            if (sums) (void) hipFree(sums);
            if (scales) (void) hipFree(scales);
            if (quantized) (void) hipFree(quantized);
            if (input) (void) hipFree(input);
        }
        void reserve(std::size_t inputs, std::size_t outputs) {
            if (inputs > input_values) {
                if (output) (void) hipFree(output);
                if (sums) (void) hipFree(sums);
                if (scales) (void) hipFree(scales);
                if (quantized) (void) hipFree(quantized);
                if (input) (void) hipFree(input);
                input = nullptr; quantized = nullptr; scales = nullptr; sums = nullptr; output = nullptr;
                input_values = 0; output_values = 0;
                hip_check(hipMalloc(reinterpret_cast<void **>(&input), inputs * sizeof(float)), "hipMalloc N23Q4 draft input");
                hip_check(hipMalloc(reinterpret_cast<void **>(&quantized), inputs), "hipMalloc N23Q4 draft q8 input");
                hip_check(hipMalloc(reinterpret_cast<void **>(&scales), (inputs / 32) * sizeof(float)), "hipMalloc N23Q4 draft scales");
                hip_check(hipMalloc(reinterpret_cast<void **>(&sums), (inputs / 32) * sizeof(int)), "hipMalloc N23Q4 draft sums");
                input_values = inputs;
            }
            if (outputs > output_values) {
                if (output) (void) hipFree(output);
                output = nullptr;
                hip_check(hipMalloc(reinterpret_cast<void **>(&output), outputs * sizeof(float)), "hipMalloc N23Q4 draft output");
                output_values = outputs;
            }
        }
    };
    static Workspace workspace;
    workspace.reserve(batch_size * input_size, batch_size * output_size);
    hip_check(hipMemcpy(workspace.input, input, batch_size * input_size * sizeof(float), hipMemcpyHostToDevice),
              "hipMemcpy N23Q4 draft input");
    const std::size_t groups = batch_size * input_size / 32;
    hipLaunchKernelGGL(quantize_q8_groups_kernel, dim3(static_cast<unsigned int>(groups)), dim3(kWarpSize), 0, nullptr,
                       workspace.input, workspace.quantized, workspace.scales, workspace.sums,
                       static_cast<int>(input_size / 32));
    launch_n23q4_matvec_batch8(static_cast<const N23Q4TileGroup *>(entry.device), workspace.quantized,
                               workspace.scales, workspace.sums, workspace.output, input_size, output_size);
    hip_check(hipGetLastError(), "N23Q4 draft batch launch");
    hip_check(hipMemcpy(output, workspace.output, batch_size * output_size * sizeof(float), hipMemcpyDeviceToHost),
              "hipMemcpy N23Q4 draft output");
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

// ─── Device-resident pipelined executor ───────────────────────────────────────

__global__ void rms_norm_batch8_kernel(float *__restrict__ data,
                                       const float *__restrict__ weights,
                                       int width, float epsilon) {
    const int batch = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    extern __shared__ float shared[];

    float *row = data + static_cast<std::size_t>(batch) * width;
    float local_sum = 0.0F;
    for (int i = tid; i < width; i += static_cast<int>(blockDim.x)) {
        const float v = row[i];
        local_sum = fmaf(v, v, local_sum);
    }
    shared[tid] = local_sum;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(shared[0] / static_cast<float>(width) + epsilon);
    for (int i = tid; i < width; i += static_cast<int>(blockDim.x)) {
        row[i] *= scale * weights[i];
    }
}

__global__ void vector_add_batch8_kernel(float *__restrict__ dest,
                                         const float *__restrict__ source,
                                         int count) {
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    if (idx < count) {
        dest[idx] += source[idx];
    }
}

__global__ void vector_copy_kernel(float *__restrict__ dest,
                                   const float *__restrict__ source, int count) {
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    if (idx < count) {
        dest[idx] = source[idx];
    }
}

__global__ void silu_multiply_batch8_kernel(float *__restrict__ gate,
                                            const float *__restrict__ up, int count) {
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    if (idx < count) {
        const float v = gate[idx];
        gate[idx] = (v / (1.0F + expf(-v))) * up[idx];
    }
}

struct DevicePipeline {
    hipStream_t compute_stream = nullptr;
    hipStream_t transfer_stream = nullptr;

    float *hidden = nullptr;
    float *slots[4] = {};
    float *norm_weights_device = nullptr;

    void *weight_buf[2] = {};
    std::size_t weight_buf_capacity = 0;
    int active_weight_buf = 0;
    const void *prefetched_host_ptr[2] = {};
    std::size_t prefetched_bytes[2] = {};
    hipEvent_t weight_consumed[2] = {};
    bool weight_consumed_valid[2] = {};
    hipEvent_t transfer_done[2] = {};
    bool transfer_done_valid[2] = {};

    void *inline_buf = nullptr;
    std::size_t inline_buf_capacity = 0;
    const void *inline_prefetched_ptr = nullptr;
    std::size_t inline_prefetched_bytes = 0;
    hipEvent_t inline_done = nullptr;
    bool inline_done_valid = false;

    std::size_t embedding_size = 0;
    std::size_t ffn_size = 0;
    std::size_t max_weight_bytes = 0;

    ~DevicePipeline() {
        if (hidden) (void)hipFree(hidden);
        for (auto &s : slots) if (s) (void)hipFree(s);
        if (norm_weights_device) (void)hipFree(norm_weights_device);
        for (auto &w : weight_buf) if (w) (void)hipFree(w);
        for (auto &e : weight_consumed) if (e) (void)hipEventDestroy(e);
        for (auto &e : transfer_done) if (e) (void)hipEventDestroy(e);
        if (inline_buf) (void)hipFree(inline_buf);
        if (inline_done) (void)hipEventDestroy(inline_done);
        if (compute_stream) (void)hipStreamDestroy(compute_stream);
        if (transfer_stream) (void)hipStreamDestroy(transfer_stream);
    }
};

DevicePipeline *create_device_pipeline(std::size_t embedding_size, std::size_t ffn_size,
                                       std::size_t max_weight_bytes) {
    auto *p = new DevicePipeline();
    p->embedding_size = embedding_size;
    p->ffn_size = ffn_size;
    p->max_weight_bytes = max_weight_bytes;

    hip_check(hipStreamCreate(&p->compute_stream), "pipeline compute stream");
    hip_check(hipStreamCreate(&p->transfer_stream), "pipeline transfer stream");

    const std::size_t hidden_bytes = 8 * std::max(embedding_size, ffn_size) * sizeof(float);
    const std::size_t ffn_bytes = 8 * ffn_size * sizeof(float);
    hip_check(hipMalloc(reinterpret_cast<void **>(&p->hidden), hidden_bytes), "pipeline hidden");
    for (auto &s : p->slots) {
        hip_check(hipMalloc(reinterpret_cast<void **>(&s), ffn_bytes), "pipeline slot");
    }
    const std::size_t norm_bytes = std::max(embedding_size, ffn_size) * sizeof(float);
    hip_check(hipMalloc(reinterpret_cast<void **>(&p->norm_weights_device), norm_bytes), "pipeline norm");

    p->weight_buf_capacity = max_weight_bytes;
    for (auto &w : p->weight_buf) {
        hip_check(hipMalloc(&w, max_weight_bytes), "pipeline weight buf");
    }
    for (auto &e : p->weight_consumed) {
        hip_check(hipEventCreateWithFlags(&e, hipEventDisableTiming), "pipeline weight event");
    }
    for (auto &e : p->transfer_done) {
        hip_check(hipEventCreateWithFlags(&e, hipEventDisableTiming), "pipeline transfer done event");
    }
    const std::size_t inline_capacity = 36ULL * 1024 * 1024;
    hip_check(hipMalloc(&p->inline_buf, inline_capacity), "pipeline inline buf");
    p->inline_buf_capacity = inline_capacity;
    hip_check(hipEventCreateWithFlags(&p->inline_done, hipEventDisableTiming), "pipeline inline event");
    return p;
}

void destroy_device_pipeline(DevicePipeline *pipeline) {
    delete pipeline;
}

void pipeline_upload_hidden(DevicePipeline *pipeline, const float *host, std::size_t count) {
    hip_check(hipMemcpyAsync(pipeline->hidden, host, count * sizeof(float),
                             hipMemcpyHostToDevice, pipeline->compute_stream),
              "pipeline upload hidden");
}

void pipeline_download_hidden(DevicePipeline *pipeline, float *host, std::size_t count) {
    hip_check(hipStreamSynchronize(pipeline->compute_stream), "pipeline sync before download");
    hip_check(hipMemcpy(host, pipeline->hidden, count * sizeof(float), hipMemcpyDeviceToHost),
              "pipeline download hidden");
}

void pipeline_rms_norm(DevicePipeline *pipeline, const float *norm_weights,
                       std::size_t width, float epsilon) {
    hip_check(hipMemcpyAsync(pipeline->norm_weights_device, norm_weights,
                             width * sizeof(float), hipMemcpyHostToDevice,
                             pipeline->compute_stream),
              "pipeline upload norm weights");
    constexpr int threads = 256;
    const std::size_t shared = threads * sizeof(float);
    hipLaunchKernelGGL(rms_norm_batch8_kernel, dim3(8), dim3(threads), shared,
                       pipeline->compute_stream,
                       pipeline->hidden, pipeline->norm_weights_device,
                       static_cast<int>(width), epsilon);
}

void pipeline_residual_add(DevicePipeline *pipeline, int source_slot) {
    const int count = static_cast<int>(8 * pipeline->embedding_size);
    constexpr int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    hipLaunchKernelGGL(vector_add_batch8_kernel, dim3(blocks), dim3(threads), 0,
                       pipeline->compute_stream,
                       pipeline->hidden, pipeline->slots[source_slot], count);
}

void pipeline_save_hidden(DevicePipeline *pipeline, int slot, std::size_t count) {
    hip_check(hipMemcpyAsync(pipeline->slots[slot], pipeline->hidden,
                             count * sizeof(float), hipMemcpyDeviceToDevice,
                             pipeline->compute_stream),
              "pipeline save hidden");
}

void pipeline_slot_to_hidden(DevicePipeline *pipeline, int slot, std::size_t count) {
    hip_check(hipMemcpyAsync(pipeline->hidden, pipeline->slots[slot],
                             count * sizeof(float), hipMemcpyDeviceToDevice,
                             pipeline->compute_stream),
              "pipeline slot to hidden");
}

void pipeline_download_slot(DevicePipeline *pipeline, int slot, float *host, std::size_t count) {
    hip_check(hipStreamSynchronize(pipeline->compute_stream), "pipeline sync before slot download");
    hip_check(hipMemcpy(host, pipeline->slots[slot], count * sizeof(float), hipMemcpyDeviceToHost),
              "pipeline download slot");
}

void pipeline_download_slots3(DevicePipeline *pipeline,
                              int slot0, float *host0, std::size_t count0,
                              int slot1, float *host1, std::size_t count1,
                              int slot2, float *host2, std::size_t count2) {
    hip_check(hipStreamSynchronize(pipeline->compute_stream), "pipeline sync before multi-slot download");
    hip_check(hipMemcpy(host0, pipeline->slots[slot0], count0 * sizeof(float), hipMemcpyDeviceToHost),
              "pipeline download slot0");
    hip_check(hipMemcpy(host1, pipeline->slots[slot1], count1 * sizeof(float), hipMemcpyDeviceToHost),
              "pipeline download slot1");
    hip_check(hipMemcpy(host2, pipeline->slots[slot2], count2 * sizeof(float), hipMemcpyDeviceToHost),
              "pipeline download slot2");
}

void pipeline_project_batch8(DevicePipeline *pipeline, const void *weights,
                             std::uint32_t type, std::size_t input_size,
                             std::size_t output_size, bool keep_resident, int output_slot) {
    const std::size_t weight_bytes = k_quant_weight_bytes(type, input_size, output_size);
    auto &cache = resident_weight_cache();
    const void *device_weights = cache.find_or_add(weights, weight_bytes, keep_resident);
    int used_buf = -1;

    if (!device_weights) {
        int found_buf = -1;
        for (int b = 0; b < 2; ++b) {
            if (pipeline->prefetched_host_ptr[b] == weights &&
                pipeline->prefetched_bytes[b] == weight_bytes) {
                found_buf = b;
                break;
            }
        }
        if (found_buf >= 0) {
            if (pipeline->transfer_done_valid[found_buf]) {
                hip_check(hipEventSynchronize(pipeline->transfer_done[found_buf]),
                          "pipeline wait prefetch buf");
                pipeline->transfer_done_valid[found_buf] = false;
            }
            device_weights = pipeline->weight_buf[found_buf];
            pipeline->prefetched_host_ptr[found_buf] = nullptr;
            used_buf = found_buf;
        } else if (pipeline->inline_prefetched_ptr == weights &&
                   pipeline->inline_prefetched_bytes == weight_bytes) {
            if (pipeline->inline_done_valid) {
                hip_check(hipEventSynchronize(pipeline->inline_done),
                          "pipeline wait inline prefetch");
                pipeline->inline_done_valid = false;
            }
            device_weights = pipeline->inline_buf;
            pipeline->inline_prefetched_ptr = nullptr;
        } else {
            int buf = -1;
            for (int b = 0; b < 2; ++b) {
                if (pipeline->prefetched_host_ptr[b] == nullptr) {
                    buf = b;
                    break;
                }
            }
            if (buf < 0) {
                hip_check(hipStreamSynchronize(pipeline->transfer_stream), "pipeline wait all prefetch");
                pipeline->prefetched_host_ptr[0] = nullptr;
                pipeline->prefetched_host_ptr[1] = nullptr;
                buf = 0;
            }
            if (pipeline->weight_consumed_valid[buf]) {
                hip_check(hipEventSynchronize(pipeline->weight_consumed[buf]),
                          "pipeline sync event before upload");
            }
            hip_check(hipMemcpyAsync(pipeline->weight_buf[buf], weights, weight_bytes,
                                     hipMemcpyHostToDevice, pipeline->compute_stream),
                      "pipeline weight upload");
            device_weights = pipeline->weight_buf[buf];
            used_buf = buf;
        }
    }

    float *out = pipeline->slots[output_slot];
    const dim3 grid(static_cast<unsigned>(output_size));
    const dim3 block(kWarpSize);
    switch (type) {
    case 8:
        hipLaunchKernelGGL(q8_0_matvec_batch8_kernel, grid, block, 0, pipeline->compute_stream,
                           static_cast<const Q8_0Block *>(device_weights), pipeline->hidden,
                           out, static_cast<int>(input_size),
                           static_cast<int>(input_size / kQ8Block),
                           static_cast<int>(output_size));
        break;
    case 11:
        hipLaunchKernelGGL(q3k_matvec_batch8_kernel, grid, block, 0, pipeline->compute_stream,
                           static_cast<const Q3KBlock *>(device_weights), pipeline->hidden,
                           out, static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        break;
    case 12:
        hipLaunchKernelGGL(q4k_matvec_batch8_kernel, grid, block, 0, pipeline->compute_stream,
                           static_cast<const Q4KBlock *>(device_weights), pipeline->hidden,
                           out, static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        break;
    case 13:
        hipLaunchKernelGGL(q5k_matvec_batch8_kernel, grid, block, 0, pipeline->compute_stream,
                           static_cast<const Q5KBlock *>(device_weights), pipeline->hidden,
                           out, static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        break;
    case 14:
        hipLaunchKernelGGL(q6k_matvec_batch8_kernel, grid, block, 0, pipeline->compute_stream,
                           static_cast<const Q6KBlock *>(device_weights), pipeline->hidden,
                           out, static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        break;
    case 20:
        hipLaunchKernelGGL(iq4nl_matvec_batch8_kernel, grid, block, 0, pipeline->compute_stream,
                           static_cast<const IQ4NLBlock *>(device_weights), pipeline->hidden,
                           out, static_cast<int>(input_size),
                           static_cast<int>(input_size / kIQ4NLBlock),
                           static_cast<int>(output_size));
        break;
    case 21:
        hipLaunchKernelGGL(q3k_matvec_batch8_kernel, grid, block, 0, pipeline->compute_stream,
                           static_cast<const Q3KBlock *>(device_weights), pipeline->hidden,
                           out, static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        break;
    case 23:
        hipLaunchKernelGGL(iq4xs_matvec_batch8_kernel, grid, block, 0, pipeline->compute_stream,
                           static_cast<const IQ4XSBlock *>(device_weights), pipeline->hidden,
                           out, static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        break;
    default:
        throw std::invalid_argument("pipeline projection: unsupported type");
    }
    if (used_buf >= 0) {
        hip_check(hipEventRecord(pipeline->weight_consumed[used_buf], pipeline->compute_stream),
                  "pipeline record weight consumed");
        pipeline->weight_consumed_valid[used_buf] = true;
    }
}

void pipeline_silu_multiply(DevicePipeline *pipeline, int gate_slot, int up_slot,
                            std::size_t count) {
    constexpr int threads = 256;
    const int blocks = (static_cast<int>(count) + threads - 1) / threads;
    hipLaunchKernelGGL(silu_multiply_batch8_kernel, dim3(blocks), dim3(threads), 0,
                       pipeline->compute_stream,
                       pipeline->slots[gate_slot], pipeline->slots[up_slot],
                       static_cast<int>(count));
}

void pipeline_sync(DevicePipeline *pipeline) {
    hip_check(hipStreamSynchronize(pipeline->compute_stream), "pipeline sync compute");
    hip_check(hipStreamSynchronize(pipeline->transfer_stream), "pipeline sync transfer");
}

void pipeline_prefetch_weights(DevicePipeline *pipeline, const void *weights,
                               std::uint32_t type, std::size_t input_size,
                               std::size_t output_size) {
    const std::size_t weight_bytes = k_quant_weight_bytes(type, input_size, output_size);
    if (resident_weight_cache().find_or_add(weights, weight_bytes, false) != nullptr) {
        return;
    }
    if (weight_bytes > pipeline->weight_buf_capacity) {
        return;
    }
    int buf = pipeline->active_weight_buf;
    pipeline->active_weight_buf = 1 - buf;
    if (pipeline->weight_consumed_valid[buf]) {
        hip_check(hipStreamWaitEvent(pipeline->transfer_stream, pipeline->weight_consumed[buf], 0),
                  "pipeline prefetch wait event");
    }
    hip_check(hipMemcpyAsync(pipeline->weight_buf[buf], weights,
                             weight_bytes, hipMemcpyHostToDevice, pipeline->transfer_stream),
              "pipeline prefetch weights");
    hip_check(hipEventRecord(pipeline->transfer_done[buf], pipeline->transfer_stream),
              "pipeline prefetch record done");
    pipeline->transfer_done_valid[buf] = true;
    pipeline->prefetched_host_ptr[buf] = weights;
    pipeline->prefetched_bytes[buf] = weight_bytes;
}

void pipeline_prefetch_inline(DevicePipeline *pipeline, const void *weights,
                              std::uint32_t type, std::size_t input_size,
                              std::size_t output_size) {
    const std::size_t weight_bytes = k_quant_weight_bytes(type, input_size, output_size);
    if (resident_weight_cache().find_or_add(weights, weight_bytes, false) != nullptr) {
        return;
    }
    if (weight_bytes > pipeline->inline_buf_capacity) {
        return;
    }
    hip_check(hipMemcpyAsync(pipeline->inline_buf, weights,
                             weight_bytes, hipMemcpyHostToDevice, pipeline->transfer_stream),
              "pipeline inline prefetch");
    hip_check(hipEventRecord(pipeline->inline_done, pipeline->transfer_stream),
              "pipeline inline record done");
    pipeline->inline_done_valid = true;
    pipeline->inline_prefetched_ptr = weights;
    pipeline->inline_prefetched_bytes = weight_bytes;
}

void pipeline_fused_ffn_batch8(DevicePipeline *pipeline, const float *norm_weights,
                               std::size_t embedding_size, std::size_t ffn_size, float epsilon,
                               const void *gate_weights, std::uint32_t gate_type,
                               const void *up_weights, std::uint32_t up_type,
                               const void *down_weights, std::uint32_t down_type,
                               bool keep_gate_up_resident) {
    // Prefetch down weight early (overlaps with save/norm/gate compute)
    pipeline_prefetch_weights(pipeline, down_weights, down_type, ffn_size, embedding_size);

    // Save pre-FFN hidden for residual
    pipeline_save_hidden(pipeline, 3, 8 * embedding_size);

    // RMS norm in-place on hidden
    pipeline_rms_norm(pipeline, norm_weights, embedding_size, epsilon);

    // Gate projection: hidden → slot[0]
    pipeline_project_batch8(pipeline, gate_weights, gate_type, embedding_size, ffn_size,
                            keep_gate_up_resident, 0);

    // Up projection: hidden → slot[1]
    pipeline_project_batch8(pipeline, up_weights, up_type, embedding_size, ffn_size,
                            keep_gate_up_resident, 1);

    // SiLU gate * up → slot[0] (in-place on gate)
    pipeline_silu_multiply(pipeline, 0, 1, 8 * ffn_size);

    // Move gate result to hidden buffer for down projection
    // (down reads from hidden, writes to slot[2])
    hip_check(hipMemcpyAsync(pipeline->hidden, pipeline->slots[0],
                             8 * ffn_size * sizeof(float), hipMemcpyDeviceToDevice,
                             pipeline->compute_stream),
              "pipeline ffn move to hidden");
    pipeline_project_batch8(pipeline, down_weights, down_type, ffn_size, embedding_size,
                            false, 2);

    // Restore pre-FFN hidden and add down output
    hip_check(hipMemcpyAsync(pipeline->hidden, pipeline->slots[3],
                             8 * embedding_size * sizeof(float), hipMemcpyDeviceToDevice,
                             pipeline->compute_stream),
              "pipeline ffn restore hidden");
    pipeline_residual_add(pipeline, 2);
}

// ============================================================================
// GPU attention for the batch-8 target verifier (short-context GQA)
// ============================================================================

struct AttentionDeviceState {
    static constexpr int kHeads = 24;
    static constexpr int kKvHeads = 4;
    static constexpr int kHeadSize = 256;
    static constexpr int kKvWidth = kKvHeads * kHeadSize;
    static constexpr int kRotaryDims = 64;

    int max_context = 0;
    int active_tokens = 0;
    int global_position = 0;

    __half *kv_cache = nullptr;
    float *q_norm_weights = nullptr;
    float *k_norm_weights = nullptr;
    float *attn_output = nullptr;
    float *rope_cos = nullptr;

    ~AttentionDeviceState() {
        if (kv_cache) (void)hipFree(kv_cache);
        if (q_norm_weights) (void)hipFree(q_norm_weights);
        if (k_norm_weights) (void)hipFree(k_norm_weights);
        if (attn_output) (void)hipFree(attn_output);
        if (rope_cos) (void)hipFree(rope_cos);
    }
};

__global__ void attn_prep_kernel(const float *__restrict__ q_slot,
                                  const float *__restrict__ k_slot,
                                  const float *__restrict__ v_slot,
                                  const float *__restrict__ q_norm_w,
                                  const float *__restrict__ k_norm_w,
                                  const float *__restrict__ rope_freqs,
                                  __half *__restrict__ kv_cache,
                                  float *__restrict__ q_out,
                                  float *__restrict__ gate_out,
                                  int max_context, int write_offset,
                                  int rope_base_position, float epsilon) {
    constexpr int kHeads = 24;
    constexpr int kKvHeads = 4;
    constexpr int kHeadSize = 256;
    constexpr int kRotaryPairs = 32;
    constexpr int kQWidth = 2 * kHeads * kHeadSize;
    constexpr int kKvWidth = kKvHeads * kHeadSize;

    const int batch = static_cast<int>(blockIdx.x);
    const int position = rope_base_position + batch;

    const float *q_row = q_slot + static_cast<std::size_t>(batch) * kQWidth;
    const float *k_row = k_slot + static_cast<std::size_t>(batch) * kKvWidth;
    const float *v_row = v_slot + static_cast<std::size_t>(batch) * kKvWidth;

    const int tid = static_cast<int>(threadIdx.x);

    __shared__ float rope_cos[kRotaryPairs];
    __shared__ float rope_sin[kRotaryPairs];
    if (tid < kRotaryPairs) {
        float theta = static_cast<float>(position) * rope_freqs[tid];
        rope_cos[tid] = cosf(theta);
        rope_sin[tid] = sinf(theta);
    }
    __syncthreads();

    for (int qh = 0; qh < kHeads; ++qh) {
        const float *q_head = q_row + qh * 2 * kHeadSize;
        float *out_q = q_out + (static_cast<std::size_t>(batch) * kHeads + qh) * kHeadSize;
        float *out_gate = gate_out + (static_cast<std::size_t>(batch) * kHeads + qh) * kHeadSize;

        float sum_sq = 0.0F;
        for (int d = tid; d < kHeadSize; d += static_cast<int>(blockDim.x)) {
            float val = q_head[d];
            sum_sq += val * val;
        }
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
            sum_sq += __shfl_down(sum_sq, offset, kWarpSize);
        sum_sq = __shfl(sum_sq, 0, kWarpSize);
        const float q_rms = rsqrtf(sum_sq / static_cast<float>(kHeadSize) + epsilon);

        for (int d = tid; d < kHeadSize; d += static_cast<int>(blockDim.x)) {
            float normed = q_head[d] * q_rms * q_norm_w[d];
            if (d < kRotaryPairs) {
                float right = q_head[d + kRotaryPairs] * q_rms * q_norm_w[d + kRotaryPairs];
                out_q[d] = normed * rope_cos[d] - right * rope_sin[d];
                out_q[d + kRotaryPairs] = normed * rope_sin[d] + right * rope_cos[d];
            } else if (d >= 2 * kRotaryPairs) {
                out_q[d] = normed;
            }
            out_gate[d] = q_head[kHeadSize + d];
        }
    }

    const int slot = write_offset + batch;
    for (int kvh = 0; kvh < kKvHeads; ++kvh) {
        const float *k_head = k_row + kvh * kHeadSize;
        const std::size_t cache_offset = (static_cast<std::size_t>(kvh) * max_context + slot) * kHeadSize;

        float sum_sq = 0.0F;
        for (int d = tid; d < kHeadSize; d += static_cast<int>(blockDim.x)) {
            float val = k_head[d];
            sum_sq += val * val;
        }
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
            sum_sq += __shfl_down(sum_sq, offset, kWarpSize);
        sum_sq = __shfl(sum_sq, 0, kWarpSize);
        const float k_rms = rsqrtf(sum_sq / static_cast<float>(kHeadSize) + epsilon);

        for (int d = tid; d < kHeadSize; d += static_cast<int>(blockDim.x)) {
            float normed = k_head[d] * k_rms * k_norm_w[d];
            if (d < kRotaryPairs) {
                float right = k_head[d + kRotaryPairs] * k_rms * k_norm_w[d + kRotaryPairs];
                kv_cache[cache_offset + d] = __float2half(normed * rope_cos[d] - right * rope_sin[d]);
                kv_cache[cache_offset + d + kRotaryPairs] = __float2half(normed * rope_sin[d] + right * rope_cos[d]);
            } else if (d >= 2 * kRotaryPairs) {
                kv_cache[cache_offset + d] = __float2half(normed);
            }
        }

        const float *v_head = v_row + kvh * kHeadSize;
        const std::size_t v_offset = (static_cast<std::size_t>(kKvHeads) * max_context * kHeadSize)
                                   + (static_cast<std::size_t>(kvh) * max_context + slot) * kHeadSize;
        for (int d = tid; d < kHeadSize; d += static_cast<int>(blockDim.x)) {
            kv_cache[v_offset + d] = __float2half(v_head[d]);
        }
    }
}

__global__ void attn_score_kernel(const float *__restrict__ queries,
                                   const float *__restrict__ gates,
                                   const __half *__restrict__ kv_cache,
                                   float *__restrict__ output,
                                   int base_position, int max_context) {
    constexpr int kHeads = 24;
    constexpr int kKvHeads = 4;
    constexpr int kHeadSize = 256;
    constexpr int kQPerKv = kHeads / kKvHeads;
    constexpr int kThreads = 256;

    const int batch = static_cast<int>(blockIdx.x);
    const int query_head = static_cast<int>(blockIdx.y);
    const int kv_head = query_head / kQPerKv;
    const int tid = static_cast<int>(threadIdx.x);
    const int context_size = base_position + batch + 1;

    const float *q = queries + (static_cast<std::size_t>(batch) * kHeads + query_head) * kHeadSize;
    const float *gate = gates + (static_cast<std::size_t>(batch) * kHeads + query_head) * kHeadSize;
    float *out = output + (static_cast<std::size_t>(batch) * kHeads + query_head) * kHeadSize;

    const __half *keys = kv_cache + static_cast<std::size_t>(kv_head) * max_context * kHeadSize;
    const __half *vals = kv_cache
        + static_cast<std::size_t>(kKvHeads) * max_context * kHeadSize
        + static_cast<std::size_t>(kv_head) * max_context * kHeadSize;

    constexpr float scale = 1.0F / 16.0F;

    extern __shared__ float smem[];
    float *scores = smem;

    // Phase 1: compute all attention scores (threads tile over tokens)
    for (int t = tid; t < context_size; t += kThreads) {
        const __half *k_token = keys + static_cast<std::size_t>(t) * kHeadSize;
        float dot = 0.0F;
        for (int d = 0; d < kHeadSize; ++d) {
            dot = fmaf(q[d], __half2float(k_token[d]), dot);
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Phase 2: softmax (first warp finds max and sum)
    __shared__ float s_max;
    __shared__ float s_inv_sum;
    if (tid < kWarpSize) {
        float local_max = -1.0e30F;
        for (int t = tid; t < context_size; t += kWarpSize) {
            if (scores[t] > local_max) local_max = scores[t];
        }
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
            float other = __shfl_down(local_max, offset, kWarpSize);
            if (other > local_max) local_max = other;
        }
        local_max = __shfl(local_max, 0, kWarpSize);
        if (tid == 0) s_max = local_max;
    }
    __syncthreads();

    for (int t = tid; t < context_size; t += kThreads) {
        scores[t] = expf(scores[t] - s_max);
    }
    __syncthreads();

    if (tid < kWarpSize) {
        float local_sum = 0.0F;
        for (int t = tid; t < context_size; t += kWarpSize) {
            local_sum += scores[t];
        }
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
            local_sum += __shfl_down(local_sum, offset, kWarpSize);
        if (tid == 0) s_inv_sum = 1.0F / __shfl(local_sum, 0, kWarpSize);
    }
    __syncthreads();

    // Phase 3: weighted sum of values (threads tile over output dimensions)
    const float inv_sum = s_inv_sum;
    for (int d = tid; d < kHeadSize; d += kThreads) {
        float accum = 0.0F;
        for (int t = 0; t < context_size; ++t) {
            accum += scores[t] * __half2float(vals[static_cast<std::size_t>(t) * kHeadSize + d]);
        }
        float g = gate[d];
        g = 1.0F / (1.0F + expf(-g));
        out[d] = accum * inv_sum * g;
    }
}

AttentionDeviceState *create_attention_device_state(int num_layers, int max_context,
                                                     float rope_freq_base) {
    auto *s = new AttentionDeviceState();
    s->max_context = max_context;
    s->active_tokens = 0;

    constexpr int kKvHeads = AttentionDeviceState::kKvHeads;
    constexpr int kHeadSize = AttentionDeviceState::kHeadSize;
    constexpr int kHeads = AttentionDeviceState::kHeads;
    constexpr int kRotaryPairs = AttentionDeviceState::kRotaryDims / 2;

    const std::size_t kv_per_layer = 2ULL * kKvHeads * max_context * kHeadSize;
    const std::size_t kv_bytes = num_layers * kv_per_layer * sizeof(__half);
    hip_check(hipMalloc(reinterpret_cast<void **>(&s->kv_cache), kv_bytes), "attn kv cache");
    hip_check(hipMemset(s->kv_cache, 0, kv_bytes), "attn kv cache zero");

    hip_check(hipMalloc(reinterpret_cast<void **>(&s->q_norm_weights), kHeadSize * sizeof(float)), "attn q norm");
    hip_check(hipMalloc(reinterpret_cast<void **>(&s->k_norm_weights), kHeadSize * sizeof(float)), "attn k norm");

    const std::size_t out_bytes = 2ULL * 8 * kHeads * kHeadSize * sizeof(float);
    hip_check(hipMalloc(reinterpret_cast<void **>(&s->attn_output), out_bytes), "attn output");

    hip_check(hipMalloc(reinterpret_cast<void **>(&s->rope_cos), kRotaryPairs * sizeof(float)), "attn rope freq");

    std::array<float, kRotaryPairs> freq_table{};
    float freq = 1.0F;
    const float freq_step = std::pow(rope_freq_base, -2.0F / static_cast<float>(kRotaryPairs * 2));
    for (int p = 0; p < kRotaryPairs; ++p) {
        freq_table[p] = freq;
        freq *= freq_step;
    }
    hip_check(hipMemcpy(s->rope_cos, freq_table.data(),
                        kRotaryPairs * sizeof(float), hipMemcpyHostToDevice), "attn rope freq upload");

    return s;
}

void destroy_attention_device_state(AttentionDeviceState *state) {
    delete state;
}

void attention_reset(AttentionDeviceState *state) {
    state->active_tokens = 0;
    state->global_position = 0;
}

static void attention_shift_all_layers(AttentionDeviceState *state, hipStream_t stream) {
    constexpr int kKvHeads = AttentionDeviceState::kKvHeads;
    constexpr int kHeadSize = AttentionDeviceState::kHeadSize;
    constexpr int kBatch = 8;
    constexpr int kNumLayers = 16;

    const int entries_to_keep = state->max_context - kBatch;
    const std::size_t entry_bytes = static_cast<std::size_t>(kHeadSize) * sizeof(__half);

    for (int layer = 0; layer < kNumLayers; ++layer) {
        const std::size_t layer_offset = static_cast<std::size_t>(layer) * 2ULL
            * kKvHeads * state->max_context * kHeadSize;
        __half *layer_kv = state->kv_cache + layer_offset;
        for (int kv = 0; kv < 2; ++kv) {
            for (int h = 0; h < kKvHeads; ++h) {
                __half *head_base = layer_kv + (static_cast<std::size_t>(kv) * kKvHeads + h)
                    * state->max_context * kHeadSize;
                hip_check(hipMemcpyAsync(head_base, head_base + kBatch * kHeadSize,
                    static_cast<std::size_t>(entries_to_keep) * entry_bytes,
                    hipMemcpyDeviceToDevice, stream), "attn shift");
            }
        }
    }
    state->active_tokens = entries_to_keep;
}

void pipeline_attention_batch8(DevicePipeline *pipeline, AttentionDeviceState *attn,
                               int layer_index,
                               const float *q_norm_weights, const float *k_norm_weights,
                               float epsilon) {
    constexpr int kHeads = AttentionDeviceState::kHeads;
    constexpr int kKvHeads = AttentionDeviceState::kKvHeads;
    constexpr int kHeadSize = AttentionDeviceState::kHeadSize;

    if (layer_index == 0 && attn->active_tokens + 8 > attn->max_context) {
        attention_shift_all_layers(attn, pipeline->compute_stream);
    }

    const int write_offset = attn->active_tokens;
    const int rope_base = attn->global_position;
    const int context_size = write_offset + 8;

    hip_check(hipMemcpyAsync(attn->q_norm_weights, q_norm_weights,
                             kHeadSize * sizeof(float), hipMemcpyHostToDevice,
                             pipeline->compute_stream), "attn upload q norm");
    hip_check(hipMemcpyAsync(attn->k_norm_weights, k_norm_weights,
                             kHeadSize * sizeof(float), hipMemcpyHostToDevice,
                             pipeline->compute_stream), "attn upload k norm");

    const std::size_t layer_kv_offset = static_cast<std::size_t>(layer_index) * 2ULL
        * kKvHeads * attn->max_context * kHeadSize;
    __half *layer_kv = attn->kv_cache + layer_kv_offset;

    float *q_buf = attn->attn_output;
    float *gate_buf = attn->attn_output + 8ULL * kHeads * kHeadSize;

    hipLaunchKernelGGL(attn_prep_kernel, dim3(8), dim3(kWarpSize), 0, pipeline->compute_stream,
                       pipeline->slots[0], pipeline->slots[1], pipeline->slots[2],
                       attn->q_norm_weights, attn->k_norm_weights,
                       attn->rope_cos,
                       layer_kv, q_buf, gate_buf,
                       attn->max_context, write_offset, rope_base, epsilon);

    const std::size_t smem_bytes = static_cast<std::size_t>(context_size) * sizeof(float);
    hipLaunchKernelGGL(attn_score_kernel, dim3(8, kHeads), dim3(256), smem_bytes,
                       pipeline->compute_stream,
                       q_buf, gate_buf, layer_kv, pipeline->slots[0],
                       write_offset, attn->max_context);
}

void attention_advance(AttentionDeviceState *state, int tokens) {
    state->active_tokens += tokens;
    state->global_position += tokens;
}

struct RecurrentDeviceState {
    int num_layers = 0;
    int qkv_width = 0;
    int recurrent_width = 0;
    int embedding_size = 0;
    int value_heads = 0;
    int head_size = 0;

    std::vector<float *> conv_states;
    std::vector<float *> conv_weights;
    std::vector<float *> beta_weights;
    std::vector<float *> alpha_weights;
    std::vector<float *> dt_bias;
    std::vector<float *> a_weights;
    std::vector<float *> norm_weights;

    float *queries = nullptr;
    float *keys = nullptr;
    float *values = nullptr;
    float *gdn_output = nullptr;
    float *decays = nullptr;
    float *betas = nullptr;
};

RecurrentDeviceState *create_recurrent_device_state(int num_layers, int qkv_width,
                                                    int recurrent_width, int embedding_size,
                                                    int value_heads, int head_size) {
    auto *s = new RecurrentDeviceState();
    s->num_layers = num_layers;
    s->qkv_width = qkv_width;
    s->recurrent_width = recurrent_width;
    s->embedding_size = embedding_size;
    s->value_heads = value_heads;
    s->head_size = head_size;

    s->conv_states.resize(num_layers);
    s->conv_weights.resize(num_layers);
    s->beta_weights.resize(num_layers);
    s->alpha_weights.resize(num_layers);
    s->dt_bias.resize(num_layers);
    s->a_weights.resize(num_layers);
    s->norm_weights.resize(num_layers);

    const std::size_t conv_state_bytes = 3ULL * qkv_width * sizeof(float);
    const std::size_t conv_weight_bytes = 4ULL * qkv_width * sizeof(float);
    const std::size_t proj_weight_bytes = static_cast<std::size_t>(embedding_size) * value_heads * sizeof(float);
    const std::size_t small_bytes = value_heads * sizeof(float);
    const std::size_t norm_bytes = head_size * sizeof(float);

    for (int i = 0; i < num_layers; ++i) {
        hip_check(hipMalloc(reinterpret_cast<void **>(&s->conv_states[i]), conv_state_bytes), "recurrent conv_state");
        hip_check(hipMemset(s->conv_states[i], 0, conv_state_bytes), "recurrent conv_state zero");
        hip_check(hipMalloc(reinterpret_cast<void **>(&s->conv_weights[i]), conv_weight_bytes), "recurrent conv_weights");
        hip_check(hipMalloc(reinterpret_cast<void **>(&s->beta_weights[i]), proj_weight_bytes), "recurrent beta_weights");
        hip_check(hipMalloc(reinterpret_cast<void **>(&s->alpha_weights[i]), proj_weight_bytes), "recurrent alpha_weights");
        hip_check(hipMalloc(reinterpret_cast<void **>(&s->dt_bias[i]), small_bytes), "recurrent dt_bias");
        hip_check(hipMalloc(reinterpret_cast<void **>(&s->a_weights[i]), small_bytes), "recurrent a_weights");
        hip_check(hipMalloc(reinterpret_cast<void **>(&s->norm_weights[i]), norm_bytes), "recurrent norm_weights");
    }

    const std::size_t qkv_batch = 8ULL * recurrent_width * sizeof(float);
    hip_check(hipMalloc(reinterpret_cast<void **>(&s->queries), qkv_batch), "recurrent queries");
    hip_check(hipMalloc(reinterpret_cast<void **>(&s->keys), qkv_batch), "recurrent keys");
    hip_check(hipMalloc(reinterpret_cast<void **>(&s->values), qkv_batch), "recurrent values");
    hip_check(hipMalloc(reinterpret_cast<void **>(&s->gdn_output), qkv_batch), "recurrent gdn_output");
    const std::size_t db_bytes = 8ULL * value_heads * sizeof(float);
    hip_check(hipMalloc(reinterpret_cast<void **>(&s->decays), db_bytes), "recurrent decays");
    hip_check(hipMalloc(reinterpret_cast<void **>(&s->betas), db_bytes), "recurrent betas");

    return s;
}

void destroy_recurrent_device_state(RecurrentDeviceState *s) {
    if (!s) return;
    for (auto *p : s->conv_states) if (p) (void)hipFree(p);
    for (auto *p : s->conv_weights) if (p) (void)hipFree(p);
    for (auto *p : s->beta_weights) if (p) (void)hipFree(p);
    for (auto *p : s->alpha_weights) if (p) (void)hipFree(p);
    for (auto *p : s->dt_bias) if (p) (void)hipFree(p);
    for (auto *p : s->a_weights) if (p) (void)hipFree(p);
    for (auto *p : s->norm_weights) if (p) (void)hipFree(p);
    if (s->queries) (void)hipFree(s->queries);
    if (s->keys) (void)hipFree(s->keys);
    if (s->values) (void)hipFree(s->values);
    if (s->gdn_output) (void)hipFree(s->gdn_output);
    if (s->decays) (void)hipFree(s->decays);
    if (s->betas) (void)hipFree(s->betas);
    delete s;
}

void recurrent_upload_weights(RecurrentDeviceState *s, int layer,
                              const float *conv_weights, const float *beta_weights,
                              const float *alpha_weights, const float *dt_bias,
                              const float *a_weights, const float *norm_weights) {
    const std::size_t conv_bytes = 4ULL * s->qkv_width * sizeof(float);
    const std::size_t proj_bytes = static_cast<std::size_t>(s->embedding_size) * s->value_heads * sizeof(float);
    const std::size_t small_bytes = s->value_heads * sizeof(float);
    const std::size_t norm_bytes = s->head_size * sizeof(float);
    hip_check(hipMemcpy(s->conv_weights[layer], conv_weights, conv_bytes, hipMemcpyHostToDevice), "upload conv_weights");
    hip_check(hipMemcpy(s->beta_weights[layer], beta_weights, proj_bytes, hipMemcpyHostToDevice), "upload beta_weights");
    hip_check(hipMemcpy(s->alpha_weights[layer], alpha_weights, proj_bytes, hipMemcpyHostToDevice), "upload alpha_weights");
    hip_check(hipMemcpy(s->dt_bias[layer], dt_bias, small_bytes, hipMemcpyHostToDevice), "upload dt_bias");
    hip_check(hipMemcpy(s->a_weights[layer], a_weights, small_bytes, hipMemcpyHostToDevice), "upload a_weights");
    hip_check(hipMemcpy(s->norm_weights[layer], norm_weights, norm_bytes, hipMemcpyHostToDevice), "upload norm_weights");
}

void recurrent_reset_conv_states(RecurrentDeviceState *s) {
    const std::size_t conv_state_bytes = 3ULL * s->qkv_width * sizeof(float);
    for (int i = 0; i < s->num_layers; ++i) {
        hip_check(hipMemset(s->conv_states[i], 0, conv_state_bytes), "reset conv_state");
    }
}

void pipeline_recurrent_batch8(DevicePipeline *pipeline, RecurrentDeviceState *recurrent,
                               int layer_index, void *gdn_state, float epsilon) {
    const int qkv_width = recurrent->qkv_width;
    const int recurrent_width = recurrent->recurrent_width;
    const int value_heads = recurrent->value_heads;
    const int head_size = recurrent->head_size;
    const int actual_key_heads = (qkv_width - value_heads * head_size) / (2 * head_size);

    // Conv1d kernel: process QKV from slot[0], write queries/keys/values
    {
        constexpr int threads = 256;
        const int blocks = (qkv_width + threads - 1) / threads;
        hipLaunchKernelGGL(recurrent_conv1d_batch8_kernel, dim3(blocks), dim3(threads),
                           0, pipeline->compute_stream,
                           pipeline->slots[0],
                           recurrent->conv_states[layer_index],
                           recurrent->conv_weights[layer_index],
                           recurrent->queries, recurrent->keys, recurrent->values,
                           qkv_width, actual_key_heads, value_heads, head_size);
        hip_check(hipGetLastError(), "recurrent conv1d launch");
    }


    // L2 normalize queries and keys
    hipLaunchKernelGGL(recurrent_l2_norm_batch8_kernel, dim3(value_heads), dim3(128),
                       0, pipeline->compute_stream,
                       recurrent->queries, recurrent->keys,
                       recurrent_width, head_size, epsilon);
    hip_check(hipGetLastError(), "recurrent l2_norm launch");

    // Decay/beta from hidden state
    hipLaunchKernelGGL(recurrent_decay_beta_batch8_kernel, dim3(value_heads), dim3(256),
                       0, pipeline->compute_stream,
                       pipeline->hidden,
                       recurrent->beta_weights[layer_index],
                       recurrent->alpha_weights[layer_index],
                       recurrent->dt_bias[layer_index],
                       recurrent->a_weights[layer_index],
                       recurrent->decays, recurrent->betas,
                       recurrent->embedding_size);
    hip_check(hipGetLastError(), "recurrent decay_beta launch");


    // Gated delta step on device (no memcpy!)
    hipLaunchKernelGGL(gated_delta_step_batch8_kernel, dim3(48), dim3(256),
                       0, pipeline->compute_stream,
                       recurrent->queries, recurrent->keys, recurrent->values,
                       recurrent->decays, recurrent->betas,
                       static_cast<__half *>(gdn_state),
                       recurrent->gdn_output, head_size, 8);
    hip_check(hipGetLastError(), "recurrent gdn launch");


    // SSM norm + silu(gate) → pipeline hidden
    hipLaunchKernelGGL(recurrent_ssm_norm_gate_batch8_kernel, dim3(value_heads), dim3(128),
                       0, pipeline->compute_stream,
                       recurrent->gdn_output, pipeline->slots[1],
                       recurrent->norm_weights[layer_index],
                       pipeline->hidden,
                       recurrent_width, value_heads, head_size, epsilon);
    hip_check(hipGetLastError(), "recurrent ssm_norm_gate launch");
}

} // namespace qwen_native

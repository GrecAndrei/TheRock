#include "kernels.h"
#include "cpu_matvec.h"
#include "iq3s_codebook.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "attention_tile.h"
#include "runtime_options.h"
#include "single_job_worker.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
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

#pragma pack(push, 1)
struct IQ3SBlock {
    std::uint16_t d;
    std::uint8_t qs[64];
    std::uint8_t qh[8];
    std::uint8_t signs[32];
    std::uint8_t scales[4];
};
#pragma pack(pop)
static_assert(sizeof(IQ3SBlock) == 110);

__device__ __constant__ std::uint16_t kIq3SCodebookDevice[512] = {
    QWEN_IQ3S_CODEBOOK_VALUES
};
#undef QWEN_IQ3S_CODEBOOK_VALUES

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
    int *topk_indices = nullptr;
    float *topk_scores = nullptr;
    std::size_t topk_capacity = 0;

    ~MatvecWorkspace() {
        if (topk_scores) (void) hipFree(topk_scores);
        if (topk_indices) (void) hipFree(topk_indices);
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

    void reserve_topk(std::size_t entries) {
        if (entries <= topk_capacity) return;
        if (topk_indices) {
            hip_check(hipFree(topk_indices), "hipFree native matvec topk indices");
            topk_indices = nullptr;
        }
        if (topk_scores) {
            hip_check(hipFree(topk_scores), "hipFree native matvec topk scores");
            topk_scores = nullptr;
        }
        topk_capacity = 0;
        hip_check(hipMalloc(reinterpret_cast<void **>(&topk_indices), entries * sizeof(int)), "hipMalloc native matvec topk indices");
        hip_check(hipMalloc(reinterpret_cast<void **>(&topk_scores), entries * sizeof(float)), "hipMalloc native matvec topk scores");
        topk_capacity = entries;
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
        for (const auto &[host, entry] : temporary_entries) {
            (void) host;
            if (entry.device) (void) hipFree(entry.device);
        }
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
        const auto staged = temporary_entries.find(host);
        if (staged != temporary_entries.end()) {
            if (staged->second.bytes != bytes) {
                throw std::runtime_error("temporary weight-cache pointer has inconsistent size");
            }
            return staged->second.device;
        }
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

    bool add_temporary(const void *host, std::size_t bytes) {
        if (find_or_add(host, bytes, false)) return false;
        void *device = nullptr;
        hip_check(hipMalloc(&device, bytes), "allocate temporary layer weights");
        const auto status = hipMemcpy(device, host, bytes, hipMemcpyHostToDevice);
        if (status != hipSuccess) {
            (void)hipFree(device);
            hip_check(status, "upload temporary layer weights");
        }
        try {
            temporary_entries.emplace(host, Entry{device, bytes});
        } catch (...) {
            (void)hipFree(device);
            throw;
        }
        return true;
    }

    void release_temporary(const void *host) noexcept {
        const auto entry = temporary_entries.find(host);
        if (entry == temporary_entries.end()) return;
        (void)hipFree(entry->second.device);
        temporary_entries.erase(entry);
    }

    void clear() {
        for (const auto &[host, entry] : temporary_entries) {
            (void)host;
            if (entry.device) hip_check(hipFree(entry.device), "clear temporary weight cache");
        }
        temporary_entries.clear();
        for (const auto &[host, entry] : entries) {
            (void) host;
            if (entry.device) hip_check(hipFree(entry.device), "clear resident weight cache");
        }
        entries.clear();
        resident_bytes = 0;
        budget_bytes = 0;
    }

    ResidentWeightCacheStats stats() const {
        return {budget_bytes, resident_bytes, entries.size()};
    }

    std::size_t budget_bytes = 0;
    std::size_t resident_bytes = 0;
    std::unordered_map<const void *, Entry> entries;
    std::unordered_map<const void *, Entry> temporary_entries;
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
    const int row = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
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
            // Each 128-value stripe occupies 64 packed low-nibble bytes but
            // advances 128 values in the activation row.
            const int quant_offset = half * 64;
            const int value_offset = half * 128;
            const std::uint8_t low0 = block.low_quants[quant_offset + lane];
            const std::uint8_t low1 = block.low_quants[quant_offset + lane + 32];
            const std::uint8_t high = block.high_quants[half * 32 + lane];
            // Q6_K stores one signed scale per 16 weights. Lanes 0..15
            // therefore use the even scale slots and lanes 16..31 the odd
            // slots for each of the four 32-value stripes.
            const int scale_offset = half * 8 + lane / 16;
            const int code0 = static_cast<int>(low0 & 0x0F) + ((high & 0x03) << 4) - 32;
            const int code1 = static_cast<int>(low1 & 0x0F) + (((high >> 2) & 0x03) << 4) - 32;
            const int code2 = static_cast<int>(low0 >> 4) + (((high >> 4) & 0x03) << 4) - 32;
            const int code3 = static_cast<int>(low1 >> 4) + (((high >> 6) & 0x03) << 4) - 32;
            sum = fmaf(scale_all * static_cast<float>(block.scales[scale_offset]) * static_cast<float>(code0),
                       block_input[value_offset + lane], sum);
            sum = fmaf(scale_all * static_cast<float>(block.scales[scale_offset + 2]) * static_cast<float>(code1),
                       block_input[value_offset + lane + 32], sum);
            sum = fmaf(scale_all * static_cast<float>(block.scales[scale_offset + 4]) * static_cast<float>(code2),
                       block_input[value_offset + lane + 64], sum);
            sum = fmaf(scale_all * static_cast<float>(block.scales[scale_offset + 6]) * static_cast<float>(code3),
                       block_input[value_offset + lane + 96], sum);
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
            const int quant_offset = half * 64;
            const int value_offset = half * 128;
            const std::uint8_t low0 = block.low_quants[quant_offset + lane];
            const std::uint8_t low1 = block.low_quants[quant_offset + lane + 32];
            const std::uint8_t high = block.high_quants[half * 32 + lane];
            const int scale_offset = half * 8 + lane / 16;
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
                const int base = block_index * kQuantBlock + value_offset + lane;
                sums[batch] = fmaf(weight0, batch_input[base], sums[batch]);
                sums[batch] = fmaf(weight1, batch_input[base + 32], sums[batch]);
                sums[batch] = fmaf(weight2, batch_input[base + 64], sums[batch]);
                sums[batch] = fmaf(weight3, batch_input[base + 96], sums[batch]);
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

// GPU top-16 reduction for output.head logits: each block handles one of the
// 8 batch positions. Each thread scans its strided vocab slice maintaining a
// local 16-element heap, then block-shared merge yields global top-16.
// Downloads 128 floats + 128 ints (640 bytes) instead of 8*248k floats (~8 MB).
__global__ void logits_top16_kernel(const float *__restrict__ logits,
                                    int vocab, int *__restrict__ out_indices,
                                    float *__restrict__ out_scores) {
    const int batch = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x);
    const float *row = logits + static_cast<std::size_t>(batch) * vocab;
    // Local top-16 per thread in registers (score, idx), sorted descending.
    float local_scores[16];
    int local_indices[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) { local_scores[k] = -1e30f; local_indices[k] = -1; }
    for (int i = tid; i < vocab; i += threads) {
        float v = row[i];
        if (v > local_scores[15]) {
            int p = 15;
            while (p > 0 && v > local_scores[p-1]) {
                local_scores[p] = local_scores[p-1];
                local_indices[p] = local_indices[p-1];
                --p;
            }
            local_scores[p] = v;
            local_indices[p] = i;
        }
    }
    // Shared merge area: 256*16 = 4096 entries per block (if 256 threads)
    __shared__ float shared_scores[256*16];
    __shared__ int shared_indices[256*16];
    const int base = tid * 16;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        shared_scores[base + k] = local_scores[k];
        shared_indices[base + k] = local_indices[k];
    }
    __syncthreads();
    if (tid == 0) {
        float global_scores[16];
        int global_indices[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) { global_scores[k] = -1e30f; global_indices[k] = -1; }
        const int total = threads * 16;
        for (int i = 0; i < total; ++i) {
            float v = shared_scores[i];
            int idx = shared_indices[i];
            if (idx < 0) continue;
            if (v > global_scores[15]) {
                int p = 15;
                while (p > 0 && v > global_scores[p-1]) {
                    global_scores[p] = global_scores[p-1];
                    global_indices[p] = global_indices[p-1];
                    --p;
                }
                global_scores[p] = v;
                global_indices[p] = idx;
            }
        }
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            out_indices[static_cast<std::size_t>(batch) * 16 + k] = global_indices[k];
            out_scores[static_cast<std::size_t>(batch) * 16 + k] = global_scores[k];
        }
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
        low4 = scales[4 + j] & 0xf;
        high2 = (scales[8 + j] >> 2) & 3;
    } else if (sb < 12) {
        low4 = scales[j] >> 4;
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
            // Each 32-byte q stripe carries four successive 2-bit planes.
            // The second stripe starts at value 128; hmask advances once per
            // 32 decoded weights.
            const int stripe = p / 32;
            const int qs_index = (stripe / 4) * 32 + lane;
            const int shift = (stripe & 3) * 2;
            const int low2 = (block.qs[qs_index] >> shift) & 3;
            const int high1 = (block.hmask[lane] >> stripe) & 1;
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

__device__ float iq3s_decode_weight(const IQ3SBlock &block, int position) {
    const int block32 = position / 32;
    const int pair = block32 / 2;
    const int half = block32 & 1;
    const int local = position & 31;
    const int group8 = local / 8;
    const int item8 = local & 7;
    const int grid_half = item8 / 4;
    const int coordinate = item8 & 3;
    const int q_offset = pair * 16 + half * 8 + group8 * 2 + grid_half;
    const int high_bit = (block.qh[pair * 2 + half]
                        >> (2 * group8 + grid_half)) & 1;
    const int codebook_index = block.qs[q_offset] | (high_bit << 8);
    const std::uint16_t lattice = kIq3SCodebookDevice[codebook_index];
    int magnitude = 2 * ((lattice >> (3 * coordinate)) & 7) + 1;
    if (block.signs[pair * 8 + half * 4 + group8] & (1U << item8)) {
        magnitude = -magnitude;
    }
    const int nibble = half == 0 ? block.scales[pair] & 15
                                 : block.scales[pair] >> 4;
    const float d = __half2float(*reinterpret_cast<const __half *>(&block.d));
    return d * static_cast<float>(1 + 2 * nibble) * static_cast<float>(magnitude);
}

__global__ void iq3s_matvec_batch8_kernel(const IQ3SBlock *__restrict__ weights,
                                          const float *__restrict__ input,
                                          float *__restrict__ output,
                                          int input_size, int blocks_per_row, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kWarpSize) return;
    float sums[8] = {};
    const IQ3SBlock *row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const IQ3SBlock &block = row_weights[block_index];
#pragma unroll
        for (int iteration = 0; iteration < 8; ++iteration) {
            const int position = iteration * 32 + lane;
            const float weight = iq3s_decode_weight(block, position);
            const int input_index = block_index * kQuantBlock + position;
#pragma unroll
            for (int batch = 0; batch < 8; ++batch) {
                sums[batch] = fmaf(
                    weight, input[static_cast<std::size_t>(batch) * input_size + input_index],
                    sums[batch]);
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

__global__ void iq3s_matvec_kernel(const IQ3SBlock *__restrict__ weights,
                                   const float *__restrict__ input,
                                   float *__restrict__ output,
                                   int blocks_per_row, int rows) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kWarpSize) return;
    float sum = 0.0F;
    const IQ3SBlock *row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        const IQ3SBlock &block = row_weights[block_index];
#pragma unroll
        for (int iteration = 0; iteration < 8; ++iteration) {
            const int position = iteration * 32 + lane;
            sum = fmaf(iq3s_decode_weight(block, position),
                       input[block_index * kQuantBlock + position], sum);
        }
    }
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, kWarpSize);
    }
    if (lane == 0) output[row] = sum;
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
            const int stripe = p / 32;
            const int qs_index = (stripe / 4) * 32 + lane;
            const int shift = (stripe & 3) * 2;
            const int low2 = (block.qs[qs_index] >> shift) & 3;
            const int high1 = (block.hmask[lane] >> stripe) & 1;
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
                                        float *__restrict__ states,
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
            predicted = fmaf(states[state_offset + static_cast<std::size_t>(row) * head_size + thread],
                             keys[vector_offset + row], predicted);
        }
        delta[thread] = (values[vector_offset + thread] - decay * predicted) * betas[head];
    }
    __syncthreads();

    for (int element = thread; element < head_size * head_size; element += static_cast<int>(blockDim.x)) {
        const int row = element / head_size;
        const int column = element % head_size;
        const float updated = decay * states[state_offset + element]
                            + keys[vector_offset + row] * delta[column];
        states[state_offset + element] = updated;
    }
    __syncthreads();

    if (thread < head_size) {
        float sum = 0.0F;
        for (int row = 0; row < head_size; ++row) {
            sum = fmaf(states[state_offset + static_cast<std::size_t>(row) * head_size + thread],
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
                                               float *__restrict__ states,
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
                predicted = fmaf(states[state_offset + static_cast<std::size_t>(row) * head_size + thread],
                                 t_keys[vector_offset + row], predicted);
            }
            delta[thread] = (t_values[vector_offset + thread] - t_decay * predicted) * t_beta;
        }
        __syncthreads();

        for (int element = thread; element < head_size * head_size; element += static_cast<int>(blockDim.x)) {
            const int row = element / head_size;
            const int column = element % head_size;
            const float updated = t_decay * states[state_offset + element]
                                + t_keys[vector_offset + row] * delta[column];
            states[state_offset + element] = updated;
        }
        __syncthreads();

        if (thread < head_size) {
            float sum = 0.0F;
            for (int row = 0; row < head_size; ++row) {
                sum = fmaf(states[state_offset + static_cast<std::size_t>(row) * head_size + thread],
                           t_queries[vector_offset + row], sum);
            }
            output[token * vectors_per_batch + vector_offset + thread] = sum * 0.08838834764831843F;
        }
        __syncthreads();
    }
}

// GPU-native recurrent: conv1d + SiLU + split QKV + L2 norm.
// One block per channel group, iterates over 8 tokens sequentially (state is serial).
// qkv_width = 2*key_heads*head_size + value_heads*head_size = 10240 for this model.
// Output: queries[8*recurrent_width], keys[8*recurrent_width], values[8*recurrent_width]
// where recurrent_width = value_heads * head_size = 6144.
// Rebuild convolution history from its checkpoint and the already projected
// prefix. The caller restores the checkpoint history before this launch.
__global__ void recurrent_conv_commit_prefix_kernel(
        const float *__restrict__ qkv_input,
        float *__restrict__ conv_state,
        int qkv_width, int committed_tokens) {
    const int channel = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                      + static_cast<int>(threadIdx.x);
    if (channel >= qkv_width) return;
    float s0 = conv_state[channel * 3 + 0];
    float s1 = conv_state[channel * 3 + 1];
    float s2 = conv_state[channel * 3 + 2];
    for (int token = 0; token < committed_tokens; ++token) {
        s0 = s1;
        s1 = s2;
        s2 = qkv_input[static_cast<std::size_t>(token) * qkv_width + channel];
    }
    conv_state[channel * 3 + 0] = s0;
    conv_state[channel * 3 + 1] = s1;
    conv_state[channel * 3 + 2] = s2;
}

__global__ void recurrent_gdn_commit_prefix_kernel(
        const float *__restrict__ keys,
        const float *__restrict__ values,
        const float *__restrict__ decays,
        const float *__restrict__ betas,
        float *__restrict__ states,
        int head_size, int committed_tokens) {
    const int head = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    const std::size_t vector_offset = static_cast<std::size_t>(head) * head_size;
    const std::size_t state_offset = vector_offset * head_size;
    const std::size_t vectors_per_batch = 48ULL * head_size;
    __shared__ float delta[128];
    for (int token = 0; token < committed_tokens; ++token) {
        const float *t_keys = keys + static_cast<std::size_t>(token) * vectors_per_batch;
        const float *t_values = values + static_cast<std::size_t>(token) * vectors_per_batch;
        const float decay = decays[token * 48 + head];
        const float beta = betas[token * 48 + head];
        if (thread < head_size) {
            float predicted = 0.0F;
            for (int row = 0; row < head_size; ++row) {
                predicted = fmaf(states[state_offset + static_cast<std::size_t>(row) * head_size + thread],
                                 t_keys[vector_offset + row], predicted);
            }
            delta[thread] = (t_values[vector_offset + thread] - decay * predicted) * beta;
        }
        __syncthreads();
        for (int element = thread; element < head_size * head_size;
             element += static_cast<int>(blockDim.x)) {
            const int row = element / head_size;
            const int column = element % head_size;
            states[state_offset + element] = decay * states[state_offset + element]
                + t_keys[vector_offset + row] * delta[column];
        }
        __syncthreads();
    }
}

__global__ void recurrent_conv1d_batch8_kernel(
        const float *__restrict__ qkv_input,   // slot[0]: 8 × qkv_width
        float *__restrict__ conv_state,         // 3 × qkv_width (persistent per layer)
        const float *__restrict__ conv_weights, // 4 × qkv_width
        float *__restrict__ queries,            // 8 × recurrent_width output
        float *__restrict__ keys,               // 8 × recurrent_width output
        float *__restrict__ values,             // 8 × recurrent_width output
        int qkv_width, int key_heads, int value_heads, int head_size, int batch_count) {
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

    for (int token = 0; token < batch_count; ++token) {
        const float input_val = qkv_input[token * qkv_width + channel];
        float sum = fmaf(s0, w0, fmaf(s1, w1, fmaf(s2, w2, input_val * w3)));
        float activated = sum / (1.0f + expf(-sum)); // SiLU

        s0 = s1;
        s1 = s2;
        s2 = input_val;

        // Split into Q/K/V based on channel position
        const int key_region = key_heads * head_size;
        if (channel < key_region) {
            // The GGUF converter tiles V heads as [replica][K head] and
            // applies the same permutation to V/Z/A/B/conv/out weights.
            const int key_head = channel / head_size;
            const int dim = channel % head_size;
            const int heads_per_key = value_heads / key_heads;
            for (int rep = 0; rep < heads_per_key; ++rep) {
                const int out_head = rep * key_heads + key_head;
                queries[token * recurrent_width + out_head * head_size + dim] = activated;
            }
        } else if (channel < 2 * key_region) {
            // K uses the same tiled broadcast as Q.
            const int key_head = (channel - key_region) / head_size;
            const int dim = (channel - key_region) % head_size;
            const int heads_per_key = value_heads / key_heads;
            for (int rep = 0; rep < heads_per_key; ++rep) {
                const int out_head = rep * key_heads + key_head;
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
        int recurrent_width, int head_size, float epsilon, int batch_count) {
    const int head = blockIdx.x;
    const int dim = threadIdx.x;
    if (dim >= head_size) return;

    for (int token = 0; token < batch_count; ++token) {
        const int base = token * recurrent_width + head * head_size;

        // L2 norm queries
        __shared__ float q_sum_sq[128];
        q_sum_sq[dim] = queries[base + dim] * queries[base + dim];
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (dim < stride) q_sum_sq[dim] += q_sum_sq[dim + stride];
            __syncthreads();
        }
        // Match the scalar reference's L2 normalization exactly.  This is
        // deliberately not RMS normalization: dividing by head_size here
        // changes every recurrent update and quickly changes greedy tokens.
        float q_scale = rsqrtf(q_sum_sq[0] + epsilon);
        queries[base + dim] *= q_scale;

        // L2 norm keys
        __shared__ float k_sum_sq[128];
        k_sum_sq[dim] = keys[base + dim] * keys[base + dim];
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (dim < stride) k_sum_sq[dim] += k_sum_sq[dim + stride];
            __syncthreads();
        }
        float k_scale = rsqrtf(k_sum_sq[0] + epsilon);
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
        int embedding_size, int batch_count) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float partial_beta[256];
    __shared__ float partial_alpha[256];

    const float *beta_row = beta_weights + head * embedding_size;
    const float *alpha_row = alpha_weights + head * embedding_size;
    const float bias = dt_bias[head];
    const float a_val = a_weights[head];

    for (int token = 0; token < batch_count; ++token) {
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
        int recurrent_width, int /*value_heads*/, int head_size, float epsilon, int batch_count) {
    const int head = blockIdx.x;
    const int dim = threadIdx.x;
    if (dim >= head_size) return;

    __shared__ float sum_sq_shared[128];

    for (int token = 0; token < batch_count; ++token) {
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
                                            int context_size, int cache_stride, int base_position, int partitions) {
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
    const int query_context_size = base_position + batch + 1;
    const int end = min(query_context_size, begin + chunk);
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
            + (static_cast<std::size_t>(kv_head) * cache_stride + token) * groups_per_head;
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
            const float score = tile_scores[thread];
            const bool raises_maximum = score > old_max;
            const float new_max = raises_maximum ? score : old_max;
            const float scale_old = raises_maximum
                ? (isinf(old_max) ? 0.0F : expf(old_max - score)) : 1.0F;
            const float probability = raises_maximum ? 1.0F : expf(score - old_max);
            rescale[thread] = scale_old;
            probabilities[thread] = probability;
            state_sum[thread] = state_sum[thread] * scale_old + probability;
            state_max[thread] = new_max;
        }
        __syncthreads();

        const int value_group_index = thread / 32;
        const int value_index = thread & 31;
        const auto &value_group = values[(static_cast<std::size_t>(kv_head) * cache_stride + token)
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

// Exact-layout Q8 KV flash attention for the correctness-critical long target
// path. One workgroup evaluates eight (batch, query-head) pairs while loading
// each K/V vector once. Six workgroups cover the 8x6 queries sharing one KV
// head, reducing full-context cache traffic by 8x without changing Q8 values.
__global__ void q8kv_flash_partial_kernel(const Q8_0Block *__restrict__ keys,
                                          const Q8_0Block *__restrict__ values,
                                          const float *__restrict__ queries,
                                          float *__restrict__ partials,
                                          int context_size, int cache_stride, int base_position, int partitions) {
    constexpr int kGroups = 8;
    constexpr int kHeads = 24;
    constexpr int kHeadsPerKv = 6;
    constexpr int kQueryGroupsPerKv = 6;
    constexpr int kPartialStride = 258;
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / kWarpSize;
    const int lane = thread & (kWarpSize - 1);
    const int kv_head = static_cast<int>(blockIdx.y) / kQueryGroupsPerKv;
    const int local_query_group = static_cast<int>(blockIdx.y) % kQueryGroupsPerKv;
    const int local_query = local_query_group * 8 + warp;
    const int batch = local_query / kHeadsPerKv;
    const int query_head = kv_head * kHeadsPerKv + local_query % kHeadsPerKv;
    const int partition = static_cast<int>(blockIdx.x);
    const int chunk = (context_size + partitions - 1) / partitions;
    const int begin = partition * chunk;
    const int end = min(context_size, begin + chunk);

    __shared__ float key_values[256];
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
        const int group = thread / 32;
        const int group_dimension = thread & 31;
        const auto &key_block = keys[(static_cast<std::size_t>(kv_head) * cache_stride + token)
                                     * kGroups + group];
        key_values[thread] = static_cast<float>(key_block.qs[group_dimension])
                           * __half2float(*reinterpret_cast<const __half *>(&key_block.d));
        __syncthreads();

        float score = 0.0F;
#pragma unroll
        for (int query_group = 0; query_group < kGroups; ++query_group) {
            const int dimension = query_group * 32 + lane;
            score = fmaf(queries[(static_cast<std::size_t>(batch) * kHeads + query_head) * 256
                                 + dimension], key_values[dimension], score);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            score += __shfl_down(score, offset, kWarpSize);
        }
        if (lane == 0) tile_scores[warp] = score * 0.0625F;
        __syncthreads();

        if (thread < 8) {
            const int local = local_query_group * 8 + thread;
            const int query_batch = local / kHeadsPerKv;
            if (token < base_position + query_batch + 1) {
                const float old_max = state_max[thread];
                const float score = tile_scores[thread];
                const bool raises_maximum = score > old_max;
                const float new_max = raises_maximum ? score : old_max;
                const float scale_old = raises_maximum
                    ? (isinf(old_max) ? 0.0F : expf(old_max - score)) : 1.0F;
                const float probability = raises_maximum ? 1.0F : expf(score - old_max);
                rescale[thread] = scale_old;
                probabilities[thread] = probability;
                state_sum[thread] = state_sum[thread] * scale_old + probability;
                state_max[thread] = new_max;
            } else {
                rescale[thread] = 1.0F;
                probabilities[thread] = 0.0F;
            }
        }
        __syncthreads();

        const auto &value_block = values[(static_cast<std::size_t>(kv_head) * cache_stride + token)
                                          * kGroups + group];
        const float value = static_cast<float>(value_block.qs[group_dimension])
                          * __half2float(*reinterpret_cast<const __half *>(&value_block.d));
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
        const int query_batch = local / kHeadsPerKv;
        const int query_head_out = kv_head * kHeadsPerKv + local % kHeadsPerKv;
        const int query_index = query_batch * kHeads + query_head_out;
        float *destination = partials
            + (static_cast<std::size_t>(query_index) * partitions + partition) * kPartialStride;
        destination[thread] = accumulator[query];
        if (thread == 0) {
            destination[256] = state_max[query];
            destination[257] = state_sum[query];
        }
    }
}

// Barrier-amortized exact-Q8 variant. It preserves the same eight-query
// grouping and token-order recurrence as q8kv_flash_partial_kernel, but stages
// eight complete K/V vectors per LDS tile and pays four workgroup barriers per
// tile instead of per token.
__global__ void q8kv_flash_partial_tiled8_kernel(const Q8_0Block *__restrict__ keys,
                                                 const Q8_0Block *__restrict__ values,
                                                 const float *__restrict__ queries,
                                                 float *__restrict__ partials,
                                                 int context_size, int cache_stride,
                                                 int base_position, int partitions) {
    constexpr int kGroups = 8;
    constexpr int kHeads = 24;
    constexpr int kHeadsPerKv = 6;
    constexpr int kQueryGroupsPerKv = 6;
    constexpr int kPartialStride = 258;
    constexpr int kTileTokens = 8;
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / kWarpSize;
    const int lane = thread & (kWarpSize - 1);
    const int kv_head = static_cast<int>(blockIdx.y) / kQueryGroupsPerKv;
    const int local_query_group = static_cast<int>(blockIdx.y) % kQueryGroupsPerKv;
    const int local_query = local_query_group * 8 + warp;
    const int batch = local_query / kHeadsPerKv;
    const int query_head = kv_head * kHeadsPerKv + local_query % kHeadsPerKv;
    const int partition = static_cast<int>(blockIdx.x);
    const int chunk = (context_size + partitions - 1) / partitions;
    const int begin = partition * chunk;
    const int end = min(context_size, begin + chunk);

    __shared__ float key_tiles[kTileTokens * 256];
    __shared__ float value_tiles[kTileTokens * 256];
    __shared__ float tile_scores[8 * kTileTokens];
    __shared__ float tile_rescale[8 * kTileTokens];
    __shared__ float tile_probability[8 * kTileTokens];
    __shared__ float state_max[8];
    __shared__ float state_sum[8];
    if (thread < 8) {
        state_max[thread] = -INFINITY;
        state_sum[thread] = 0.0F;
    }
    __syncthreads();

    float accumulator[8] = {};
    for (int tile_begin = begin; tile_begin < end; tile_begin += kTileTokens) {
        const int tile_count = min(kTileTokens, end - tile_begin);
        const int group = thread / 32;
        const int group_dimension = thread & 31;
        for (int tile_token = 0; tile_token < tile_count; ++tile_token) {
            const int token = tile_begin + tile_token;
            const auto &key_block = keys[(static_cast<std::size_t>(kv_head) * cache_stride + token)
                                         * kGroups + group];
            const auto &value_block = values[(static_cast<std::size_t>(kv_head) * cache_stride + token)
                                             * kGroups + group];
            key_tiles[tile_token * 256 + thread]
                = static_cast<float>(key_block.qs[group_dimension])
                * __half2float(*reinterpret_cast<const __half *>(&key_block.d));
            value_tiles[tile_token * 256 + thread]
                = static_cast<float>(value_block.qs[group_dimension])
                * __half2float(*reinterpret_cast<const __half *>(&value_block.d));
        }
        __syncthreads();

        for (int tile_token = 0; tile_token < tile_count; ++tile_token) {
            float score = 0.0F;
#pragma unroll
            for (int query_group = 0; query_group < kGroups; ++query_group) {
                const int dimension = query_group * 32 + lane;
                score = fmaf(queries[(static_cast<std::size_t>(batch) * kHeads + query_head) * 256
                                     + dimension],
                             key_tiles[tile_token * 256 + dimension], score);
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                score += __shfl_down(score, offset, kWarpSize);
            }
            if (lane == 0) tile_scores[warp * kTileTokens + tile_token] = score * 0.0625F;
        }
        __syncthreads();

        if (thread < 8) {
            const int local = local_query_group * 8 + thread;
            const int query_batch = local / kHeadsPerKv;
            float maximum = state_max[thread];
            float sum = state_sum[thread];
            for (int tile_token = 0; tile_token < tile_count; ++tile_token) {
                const int token = tile_begin + tile_token;
                float rescale = 1.0F;
                float probability = 0.0F;
                if (token < base_position + query_batch + 1) {
                    const float score = tile_scores[thread * kTileTokens + tile_token];
                    if (score > maximum) {
                        rescale = isinf(maximum) ? 0.0F : expf(maximum - score);
                        probability = 1.0F;
                        maximum = score;
                    } else {
                        probability = expf(score - maximum);
                    }
                    sum = sum * rescale + probability;
                }
                tile_rescale[thread * kTileTokens + tile_token] = rescale;
                tile_probability[thread * kTileTokens + tile_token] = probability;
            }
            state_max[thread] = maximum;
            state_sum[thread] = sum;
        }
        __syncthreads();

        for (int tile_token = 0; tile_token < tile_count; ++tile_token) {
            const float value = value_tiles[tile_token * 256 + thread];
#pragma unroll
            for (int query = 0; query < 8; ++query) {
                accumulator[query]
                    = accumulator[query] * tile_rescale[query * kTileTokens + tile_token]
                    + tile_probability[query * kTileTokens + tile_token] * value;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int query = 0; query < 8; ++query) {
        const int local = local_query_group * 8 + query;
        const int query_batch = local / kHeadsPerKv;
        const int query_head_out = kv_head * kHeadsPerKv + local % kHeadsPerKv;
        const int query_index = query_batch * kHeads + query_head_out;
        float *destination = partials
            + (static_cast<std::size_t>(query_index) * partitions + partition) * kPartialStride;
        destination[thread] = accumulator[query];
        if (thread == 0) {
            destination[256] = state_max[query];
            destination[257] = state_sum[query];
        }
    }
}

// Two-read GQA variant: each workgroup owns 24 of the 48 queries sharing a KV
// head, so the complete K/V history is read twice rather than six times.
__global__ void q8kv_flash_partial_tiled24_kernel(const Q8_0Block *__restrict__ keys,
                                                  const Q8_0Block *__restrict__ values,
                                                  const float *__restrict__ queries,
                                                  float *__restrict__ partials,
                                                  int context_size, int cache_stride,
                                                  int base_position, int partitions) {
    constexpr int kGroups = 8;
    constexpr int kHeads = 24;
    constexpr int kHeadsPerKv = 6;
    constexpr int kQueryGroupsPerKv = 2;
    constexpr int kQueriesPerGroup = 24;
    constexpr int kPartialStride = 258;
    constexpr int kTileTokens = 8;
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / kWarpSize;
    const int lane = thread & (kWarpSize - 1);
    const int kv_head = static_cast<int>(blockIdx.y) / kQueryGroupsPerKv;
    const int query_supergroup = static_cast<int>(blockIdx.y) % kQueryGroupsPerKv;
    const int partition = static_cast<int>(blockIdx.x);
    const int chunk = (context_size + partitions - 1) / partitions;
    const int begin = partition * chunk;
    const int end = min(context_size, begin + chunk);

    __shared__ float key_tiles[kTileTokens * 256];
    __shared__ float value_tiles[kTileTokens * 256];
    __shared__ float tile_scores[kQueriesPerGroup * kTileTokens];
    __shared__ float tile_rescale[kQueriesPerGroup * kTileTokens];
    __shared__ float tile_probability[kQueriesPerGroup * kTileTokens];
    __shared__ float state_max[kQueriesPerGroup];
    __shared__ float state_sum[kQueriesPerGroup];
    if (thread < kQueriesPerGroup) {
        state_max[thread] = -INFINITY;
        state_sum[thread] = 0.0F;
    }
    __syncthreads();

    float accumulator[kQueriesPerGroup] = {};
    for (int tile_begin = begin; tile_begin < end; tile_begin += kTileTokens) {
        const int tile_count = min(kTileTokens, end - tile_begin);
        const int group = thread / 32;
        const int group_dimension = thread & 31;
        for (int tile_token = 0; tile_token < tile_count; ++tile_token) {
            const int token = tile_begin + tile_token;
            const auto &key_block = keys[(static_cast<std::size_t>(kv_head) * cache_stride + token)
                                         * kGroups + group];
            const auto &value_block = values[(static_cast<std::size_t>(kv_head) * cache_stride + token)
                                             * kGroups + group];
            key_tiles[tile_token * 256 + thread]
                = static_cast<float>(key_block.qs[group_dimension])
                * __half2float(*reinterpret_cast<const __half *>(&key_block.d));
            value_tiles[tile_token * 256 + thread]
                = static_cast<float>(value_block.qs[group_dimension])
                * __half2float(*reinterpret_cast<const __half *>(&value_block.d));
        }
        __syncthreads();

#pragma unroll
        for (int query_wave = 0; query_wave < 3; ++query_wave) {
            const int group_query = query_wave * 8 + warp;
            const int local_query = query_supergroup * kQueriesPerGroup + group_query;
            const int batch = local_query / kHeadsPerKv;
            const int query_head = kv_head * kHeadsPerKv + local_query % kHeadsPerKv;
            for (int tile_token = 0; tile_token < tile_count; ++tile_token) {
                float score = 0.0F;
#pragma unroll
                for (int query_group = 0; query_group < kGroups; ++query_group) {
                    const int dimension = query_group * 32 + lane;
                    score = fmaf(queries[(static_cast<std::size_t>(batch) * kHeads + query_head) * 256
                                         + dimension],
                                 key_tiles[tile_token * 256 + dimension], score);
                }
#pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    score += __shfl_down(score, offset, kWarpSize);
                }
                if (lane == 0) {
                    tile_scores[group_query * kTileTokens + tile_token] = score * 0.0625F;
                }
            }
        }
        __syncthreads();

        if (thread < kQueriesPerGroup) {
            const int local_query = query_supergroup * kQueriesPerGroup + thread;
            const int query_batch = local_query / kHeadsPerKv;
            float maximum = state_max[thread];
            float sum = state_sum[thread];
            for (int tile_token = 0; tile_token < tile_count; ++tile_token) {
                const int token = tile_begin + tile_token;
                float rescale = 1.0F;
                float probability = 0.0F;
                if (token < base_position + query_batch + 1) {
                    const float score = tile_scores[thread * kTileTokens + tile_token];
                    if (score > maximum) {
                        rescale = isinf(maximum) ? 0.0F : expf(maximum - score);
                        probability = 1.0F;
                        maximum = score;
                    } else {
                        probability = expf(score - maximum);
                    }
                    sum = sum * rescale + probability;
                }
                tile_rescale[thread * kTileTokens + tile_token] = rescale;
                tile_probability[thread * kTileTokens + tile_token] = probability;
            }
            state_max[thread] = maximum;
            state_sum[thread] = sum;
        }
        __syncthreads();

        for (int tile_token = 0; tile_token < tile_count; ++tile_token) {
            const float value = value_tiles[tile_token * 256 + thread];
#pragma unroll
            for (int query = 0; query < kQueriesPerGroup; ++query) {
                accumulator[query]
                    = accumulator[query] * tile_rescale[query * kTileTokens + tile_token]
                    + tile_probability[query * kTileTokens + tile_token] * value;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int query = 0; query < kQueriesPerGroup; ++query) {
        const int local_query = query_supergroup * kQueriesPerGroup + query;
        const int query_batch = local_query / kHeadsPerKv;
        const int query_head_out = kv_head * kHeadsPerKv + local_query % kHeadsPerKv;
        const int query_index = query_batch * kHeads + query_head_out;
        float *destination = partials
            + (static_cast<std::size_t>(query_index) * partitions + partition) * kPartialStride;
        destination[thread] = accumulator[query];
        if (thread == 0) {
            destination[256] = state_max[query];
            destination[257] = state_sum[query];
        }
    }
}

// Barrier-free query-warp variant. Each wave owns one query and keeps all 256
// output dimensions in eight lane-local accumulators. It intentionally relies
// on L2 reuse between the 48 queries sharing a KV head rather than coordinating
// them through per-token workgroup barriers.
__global__ void q8kv_flash_partial_warp_kernel(const Q8_0Block *__restrict__ keys,
                                               const Q8_0Block *__restrict__ values,
                                               const float *__restrict__ queries,
                                               float *__restrict__ partials,
                                               int context_size, int cache_stride,
                                               int base_position, int partitions) {
    constexpr int kGroups = 8;
    constexpr int kHeads = 24;
    constexpr int kHeadsPerKv = 6;
    constexpr int kPartialStride = 258;
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / kWarpSize;
    const int lane = thread & (kWarpSize - 1);
    const int queries_per_group = static_cast<int>(blockDim.x) / kWarpSize;
    const int query_groups_per_kv = 48 / queries_per_group;
    const int kv_head = static_cast<int>(blockIdx.y) / query_groups_per_kv;
    const int local_query_group = static_cast<int>(blockIdx.y) % query_groups_per_kv;
    const int local_query = local_query_group * queries_per_group + warp;
    const int batch = local_query / kHeadsPerKv;
    const int query_head = kv_head * kHeadsPerKv + local_query % kHeadsPerKv;
    const int query_index = batch * kHeads + query_head;
    const int partition = static_cast<int>(blockIdx.x);
    const int chunk = (context_size + partitions - 1) / partitions;
    const int begin = partition * chunk;
    const int end = min(min(context_size, begin + chunk), base_position + batch + 1);
    const float *query = queries + static_cast<std::size_t>(query_index) * 256;

    float maximum = -INFINITY;
    float sum = 0.0F;
    float accumulator[kGroups] = {};
    for (int token = begin; token < end; ++token) {
        float score = 0.0F;
#pragma unroll
        for (int group = 0; group < kGroups; ++group) {
            const auto &key_block = keys[(static_cast<std::size_t>(kv_head) * cache_stride + token)
                                         * kGroups + group];
            const float key = static_cast<float>(key_block.qs[lane])
                            * __half2float(*reinterpret_cast<const __half *>(&key_block.d));
            score = fmaf(query[group * 32 + lane], key, score);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            score += __shfl_down(score, offset, kWarpSize);
        }
        score = __shfl(score, 0, kWarpSize) * 0.0625F;
        float rescale = 0.0F;
        float probability = 0.0F;
        if (lane == 0) {
            if (score > maximum) {
                rescale = isinf(maximum) ? 0.0F : expf(maximum - score);
                probability = 1.0F;
                maximum = score;
            } else {
                rescale = 1.0F;
                probability = expf(score - maximum);
            }
            sum = sum * rescale + probability;
        }
        rescale = __shfl(rescale, 0, kWarpSize);
        probability = __shfl(probability, 0, kWarpSize);
        maximum = __shfl(maximum, 0, kWarpSize);
        sum = __shfl(sum, 0, kWarpSize);
#pragma unroll
        for (int group = 0; group < kGroups; ++group) {
            const auto &value_block = values[(static_cast<std::size_t>(kv_head) * cache_stride + token)
                                             * kGroups + group];
            const float value = static_cast<float>(value_block.qs[lane])
                              * __half2float(*reinterpret_cast<const __half *>(&value_block.d));
            accumulator[group] = accumulator[group] * rescale + probability * value;
        }
    }

    float *destination = partials
        + (static_cast<std::size_t>(query_index) * partitions + partition) * kPartialStride;
#pragma unroll
    for (int group = 0; group < kGroups; ++group) {
        destination[group * 32 + lane] = accumulator[group];
    }
    if (lane == 0) {
        destination[256] = maximum;
        destination[257] = sum;
    }
}

// LDS-tiled companion to the barrier-free query-wave kernel. Sixteen query
// waves share each dequantized K/V tile, reducing cache requests while paying
// only two workgroup barriers per sixteen history tokens. The opt-in packed
// representation reduces LDS from 32 KiB to 9 KiB, trading repeated coefficient
// multiplication for possible occupancy/cache gains; no speedup is assumed.
template <bool Packed>
__global__ void q8kv_flash_partial_warp16_lds_kernel(
    const Q8_0Block *__restrict__ keys,
    const Q8_0Block *__restrict__ values,
    const float *__restrict__ queries,
    float *__restrict__ partials,
    int context_size, int cache_stride,
    int base_position, int partitions) {
    constexpr int kGroups = 8;
    constexpr int kHeads = 24;
    constexpr int kHeadsPerKv = 6;
    constexpr int kQueriesPerGroup = 16;
    constexpr int kQueryGroupsPerKv = 3;
    constexpr int kTileTokens = 16;
    constexpr int kPartialStride = 258;
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / kWarpSize;
    const int lane = thread & (kWarpSize - 1);
    const int kv_head = static_cast<int>(blockIdx.y) / kQueryGroupsPerKv;
    const int query_group = static_cast<int>(blockIdx.y) % kQueryGroupsPerKv;
    const int local_query = query_group * kQueriesPerGroup + warp;
    const int batch = local_query / kHeadsPerKv;
    const int query_head = kv_head * kHeadsPerKv + local_query % kHeadsPerKv;
    const int query_index = batch * kHeads + query_head;
    const int partition = static_cast<int>(blockIdx.x);
    const int chunk = (context_size + partitions - 1) / partitions;
    const int begin = partition * chunk;
    const int partition_end = min(context_size, begin + chunk);
    const int query_end = min(partition_end, base_position + batch + 1);
    const float *query = queries + static_cast<std::size_t>(query_index) * 256;

    __shared__ AttentionSharedTile<kTileTokens, Packed> tile;
    float maximum = -INFINITY;
    float sum = 0.0F;
    float accumulator[kGroups] = {};

    for (int tile_begin = begin; tile_begin < partition_end; tile_begin += kTileTokens) {
        const int tile_count = min(kTileTokens, partition_end - tile_begin);
        const int tile_values = tile_count * 256;
        for (int item = thread; item < tile_values; item += static_cast<int>(blockDim.x)) {
            const int tile_token = item / 256;
            const int dimension = item - tile_token * 256;
            const int group = dimension / 32;
            const int group_lane = dimension & 31;
            const int token = tile_begin + tile_token;
            const std::size_t block_index =
                (static_cast<std::size_t>(kv_head) * cache_stride + token) * kGroups + group;
            const auto &key_block = keys[block_index];
            const auto &value_block = values[block_index];
            tile.stage(item, key_block.qs[group_lane], value_block.qs[group_lane],
                       __half2float(*reinterpret_cast<const __half *>(&key_block.d)),
                       __half2float(*reinterpret_cast<const __half *>(&value_block.d)));
        }
        __syncthreads();

        for (int tile_token = 0; tile_token < tile_count; ++tile_token) {
            const int token = tile_begin + tile_token;
            if (token >= query_end) continue;
            float score = 0.0F;
#pragma unroll
            for (int group = 0; group < kGroups; ++group) {
                score = fmaf(query[group * 32 + lane],
                             tile.key(tile_token * 256 + group * 32 + lane), score);
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                score += __shfl_down(score, offset, kWarpSize);
            }
            score = __shfl(score, 0, kWarpSize) * 0.0625F;
            float rescale = 0.0F;
            float probability = 0.0F;
            if (lane == 0) {
                if (score > maximum) {
                    rescale = isinf(maximum) ? 0.0F : expf(maximum - score);
                    probability = 1.0F;
                    maximum = score;
                } else {
                    rescale = 1.0F;
                    probability = expf(score - maximum);
                }
                sum = sum * rescale + probability;
            }
            rescale = __shfl(rescale, 0, kWarpSize);
            probability = __shfl(probability, 0, kWarpSize);
            maximum = __shfl(maximum, 0, kWarpSize);
            sum = __shfl(sum, 0, kWarpSize);
#pragma unroll
            for (int group = 0; group < kGroups; ++group) {
                const float value = tile.value(tile_token * 256 + group * 32 + lane);
                accumulator[group] = accumulator[group] * rescale + probability * value;
            }
        }
        __syncthreads();
    }

    float *destination = partials
        + (static_cast<std::size_t>(query_index) * partitions + partition) * kPartialStride;
#pragma unroll
    for (int group = 0; group < kGroups; ++group) {
        destination[group * 32 + lane] = accumulator[group];
    }
    if (lane == 0) {
        destination[256] = maximum;
        destination[257] = sum;
    }
}

// Long-context target attention keeps values in Q8_0. Key-score quantization
// is still N23KV4, but preserving value precision is necessary to retain the
// target's greedy decisions through stacked full-attention layers.
__global__ void n23k_q8v_flash_partial_kernel(const N23KV4Group *__restrict__ keys,
                                               const Q8_0Block *__restrict__ values,
                                               const std::int8_t *__restrict__ queries,
                                               const float *__restrict__ query_scales,
                                               float *__restrict__ partials,
                                               int context_size, int cache_stride, int base_position, int partitions) {
    constexpr int kGroups = 8;
    constexpr int kHeads = 24;
    constexpr int kHeadsPerKv = 6;
    constexpr int kPartialStride = 258;
    const int dimension = static_cast<int>(threadIdx.x);
    const int query_index = static_cast<int>(blockIdx.y);
    const int batch = query_index / kHeads;
    const int query_head = query_index % kHeads;
    const int kv_head = query_head / kHeadsPerKv;
    const int partition = static_cast<int>(blockIdx.x);
    const int chunk = (context_size + partitions - 1) / partitions;
    const int begin = partition * chunk;
    const int query_context_size = base_position + batch + 1;
    const int end = min(query_context_size, begin + chunk);
    const int group = dimension / 32;
    const int group_dimension = dimension & 31;
    const std::size_t query_group = (static_cast<std::size_t>(batch) * kHeads + query_head) * kGroups + group;
    const float query_scale = query_scales[query_group];
    const std::int8_t query_value = queries[query_group * 32 + group_dimension];
    __shared__ float reduction[256];
    __shared__ float state_max;
    __shared__ float state_sum;
    __shared__ float probability;
    __shared__ float rescale;
    if (dimension == 0) {
        state_max = -INFINITY;
        state_sum = 0.0F;
    }
    __syncthreads();
    float accumulator = 0.0F;
    for (int token = begin; token < end; ++token) {
        const auto &key_group = keys[(static_cast<std::size_t>(kv_head) * cache_stride + token) * kGroups + group];
        const std::uint8_t packed = key_group.quants[group_dimension / 2];
        const int nibble = (packed >> ((group_dimension & 1) * 4)) & 0x0f;
        const int key_value = nibble >= 8 ? nibble - 16 : nibble;
        const float key_scale = __half2float(*reinterpret_cast<const __half *>(&key_group.scale));
        reduction[dimension] = static_cast<float>(key_value) * static_cast<float>(query_value)
                             * key_scale * query_scale;
        __syncthreads();
        for (int offset = 128; offset > 0; offset /= 2) {
            if (dimension < offset) reduction[dimension] += reduction[dimension + offset];
            __syncthreads();
        }
        if (dimension == 0) {
            const float score = reduction[0] * 0.0625F;
            if (score > state_max) {
                rescale = isinf(state_max) ? 0.0F : expf(state_max - score);
                probability = 1.0F;
                state_max = score;
            } else {
                rescale = 1.0F;
                probability = expf(score - state_max);
            }
            state_sum = state_sum * rescale + probability;
        }
        __syncthreads();
        const auto &value_block = values[(static_cast<std::size_t>(kv_head) * cache_stride + token)
                                          * kGroups + group];
        const float value = static_cast<float>(value_block.qs[group_dimension])
                          * __half2float(*reinterpret_cast<const __half *>(&value_block.d));
        accumulator = accumulator * rescale + probability * value;
        __syncthreads();
    }
    float *destination = partials + (static_cast<std::size_t>(query_index) * partitions + partition)
                                 * kPartialStride;
    destination[dimension] = accumulator;
    if (dimension == 0) {
        destination[256] = state_max;
        destination[257] = state_sum;
    }
}

// Explicit-GQA Q4 flash-attention kernel for gfx1032/RDNA2.
//
// One 192-thread workgroup owns one (KV head, token partition). Its six
// wave32s cover all 48 causal queries that share that KV head. Each wave is
// divided into eight 4-lane subgroups, so it scores eight queries at once.
// K/V are decoded once per workgroup into a 16-token LDS tile; no correctness
// or performance assumption is made about inter-workgroup Infinity-Cache reuse.
__device__ __forceinline__ int q4x4_sign_extend_bytes(std::uint32_t x) {
    x &= 0x0f0f0f0fU;
    return static_cast<int>(x | ((x & 0x08080808U) * 0x1eU));
}

__device__ __forceinline__ std::uint32_t load_q4_dword_unaligned(
    const std::uint8_t *__restrict__ p) {
    std::uint32_t result;
    __builtin_memcpy(&result, p, sizeof(result));
    return result;
}

__device__ __forceinline__ int packed_s8_byte(int packed, int byte_index) {
    return static_cast<int>(static_cast<std::int8_t>(
        static_cast<std::uint32_t>(packed) >> (byte_index * 8)));
}

__global__ __launch_bounds__(192)
void n23kv4_flash_gqa48_tile16_kernel(
    const N23KV4Group *__restrict__ keys,
    const N23KV4Group *__restrict__ values,
    const std::int8_t *__restrict__ queries,
    const float *__restrict__ query_scales,
    float *__restrict__ partials,
    int context_size, int cache_stride,
    int base_position, int partitions) {
    constexpr int kGroups = 8;
    constexpr int kHeads = 24;
    constexpr int kHeadsPerKv = 6;
    constexpr int kQueriesPerWave = 8;
    constexpr int kTileTokens = 16;
    constexpr int kPartialStride = 258;

    const int thread = static_cast<int>(threadIdx.x);
    const int wave = thread >> 5;
    const int lane = thread & 31;
    const int subgroup = lane >> 2;
    const int lane4 = lane & 3;

    const int partition = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int chunk = (context_size + partitions - 1) / partitions;
    const int begin = partition * chunk;
    const int partition_end = min(context_size, begin + chunk);

    const int owned_local_query = wave * kQueriesPerWave + subgroup;
    const int owned_batch = owned_local_query / kHeadsPerKv;
    const int owned_query_head = kv_head * kHeadsPerKv
                               + owned_local_query % kHeadsPerKv;
    const int owned_query_index = owned_batch * kHeads + owned_query_head;

    int query_words[kGroups][2];
    float query_group_scales[kGroups];
#pragma unroll
    for (int group = 0; group < kGroups; ++group) {
        const std::size_t qg = static_cast<std::size_t>(owned_query_index)
                             * kGroups + group;
        const auto *q = queries + qg * 32 + lane4 * 8;
        query_words[group][0] = static_cast<int>(
              static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[0]))
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[2])) << 8)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[4])) << 16)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[6])) << 24));
        query_words[group][1] = static_cast<int>(
              static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[1]))
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[3])) << 8)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[5])) << 16)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[7])) << 24));
        query_group_scales[group] = query_scales[qg] * 0.0625F;
    }

    // 4,096 + 512 + 16,384 = 20,992 bytes of LDS.
    __shared__ int key_words[kTileTokens][kGroups][8];
    __shared__ float key_group_scales[kTileTokens][kGroups];
    __shared__ float value_tile[kTileTokens][kGroups][32];

    float maximum = -INFINITY;
    float sum = 0.0F;
    float accumulator[kQueriesPerWave][kGroups] = {};

    for (int tile_begin = begin; tile_begin < partition_end;
         tile_begin += kTileTokens) {
        const int tile_count = min(kTileTokens, partition_end - tile_begin);
        const int loader_tasks = tile_count * kGroups * 3;
        for (int task = thread; task < loader_tasks;
             task += static_cast<int>(blockDim.x)) {
            const int part = task % 3;
            const int token_group = task / 3;
            const int tile_token = token_group / kGroups;
            const int group = token_group - tile_token * kGroups;
            const int token = tile_begin + tile_token;
            const std::size_t cache_group =
                (static_cast<std::size_t>(kv_head) * cache_stride + token)
                * kGroups + group;

            if (part == 0) {
                const auto &key_group = keys[cache_group];
                key_group_scales[tile_token][group] = __half2float(
                    *reinterpret_cast<const __half *>(&key_group.scale));
#pragma unroll
                for (int chunk8 = 0; chunk8 < 4; ++chunk8) {
                    const std::uint32_t packed = load_q4_dword_unaligned(
                        key_group.quants + chunk8 * 4);
                    key_words[tile_token][group][chunk8 * 2]
                        = q4x4_sign_extend_bytes(packed);
                    key_words[tile_token][group][chunk8 * 2 + 1]
                        = q4x4_sign_extend_bytes(packed >> 4);
                }
            } else {
                const auto &value_group = values[cache_group];
                const float value_scale = __half2float(
                    *reinterpret_cast<const __half *>(&value_group.scale));
                const int half = part - 1;
                const int half_base = half * 16;
#pragma unroll
                for (int chunk8 = 0; chunk8 < 2; ++chunk8) {
                    const std::uint32_t packed = load_q4_dword_unaligned(
                        value_group.quants + half * 8 + chunk8 * 4);
                    const int even = q4x4_sign_extend_bytes(packed);
                    const int odd = q4x4_sign_extend_bytes(packed >> 4);
                    const int base = half_base + chunk8 * 8;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        value_tile[tile_token][group][base + i * 2]
                            = static_cast<float>(packed_s8_byte(even, i)) * value_scale;
                        value_tile[tile_token][group][base + i * 2 + 1]
                            = static_cast<float>(packed_s8_byte(odd, i)) * value_scale;
                    }
                }
            }
        }
        __syncthreads();

#pragma unroll 1
        for (int tile_token = 0; tile_token < tile_count; ++tile_token) {
            const int token = tile_begin + tile_token;
            float score = 0.0F;
#pragma unroll
            for (int group = 0; group < kGroups; ++group) {
                int dot = __builtin_amdgcn_sdot4(
                    key_words[tile_token][group][lane4 * 2],
                    query_words[group][0], 0, false);
                dot = __builtin_amdgcn_sdot4(
                    key_words[tile_token][group][lane4 * 2 + 1],
                    query_words[group][1], dot, false);
                score = fmaf(static_cast<float>(dot) * key_group_scales[tile_token][group],
                             query_group_scales[group], score);
            }
            score += __shfl_down(score, 2, 4);
            score += __shfl_down(score, 1, 4);

            float probability = 0.0F;
            float rescale = 1.0F;
            bool raises_maximum = false;
            if (lane4 == 0 && token < base_position + owned_batch + 1) {
                raises_maximum = score > maximum;
                if (raises_maximum) {
                    rescale = isinf(maximum) ? 0.0F : expf(maximum - score);
                    probability = 1.0F;
                    maximum = score;
                } else {
                    probability = expf(score - maximum);
                }
                sum = fmaf(sum, rescale, probability);
            }

            if (__any(raises_maximum)) {
#pragma unroll
                for (int query = 0; query < kQueriesPerWave; ++query) {
                    const float r = __shfl(rescale, query * 4, 32);
                    if (r != 1.0F) {
#pragma unroll
                        for (int group = 0; group < kGroups; ++group) {
                            accumulator[query][group] *= r;
                        }
                    }
                }
            }

            float value[kGroups];
#pragma unroll
            for (int group = 0; group < kGroups; ++group) {
                value[group] = value_tile[tile_token][group][lane];
            }
#pragma unroll
            for (int query = 0; query < kQueriesPerWave; ++query) {
                const float p = __shfl(probability, query * 4, 32);
#pragma unroll
                for (int group = 0; group < kGroups; ++group) {
                    accumulator[query][group] = fmaf(p, value[group], accumulator[query][group]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int query = 0; query < kQueriesPerWave; ++query) {
        const int local_query = wave * kQueriesPerWave + query;
        const int batch = local_query / kHeadsPerKv;
        const int query_head = kv_head * kHeadsPerKv + local_query % kHeadsPerKv;
        const int query_index = batch * kHeads + query_head;
        float *destination = partials
            + (static_cast<std::size_t>(query_index) * partitions + partition)
              * kPartialStride;
#pragma unroll
        for (int group = 0; group < kGroups; ++group) {
            destination[group * 32 + lane] = accumulator[query][group];
        }
        if (lane == query * 4) {
            destination[256] = maximum;
            destination[257] = sum;
        }
    }
}

// Occupancy-oriented companion to GQA48. Two six-wave workgroups cover one
// KV head, so each lane keeps 32 rather than 64 value accumulators. The K/V
// tile is consequently traversed twice, which is the intentional trade-off.
__device__ __forceinline__ int q4x4_expand_sequential(std::uint32_t packed2) {
    const std::uint32_t nibbles = packed2 & 0xffffU;
    const std::uint32_t bytes = (nibbles & 0x000fU)
                              | ((nibbles & 0x00f0U) << 4)
                              | ((nibbles & 0x0f00U) << 8)
                              | ((nibbles & 0xf000U) << 12);
    return q4x4_sign_extend_bytes(bytes);
}

__global__ __launch_bounds__(192)
void n23kv4_flash_gqa24_tile16_kernel(
    const N23KV4Group *__restrict__ keys,
    const N23KV4Group *__restrict__ values,
    const std::int8_t *__restrict__ queries,
    const float *__restrict__ query_scales,
    float *__restrict__ partials,
    int context_size, int cache_stride,
    int base_position, int partitions) {
    constexpr int kGroups = 8;
    constexpr int kHeads = 24;
    constexpr int kHeadsPerKv = 6;
    constexpr int kQueriesPerWave = 4;
    constexpr int kQueriesPerWorkgroup = 24;
    constexpr int kQueryGroupsPerKv = 2;
    constexpr int kTileTokens = 16;
    constexpr int kPartialStride = 258;

    const int thread = static_cast<int>(threadIdx.x);
    const int wave = thread >> 5;
    const int lane = thread & 31;
    const int subgroup = lane >> 3;
    const int lane8 = lane & 7;
    const int partition = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y) / kQueryGroupsPerKv;
    const int query_group = static_cast<int>(blockIdx.y) % kQueryGroupsPerKv;
    const int chunk = (context_size + partitions - 1) / partitions;
    const int begin = partition * chunk;
    const int partition_end = min(context_size, begin + chunk);

    const int owned_local_query = query_group * kQueriesPerWorkgroup
                                + wave * kQueriesPerWave + subgroup;
    const int owned_batch = owned_local_query / kHeadsPerKv;
    const int owned_query_head = kv_head * kHeadsPerKv
                               + owned_local_query % kHeadsPerKv;
    const int owned_query_index = owned_batch * kHeads + owned_query_head;

    int query_words[kGroups];
    float query_group_scales[kGroups];
#pragma unroll
    for (int group = 0; group < kGroups; ++group) {
        const std::size_t qg = static_cast<std::size_t>(owned_query_index) * kGroups + group;
        const auto *q = queries + qg * 32 + lane8 * 4;
        query_words[group] = static_cast<int>(
              static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[0]))
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[1])) << 8)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[2])) << 16)
            | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[3])) << 24));
        query_group_scales[group] = query_scales[qg] * 0.0625F;
    }

    __shared__ int key_words[kTileTokens][kGroups][8];
    __shared__ float key_group_scales[kTileTokens][kGroups];
    __shared__ float value_tile[kTileTokens][kGroups][32];
    float maximum = -INFINITY;
    float sum = 0.0F;
    float accumulator[kQueriesPerWave][kGroups] = {};

    for (int tile_begin = begin; tile_begin < partition_end; tile_begin += kTileTokens) {
        const int tile_count = min(kTileTokens, partition_end - tile_begin);
        const int loader_tasks = tile_count * kGroups * 3;
        for (int task = thread; task < loader_tasks; task += static_cast<int>(blockDim.x)) {
            const int part = task % 3;
            const int token_group = task / 3;
            const int tile_token = token_group / kGroups;
            const int group = token_group - tile_token * kGroups;
            const int token = tile_begin + tile_token;
            const std::size_t cache_group =
                (static_cast<std::size_t>(kv_head) * cache_stride + token) * kGroups + group;
            if (part == 0) {
                const auto &key_group = keys[cache_group];
                key_group_scales[tile_token][group] = __half2float(
                    *reinterpret_cast<const __half *>(&key_group.scale));
#pragma unroll
                for (int word = 0; word < 8; ++word) {
                    std::uint16_t packed;
                    __builtin_memcpy(&packed, key_group.quants + word * 2, sizeof(packed));
                    key_words[tile_token][group][word] = q4x4_expand_sequential(packed);
                }
            } else {
                const auto &value_group = values[cache_group];
                const float value_scale = __half2float(
                    *reinterpret_cast<const __half *>(&value_group.scale));
                const int half = part - 1;
                const int half_base = half * 16;
#pragma unroll
                for (int chunk8 = 0; chunk8 < 2; ++chunk8) {
                    const std::uint32_t packed = load_q4_dword_unaligned(
                        value_group.quants + half * 8 + chunk8 * 4);
                    const int even = q4x4_sign_extend_bytes(packed);
                    const int odd = q4x4_sign_extend_bytes(packed >> 4);
                    const int base = half_base + chunk8 * 8;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        value_tile[tile_token][group][base + i * 2]
                            = static_cast<float>(packed_s8_byte(even, i)) * value_scale;
                        value_tile[tile_token][group][base + i * 2 + 1]
                            = static_cast<float>(packed_s8_byte(odd, i)) * value_scale;
                    }
                }
            }
        }
        __syncthreads();
#pragma unroll 1
        for (int tile_token = 0; tile_token < tile_count; ++tile_token) {
            const int token = tile_begin + tile_token;
            float score = 0.0F;
#pragma unroll
            for (int group = 0; group < kGroups; ++group) {
                const int dot = __builtin_amdgcn_sdot4(
                    key_words[tile_token][group][lane8], query_words[group], 0, false);
                score = fmaf(static_cast<float>(dot) * key_group_scales[tile_token][group],
                             query_group_scales[group], score);
            }
#pragma unroll
            for (int offset = 4; offset > 0; offset /= 2) score += __shfl_down(score, offset, 8);

            float probability = 0.0F;
            float rescale = 1.0F;
            bool raises_maximum = false;
            if (lane8 == 0 && token < base_position + owned_batch + 1) {
                raises_maximum = score > maximum;
                if (raises_maximum) {
                    rescale = isinf(maximum) ? 0.0F : expf(maximum - score);
                    probability = 1.0F;
                    maximum = score;
                } else {
                    probability = expf(score - maximum);
                }
                sum = fmaf(sum, rescale, probability);
            }
            if (__any(raises_maximum)) {
#pragma unroll
                for (int query = 0; query < kQueriesPerWave; ++query) {
                    const float r = __shfl(rescale, query * 8, 32);
                    if (r != 1.0F) {
#pragma unroll
                        for (int group = 0; group < kGroups; ++group) accumulator[query][group] *= r;
                    }
                }
            }
            float value[kGroups];
#pragma unroll
            for (int group = 0; group < kGroups; ++group) value[group] = value_tile[tile_token][group][lane];
#pragma unroll
            for (int query = 0; query < kQueriesPerWave; ++query) {
                const float p = __shfl(probability, query * 8, 32);
#pragma unroll
                for (int group = 0; group < kGroups; ++group) {
                    accumulator[query][group] = fmaf(p, value[group], accumulator[query][group]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int query = 0; query < kQueriesPerWave; ++query) {
        const int local_query = query_group * kQueriesPerWorkgroup
                              + wave * kQueriesPerWave + query;
        const int batch = local_query / kHeadsPerKv;
        const int query_head = kv_head * kHeadsPerKv + local_query % kHeadsPerKv;
        const int query_index = batch * kHeads + query_head;
        float *destination = partials
            + (static_cast<std::size_t>(query_index) * partitions + partition) * kPartialStride;
#pragma unroll
        for (int group = 0; group < kGroups; ++group) destination[group * 32 + lane] = accumulator[query][group];
        if (lane == query * 8) {
            destination[256] = maximum;
            destination[257] = sum;
        }
    }
}

__global__ void n23kv4_flash_warp16_kernel(const N23KV4Group *__restrict__ keys,
                                            const N23KV4Group *__restrict__ values,
                                            const std::int8_t *__restrict__ queries,
                                            const float *__restrict__ query_scales,
                                            float *__restrict__ partials,
                                            int context_size, int cache_stride,
                                            int base_position, int partitions) {
    constexpr int kGroups = 8;
    constexpr int kHeads = 24;
    constexpr int kHeadsPerKv = 6;
    constexpr int kPartialStride = 258;
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / kWarpSize;
    const int lane = thread & (kWarpSize - 1);
    const int queries_per_group = static_cast<int>(blockDim.x) / kWarpSize;
    const int query_groups_per_kv = 48 / queries_per_group;
    const int kv_head = static_cast<int>(blockIdx.x) / query_groups_per_kv;
    const int local_query_group = static_cast<int>(blockIdx.x) % query_groups_per_kv;
    const int local_query = local_query_group * queries_per_group + warp;
    const int batch = local_query / kHeadsPerKv;
    const int query_head = kv_head * kHeadsPerKv + local_query % kHeadsPerKv;
    const int query_index = batch * kHeads + query_head;
    const int partition = static_cast<int>(blockIdx.y);
    const int chunk = (context_size + partitions - 1) / partitions;
    const int begin = partition * chunk;
    const int end = min(min(context_size, begin + chunk), base_position + batch + 1);

    const int lane_group = lane / 8;
    const int lane_in_group = lane & 7;

    int preloaded_query[2];
    float preloaded_qscale[2];
    for (int half = 0; half < 2; ++half) {
        const int group = half * 4 + lane_group;
        const std::size_t qg = static_cast<std::size_t>(query_index) * kGroups + group;
        preloaded_query[half] = *reinterpret_cast<const int *>(
            queries + qg * 32 + lane_in_group * 4);
        preloaded_qscale[half] = query_scales[qg] * 0.0625F;
    }

    float maximum = -INFINITY;
    float sum = 0.0F;
    float accumulator[kGroups] = {};

    for (int token = begin; token < end; ++token) {
        float score = 0.0F;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int group = half * 4 + lane_group;
            const auto &key_group = keys[(static_cast<std::size_t>(kv_head) * cache_stride + token)
                                         * kGroups + group];
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
            const int dot = __builtin_amdgcn_sdot4(static_cast<int>(key_word),
                                                    preloaded_query[half], 0, false);
            const float key_scale = __half2float(
                *reinterpret_cast<const __half *>(&key_group.scale));
            score += static_cast<float>(dot) * key_scale * preloaded_qscale[half];
        }
#pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            const float other = __shfl_down(score, offset, kWarpSize);
            if (lane_in_group + offset < 8) {
                score += other;
            }
        }
        score = __shfl(score, 0, kWarpSize) + __shfl(score, 8, kWarpSize)
              + __shfl(score, 16, kWarpSize) + __shfl(score, 24, kWarpSize);

        float rescale = 1.0F;
        float probability = 0.0F;
        if (lane == 0) {
            if (score > maximum) {
                rescale = isinf(maximum) ? 0.0F : expf(maximum - score);
                probability = 1.0F;
                maximum = score;
            } else {
                probability = expf(score - maximum);
            }
            sum = sum * rescale + probability;
        }
        rescale = __shfl(rescale, 0, kWarpSize);
        probability = __shfl(probability, 0, kWarpSize);

        if (rescale != 1.0F) {
#pragma unroll
            for (int group = 0; group < kGroups; ++group) {
                accumulator[group] *= rescale;
            }
            maximum = __shfl(maximum, 0, kWarpSize);
            sum = __shfl(sum, 0, kWarpSize);
        }

#pragma unroll
        for (int group = 0; group < kGroups; ++group) {
            const auto &value_group = values[(static_cast<std::size_t>(kv_head) * cache_stride + token)
                                             * kGroups + group];
            const auto value_byte = value_group.quants[lane / 2];
            const int value_nibble = (value_byte >> ((lane & 1) * 4)) & 0x0F;
            const float value = static_cast<float>((value_nibble ^ 8) - 8)
                              * __half2float(*reinterpret_cast<const __half *>(&value_group.scale));
            accumulator[group] += probability * value;
        }
    }

    float *destination = partials
        + (static_cast<std::size_t>(query_index) * partitions + partition) * kPartialStride;
#pragma unroll
    for (int group = 0; group < kGroups; ++group) {
        destination[group * 32 + lane] = accumulator[group];
    }
    if (lane == 0) {
        destination[256] = maximum;
        destination[257] = sum;
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
            const int quant_offset = half * 64;
            const int value_offset = half * 128;
            for (int lane = 0; lane < kWarpSize; ++lane) {
                const int scale_offset = half * 8 + lane / 16;
                const std::uint8_t low0 = block.low_quants[quant_offset + lane];
                const std::uint8_t low1 = block.low_quants[quant_offset + lane + 32];
                const std::uint8_t high = block.high_quants[half * 32 + lane];
                const int code0 = static_cast<int>(low0 & 0x0F) + ((high & 0x03) << 4) - 32;
                const int code1 = static_cast<int>(low1 & 0x0F) + (((high >> 2) & 0x03) << 4) - 32;
                const int code2 = static_cast<int>(low0 >> 4) + (((high >> 4) & 0x03) << 4) - 32;
                const int code3 = static_cast<int>(low1 >> 4) + (((high >> 6) & 0x03) << 4) - 32;
                const std::size_t base = block_index * kQuantBlock + value_offset + lane;
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
    case 21: return (input_size / kQuantBlock) * output_size * sizeof(IQ3SBlock);
    case 23: return (input_size / kQuantBlock) * output_size * sizeof(IQ4XSBlock);
    default: throw std::invalid_argument("matrix is not a supported quant type");
    }
}

int hybrid_cpu_worker_count() {
    if (const char *requested = std::getenv("QWEN_NATIVE_CPU_THREADS")) {
        return std::clamp(std::atoi(requested), 1, 24);
    }
    // Match the i5-14600K's 20 schedulable CPUs. Oversubscribing 24 workers
    // helps isolated CPU-only microbenchmarks but slows concurrent HIP service.
    return 20;
}

void launch_k_matvec_batch8(std::uint32_t type, const void *weights, const float *input,
                            float *output, std::size_t input_size, std::size_t output_size,
                            hipStream_t stream = nullptr) {
    const dim3 grid(static_cast<unsigned int>(output_size));
    const dim3 block(kWarpSize);
    switch (type) {
    case 8:
        hipLaunchKernelGGL(q8_0_matvec_batch8_kernel, grid, block, 0, stream,
                           static_cast<const Q8_0Block *>(weights), input, output,
                           static_cast<int>(input_size),
                           static_cast<int>(input_size / kQ8Block),
                           static_cast<int>(output_size));
        return;
    case 11:
        hipLaunchKernelGGL(q3k_matvec_batch8_kernel, grid, block, 0, stream,
                           static_cast<const Q3KBlock *>(weights), input, output,
                           static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        return;
    case 12: {
        const dim3 grid4(static_cast<unsigned int>((output_size + 3) / 4));
        const dim3 block4(kWarpSize, 4);
        hipLaunchKernelGGL(q4k_matvec_batch8_kernel, grid4, block4, 0, stream,
                           static_cast<const Q4KBlock *>(weights), input, output,
                           static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        return;
    }
    case 13:
        hipLaunchKernelGGL(q5k_matvec_batch8_kernel, grid, block, 0, stream,
                           static_cast<const Q5KBlock *>(weights), input, output,
                           static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        return;
    case 14:
        hipLaunchKernelGGL(q6k_matvec_batch8_kernel, grid, block, 0, stream,
                           static_cast<const Q6KBlock *>(weights), input, output,
                           static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        return;
    case 20:
        hipLaunchKernelGGL(iq4nl_matvec_batch8_kernel, grid, block, 0, stream,
                           static_cast<const IQ4NLBlock *>(weights), input, output,
                           static_cast<int>(input_size),
                           static_cast<int>(input_size / kIQ4NLBlock),
                           static_cast<int>(output_size));
        return;
    case 21:
        hipLaunchKernelGGL(iq3s_matvec_batch8_kernel, grid, block, 0, stream,
                           static_cast<const IQ3SBlock *>(weights), input, output,
                           static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        return;
    case 23:
        hipLaunchKernelGGL(iq4xs_matvec_batch8_kernel, grid, block, 0, stream,
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
        hipLaunchKernelGGL(q3k_matvec_kernel, grid, block, 0, nullptr,
                           static_cast<const Q3KBlock *>(weights), input, output,
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        return;
    case 21:
        hipLaunchKernelGGL(iq3s_matvec_kernel, grid, block, 0, nullptr,
                           static_cast<const IQ3SBlock *>(weights), input, output,
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

Q4KBenchmarkResult benchmark_host_mapped_batch8(const void *weights, std::uint32_t type,
                                                std::size_t input_size,
                                                std::size_t output_size, int iterations) {
    if (!weights || input_size == 0 || output_size == 0 || iterations <= 0) {
        throw std::invalid_argument("invalid mapped-host benchmark arguments");
    }
    void *device_weights = nullptr;
    hip_check(hipHostGetDevicePointer(&device_weights, const_cast<void *>(weights), 0),
              "mapped-host weight pointer");
    std::vector<float> input(8 * input_size);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::sin(static_cast<float>(index + 1) * 0.00137F);
    }
    float *device_input = nullptr;
    float *device_output = nullptr;
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    hip_check(hipMalloc(reinterpret_cast<void **>(&device_input), input.size() * sizeof(float)),
              "mapped-host benchmark input");
    hip_check(hipMalloc(reinterpret_cast<void **>(&device_output),
                        8 * output_size * sizeof(float)), "mapped-host benchmark output");
    hip_check(hipMemcpy(device_input, input.data(), input.size() * sizeof(float),
                        hipMemcpyHostToDevice), "mapped-host benchmark upload input");
    hip_check(hipEventCreate(&begin), "mapped-host benchmark begin event");
    hip_check(hipEventCreate(&end), "mapped-host benchmark end event");
    launch_k_matvec_batch8(type, device_weights, device_input, device_output,
                           input_size, output_size);
    hip_check(hipDeviceSynchronize(), "mapped-host benchmark warmup");
    hip_check(hipEventRecord(begin), "mapped-host benchmark record begin");
    for (int iteration = 0; iteration < iterations; ++iteration) {
        launch_k_matvec_batch8(type, device_weights, device_input, device_output,
                               input_size, output_size);
    }
    hip_check(hipEventRecord(end), "mapped-host benchmark record end");
    hip_check(hipEventSynchronize(end), "mapped-host benchmark wait");
    float elapsed_ms = 0.0F;
    hip_check(hipEventElapsedTime(&elapsed_ms, begin, end),
              "mapped-host benchmark elapsed");
    Q4KBenchmarkResult result;
    result.milliseconds = elapsed_ms / iterations;
    result.bandwidth_gbps = static_cast<float>(k_quant_weight_bytes(type, input_size, output_size))
                          / (result.milliseconds * 1.0e6F);
    hip_check(hipEventDestroy(end), "mapped-host destroy end");
    hip_check(hipEventDestroy(begin), "mapped-host destroy begin");
    hip_check(hipFree(device_output), "mapped-host free output");
    hip_check(hipFree(device_input), "mapped-host free input");
    return result;
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

ScopedWeightResidency::~ScopedWeightResidency() {
    for (const auto *weights : owned_) resident_weight_cache().release_temporary(weights);
}

void ScopedWeightResidency::add(const void *weights, std::uint32_t type,
                               std::size_t input_size, std::size_t output_size) {
    // Reserve ownership storage before any device allocation so a failed host
    // allocation cannot strand a successfully staged matrix.
    owned_.reserve(owned_.size() + 1);
    if (resident_weight_cache().add_temporary(
            weights, k_quant_weight_bytes(type, input_size, output_size))) {
        owned_.push_back(weights);
    }
}

void set_resident_weight_cache_budget(std::size_t bytes) {
    resident_weight_cache().set_budget(bytes);
}

void clear_resident_weight_cache() {
    resident_weight_cache().clear();
    clear_cpu_packed_matrix_cache();
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

void execute_k_matvec_cached_batch8_top16(const void *weights, std::uint32_t weight_type,
                                           const float *input, std::size_t input_size,
                                           int *out_indices, float *out_scores,
                                           std::size_t output_size, bool keep_resident) {
    constexpr std::size_t batch_size = 8;
    constexpr int kTopK = 16;
    const std::size_t weight_bytes = k_quant_weight_bytes(weight_type, input_size, output_size);
    auto &workspace = matvec_workspace();
    const void *device_weights = resident_weight_cache().find_or_add(weights, weight_bytes, keep_resident);
    if (!device_weights) {
        workspace.reserve_weights(weight_bytes);
        hip_check(hipMemcpy(workspace.weights, weights, weight_bytes, hipMemcpyHostToDevice),
                  "hipMemcpy native cached top16 matvec weights");
        device_weights = workspace.weights;
    }
    workspace.reserve_input(batch_size * input_size);
    workspace.reserve_output(batch_size * output_size);
    workspace.reserve_topk(batch_size * kTopK);
    hip_check(hipMemcpy(workspace.input, input, batch_size * input_size * sizeof(float), hipMemcpyHostToDevice),
              "hipMemcpy native cached top16 matvec input");
    launch_k_matvec_batch8(weight_type, device_weights, workspace.input, workspace.output,
                           input_size, output_size);
    hip_check(hipGetLastError(), "native cached top16 matvec launch");
    hipLaunchKernelGGL(logits_top16_kernel, dim3(batch_size), dim3(256), 0, nullptr,
                       workspace.output, static_cast<int>(output_size),
                       workspace.topk_indices, workspace.topk_scores);
    hip_check(hipGetLastError(), "logits top16 kernel launch");
    hip_check(hipMemcpy(out_indices, workspace.topk_indices, batch_size * kTopK * sizeof(int), hipMemcpyDeviceToHost),
              "hipMemcpy top16 indices");
    hip_check(hipMemcpy(out_scores, workspace.topk_scores, batch_size * kTopK * sizeof(float), hipMemcpyDeviceToHost),
              "hipMemcpy top16 scores");
}

void execute_k_matvec_hybrid_cached_batch8(const void *weights, std::uint32_t weight_type,
                                           const float *input, std::size_t input_size,
                                           float *output, std::size_t output_size,
                                           bool keep_resident) {
    constexpr std::size_t batch_size = 8;
    const std::size_t weight_bytes = k_quant_weight_bytes(weight_type, input_size, output_size);
    auto &cache = resident_weight_cache();
    if (cache.find_or_add(weights, weight_bytes, keep_resident) != nullptr || output_size < 256) {
        execute_k_matvec_cached_batch8(weights, weight_type, input, input_size,
                                       output, output_size, keep_resident);
        return;
    }
    std::size_t cpu_rows = output_size * 3 / 10;
    if (weight_type == 12 || weight_type == 13 || weight_type == 14) {
        cpu_rows = output_size * 2 / 3;
    } else if (weight_type == 23) {
        cpu_rows = output_size * 3 / 5;
    }
    cpu_rows -= cpu_rows % 8;
    const std::size_t gpu_rows = output_size - cpu_rows;
    const std::size_t row_bytes = weight_bytes / output_size;
    std::vector<float> cpu_output(batch_size * cpu_rows);
    std::vector<float> gpu_output(batch_size * gpu_rows);
    auto cpu_work = std::async(std::launch::async, [&] {
        cpu_quant_matvec_batch8_rows_packed(weights, weight_type, input, input_size,
                                            cpu_output.data(), cpu_rows,
                                            hybrid_cpu_worker_count());
    });

    auto &workspace = matvec_workspace();
    workspace.reserve_weights(gpu_rows * row_bytes);
    workspace.reserve_input(batch_size * input_size);
    workspace.reserve_output(batch_size * gpu_rows);
    const auto *gpu_weights = static_cast<const std::uint8_t *>(weights) + cpu_rows * row_bytes;
    hip_check(hipMemcpy(workspace.weights, gpu_weights, gpu_rows * row_bytes, hipMemcpyHostToDevice),
              "hybrid cached batch8 weights");
    hip_check(hipMemcpy(workspace.input, input, batch_size * input_size * sizeof(float), hipMemcpyHostToDevice),
              "hybrid cached batch8 input");
    launch_k_matvec_batch8(weight_type, workspace.weights, workspace.input, workspace.output,
                           input_size, gpu_rows);
    hip_check(hipGetLastError(), "hybrid cached batch8 GPU launch");
    hip_check(hipMemcpy(gpu_output.data(), workspace.output,
                        batch_size * gpu_rows * sizeof(float), hipMemcpyDeviceToHost),
              "hybrid cached batch8 GPU output");
    cpu_work.get();
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        std::copy_n(cpu_output.data() + batch * cpu_rows, cpu_rows,
                    output + batch * output_size);
        std::copy_n(gpu_output.data() + batch * gpu_rows, gpu_rows,
                    output + batch * output_size + cpu_rows);
    }
}

void *create_gated_delta_state() {
    constexpr std::size_t state_values = 48ULL * 128ULL * 128ULL;
    void *state = nullptr;
    hip_check(hipMalloc(&state, state_values * sizeof(float)), "hipMalloc gated-delta state");
    hip_check(hipMemset(state, 0, state_values * sizeof(float)), "hipMemset gated-delta state");
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
    hip_check(hipMemcpy(state, host_state, state_values * sizeof(float), hipMemcpyHostToDevice),
              "hipMemcpy gated-delta state upload");
}

void download_gated_delta_state(const void *state, float *host_state) {
    if (!state || !host_state) {
        throw std::invalid_argument("gated-delta download requires state and host data");
    }
    constexpr std::size_t state_values = 48ULL * 128ULL * 128ULL;
    hip_check(hipMemcpy(host_state, state, state_values * sizeof(float), hipMemcpyDeviceToHost),
              "hipMemcpy gated-delta state download");
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
                       workspace.decays, workspace.betas, static_cast<float *>(state),
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
                       workspace.decays, workspace.betas, static_cast<float *>(state),
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

bool has_n23q4_matrix(const void *weights) {
    return n23_draft_cache().entries.count(weights) != 0;
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
    std::vector<float> reference_state(state_values, 0.0F);
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
                    sum = std::fma(reference_state[state_offset + static_cast<std::size_t>(row) * head_size + column],
                                   keys[vector_offset + column], sum);
                }
                state_key[row] = sum;
            }
            for (int row = 0; row < head_size; ++row) {
                const float update = (values[vector_offset + row] - decays[head] * state_key[row])
                                   * betas[head];
                for (int column = 0; column < head_size; ++column) {
                    const std::size_t offset = state_offset + static_cast<std::size_t>(row) * head_size + column;
                    const float next = decays[head] * reference_state[offset]
                                     + update * keys[vector_offset + column];
                    reference_state[offset] = next;
                }
            }
            for (int row = 0; row < head_size; ++row) {
                float sum = 0.0F;
                for (int column = 0; column < head_size; ++column) {
                    sum = std::fma(reference_state[state_offset + static_cast<std::size_t>(row) * head_size + column],
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
    float *device_states = nullptr;
    float *device_output = nullptr;
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    hip_check(hipMalloc(reinterpret_cast<void **>(&device_queries), queries.size() * sizeof(float)), "hipMalloc GDN queries");
    try {
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_keys), keys.size() * sizeof(float)), "hipMalloc GDN keys");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_values), values.size() * sizeof(float)), "hipMalloc GDN values");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_decays), decays.size() * sizeof(float)), "hipMalloc GDN decays");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_betas), betas.size() * sizeof(float)), "hipMalloc GDN betas");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_states), state_values * sizeof(float)), "hipMalloc GDN states");
        hip_check(hipMalloc(reinterpret_cast<void **>(&device_output), output.size() * sizeof(float)), "hipMalloc GDN output");
        hip_check(hipMemcpy(device_queries, queries.data(), queries.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy GDN queries");
        hip_check(hipMemcpy(device_keys, keys.data(), keys.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy GDN keys");
        hip_check(hipMemcpy(device_values, values.data(), values.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy GDN values");
        hip_check(hipMemcpy(device_decays, decays.data(), decays.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy GDN decays");
        hip_check(hipMemcpy(device_betas, betas.data(), betas.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy GDN betas");
        hip_check(hipMemset(device_states, 0, state_values * sizeof(float)), "hipMemset GDN states");
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
        const std::size_t state_bytes = state_values * sizeof(float);
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
                               device_partials, static_cast<int>(context_size), static_cast<int>(context_size),
                               static_cast<int>(context_size) - 1, partitions);
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

__global__ void rms_norm_batch8_kernel(const float *__restrict__ src,
                                       float *__restrict__ dst,
                                       const float *__restrict__ weights,
                                       int width, float epsilon) {
    const int batch = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    extern __shared__ float shared[];

    const float *in_row = src + static_cast<std::size_t>(batch) * width;
    float *out_row = dst + static_cast<std::size_t>(batch) * width;
    float local_sum = 0.0F;
    for (int i = tid; i < width; i += static_cast<int>(blockDim.x)) {
        const float v = in_row[i];
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
        out_row[i] = in_row[i] * (scale * weights[i]);
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

__global__ void scatter_hybrid_rows_kernel(const float *__restrict__ packed,
                                           float *__restrict__ output,
                                           int row_offset, int packed_rows, int output_rows) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                    + static_cast<int>(threadIdx.x);
    if (index < 8 * packed_rows) {
        const int batch = index / packed_rows;
        const int row = index % packed_rows;
        output[static_cast<std::size_t>(batch) * output_rows + row_offset + row] = packed[index];
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

using AsyncWorker = SingleJobWorker;

struct DevicePipeline {
    hipStream_t compute_stream = nullptr;
    hipStream_t transfer_stream = nullptr;
    hipStream_t activation_stream = nullptr;
    hipEvent_t activation_ready = nullptr;
    hipEvent_t activation_copied = nullptr;
    bool async_activations = false;
    int device_ordinal = 0;
    AsyncWorker async_worker;

    float *hidden = nullptr;
    float *slots[5] = {};
    float *norm_weights_device = nullptr;
    float *hybrid_output = nullptr;
    void *hybrid_weights = nullptr;
    float *hybrid_host_input = nullptr;
    float *hybrid_host_output = nullptr;
    float *hybrid_host_output2 = nullptr;
    float *hybrid_host_output3 = nullptr;

    void *weight_buf[2] = {};
    std::size_t weight_buf_capacity = 0;
    bool pinned_staging = false;
    bool coarse_staging = false;
    void *host_weight_buf[2] = {};
    bool host_weight_pending[2] = {};
    int active_weight_buf = 0;
    const void *prefetched_host_ptr[2] = {};
    std::size_t prefetched_bytes[2] = {};
    hipEvent_t weight_consumed[2] = {};
    bool weight_consumed_valid[2] = {};
    hipEvent_t transfer_done[2] = {};
    bool transfer_done_valid[2] = {};

    void *inline_buf = nullptr;
    std::size_t inline_buf_capacity = 0;
    void *host_inline_buf = nullptr;
    bool host_inline_pending = false;
    const void *inline_prefetched_ptr = nullptr;
    std::size_t inline_prefetched_bytes = 0;
    hipEvent_t inline_done = nullptr;
    bool inline_done_valid = false;
    hipEvent_t inline_consumed = nullptr;
    bool inline_consumed_valid = false;

    std::size_t embedding_size = 0;
    std::size_t ffn_size = 0;
    std::size_t max_weight_bytes = 0;

    ~DevicePipeline() {
        // An unwinding GPU error must not release buffers still read/written
        // by an accepted CPU job. drain() does not rethrow that job's error.
        async_worker.drain();
        if (compute_stream) (void)hipStreamSynchronize(compute_stream);
        if (transfer_stream) (void)hipStreamSynchronize(transfer_stream);
        if (activation_stream) (void)hipStreamSynchronize(activation_stream);
        if (hidden) (void)hipFree(hidden);
        for (auto &s : slots) if (s) (void)hipFree(s);
        if (norm_weights_device) (void)hipFree(norm_weights_device);
        if (hybrid_output) (void)hipFree(hybrid_output);
        if (hybrid_weights) (void)hipFree(hybrid_weights);
        if (hybrid_host_input) (void)hipHostFree(hybrid_host_input);
        if (hybrid_host_output) (void)hipHostFree(hybrid_host_output);
        if (hybrid_host_output2) (void)hipHostFree(hybrid_host_output2);
        if (hybrid_host_output3) (void)hipHostFree(hybrid_host_output3);
        for (auto &w : weight_buf) if (w) (void)hipFree(w);
        for (auto &w : host_weight_buf) if (w) (void)hipHostFree(w);
        if (host_inline_buf) (void)hipHostFree(host_inline_buf);
        for (auto &e : weight_consumed) if (e) (void)hipEventDestroy(e);
        for (auto &e : transfer_done) if (e) (void)hipEventDestroy(e);
        if (inline_buf) (void)hipFree(inline_buf);
        if (inline_done) (void)hipEventDestroy(inline_done);
        if (inline_consumed) (void)hipEventDestroy(inline_consumed);
        if (activation_ready) (void)hipEventDestroy(activation_ready);
        if (activation_copied) (void)hipEventDestroy(activation_copied);
        if (compute_stream) (void)hipStreamDestroy(compute_stream);
        if (transfer_stream) (void)hipStreamDestroy(transfer_stream);
        if (activation_stream) (void)hipStreamDestroy(activation_stream);
    }
};

DevicePipeline *create_device_pipeline(std::size_t embedding_size, std::size_t ffn_size,
                                       std::size_t max_weight_bytes) {
    auto p = std::make_unique<DevicePipeline>();
    p->async_activations = environment_flag_enabled("QWEN_NATIVE_ASYNC_ACTIVATIONS");
    p->pinned_staging = environment_flag_enabled("QWEN_NATIVE_PINNED_STAGING");
    p->coarse_staging = environment_flag_enabled("QWEN_NATIVE_PINNED_STAGING_COARSE");
    hip_check(hipGetDevice(&p->device_ordinal), "pipeline current device");
    p->embedding_size = embedding_size;
    p->ffn_size = ffn_size;
    p->max_weight_bytes = max_weight_bytes;

    hip_check(hipStreamCreate(&p->compute_stream), "pipeline compute stream");
    hip_check(hipStreamCreate(&p->transfer_stream), "pipeline transfer stream");
    if (p->async_activations) {
        hip_check(hipStreamCreate(&p->activation_stream), "pipeline activation stream");
        hip_check(hipEventCreateWithFlags(&p->activation_ready, hipEventDisableTiming),
                  "pipeline activation ready event");
        hip_check(hipEventCreateWithFlags(&p->activation_copied, hipEventDisableTiming),
                  "pipeline activation copied event");
    }

    const std::size_t hidden_bytes = 8 * std::max(embedding_size, ffn_size) * sizeof(float);
    const std::size_t ffn_bytes = 8 * ffn_size * sizeof(float);
    hip_check(hipMalloc(reinterpret_cast<void **>(&p->hidden), hidden_bytes), "pipeline hidden");
    for (auto &s : p->slots) {
        hip_check(hipMalloc(reinterpret_cast<void **>(&s), ffn_bytes), "pipeline slot");
    }
    const std::size_t norm_bytes = std::max(embedding_size, ffn_size) * sizeof(float);
    hip_check(hipMalloc(reinterpret_cast<void **>(&p->norm_weights_device), norm_bytes), "pipeline norm");
    const bool use_hybrid_rows = std::getenv("QWEN_NATIVE_HYBRID") != nullptr;
    const bool use_hybrid_ffn = std::getenv("QWEN_NATIVE_HYBRID_FFN") != nullptr;
    const bool use_cpu_gate = !std::getenv("QWEN_NATIVE_GPU_GATE");
    if (use_hybrid_rows || use_hybrid_ffn || use_cpu_gate) {
        const std::size_t hybrid_bytes = 8 * std::max(embedding_size, ffn_size) * sizeof(float);
        hip_check(hipHostMalloc(reinterpret_cast<void **>(&p->hybrid_host_input), hybrid_bytes, hipHostMallocDefault),
                  "pipeline pinned host input");
        hip_check(hipHostMalloc(reinterpret_cast<void **>(&p->hybrid_host_output), hybrid_bytes, hipHostMallocDefault),
                  "pipeline pinned host output");
        hip_check(hipHostMalloc(reinterpret_cast<void **>(&p->hybrid_host_output2), hybrid_bytes, hipHostMallocDefault),
                  "pipeline pinned host output2");
        hip_check(hipHostMalloc(reinterpret_cast<void **>(&p->hybrid_host_output3), hybrid_bytes, hipHostMallocDefault),
                  "pipeline pinned host output3");
    }

    p->weight_buf_capacity = max_weight_bytes;
    for (auto &w : p->weight_buf) {
        hip_check(hipMalloc(&w, max_weight_bytes), "pipeline weight buf");
    }
    if (p->pinned_staging) {
        const unsigned flags = p->coarse_staging ? hipHostMallocNonCoherent : hipHostMallocDefault;
        for (auto &w : p->host_weight_buf) {
            hip_check(hipHostMalloc(&w, max_weight_bytes, flags),
                      "pipeline bounded pinned weight staging");
        }
        hip_check(hipHostMalloc(&p->host_inline_buf, max_weight_bytes, flags),
                  "pipeline bounded pinned inline staging");
    }
    for (auto &e : p->weight_consumed) {
        hip_check(hipEventCreateWithFlags(&e, hipEventDisableTiming), "pipeline weight event");
    }
    const unsigned copy_event_flags = hipEventDisableTiming |
        (p->pinned_staging && p->coarse_staging ? hipEventReleaseToSystem : 0);
    for (auto &e : p->transfer_done) {
        hip_check(hipEventCreateWithFlags(&e, copy_event_flags), "pipeline transfer done event");
    }
    hip_check(hipMalloc(&p->inline_buf, max_weight_bytes), "pipeline inline buf");
    p->inline_buf_capacity = max_weight_bytes;
    hip_check(hipEventCreateWithFlags(&p->inline_done, copy_event_flags), "pipeline inline event");
    hip_check(hipEventCreateWithFlags(&p->inline_consumed, hipEventDisableTiming),
              "pipeline inline consumed event");
    return p.release();
}

void destroy_device_pipeline(DevicePipeline *pipeline) {
    delete pipeline;
}

const void *pipeline_stage_host_weights(DevicePipeline *pipeline, const void *source,
                                        std::size_t bytes, int buffer) {
    if (!pipeline->pinned_staging) return source;
    if (!source || bytes == 0 || buffer < -1 || buffer > 1 || bytes > pipeline->max_weight_bytes) {
        throw std::invalid_argument("weight copy exceeds bounded host staging capacity");
    }
    if (buffer >= 0) {
        if (pipeline->host_weight_pending[buffer]) {
            hip_check(hipEventSynchronize(pipeline->transfer_done[buffer]),
                      "wait previous pinned weight DMA before host reuse");
        }
        std::memcpy(pipeline->host_weight_buf[buffer], source, bytes);
        pipeline->host_weight_pending[buffer] = true;
        return pipeline->host_weight_buf[buffer];
    }
    if (pipeline->host_inline_pending) {
        hip_check(hipEventSynchronize(pipeline->inline_done),
                  "wait previous pinned inline DMA before host reuse");
    }
    std::memcpy(pipeline->host_inline_buf, source, bytes);
    pipeline->host_inline_pending = true;
    return pipeline->host_inline_buf;
}

namespace {

// Input is read-only throughout each projection/grouped job below. The worker
// owns pinned host input until wait(); callers cannot reuse it before then.
// A separate D2H stream avoids blocking activation delivery behind weight H2D.
void pipeline_stage_cpu_input(DevicePipeline *pipeline, const float *source,
                              std::size_t count) {
    if (!source || !pipeline->hybrid_host_input || count == 0 ||
        count > 8 * std::max(pipeline->embedding_size, pipeline->ffn_size)) {
        throw std::invalid_argument("invalid pipeline CPU activation transfer");
    }
    if (!pipeline->async_activations) {
        hip_check(hipStreamSynchronize(pipeline->compute_stream), "CPU activation wait input");
        hip_check(hipMemcpy(pipeline->hybrid_host_input, source, count * sizeof(float),
                           hipMemcpyDeviceToHost), "CPU activation download input");
        return;
    }
    hip_check(hipEventRecord(pipeline->activation_ready, pipeline->compute_stream),
              "CPU activation record ready");
    hip_check(hipStreamWaitEvent(pipeline->activation_stream, pipeline->activation_ready, 0),
              "CPU activation wait producer");
    hip_check(hipMemcpyAsync(pipeline->hybrid_host_input, source, count * sizeof(float),
                            hipMemcpyDeviceToHost, pipeline->activation_stream),
              "CPU activation async download");
    hip_check(hipEventRecord(pipeline->activation_copied, pipeline->activation_stream),
              "CPU activation record copied");
}

void pipeline_wait_cpu_input(DevicePipeline *pipeline) {
    if (pipeline->async_activations) {
        hip_check(hipSetDevice(pipeline->device_ordinal), "CPU activation select device");
        hip_check(hipEventSynchronize(pipeline->activation_copied), "CPU activation await copy");
    }
}

} // namespace

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

void pipeline_download_hidden_async(DevicePipeline *pipeline, float *host, std::size_t count) {
    hip_check(hipMemcpyAsync(host, pipeline->hidden, count * sizeof(float), hipMemcpyDeviceToHost,
                             pipeline->compute_stream),
              "pipeline download hidden async");
}

void pipeline_rms_norm(DevicePipeline *pipeline, const float *norm_weights,
                       std::size_t width, float epsilon,
                       float *destination, const float *source) {
    hip_check(hipMemcpyAsync(pipeline->norm_weights_device, norm_weights,
                             width * sizeof(float), hipMemcpyHostToDevice,
                             pipeline->compute_stream),
              "pipeline upload norm weights");
    constexpr int threads = 256;
    const std::size_t shared = threads * sizeof(float);
    const float *src = source ? source : pipeline->hidden;
    float *dst = destination ? destination : pipeline->hidden;
    hipLaunchKernelGGL(rms_norm_batch8_kernel, dim3(8), dim3(threads), shared,
                       pipeline->compute_stream,
                       src, dst, pipeline->norm_weights_device,
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

float *pipeline_slot_ptr(DevicePipeline *pipeline, int slot) {
    return pipeline->slots[slot];
}

float *pipeline_hidden_ptr(DevicePipeline *pipeline) {
    return pipeline->hidden;
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
                             std::size_t output_size, bool keep_resident, int output_slot,
                             const float *input_buffer) {
    const std::size_t weight_bytes = k_quant_weight_bytes(type, input_size, output_size);
    auto &cache = resident_weight_cache();
    const void *device_weights = cache.find_or_add(weights, weight_bytes, keep_resident);
    int used_buf = -1;
    bool used_inline = false;

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
                hip_check(hipStreamWaitEvent(pipeline->compute_stream, pipeline->transfer_done[found_buf], 0),
                          "pipeline wait prefetch buf");
                pipeline->transfer_done_valid[found_buf] = false;
            }
            device_weights = pipeline->weight_buf[found_buf];
            pipeline->prefetched_host_ptr[found_buf] = nullptr;
            used_buf = found_buf;
        } else if (pipeline->inline_prefetched_ptr == weights &&
                   pipeline->inline_prefetched_bytes == weight_bytes) {
            if (pipeline->inline_done_valid) {
                hip_check(hipStreamWaitEvent(pipeline->compute_stream, pipeline->inline_done, 0),
                          "pipeline wait inline prefetch");
                pipeline->inline_done_valid = false;
            }
            device_weights = pipeline->inline_buf;
            pipeline->inline_prefetched_ptr = nullptr;
            used_inline = true;
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
            const auto *host_weights = pipeline_stage_host_weights(pipeline, weights, weight_bytes, buf);
            hip_check(hipMemcpyAsync(pipeline->weight_buf[buf], host_weights, weight_bytes,
                                     hipMemcpyHostToDevice, pipeline->compute_stream),
                      "pipeline weight upload");
            if (pipeline->pinned_staging) {
                hip_check(hipEventRecord(pipeline->transfer_done[buf], pipeline->compute_stream),
                          "pipeline direct pinned upload complete");
            }
            device_weights = pipeline->weight_buf[buf];
            used_buf = buf;
        }
    }

    const float *in_act = input_buffer ? input_buffer : pipeline->hidden;
    float *out = pipeline->slots[output_slot];
    const dim3 grid(static_cast<unsigned>(output_size));
    const dim3 block(kWarpSize);
    switch (type) {
    case 8:
        hipLaunchKernelGGL(q8_0_matvec_batch8_kernel, grid, block, 0, pipeline->compute_stream,
                           static_cast<const Q8_0Block *>(device_weights), in_act,
                           out, static_cast<int>(input_size),
                           static_cast<int>(input_size / kQ8Block),
                           static_cast<int>(output_size));
        break;
    case 11:
        hipLaunchKernelGGL(q3k_matvec_batch8_kernel, grid, block, 0, pipeline->compute_stream,
                           static_cast<const Q3KBlock *>(device_weights), in_act,
                           out, static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        break;
    case 12: {
        const dim3 grid4(static_cast<unsigned>((output_size + 3) / 4));
        const dim3 block4(kWarpSize, 4);
        hipLaunchKernelGGL(q4k_matvec_batch8_kernel, grid4, block4, 0, pipeline->compute_stream,
                           static_cast<const Q4KBlock *>(device_weights), in_act,
                           out, static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        break;
    }
    case 13:
        hipLaunchKernelGGL(q5k_matvec_batch8_kernel, grid, block, 0, pipeline->compute_stream,
                           static_cast<const Q5KBlock *>(device_weights), in_act,
                           out, static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        break;
    case 14:
        hipLaunchKernelGGL(q6k_matvec_batch8_kernel, grid, block, 0, pipeline->compute_stream,
                           static_cast<const Q6KBlock *>(device_weights), in_act,
                           out, static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        break;
    case 20:
        hipLaunchKernelGGL(iq4nl_matvec_batch8_kernel, grid, block, 0, pipeline->compute_stream,
                           static_cast<const IQ4NLBlock *>(device_weights), in_act,
                           out, static_cast<int>(input_size),
                           static_cast<int>(input_size / kIQ4NLBlock),
                           static_cast<int>(output_size));
        break;
    case 21:
        hipLaunchKernelGGL(iq3s_matvec_batch8_kernel, grid, block, 0, pipeline->compute_stream,
                           static_cast<const IQ3SBlock *>(device_weights), in_act,
                           out, static_cast<int>(input_size),
                           static_cast<int>(input_size / kQuantBlock),
                           static_cast<int>(output_size));
        break;
    case 23:
        hipLaunchKernelGGL(iq4xs_matvec_batch8_kernel, grid, block, 0, pipeline->compute_stream,
                           static_cast<const IQ4XSBlock *>(device_weights), in_act,
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
    if (used_inline) {
        hip_check(hipEventRecord(pipeline->inline_consumed, pipeline->compute_stream),
                  "pipeline record inline consumed");
        pipeline->inline_consumed_valid = true;
    }
}

std::size_t hybrid_cpu_row_count(std::uint32_t type, std::size_t output_size) {
    // Balance exact output-row shards from measured per-format throughput.
    // Q4_K/Q5_K/IQ4_XS sustain about 24 GB/s, Q6_K about 19 GB/s, and
    // streamed GPU rows about 10 GB/s on this host. Keep row counts aligned
    // to the eight-row CPU SoA tiles.
    std::size_t numerator = 1;
    std::size_t denominator = 2;
    if (type == 12 || type == 13 || type == 23) {
        numerator = 9; denominator = 10;
    } else if (type == 14) {
        numerator = 17; denominator = 20;
    } else if (type == 11 || type == 21) {
        numerator = 1; denominator = 5;
    } else if (type == 20) {
        numerator = 2; denominator = 5;
    }
    if (const char *split = std::getenv("QWEN_NATIVE_SOA_SPLIT")) {
        const int percentage = std::clamp(std::atoi(split), 10, 90);
        if (type == 12 || type == 13 || type == 14 || type == 23) {
            numerator = static_cast<std::size_t>(percentage);
            denominator = 100;
        }
    }
    std::size_t rows = output_size * numerator / denominator;
    rows -= rows % 8;
    return rows;
}

void pipeline_project_gpu_rows_pipelined(DevicePipeline *pipeline,
                                         const void *weights, std::uint32_t type,
                                         std::size_t input_size, std::size_t output_size,
                                         std::size_t destination_row,
                                         std::size_t destination_stride,
                                         int destination_slot,
                                         const float *input_buffer = nullptr) {
    const std::size_t weight_bytes = k_quant_weight_bytes(type, input_size, output_size);
    const std::size_t row_bytes = weight_bytes / output_size;
    std::size_t target_chunk_bytes = 8ULL * 1024ULL * 1024ULL;
    if (const char *requested = std::getenv("QWEN_NATIVE_GPU_CHUNK_MB")) {
        const int mebibytes = std::clamp(std::atoi(requested), 1, 256);
        target_chunk_bytes = static_cast<std::size_t>(mebibytes) * 1024ULL * 1024ULL;
    }
    std::size_t chunk_rows = std::max<std::size_t>(8, target_chunk_bytes / row_bytes);
    chunk_rows -= chunk_rows % 8;
    chunk_rows = std::min(chunk_rows, output_size);

    // Hybrid mode does not issue matrix prefetches, but clear any stale
    // speculative prefetch bookkeeping before borrowing the two staging
    // buffers as a copy/compute pipeline.
    if (pipeline->prefetched_host_ptr[0] || pipeline->prefetched_host_ptr[1]) {
        hip_check(hipStreamSynchronize(pipeline->transfer_stream),
                  "hybrid row pipeline drain stale prefetch");
        pipeline->prefetched_host_ptr[0] = nullptr;
        pipeline->prefetched_host_ptr[1] = nullptr;
    }

    const auto *source = static_cast<const std::uint8_t *>(weights);
    const float *in_act = input_buffer ? input_buffer : pipeline->hidden;
    for (std::size_t row = 0; row < output_size; row += chunk_rows) {
        const std::size_t rows = std::min(chunk_rows, output_size - row);
        const int buffer = pipeline->active_weight_buf;
        pipeline->active_weight_buf = 1 - buffer;
        if (pipeline->weight_consumed_valid[buffer]) {
            hip_check(hipStreamWaitEvent(pipeline->transfer_stream,
                                         pipeline->weight_consumed[buffer], 0),
                      "hybrid row pipeline wait consumed");
        }
        const auto *host_weights = pipeline_stage_host_weights(
            pipeline, source + row * row_bytes, rows * row_bytes, buffer);
        hip_check(hipMemcpyAsync(pipeline->weight_buf[buffer], host_weights,
                                 rows * row_bytes, hipMemcpyHostToDevice,
                                 pipeline->transfer_stream),
                  "hybrid row pipeline upload");
        hip_check(hipEventRecord(pipeline->transfer_done[buffer],
                                 pipeline->transfer_stream),
                  "hybrid row pipeline record transfer");
        hip_check(hipStreamWaitEvent(pipeline->compute_stream,
                                     pipeline->transfer_done[buffer], 0),
                  "hybrid row pipeline compute wait");
        launch_k_matvec_batch8(type, pipeline->weight_buf[buffer], in_act,
                               pipeline->slots[4], input_size, rows,
                               pipeline->compute_stream);
        constexpr int threads = 256;
        const int elements = static_cast<int>(8 * rows);
        hipLaunchKernelGGL(scatter_hybrid_rows_kernel,
                           dim3((elements + threads - 1) / threads), dim3(threads),
                           0, pipeline->compute_stream,
                           pipeline->slots[4], pipeline->slots[destination_slot],
                           static_cast<int>(destination_row + row),
                           static_cast<int>(rows),
                           static_cast<int>(destination_stride));
        hip_check(hipGetLastError(), "hybrid row pipeline launch");
        hip_check(hipEventRecord(pipeline->weight_consumed[buffer],
                                 pipeline->compute_stream),
                  "hybrid row pipeline record consumed");
        pipeline->weight_consumed_valid[buffer] = true;
    }
}

void pipeline_project_batch8_hybrid(DevicePipeline *pipeline, const void *weights,
                                    std::uint32_t type, std::size_t input_size,
                                    std::size_t output_size, bool keep_resident,
                                    int output_slot,
                                    const float *input_buffer) {
    const std::size_t weight_bytes = k_quant_weight_bytes(type, input_size, output_size);
    if (resident_weight_cache().find_or_add(weights, weight_bytes, keep_resident) != nullptr ||
        pipeline->inline_prefetched_ptr == weights ||
        pipeline->prefetched_host_ptr[0] == weights ||
        pipeline->prefetched_host_ptr[1] == weights ||
        output_size < 256 || type == 21) {
        pipeline_project_batch8(pipeline, weights, type, input_size, output_size,
                                keep_resident, output_slot, input_buffer);
        return;
    }

    const std::size_t cpu_rows = hybrid_cpu_row_count(type, output_size);
    const std::size_t gpu_rows = output_size - cpu_rows;
    const std::size_t row_bytes = weight_bytes / output_size;
    if (gpu_rows * row_bytes > pipeline->max_weight_bytes) {
        throw std::runtime_error("hybrid projection exceeds weight staging capacity");
    }

    const float *in_act = input_buffer ? input_buffer : pipeline->hidden;
    pipeline_stage_cpu_input(pipeline, in_act, 8 * input_size);
    if (pipeline->async_activations && in_act == pipeline->slots[output_slot]) {
        hip_check(hipStreamWaitEvent(pipeline->compute_stream, pipeline->activation_copied, 0),
                  "hybrid projection preserve aliased input");
    }

    pipeline->async_worker.submit([=] {
        pipeline_wait_cpu_input(pipeline);
        cpu_quant_matvec_batch8_rows_packed(weights, type,
                                            pipeline->hybrid_host_input, input_size,
                                            pipeline->hybrid_host_output, cpu_rows,
                                            hybrid_cpu_worker_count());
    });

    const auto *gpu_weights = static_cast<const std::uint8_t *>(weights) + cpu_rows * row_bytes;
    const bool pipeline_gpu_rows = !std::getenv("QWEN_NATIVE_GPU_PIPELINE") || std::string(std::getenv("QWEN_NATIVE_GPU_PIPELINE")) != "0";
    if (pipeline_gpu_rows) {
        pipeline_project_gpu_rows_pipelined(pipeline, gpu_weights, type, input_size,
                                            gpu_rows, cpu_rows, output_size, output_slot,
                                            input_buffer);
    } else {
        pipeline_project_batch8(pipeline, gpu_weights, type, input_size, gpu_rows,
                                false, 4, input_buffer);
    }

    hip_check(hipStreamSynchronize(pipeline->compute_stream),
              "hybrid projection wait GPU rows");
    pipeline->async_worker.wait();

    for (int batch_index = 0; batch_index < 8; ++batch_index) {
        hip_check(hipMemcpyAsync(pipeline->slots[output_slot]
                                     + static_cast<std::size_t>(batch_index) * output_size,
                                 pipeline->hybrid_host_output
                                     + static_cast<std::size_t>(batch_index) * cpu_rows,
                                 cpu_rows * sizeof(float), hipMemcpyHostToDevice,
                                 pipeline->compute_stream),
                  "hybrid projection upload CPU rows");
    }
    if (!pipeline_gpu_rows) {
        constexpr int threads = 256;
        const int elements = static_cast<int>(8 * gpu_rows);
        hipLaunchKernelGGL(scatter_hybrid_rows_kernel,
                           dim3((elements + threads - 1) / threads), dim3(threads),
                           0, pipeline->compute_stream,
                           pipeline->slots[4], pipeline->slots[output_slot],
                           static_cast<int>(cpu_rows), static_cast<int>(gpu_rows),
                           static_cast<int>(output_size));
        hip_check(hipGetLastError(), "hybrid projection scatter launch");
    }
}

void pipeline_project_attention_qkv_hybrid(
    DevicePipeline *pipeline,
    const void *q_weights, std::uint32_t q_type, std::size_t q_size,
    const void *k_weights, std::uint32_t k_type, std::size_t k_size,
    const void *v_weights, std::uint32_t v_type, std::size_t v_size,
    std::size_t input_size, bool keep_resident,
    const float *input_buffer) {
    const auto soa_supported = [](std::uint32_t type) {
        return type == 12 || type == 13 || type == 14 || type == 23;
    };
    if (!soa_supported(q_type) || !soa_supported(k_type) || !soa_supported(v_type)) {
        pipeline_project_batch8_hybrid(pipeline, q_weights, q_type, input_size,
                                       q_size, keep_resident, 0, input_buffer);
        pipeline_project_batch8_hybrid(pipeline, k_weights, k_type, input_size,
                                       k_size, keep_resident, 1, input_buffer);
        pipeline_project_batch8_hybrid(pipeline, v_weights, v_type, input_size,
                                       v_size, keep_resident, 2, input_buffer);
        return;
    }

    const std::size_t q_bytes = k_quant_weight_bytes(q_type, input_size, q_size);
    const std::size_t k_bytes = k_quant_weight_bytes(k_type, input_size, k_size);
    const std::size_t v_bytes = k_quant_weight_bytes(v_type, input_size, v_size);
    auto &cache = resident_weight_cache();
    const bool q_resident = cache.find_or_add(q_weights, q_bytes, keep_resident) != nullptr;
    const bool k_resident = cache.find_or_add(k_weights, k_bytes, keep_resident) != nullptr;
    const bool v_resident = cache.find_or_add(v_weights, v_bytes, keep_resident) != nullptr;
    if (q_resident && k_resident && v_resident) {
        pipeline_project_batch8(pipeline, q_weights, q_type, input_size, q_size,
                                keep_resident, 0, input_buffer);
        pipeline_project_batch8(pipeline, k_weights, k_type, input_size, k_size,
                                keep_resident, 1, input_buffer);
        pipeline_project_batch8(pipeline, v_weights, v_type, input_size, v_size,
                                keep_resident, 2, input_buffer);
        return;
    }

    const float *in_act = input_buffer ? input_buffer : pipeline->hidden;
    pipeline_stage_cpu_input(pipeline, in_act, 8 * input_size);
    const std::size_t q_cpu_rows = q_resident ? 0 : hybrid_cpu_row_count(q_type, q_size);
    const std::size_t k_cpu_rows = k_resident ? 0 : hybrid_cpu_row_count(k_type, k_size);
    const std::size_t v_cpu_rows = v_resident ? 0 : hybrid_cpu_row_count(v_type, v_size);

    const bool pipeline_gpu_rows = !std::getenv("QWEN_NATIVE_GPU_PIPELINE") || std::string(std::getenv("QWEN_NATIVE_GPU_PIPELINE")) != "0";
    const auto launch = [&](const void *weights, std::uint32_t type,
                            std::size_t output_size, std::size_t weight_bytes,
                            std::size_t cpu_rows, bool resident, int output_slot) {
        if (resident) {
            pipeline_project_batch8(pipeline, weights, type, input_size, output_size,
                                    keep_resident, output_slot, input_buffer);
            return;
        }
        const std::size_t gpu_rows = output_size - cpu_rows;
        const std::size_t row_bytes = weight_bytes / output_size;
        const auto *gpu_weights = static_cast<const std::uint8_t *>(weights)
                                + cpu_rows * row_bytes;
        if (pipeline_gpu_rows) {
            pipeline_project_gpu_rows_pipelined(
                pipeline, gpu_weights, type, input_size, gpu_rows,
                cpu_rows, output_size, output_slot, input_buffer);
        } else {
            pipeline_project_batch8(pipeline, gpu_weights, type, input_size,
                                    gpu_rows, false, 4, input_buffer);
            constexpr int threads = 256;
            const int elements = static_cast<int>(8 * gpu_rows);
            hipLaunchKernelGGL(scatter_hybrid_rows_kernel,
                               dim3((elements + threads - 1) / threads), dim3(threads),
                               0, pipeline->compute_stream,
                               pipeline->slots[4], pipeline->slots[output_slot],
                               static_cast<int>(cpu_rows), static_cast<int>(gpu_rows),
                               static_cast<int>(output_size));
            hip_check(hipGetLastError(), "grouped attention QKV scatter launch");
        }
    };
    pipeline->async_worker.submit([=] {
        pipeline_wait_cpu_input(pipeline);
        if (!q_resident) {
            cpu_quant_matvec_batch8_rows_packed(
                q_weights, q_type, pipeline->hybrid_host_input, input_size,
                pipeline->hybrid_host_output, q_cpu_rows,
                hybrid_cpu_worker_count());
        }
        if (!k_resident) {
            cpu_quant_matvec_batch8_rows_packed(
                k_weights, k_type, pipeline->hybrid_host_input, input_size,
                pipeline->hybrid_host_output2, k_cpu_rows,
                hybrid_cpu_worker_count());
        }
        if (!v_resident) {
            cpu_quant_matvec_batch8_rows_packed(
                v_weights, v_type, pipeline->hybrid_host_input, input_size,
                pipeline->hybrid_host_output3, v_cpu_rows,
                hybrid_cpu_worker_count());
        }
    });

    launch(q_weights, q_type, q_size, q_bytes, q_cpu_rows, q_resident, 0);
    launch(k_weights, k_type, k_size, k_bytes, k_cpu_rows, k_resident, 1);
    launch(v_weights, v_type, v_size, v_bytes, v_cpu_rows, v_resident, 2);

    hip_check(hipStreamSynchronize(pipeline->compute_stream),
              "grouped attention QKV wait GPU rows");
    pipeline->async_worker.wait();

    const auto upload = [&](const float *host, std::size_t cpu_rows,
                            std::size_t output_size, int output_slot, bool resident) {
        if (resident) return;
        for (int batch = 0; batch < 8; ++batch) {
            hip_check(hipMemcpyAsync(
                          pipeline->slots[output_slot]
                              + static_cast<std::size_t>(batch) * output_size,
                          host + static_cast<std::size_t>(batch) * cpu_rows,
                          cpu_rows * sizeof(float), hipMemcpyHostToDevice,
                          pipeline->compute_stream),
                      "grouped attention QKV upload CPU rows");
        }
    };
    upload(pipeline->hybrid_host_output, q_cpu_rows, q_size, 0, q_resident);
    upload(pipeline->hybrid_host_output2, k_cpu_rows, k_size, 1, k_resident);
    upload(pipeline->hybrid_host_output3, v_cpu_rows, v_size, 2, v_resident);
}

void pipeline_project_recurrent_qkv_gate_cpu(DevicePipeline *pipeline,
                                             const void *qkv_weights, std::uint32_t qkv_type,
                                             std::size_t input_size, std::size_t qkv_size,
                                             const void *gate_weights, std::uint32_t gate_type,
                                             std::size_t gate_size, bool keep_resident,
                                             const float *input_buffer) {
    const float *in_act = input_buffer ? input_buffer : pipeline->hidden;
    pipeline_stage_cpu_input(pipeline, in_act, 8 * input_size);
    pipeline->async_worker.submit([=] {
        pipeline_wait_cpu_input(pipeline);
        cpu_quant_matvec_batch8_rows_packed(gate_weights, gate_type,
                                            pipeline->hybrid_host_input, input_size,
                                            pipeline->hybrid_host_output, gate_size,
                                            hybrid_cpu_worker_count());
    });
    pipeline_project_batch8(pipeline, qkv_weights, qkv_type, input_size, qkv_size,
                            keep_resident, 0, in_act);
    hip_check(hipStreamSynchronize(pipeline->compute_stream), "CPU gate wait QKV");
    pipeline->async_worker.wait();
    hip_check(hipMemcpyAsync(pipeline->slots[1], pipeline->hybrid_host_output,
                             8 * gate_size * sizeof(float), hipMemcpyHostToDevice,
                             pipeline->compute_stream),
              "CPU gate upload output");
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
    const auto *host_weights = pipeline_stage_host_weights(pipeline, weights, weight_bytes, buf);
    hip_check(hipMemcpyAsync(pipeline->weight_buf[buf], host_weights,
                             weight_bytes, hipMemcpyHostToDevice, pipeline->transfer_stream),
              "pipeline prefetch weights");
    hip_check(hipEventRecord(pipeline->transfer_done[buf], pipeline->transfer_stream),
              "pipeline prefetch record done");
    pipeline->transfer_done_valid[buf] = true;
    pipeline->prefetched_host_ptr[buf] = weights;
    pipeline->prefetched_bytes[buf] = weight_bytes;
}

void pipeline_prefetch_hybrid_weights(DevicePipeline *pipeline, const void *weights,
                                      std::uint32_t type, std::size_t input_size,
                                      std::size_t output_size) {
    const std::size_t full_bytes = k_quant_weight_bytes(type, input_size, output_size);
    if (resident_weight_cache().find_or_add(weights, full_bytes, false) != nullptr) return;
    const std::size_t cpu_rows = hybrid_cpu_row_count(type, output_size);
    const std::size_t gpu_rows = output_size - cpu_rows;
    const std::size_t row_bytes = full_bytes / output_size;
    const auto *gpu_weights = static_cast<const std::uint8_t *>(weights) + cpu_rows * row_bytes;
    pipeline_prefetch_weights(pipeline, gpu_weights, type, input_size, gpu_rows);
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
    if (pipeline->inline_consumed_valid) {
        hip_check(hipStreamWaitEvent(pipeline->transfer_stream, pipeline->inline_consumed, 0),
                  "pipeline inline prefetch wait consumed");
        pipeline->inline_consumed_valid = false;
    }
    const auto *host_weights = pipeline_stage_host_weights(pipeline, weights, weight_bytes, -1);
    hip_check(hipMemcpyAsync(pipeline->inline_buf, host_weights,
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
    // Gate/up occupy the two rotating buffers. Keep down in the separately
    // synchronized inline buffer so all three transfers can be pipelined.
    pipeline_prefetch_inline(pipeline, down_weights, down_type, ffn_size, embedding_size);

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

void pipeline_fused_ffn_batch8_hybrid_up(DevicePipeline *pipeline, const float *norm_weights,
                                         std::size_t embedding_size, std::size_t ffn_size,
                                         float epsilon,
                                         const void *gate_weights, std::uint32_t gate_type,
                                         const void *up_weights, std::uint32_t up_type,
                                         const void *down_weights, std::uint32_t down_type,
                                         bool keep_gate_resident) {
    pipeline_prefetch_inline(pipeline, down_weights, down_type, ffn_size, embedding_size);
    pipeline_save_hidden(pipeline, 3, 8 * embedding_size);
    pipeline_rms_norm(pipeline, norm_weights, embedding_size, epsilon);
    pipeline_stage_cpu_input(pipeline, pipeline->hidden, 8 * embedding_size);

    pipeline->async_worker.submit([=] {
        pipeline_wait_cpu_input(pipeline);
        cpu_quant_matvec_batch8_rows(up_weights, up_type,
                                     pipeline->hybrid_host_input, embedding_size,
                                     pipeline->hybrid_host_output, ffn_size,
                                     static_cast<int>(std::thread::hardware_concurrency()));
    });
    pipeline_project_batch8(pipeline, gate_weights, gate_type, embedding_size, ffn_size,
                            keep_gate_resident, 0);
    hip_check(hipStreamSynchronize(pipeline->compute_stream), "hybrid FFN wait gate");
    pipeline->async_worker.wait();
    hip_check(hipMemcpyAsync(pipeline->slots[1], pipeline->hybrid_host_output,
                             8 * ffn_size * sizeof(float), hipMemcpyHostToDevice,
                             pipeline->compute_stream),
              "hybrid FFN upload up");
    pipeline_silu_multiply(pipeline, 0, 1, 8 * ffn_size);
    hip_check(hipMemcpyAsync(pipeline->hidden, pipeline->slots[0],
                             8 * ffn_size * sizeof(float), hipMemcpyDeviceToDevice,
                             pipeline->compute_stream),
              "hybrid FFN move to hidden");
    pipeline_project_batch8(pipeline, down_weights, down_type, ffn_size, embedding_size,
                            false, 2);
    hip_check(hipMemcpyAsync(pipeline->hidden, pipeline->slots[3],
                             8 * embedding_size * sizeof(float), hipMemcpyDeviceToDevice,
                             pipeline->compute_stream),
              "hybrid FFN restore hidden");
    pipeline_residual_add(pipeline, 2);
}

void pipeline_fused_ffn_batch8_hybrid_soa(DevicePipeline *pipeline, const float *norm_weights,
                                          std::size_t embedding_size, std::size_t ffn_size,
                                          float epsilon,
                                          const void *gate_weights, std::uint32_t gate_type,
                                          const void *up_weights, std::uint32_t up_type,
                                          const void *down_weights, std::uint32_t down_type,
                                          bool keep_gate_up_resident) {
    const auto soa_supported = [](std::uint32_t type) {
        return type == 12 || type == 13 || type == 14 || type == 23;
    };
    const std::size_t gate_bytes = k_quant_weight_bytes(gate_type, embedding_size, ffn_size);
    const std::size_t up_bytes = k_quant_weight_bytes(up_type, embedding_size, ffn_size);
    const bool gate_resident = resident_weight_cache().find_or_add(
        gate_weights, gate_bytes, keep_gate_up_resident) != nullptr;
    const bool up_resident = resident_weight_cache().find_or_add(
        up_weights, up_bytes, keep_gate_up_resident) != nullptr;

    pipeline_rms_norm(pipeline, norm_weights, embedding_size, epsilon,
                      pipeline->slots[3], pipeline->hidden);

    if (std::getenv("QWEN_NATIVE_UNFUSED_FFN") || gate_resident || up_resident ||
        !soa_supported(gate_type) || !soa_supported(up_type)) {
        pipeline_project_batch8_hybrid(pipeline, gate_weights, gate_type,
                                       embedding_size, ffn_size,
                                       keep_gate_up_resident, 0, pipeline->slots[3]);
        pipeline_project_batch8_hybrid(pipeline, up_weights, up_type,
                                       embedding_size, ffn_size,
                                       keep_gate_up_resident, 1, pipeline->slots[3]);
        pipeline_prefetch_inline(pipeline, down_weights, down_type, ffn_size, embedding_size);
    } else {
        // Gate and up consume the same normalized activation. Run their CPU
        // shards in one worker-pool job and their GPU shards back-to-back so
        // neither resource waits at the matrix boundary and the activation is
        // downloaded only once.
        pipeline_stage_cpu_input(pipeline, pipeline->slots[3], 8 * embedding_size);

        const std::size_t gate_cpu_rows = hybrid_cpu_row_count(gate_type, ffn_size);
        const std::size_t up_cpu_rows = hybrid_cpu_row_count(up_type, ffn_size);
        const std::size_t gate_gpu_rows = ffn_size - gate_cpu_rows;
        const std::size_t up_gpu_rows = ffn_size - up_cpu_rows;
        const std::size_t gate_row_bytes = gate_bytes / ffn_size;
        const std::size_t up_row_bytes = up_bytes / ffn_size;

        const bool pipeline_gpu_rows = !std::getenv("QWEN_NATIVE_GPU_PIPELINE") || std::string(std::getenv("QWEN_NATIVE_GPU_PIPELINE")) != "0";
        const auto launch_gpu_shard = [&](const void *weights, std::uint32_t type,
                                          std::size_t cpu_rows, std::size_t gpu_rows,
                                          std::size_t row_bytes, int output_slot) {
            const auto *gpu_weights = static_cast<const std::uint8_t *>(weights)
                                    + cpu_rows * row_bytes;
            if (pipeline_gpu_rows) {
                pipeline_project_gpu_rows_pipelined(
                    pipeline, gpu_weights, type, embedding_size, gpu_rows,
                    cpu_rows, ffn_size, output_slot, pipeline->slots[3]);
            } else {
                pipeline_project_batch8(pipeline, gpu_weights, type, embedding_size,
                                        gpu_rows, false, 4, pipeline->slots[3]);
                constexpr int threads = 256;
                const int elements = static_cast<int>(8 * gpu_rows);
                hipLaunchKernelGGL(scatter_hybrid_rows_kernel,
                                   dim3((elements + threads - 1) / threads), dim3(threads),
                                   0, pipeline->compute_stream,
                                   pipeline->slots[4], pipeline->slots[output_slot],
                                   static_cast<int>(cpu_rows), static_cast<int>(gpu_rows),
                                   static_cast<int>(ffn_size));
                hip_check(hipGetLastError(), "fused SoA FFN scatter launch");
            }
        };
        pipeline->async_worker.submit([=] {
            pipeline_wait_cpu_input(pipeline);
            cpu_quant_matvec_batch8_rows_packed_dual(
                gate_weights, gate_type, pipeline->hybrid_host_output, gate_cpu_rows,
                up_weights, up_type, pipeline->hybrid_host_output2, up_cpu_rows,
                pipeline->hybrid_host_input, embedding_size,
                hybrid_cpu_worker_count());
        });

        launch_gpu_shard(gate_weights, gate_type, gate_cpu_rows, gate_gpu_rows,
                         gate_row_bytes, 0);
        launch_gpu_shard(up_weights, up_type, up_cpu_rows, up_gpu_rows,
                         up_row_bytes, 1);
        pipeline_prefetch_inline(pipeline, down_weights, down_type, ffn_size, embedding_size);

        hip_check(hipStreamSynchronize(pipeline->compute_stream),
                  "fused SoA FFN wait GPU gate/up");
        pipeline->async_worker.wait();

        // CPU rows are packed as [batch][row]. Upload each result in one DMA
        // and scatter it on-device instead of issuing eight small H2D copies.
        // slots[4] is no longer needed by the completed GPU shards at this
        // point and is large enough for the largest packed FFN shard.
        const auto upload_cpu_rows = [&](const float *host,
                                         std::size_t cpu_rows, int output_slot,
                                         const char *label) {
            if (cpu_rows == 0) return;
            hip_check(hipMemcpyAsync(pipeline->slots[4], host,
                                     8 * cpu_rows * sizeof(float), hipMemcpyHostToDevice,
                                     pipeline->compute_stream), label);
            constexpr int threads = 256;
            const int elements = static_cast<int>(8 * cpu_rows);
            hipLaunchKernelGGL(scatter_hybrid_rows_kernel,
                               dim3((elements + threads - 1) / threads), dim3(threads),
                               0, pipeline->compute_stream,
                               pipeline->slots[4], pipeline->slots[output_slot],
                               0, static_cast<int>(cpu_rows), static_cast<int>(ffn_size));
            hip_check(hipGetLastError(), "fused SoA FFN scatter CPU rows");
        };
        upload_cpu_rows(pipeline->hybrid_host_output, gate_cpu_rows, 0,
                        "fused SoA FFN upload CPU gate rows");
        upload_cpu_rows(pipeline->hybrid_host_output2, up_cpu_rows, 1,
                        "fused SoA FFN upload CPU up rows");
    }

    pipeline_silu_multiply(pipeline, 0, 1, 8 * ffn_size);
    pipeline_project_batch8_hybrid(pipeline, down_weights, down_type,
                                   ffn_size, embedding_size, false, 2, pipeline->slots[0]);
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

    // FP16 is kept for short-context scalar/pipeline equivalence tests. Long
    // contexts use the authoritative device-resident N23KV4 history below.
    __half *kv_cache = nullptr;
    bool use_n23kv4 = false;
    bool use_q8_long_cache = true;
    bool use_mixed_n23k_cache = false;
    bool use_q4_cache = false;
    // Long-context target KV keeps both K and V in native Q8_0 blocks.
    // N23KV4 kernels remain available for measured experiments, but are not
    // used for correctness-critical target verification.
    N23KV4Group *kv4_cache = nullptr;
    Q8_0Block *q8_key_cache = nullptr;
    Q8_0Block *q8_value_cache = nullptr;
    float *kv4_staging = nullptr;
    std::int8_t *kv4_queries = nullptr;
    float *kv4_query_scales = nullptr;
    float *kv4_partials = nullptr;
    int kv4_max_partitions = 32;
    float *q_norm_weights = nullptr;
    float *k_norm_weights = nullptr;
    float *attn_output = nullptr;
    float *rope_cos = nullptr;

    ~AttentionDeviceState() {
        if (kv_cache) (void)hipFree(kv_cache);
        if (kv4_cache) (void)hipFree(kv4_cache);
        if (q8_key_cache) (void)hipFree(q8_key_cache);
        if (q8_value_cache) (void)hipFree(q8_value_cache);
        if (kv4_staging) (void)hipFree(kv4_staging);
        if (kv4_queries) (void)hipFree(kv4_queries);
        if (kv4_query_scales) (void)hipFree(kv4_query_scales);
        if (kv4_partials) (void)hipFree(kv4_partials);
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
        // q_proj is viewed as [heads, 2 * head_size] by Qwen3.5, then
        // chunked on the final axis: [q0, gate0, q1, gate1, ...].
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

// Long-context preparation keeps normalized/rotated Q in FP32 for the
// existing pipeline but quantizes K/V into independent 32-value N23KV4 groups.
// Per-group scales bound the error while reducing a 128K, 16-layer KV cache
// from 8 GiB FP16 to 2.25 GiB.
__global__ void attn_prep_kv4_kernel(const float *__restrict__ q_slot,
                                      const float *__restrict__ k_slot,
                                      const float *__restrict__ v_slot,
                                      const float *__restrict__ q_norm_w,
                                      const float *__restrict__ k_norm_w,
                                      const float *__restrict__ rope_freqs,
                                      float *__restrict__ kv_staging,
                                      float *__restrict__ q_out,
                                      float *__restrict__ gate_out,
                                      int rope_base_position, float epsilon) {
    constexpr int kHeads = 24;
    constexpr int kKvHeads = 4;
    constexpr int kHeadSize = 256;
    constexpr int kRotaryPairs = 32;
    constexpr int kQWidth = 2 * kHeads * kHeadSize;
    constexpr int kKvWidth = kKvHeads * kHeadSize;
    const int batch = static_cast<int>(blockIdx.x);
    const int position = rope_base_position + batch;
    const int tid = static_cast<int>(threadIdx.x);
    const float *q_row = q_slot + static_cast<std::size_t>(batch) * kQWidth;
    const float *k_row = k_slot + static_cast<std::size_t>(batch) * kKvWidth;
    const float *v_row = v_slot + static_cast<std::size_t>(batch) * kKvWidth;
    __shared__ float rope_cos[kRotaryPairs];
    __shared__ float rope_sin[kRotaryPairs];
    if (tid < kRotaryPairs) {
        const float theta = static_cast<float>(position) * rope_freqs[tid];
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
            sum_sq += q_head[d] * q_head[d];
        }
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
            sum_sq += __shfl_down(sum_sq, offset, kWarpSize);
        const float q_rms = rsqrtf(__shfl(sum_sq, 0, kWarpSize) / static_cast<float>(kHeadSize) + epsilon);
        for (int d = tid; d < kHeadSize; d += static_cast<int>(blockDim.x)) {
            const float normed = q_head[d] * q_rms * q_norm_w[d];
            if (d < kRotaryPairs) {
                const float right = q_head[d + kRotaryPairs] * q_rms * q_norm_w[d + kRotaryPairs];
                out_q[d] = normed * rope_cos[d] - right * rope_sin[d];
                out_q[d + kRotaryPairs] = normed * rope_sin[d] + right * rope_cos[d];
            } else if (d >= 2 * kRotaryPairs) {
                out_q[d] = normed;
            }
            out_gate[d] = q_head[kHeadSize + d];
        }
    }

    for (int kvh = 0; kvh < kKvHeads; ++kvh) {
        const float *k_head = k_row + kvh * kHeadSize;
        float *staged_key = kv_staging + (static_cast<std::size_t>(batch) * 2 * kKvHeads + kvh) * kHeadSize;
        float *staged_value = kv_staging + (static_cast<std::size_t>(batch) * 2 * kKvHeads + kKvHeads + kvh) * kHeadSize;
        float sum_sq = 0.0F;
        for (int d = tid; d < kHeadSize; d += static_cast<int>(blockDim.x)) {
            sum_sq += k_head[d] * k_head[d];
        }
        for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
            sum_sq += __shfl_down(sum_sq, offset, kWarpSize);
        const float k_rms = rsqrtf(__shfl(sum_sq, 0, kWarpSize) / static_cast<float>(kHeadSize) + epsilon);
        const float *v_head = v_row + kvh * kHeadSize;
        for (int d = tid; d < kHeadSize; d += static_cast<int>(blockDim.x)) {
            const float normed = k_head[d] * k_rms * k_norm_w[d];
            if (d < kRotaryPairs) {
                const float right = k_head[d + kRotaryPairs] * k_rms * k_norm_w[d + kRotaryPairs];
                staged_key[d] = normed * rope_cos[d] - right * rope_sin[d];
                staged_key[d + kRotaryPairs] = normed * rope_sin[d] + right * rope_cos[d];
            } else if (d >= 2 * kRotaryPairs) {
                staged_key[d] = normed;
            }
            staged_value[d] = v_head[d];
        }
    }
}

__global__ void kv4_quantize_queries_kernel(const float *__restrict__ queries,
                                             std::int8_t *__restrict__ quantized,
                                             float *__restrict__ scales) {
    constexpr int kHeadSize = 256;
    constexpr int kGroups = 8;
    const int batch = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int group = static_cast<int>(blockIdx.z);
    const int tid = static_cast<int>(threadIdx.x);
    const float value = queries[(static_cast<std::size_t>(batch) * 24 + head) * kHeadSize
                                + group * 32 + tid];
    __shared__ float maximum[32];
    maximum[tid] = fabsf(value);
    __syncthreads();
    for (int offset = 16; offset > 0; offset /= 2) {
        if (tid < offset) maximum[tid] = fmaxf(maximum[tid], maximum[tid + offset]);
        __syncthreads();
    }
    const float scale = maximum[0] > 0.0F ? maximum[0] / 127.0F : 1.0F;
    const int q = static_cast<int>(rintf(value / scale));
    quantized[((static_cast<std::size_t>(batch) * 24 + head) * kGroups + group) * 32 + tid]
        = static_cast<std::int8_t>(max(-127, min(127, q)));
    if (tid == 0) scales[(static_cast<std::size_t>(batch) * 24 + head) * kGroups + group] = scale;
}

__global__ void kv4_quantize_cache_kernel(const float *__restrict__ staging,
                                           N23KV4Group *__restrict__ cache,
                                           int max_context, int write_offset, int kind) {
    constexpr int kKvHeads = 4;
    constexpr int kHeadSize = 256;
    constexpr int kGroups = 8;
    const int batch = static_cast<int>(blockIdx.x);
    const int packed = static_cast<int>(blockIdx.y);
    const int head = packed / kGroups;
    const int group = packed % kGroups;
    const int tid = static_cast<int>(threadIdx.x);
    const float value = staging[(static_cast<std::size_t>(batch) * 2 * kKvHeads
                                 + kind * kKvHeads + head) * kHeadSize + group * 32 + tid];
    __shared__ float maximum[32];
    maximum[tid] = fabsf(value);
    __syncthreads();
    for (int offset = 16; offset > 0; offset /= 2) {
        if (tid < offset) maximum[tid] = fmaxf(maximum[tid], maximum[tid + offset]);
        __syncthreads();
    }
    const float scale = maximum[0] > 0.0F ? maximum[0] / 7.0F : 1.0F;
    auto *destination = cache + (static_cast<std::size_t>(head) * max_context + write_offset + batch)
                                * kGroups + group;
    if (tid == 0) *reinterpret_cast<__half *>(&destination->scale) = __float2half(scale);
    if (tid < 16) {
        const float first_value = staging[(static_cast<std::size_t>(batch) * 2 * kKvHeads
                                           + kind * kKvHeads + head) * kHeadSize + group * 32 + 2 * tid];
        const float second_value = staging[(static_cast<std::size_t>(batch) * 2 * kKvHeads
                                            + kind * kKvHeads + head) * kHeadSize + group * 32 + 2 * tid + 1];
        const int lo = max(-8, min(7, static_cast<int>(rintf(first_value / scale)))) & 0x0f;
        const int hi = max(-8, min(7, static_cast<int>(rintf(second_value / scale)))) & 0x0f;
        destination->quants[tid] = static_cast<std::uint8_t>(lo | (hi << 4));
    }
}

__global__ void q8_quantize_cache_kernel(const float *__restrict__ staging,
                                          Q8_0Block *__restrict__ cache,
                                          int max_context, int write_offset, int kind) {
    constexpr int kKvHeads = 4;
    constexpr int kHeadSize = 256;
    constexpr int kGroups = 8;
    const int batch = static_cast<int>(blockIdx.x);
    const int packed = static_cast<int>(blockIdx.y);
    const int head = packed / kGroups;
    const int group = packed % kGroups;
    const int tid = static_cast<int>(threadIdx.x);
    const float value = staging[(static_cast<std::size_t>(batch) * 2 * kKvHeads
                                 + kind * kKvHeads + head) * kHeadSize + group * 32 + tid];
    __shared__ float maximum[32];
    maximum[tid] = fabsf(value);
    __syncthreads();
    for (int offset = 16; offset > 0; offset /= 2) {
        if (tid < offset) maximum[tid] = fmaxf(maximum[tid], maximum[tid + offset]);
        __syncthreads();
    }
    const float scale = maximum[0] > 0.0F ? maximum[0] / 127.0F : 1.0F;
    auto *destination = cache + (static_cast<std::size_t>(head) * max_context + write_offset + batch)
                                * kGroups + group;
    if (tid == 0) *reinterpret_cast<__half *>(&destination->d) = __float2half(scale);
    destination->qs[tid] = static_cast<std::int8_t>(
        max(-127, min(127, static_cast<int>(rintf(value / scale)))));
}

__global__ void kv4_apply_gate_kernel(const float *__restrict__ attention,
                                      const float *__restrict__ gates,
                                      float *__restrict__ output) {
    constexpr int kHeads = 24;
    constexpr int kHeadSize = 256;
    const int batch = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int dim = static_cast<int>(threadIdx.x);
    const std::size_t index = (static_cast<std::size_t>(batch) * kHeads + head) * kHeadSize + dim;
    output[index] = attention[index] / (1.0F + expf(-gates[index]));
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

    constexpr int kGroupsPerHead = kHeadSize / 32;
    // Above 4K, FP16 KV alone would consume 8 GiB at 128K. N23KV4 makes the
    // complete history fit on the 8 GiB device while retaining all positions.
    s->use_n23kv4 = max_context > 4096;
    if (s->use_n23kv4) {
        const char *kv_mode = std::getenv("QWEN_NATIVE_KV");
        s->use_q4_cache = kv_mode && std::strcmp(kv_mode, "q4") == 0;
        s->use_mixed_n23k_cache = kv_mode && std::strcmp(kv_mode, "n23k") == 0;
        s->use_q8_long_cache = !kv_mode || (std::strcmp(kv_mode, "n23") != 0
                                            && !s->use_mixed_n23k_cache && !s->use_q4_cache);
        const std::size_t groups_per_layer = static_cast<std::size_t>(kKvHeads) * max_context * kGroupsPerHead;
        if (s->use_q4_cache) {
            const std::size_t kv_bytes = static_cast<std::size_t>(num_layers) * 2 * groups_per_layer
                                       * sizeof(N23KV4Group);
            hip_check(hipMalloc(reinterpret_cast<void **>(&s->kv4_cache), kv_bytes), "attn Q4 KV cache");
            hip_check(hipMemset(s->kv4_cache, 0, kv_bytes), "attn Q4 KV cache zero");
        } else if (s->use_q8_long_cache) {
            const std::size_t kv_bytes = static_cast<std::size_t>(num_layers) * groups_per_layer
                                       * sizeof(Q8_0Block);
            hip_check(hipMalloc(reinterpret_cast<void **>(&s->q8_key_cache), kv_bytes), "attn Q8 key cache");
            hip_check(hipMemset(s->q8_key_cache, 0, kv_bytes), "attn Q8 key cache zero");
            hip_check(hipMalloc(reinterpret_cast<void **>(&s->q8_value_cache), kv_bytes), "attn Q8 value cache");
            hip_check(hipMemset(s->q8_value_cache, 0, kv_bytes), "attn Q8 value cache zero");
        } else if (s->use_mixed_n23k_cache) {
            const std::size_t key_bytes = static_cast<std::size_t>(num_layers) * groups_per_layer
                                        * sizeof(N23KV4Group);
            const std::size_t value_bytes = static_cast<std::size_t>(num_layers) * groups_per_layer
                                          * sizeof(Q8_0Block);
            hip_check(hipMalloc(reinterpret_cast<void **>(&s->kv4_cache), key_bytes), "attn N23 key cache");
            hip_check(hipMemset(s->kv4_cache, 0, key_bytes), "attn N23 key cache zero");
            hip_check(hipMalloc(reinterpret_cast<void **>(&s->q8_value_cache), value_bytes), "attn Q8 value cache");
            hip_check(hipMemset(s->q8_value_cache, 0, value_bytes), "attn Q8 value cache zero");
        } else {
            const std::size_t kv_bytes = static_cast<std::size_t>(num_layers) * 2 * groups_per_layer
                                       * sizeof(N23KV4Group);
            hip_check(hipMalloc(reinterpret_cast<void **>(&s->kv4_cache), kv_bytes), "attn N23KV4 cache");
            hip_check(hipMemset(s->kv4_cache, 0, kv_bytes), "attn N23KV4 cache zero");
        }
        hip_check(hipMalloc(reinterpret_cast<void **>(&s->kv4_staging),
                             2ULL * 8 * kKvHeads * kHeadSize * sizeof(float)), "attn N23KV4 staging");
        hip_check(hipMalloc(reinterpret_cast<void **>(&s->kv4_queries),
                             8ULL * kHeads * kHeadSize * sizeof(std::int8_t)), "attn N23KV4 queries");
        hip_check(hipMalloc(reinterpret_cast<void **>(&s->kv4_query_scales),
                             8ULL * kHeads * kGroupsPerHead * sizeof(float)), "attn N23KV4 query scales");
        constexpr std::size_t partial_stride = 258;
        hip_check(hipMalloc(reinterpret_cast<void **>(&s->kv4_partials),
                             8ULL * kHeads * s->kv4_max_partitions * partial_stride * sizeof(float)),
                  "attn N23KV4 partials");
    } else {
        const std::size_t kv_per_layer = 2ULL * kKvHeads * max_context * kHeadSize;
        const std::size_t kv_bytes = num_layers * kv_per_layer * sizeof(__half);
        hip_check(hipMalloc(reinterpret_cast<void **>(&s->kv_cache), kv_bytes), "attn kv cache");
        hip_check(hipMemset(s->kv_cache, 0, kv_bytes), "attn kv cache zero");
    }

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

void attention_upload_history(AttentionDeviceState *state, int layer_index,
                              const std::uint16_t *keys, const std::uint16_t *values,
                              std::size_t token_count) {
    if (!state || (state->use_n23kv4 && token_count != 0) || layer_index < 0 || layer_index >= 16 ||
        token_count > static_cast<std::size_t>(state->max_context) ||
        (token_count != 0 && (!keys || !values))) {
        throw std::invalid_argument("invalid attention history upload");
    }
    if (token_count == 0) return;
    constexpr std::size_t kKvHeads = AttentionDeviceState::kKvHeads;
    constexpr std::size_t kHeadSize = AttentionDeviceState::kHeadSize;
    constexpr std::size_t kKvWidth = kKvHeads * kHeadSize;
    std::vector<__half> packed(2 * kKvHeads * token_count * kHeadSize);
    for (std::size_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
        for (std::size_t token = 0; token < token_count; ++token) {
            const std::size_t source = token * kKvWidth + kv_head * kHeadSize;
            const std::size_t destination = (kv_head * token_count + token) * kHeadSize;
            std::memcpy(packed.data() + destination, keys + source, kHeadSize * sizeof(__half));
            std::memcpy(packed.data() + kKvHeads * token_count * kHeadSize + destination,
                        values + source, kHeadSize * sizeof(__half));
        }
    }
    const std::size_t layer_offset = static_cast<std::size_t>(layer_index) * 2ULL
        * kKvHeads * state->max_context * kHeadSize;
    __half *destination = state->kv_cache + layer_offset;
    for (std::size_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
        const std::size_t packed_offset = kv_head * token_count * kHeadSize;
        const std::size_t device_offset = kv_head * static_cast<std::size_t>(state->max_context) * kHeadSize;
        hip_check(hipMemcpy(destination + device_offset, packed.data() + packed_offset,
                            token_count * kHeadSize * sizeof(__half), hipMemcpyHostToDevice),
                  "attention history key upload");
        hip_check(hipMemcpy(destination + kKvHeads * static_cast<std::size_t>(state->max_context) * kHeadSize
                            + device_offset,
                            packed.data() + kKvHeads * token_count * kHeadSize + packed_offset,
                            token_count * kHeadSize * sizeof(__half), hipMemcpyHostToDevice),
                  "attention history value upload");
    }
}

void attention_download_history(const AttentionDeviceState *state, int layer_index,
                                std::uint16_t *keys, std::uint16_t *values,
                                std::size_t token_count) {
    if (!state || state->use_n23kv4 || !keys || !values || layer_index < 0 || layer_index >= 16 ||
        token_count > static_cast<std::size_t>(state->active_tokens)) {
        throw std::invalid_argument("invalid attention history download");
    }
    constexpr std::size_t kKvHeads = AttentionDeviceState::kKvHeads;
    constexpr std::size_t kHeadSize = AttentionDeviceState::kHeadSize;
    constexpr std::size_t kKvWidth = kKvHeads * kHeadSize;
    std::vector<__half> packed(2 * kKvHeads * token_count * kHeadSize);
    const std::size_t layer_offset = static_cast<std::size_t>(layer_index) * 2ULL
        * kKvHeads * state->max_context * kHeadSize;
    const __half *source = state->kv_cache + layer_offset;
    for (std::size_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
        const std::size_t packed_offset = kv_head * token_count * kHeadSize;
        const std::size_t device_offset = kv_head * static_cast<std::size_t>(state->max_context) * kHeadSize;
        hip_check(hipMemcpy(packed.data() + packed_offset, source + device_offset,
                            token_count * kHeadSize * sizeof(__half), hipMemcpyDeviceToHost),
                  "attention history key download");
        hip_check(hipMemcpy(packed.data() + kKvHeads * token_count * kHeadSize + packed_offset,
                            source + kKvHeads * static_cast<std::size_t>(state->max_context) * kHeadSize
                            + device_offset,
                            token_count * kHeadSize * sizeof(__half), hipMemcpyDeviceToHost),
                  "attention history value download");
    }
    for (std::size_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
        for (std::size_t token = 0; token < token_count; ++token) {
            const std::size_t destination = token * kKvWidth + kv_head * kHeadSize;
            const std::size_t packed_offset = (kv_head * token_count + token) * kHeadSize;
            std::memcpy(keys + destination, packed.data() + packed_offset, kHeadSize * sizeof(__half));
            std::memcpy(values + destination,
                        packed.data() + kKvHeads * token_count * kHeadSize + packed_offset,
                        kHeadSize * sizeof(__half));
        }
    }
}

void attention_set_history_length(AttentionDeviceState *state, int token_count) {
    if (!state || token_count < 0 || token_count > state->max_context) {
        throw std::invalid_argument("invalid attention history length");
    }
    state->active_tokens = token_count;
    state->global_position = token_count;
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
                               float epsilon, int batch_count) {
    if (batch_count < 1 || batch_count > 8) {
        throw std::invalid_argument("pipeline attention batch count must be in [1, 8]");
    }
    constexpr int kHeads = AttentionDeviceState::kHeads;
    constexpr int kKvHeads = AttentionDeviceState::kKvHeads;
    constexpr int kHeadSize = AttentionDeviceState::kHeadSize;

    if (layer_index == 0 && attn->active_tokens + batch_count > attn->max_context) {
        if (attn->use_n23kv4) {
            throw std::runtime_error("N23KV4 attention reached its declared context capacity");
        }
        attention_shift_all_layers(attn, pipeline->compute_stream);
    }

    const int write_offset = attn->active_tokens;
    const int rope_base = attn->global_position;

    hip_check(hipMemcpyAsync(attn->q_norm_weights, q_norm_weights,
                             kHeadSize * sizeof(float), hipMemcpyHostToDevice,
                             pipeline->compute_stream), "attn upload q norm");
    hip_check(hipMemcpyAsync(attn->k_norm_weights, k_norm_weights,
                             kHeadSize * sizeof(float), hipMemcpyHostToDevice,
                             pipeline->compute_stream), "attn upload k norm");

    float *q_buf = attn->attn_output;
    float *gate_buf = attn->attn_output + 8ULL * kHeads * kHeadSize;
    if (attn->use_n23kv4) {
        hipLaunchKernelGGL(attn_prep_kv4_kernel, dim3(batch_count), dim3(kWarpSize), 0,
                           pipeline->compute_stream,
                           pipeline->slots[0], pipeline->slots[1], pipeline->slots[2],
                           attn->q_norm_weights, attn->k_norm_weights, attn->rope_cos,
                           attn->kv4_staging, q_buf, gate_buf, rope_base, epsilon);
        hip_check(hipGetLastError(), "attn N23KV4 preparation launch");
        const std::size_t layer_groups = static_cast<std::size_t>(kKvHeads) * attn->max_context * 8;
        const int context_size = write_offset + batch_count;
        int partitions = std::max(1, std::min(attn->kv4_max_partitions,
                                              (context_size + 2047) / 2048));
        if (const char *requested = std::getenv("QWEN_NATIVE_Q8_PARTITIONS")) {
            partitions = std::clamp(std::atoi(requested), 1, attn->kv4_max_partitions);
        }
        if (attn->use_q4_cache) {
            auto *layer_cache = attn->kv4_cache + static_cast<std::size_t>(layer_index) * 2 * layer_groups;
            auto *layer_keys = layer_cache;
            auto *layer_values = layer_cache + layer_groups;
            hipLaunchKernelGGL(kv4_quantize_cache_kernel, dim3(batch_count, kKvHeads * 8), dim3(32), 0,
                               pipeline->compute_stream, attn->kv4_staging, layer_keys,
                               attn->max_context, write_offset, 0);
            hip_check(hipGetLastError(), "attn Q4 key quantization launch");
            hipLaunchKernelGGL(kv4_quantize_cache_kernel, dim3(batch_count, kKvHeads * 8), dim3(32), 0,
                               pipeline->compute_stream, attn->kv4_staging, layer_values,
                               attn->max_context, write_offset, 1);
            hip_check(hipGetLastError(), "attn Q4 value quantization launch");
            hipLaunchKernelGGL(kv4_quantize_queries_kernel, dim3(batch_count, kHeads, 8), dim3(32), 0,
                               pipeline->compute_stream, q_buf, attn->kv4_queries, attn->kv4_query_scales);
            hip_check(hipGetLastError(), "attn Q4 query quantization launch");
            const char *q4_tiled = std::getenv("QWEN_NATIVE_Q4_TILED");
            if (q4_tiled && std::strcmp(q4_tiled, "gqa48") == 0) {
                hipLaunchKernelGGL(n23kv4_flash_gqa48_tile16_kernel,
                                   dim3(partitions, kKvHeads), dim3(192), 0,
                                   pipeline->compute_stream, layer_keys, layer_values,
                                   attn->kv4_queries, attn->kv4_query_scales, attn->kv4_partials,
                                   context_size, attn->max_context, write_offset, partitions);
                hip_check(hipGetLastError(), "attn Q4 flash gqa48 launch");
            } else if (q4_tiled && std::strcmp(q4_tiled, "gqa24") == 0) {
                hipLaunchKernelGGL(n23kv4_flash_gqa24_tile16_kernel,
                                   dim3(partitions, kKvHeads * 2), dim3(192), 0,
                                   pipeline->compute_stream, layer_keys, layer_values,
                                   attn->kv4_queries, attn->kv4_query_scales, attn->kv4_partials,
                                   context_size, attn->max_context, write_offset, partitions);
                hip_check(hipGetLastError(), "attn Q4 flash gqa24 launch");
            } else {
                hipLaunchKernelGGL(n23kv4_flash_warp16_kernel,
                                   dim3(kKvHeads * 3, partitions), dim3(512), 0,
                                   pipeline->compute_stream, layer_keys, layer_values,
                                   attn->kv4_queries, attn->kv4_query_scales, attn->kv4_partials,
                                   context_size, attn->max_context, write_offset, partitions);
                hip_check(hipGetLastError(), "attn Q4 flash warp16 launch");
            }
        } else if (attn->use_q8_long_cache) {
            auto *layer_keys = attn->q8_key_cache + static_cast<std::size_t>(layer_index) * layer_groups;
            auto *layer_values = attn->q8_value_cache + static_cast<std::size_t>(layer_index) * layer_groups;
            hipLaunchKernelGGL(q8_quantize_cache_kernel, dim3(batch_count, kKvHeads * 8), dim3(32), 0,
                               pipeline->compute_stream, attn->kv4_staging, layer_keys,
                               attn->max_context, write_offset, 0);
            hip_check(hipGetLastError(), "attn Q8 key quantization launch");
            hipLaunchKernelGGL(q8_quantize_cache_kernel, dim3(batch_count, kKvHeads * 8), dim3(32), 0,
                               pipeline->compute_stream, attn->kv4_staging, layer_values,
                               attn->max_context, write_offset, 1);
            hip_check(hipGetLastError(), "attn Q8 value quantization launch");
            const char *q8_tiled = std::getenv("QWEN_NATIVE_Q8_TILED");
            if (q8_tiled && std::strcmp(q8_tiled, "warp16packed") == 0) {
                hipLaunchKernelGGL(q8kv_flash_partial_warp16_lds_kernel<true>,
                                   dim3(partitions, kKvHeads * 3), dim3(512), 0,
                                   pipeline->compute_stream, layer_keys, layer_values, q_buf,
                                   attn->kv4_partials, context_size, attn->max_context,
                                   write_offset, partitions);
            } else if (q8_tiled && std::strcmp(q8_tiled, "warp16lds") == 0) {
                hipLaunchKernelGGL(q8kv_flash_partial_warp16_lds_kernel<false>,
                                   dim3(partitions, kKvHeads * 3), dim3(512), 0,
                                   pipeline->compute_stream, layer_keys, layer_values, q_buf,
                                   attn->kv4_partials, context_size, attn->max_context,
                                   write_offset, partitions);
            } else if (q8_tiled && std::strcmp(q8_tiled, "warp12") == 0) {
                hipLaunchKernelGGL(q8kv_flash_partial_warp_kernel,
                                   dim3(partitions, kKvHeads * 4), dim3(384), 0,
                                   pipeline->compute_stream, layer_keys, layer_values, q_buf,
                                   attn->kv4_partials, context_size, attn->max_context,
                                   write_offset, partitions);
            } else if (q8_tiled && std::strcmp(q8_tiled, "warp16") == 0) {
                hipLaunchKernelGGL(q8kv_flash_partial_warp_kernel,
                                   dim3(partitions, kKvHeads * 3), dim3(512), 0,
                                   pipeline->compute_stream, layer_keys, layer_values, q_buf,
                                   attn->kv4_partials, context_size, attn->max_context,
                                   write_offset, partitions);
            } else if (q8_tiled && std::strcmp(q8_tiled, "warp24") == 0) {
                hipLaunchKernelGGL(q8kv_flash_partial_warp_kernel,
                                   dim3(partitions, kKvHeads * 2), dim3(768), 0,
                                   pipeline->compute_stream, layer_keys, layer_values, q_buf,
                                   attn->kv4_partials, context_size, attn->max_context,
                                   write_offset, partitions);
            } else if (q8_tiled && std::strcmp(q8_tiled, "warp") == 0) {
                hipLaunchKernelGGL(q8kv_flash_partial_warp_kernel,
                                   dim3(partitions, kKvHeads * 6), dim3(256), 0,
                                   pipeline->compute_stream, layer_keys, layer_values, q_buf,
                                   attn->kv4_partials, context_size, attn->max_context,
                                   write_offset, partitions);
            } else if (q8_tiled && std::strcmp(q8_tiled, "24") == 0) {
                hipLaunchKernelGGL(q8kv_flash_partial_tiled24_kernel,
                                   dim3(partitions, kKvHeads * 2), dim3(256), 0,
                                   pipeline->compute_stream, layer_keys, layer_values, q_buf,
                                   attn->kv4_partials, context_size, attn->max_context,
                                   write_offset, partitions);
            } else if (q8_tiled) {
                hipLaunchKernelGGL(q8kv_flash_partial_tiled8_kernel,
                                   dim3(partitions, kKvHeads * 6), dim3(256), 0,
                                   pipeline->compute_stream, layer_keys, layer_values, q_buf,
                                   attn->kv4_partials, context_size, attn->max_context,
                                   write_offset, partitions);
            } else {
                hipLaunchKernelGGL(q8kv_flash_partial_kernel,
                                   dim3(partitions, kKvHeads * 6), dim3(256), 0,
                                   pipeline->compute_stream, layer_keys, layer_values, q_buf,
                                   attn->kv4_partials, context_size, attn->max_context,
                                   write_offset, partitions);
            }
            hip_check(hipGetLastError(), "attn Q8 flash partial launch");
        } else {
            hipLaunchKernelGGL(kv4_quantize_queries_kernel, dim3(batch_count, kHeads, 8), dim3(32), 0,
                               pipeline->compute_stream, q_buf, attn->kv4_queries, attn->kv4_query_scales);
            hip_check(hipGetLastError(), "attn N23KV4 query quantization launch");
            if (attn->use_mixed_n23k_cache) {
                auto *layer_keys = attn->kv4_cache + static_cast<std::size_t>(layer_index) * layer_groups;
                auto *layer_values = attn->q8_value_cache + static_cast<std::size_t>(layer_index) * layer_groups;
                hipLaunchKernelGGL(kv4_quantize_cache_kernel, dim3(batch_count, kKvHeads * 8), dim3(32), 0,
                                   pipeline->compute_stream, attn->kv4_staging, layer_keys,
                                   attn->max_context, write_offset, 0);
                hip_check(hipGetLastError(), "attn N23 key quantization launch");
                hipLaunchKernelGGL(q8_quantize_cache_kernel, dim3(batch_count, kKvHeads * 8), dim3(32), 0,
                                   pipeline->compute_stream, attn->kv4_staging, layer_values,
                                   attn->max_context, write_offset, 1);
                hip_check(hipGetLastError(), "attn Q8 value quantization launch");
                hipLaunchKernelGGL(n23k_q8v_flash_partial_kernel,
                                   dim3(partitions, batch_count * kHeads), dim3(256), 0,
                                   pipeline->compute_stream, layer_keys, layer_values, attn->kv4_queries,
                                   attn->kv4_query_scales, attn->kv4_partials,
                                   context_size, attn->max_context, write_offset, partitions);
                hip_check(hipGetLastError(), "attn N23K/Q8V flash partial launch");
            } else {
                auto *layer_cache = attn->kv4_cache + static_cast<std::size_t>(layer_index) * 2 * layer_groups;
                auto *layer_keys = layer_cache;
                auto *layer_values = layer_cache + layer_groups;
                hipLaunchKernelGGL(kv4_quantize_cache_kernel, dim3(batch_count, kKvHeads * 8), dim3(32), 0,
                                   pipeline->compute_stream, attn->kv4_staging, layer_keys,
                                   attn->max_context, write_offset, 0);
                hip_check(hipGetLastError(), "attn N23KV4 key quantization launch");
                hipLaunchKernelGGL(kv4_quantize_cache_kernel, dim3(batch_count, kKvHeads * 8), dim3(32), 0,
                                   pipeline->compute_stream, attn->kv4_staging, layer_values,
                                   attn->max_context, write_offset, 1);
                hip_check(hipGetLastError(), "attn N23KV4 value quantization launch");
                hipLaunchKernelGGL(n23kv4_flash_partial_kernel, dim3(partitions, kKvHeads * 6), dim3(256), 0,
                                   pipeline->compute_stream, layer_keys, layer_values, attn->kv4_queries,
                                   attn->kv4_query_scales, attn->kv4_partials,
                                   context_size, attn->max_context, write_offset, partitions);
                hip_check(hipGetLastError(), "attn N23KV4 flash partial launch");
            }
        }
        hipLaunchKernelGGL(n23kv4_flash_reduce_kernel, dim3(batch_count * kHeads), dim3(256), 0,
                           pipeline->compute_stream, attn->kv4_partials, pipeline->slots[0], partitions);
        hip_check(hipGetLastError(), "attn N23KV4 flash reduction launch");
        hipLaunchKernelGGL(kv4_apply_gate_kernel, dim3(batch_count, kHeads), dim3(256), 0,
                           pipeline->compute_stream, pipeline->slots[0], gate_buf, pipeline->slots[0]);
        hip_check(hipGetLastError(), "attn N23KV4 gate launch");
        return;
    }

    const std::size_t layer_kv_offset = static_cast<std::size_t>(layer_index) * 2ULL
        * kKvHeads * attn->max_context * kHeadSize;
    __half *layer_kv = attn->kv_cache + layer_kv_offset;
    hipLaunchKernelGGL(attn_prep_kernel, dim3(batch_count), dim3(kWarpSize), 0, pipeline->compute_stream,
                       pipeline->slots[0], pipeline->slots[1], pipeline->slots[2],
                       attn->q_norm_weights, attn->k_norm_weights,
                       attn->rope_cos,
                       layer_kv, q_buf, gate_buf,
                       attn->max_context, write_offset, rope_base, epsilon);

    const std::size_t smem_bytes = static_cast<std::size_t>(write_offset + batch_count) * sizeof(float);
    hipLaunchKernelGGL(attn_score_kernel, dim3(batch_count, kHeads), dim3(256), smem_bytes,
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

    // Compact transactional record for the last verifier batch. Replaying the
    // recurrent update factors is far cheaper than rerunning all 64 layers.
    float *rollback_qkv = nullptr;
    float *rollback_keys = nullptr;
    float *rollback_values = nullptr;
    float *rollback_decays = nullptr;
    float *rollback_betas = nullptr;
    float *checkpoint_matrices = nullptr;
    float *checkpoint_convolutions = nullptr;
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

    const std::size_t rollback_qkv_bytes = static_cast<std::size_t>(num_layers) * 8 * qkv_width * sizeof(float);
    const std::size_t rollback_vector_bytes = static_cast<std::size_t>(num_layers) * 8 * recurrent_width * sizeof(float);
    const std::size_t rollback_db_bytes = static_cast<std::size_t>(num_layers) * 8 * value_heads * sizeof(float);
    hip_check(hipMalloc(reinterpret_cast<void **>(&s->rollback_qkv), rollback_qkv_bytes), "recurrent rollback qkv");
    hip_check(hipMalloc(reinterpret_cast<void **>(&s->rollback_keys), rollback_vector_bytes), "recurrent rollback keys");
    hip_check(hipMalloc(reinterpret_cast<void **>(&s->rollback_values), rollback_vector_bytes), "recurrent rollback values");
    hip_check(hipMalloc(reinterpret_cast<void **>(&s->rollback_decays), rollback_db_bytes), "recurrent rollback decays");
    hip_check(hipMalloc(reinterpret_cast<void **>(&s->rollback_betas), rollback_db_bytes), "recurrent rollback betas");

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
    if (s->rollback_qkv) (void)hipFree(s->rollback_qkv);
    if (s->rollback_keys) (void)hipFree(s->rollback_keys);
    if (s->rollback_values) (void)hipFree(s->rollback_values);
    if (s->rollback_decays) (void)hipFree(s->rollback_decays);
    if (s->rollback_betas) (void)hipFree(s->rollback_betas);
    if (s->checkpoint_matrices) (void)hipFree(s->checkpoint_matrices);
    if (s->checkpoint_convolutions) (void)hipFree(s->checkpoint_convolutions);
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

void recurrent_upload_convolution_state(RecurrentDeviceState *state, int layer_index,
                                        const float *history, std::size_t value_count) {
    if (!state || !history || layer_index < 0 || layer_index >= state->num_layers ||
        value_count != 3ULL * static_cast<std::size_t>(state->qkv_width)) {
        throw std::invalid_argument("invalid recurrent convolution upload");
    }
    hip_check(hipMemcpy(state->conv_states[layer_index], history,
                        value_count * sizeof(float), hipMemcpyHostToDevice),
              "recurrent convolution upload");
}

void recurrent_download_convolution_state(const RecurrentDeviceState *state, int layer_index,
                                          float *history, std::size_t value_count) {
    if (!state || !history || layer_index < 0 || layer_index >= state->num_layers ||
        value_count != 3ULL * static_cast<std::size_t>(state->qkv_width)) {
        throw std::invalid_argument("invalid recurrent convolution download");
    }
    hip_check(hipMemcpy(history, state->conv_states[layer_index],
                        value_count * sizeof(float), hipMemcpyDeviceToHost),
              "recurrent convolution download");
}

void pipeline_recurrent_batch8(DevicePipeline *pipeline, RecurrentDeviceState *recurrent,
                               int layer_index, void *gdn_state, float epsilon, int batch_count,
                               const float *normalized_hidden, float *output_buffer) {
    if (batch_count < 1 || batch_count > 8) {
        throw std::invalid_argument("pipeline recurrent batch count must be in [1, 8]");
    }
    const int qkv_width = recurrent->qkv_width;
    const int recurrent_width = recurrent->recurrent_width;
    const int value_heads = recurrent->value_heads;
    const int head_size = recurrent->head_size;
    const int actual_key_heads = (qkv_width - value_heads * head_size) / (2 * head_size);

    // Preserve the projected QKV input before convolution mutates the shared
    // pipeline slots. Prefix commit uses this only to rebuild conv history.
    const std::size_t qkv_layer_stride = 8ULL * qkv_width;
    hip_check(hipMemcpyAsync(recurrent->rollback_qkv
                                 + static_cast<std::size_t>(layer_index) * qkv_layer_stride,
                             pipeline->slots[0],
                             static_cast<std::size_t>(batch_count) * qkv_width * sizeof(float),
                             hipMemcpyDeviceToDevice, pipeline->compute_stream),
              "capture recurrent rollback qkv");

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
                           qkv_width, actual_key_heads, value_heads, head_size, batch_count);
        hip_check(hipGetLastError(), "recurrent conv1d launch");
    }


    // L2 normalize queries and keys
    hipLaunchKernelGGL(recurrent_l2_norm_batch8_kernel, dim3(value_heads), dim3(128),
                       0, pipeline->compute_stream,
                       recurrent->queries, recurrent->keys,
                       recurrent_width, head_size, epsilon, batch_count);
    hip_check(hipGetLastError(), "recurrent l2_norm launch");

    const std::size_t vector_layer_stride = 8ULL * recurrent_width;
    const std::size_t captured_vector_bytes = static_cast<std::size_t>(batch_count)
                                            * recurrent_width * sizeof(float);
    hip_check(hipMemcpyAsync(recurrent->rollback_keys
                                 + static_cast<std::size_t>(layer_index) * vector_layer_stride,
                             recurrent->keys, captured_vector_bytes,
                             hipMemcpyDeviceToDevice, pipeline->compute_stream),
              "capture recurrent rollback keys");
    hip_check(hipMemcpyAsync(recurrent->rollback_values
                                 + static_cast<std::size_t>(layer_index) * vector_layer_stride,
                             recurrent->values, captured_vector_bytes,
                             hipMemcpyDeviceToDevice, pipeline->compute_stream),
              "capture recurrent rollback values");

    const float *norm_hid = normalized_hidden ? normalized_hidden : pipeline->hidden;
    float *out_buf = output_buffer ? output_buffer : pipeline->hidden;

    // Decay/beta from hidden state
    hipLaunchKernelGGL(recurrent_decay_beta_batch8_kernel, dim3(value_heads), dim3(256),
                       0, pipeline->compute_stream,
                       norm_hid,
                       recurrent->beta_weights[layer_index],
                       recurrent->alpha_weights[layer_index],
                       recurrent->dt_bias[layer_index],
                       recurrent->a_weights[layer_index],
                       recurrent->decays, recurrent->betas,
                       recurrent->embedding_size, batch_count);
    hip_check(hipGetLastError(), "recurrent decay_beta launch");

    const std::size_t db_layer_stride = 8ULL * value_heads;
    const std::size_t captured_db_bytes = static_cast<std::size_t>(batch_count)
                                        * value_heads * sizeof(float);
    hip_check(hipMemcpyAsync(recurrent->rollback_decays
                                 + static_cast<std::size_t>(layer_index) * db_layer_stride,
                             recurrent->decays, captured_db_bytes,
                             hipMemcpyDeviceToDevice, pipeline->compute_stream),
              "capture recurrent rollback decays");
    hip_check(hipMemcpyAsync(recurrent->rollback_betas
                                 + static_cast<std::size_t>(layer_index) * db_layer_stride,
                             recurrent->betas, captured_db_bytes,
                             hipMemcpyDeviceToDevice, pipeline->compute_stream),
              "capture recurrent rollback betas");

    // Gated delta step on device (no memcpy!)
    hipLaunchKernelGGL(gated_delta_step_batch8_kernel, dim3(48), dim3(256),
                       0, pipeline->compute_stream,
                       recurrent->queries, recurrent->keys, recurrent->values,
                       recurrent->decays, recurrent->betas,
                       static_cast<float *>(gdn_state),
                       recurrent->gdn_output, head_size, batch_count);
    hip_check(hipGetLastError(), "recurrent gdn launch");


    // SSM norm + silu(gate) → pipeline output buffer
    hipLaunchKernelGGL(recurrent_ssm_norm_gate_batch8_kernel, dim3(value_heads), dim3(128),
                       0, pipeline->compute_stream,
                       recurrent->gdn_output, pipeline->slots[1],
                       recurrent->norm_weights[layer_index],
                       out_buf,
                       recurrent_width, value_heads, head_size, epsilon, batch_count);
    hip_check(hipGetLastError(), "recurrent ssm_norm_gate launch");
}

void recurrent_commit_captured_prefix(RecurrentDeviceState *state, int layer_index,
                                      void *gdn_state, int committed_tokens,
                                      DevicePipeline *pipeline) {
    if (!state || !gdn_state || layer_index < 0 || layer_index >= state->num_layers ||
        committed_tokens < 1 || committed_tokens > 8) {
        throw std::invalid_argument("invalid captured recurrent prefix commit");
    }
    constexpr int threads = 256;
    const int conv_blocks = (state->qkv_width + threads - 1) / threads;
    const std::size_t qkv_offset = static_cast<std::size_t>(layer_index) * 8 * state->qkv_width;
    const hipStream_t stream = pipeline ? pipeline->compute_stream : nullptr;
    hipLaunchKernelGGL(recurrent_conv_commit_prefix_kernel, dim3(conv_blocks), dim3(threads),
                       0, stream,
                       state->rollback_qkv + qkv_offset,
                       state->conv_states[layer_index], state->qkv_width, committed_tokens);
    hip_check(hipGetLastError(), "recurrent captured conv commit launch");

    const std::size_t vector_offset = static_cast<std::size_t>(layer_index)
                                    * 8 * state->recurrent_width;
    const std::size_t db_offset = static_cast<std::size_t>(layer_index) * 8 * state->value_heads;
    hipLaunchKernelGGL(recurrent_gdn_commit_prefix_kernel, dim3(state->value_heads), dim3(256),
                       0, stream,
                       state->rollback_keys + vector_offset,
                       state->rollback_values + vector_offset,
                       state->rollback_decays + db_offset,
                       state->rollback_betas + db_offset,
                       static_cast<float *>(gdn_state), state->head_size, committed_tokens);
    hip_check(hipGetLastError(), "recurrent captured GDN commit launch");
}

void recurrent_checkpoint_layer(DevicePipeline *pipeline, RecurrentDeviceState *state,
                                int layer_index, const void *gdn_state) {
    if (!pipeline || !state || !gdn_state || layer_index < 0 || layer_index >= state->num_layers) {
        throw std::invalid_argument("invalid device recurrent checkpoint");
    }
    const std::size_t matrix_values = static_cast<std::size_t>(state->value_heads)
                                    * state->head_size * state->head_size;
    const std::size_t conv_values = 3ULL * state->qkv_width;
    if (!state->checkpoint_matrices) {
        hip_check(hipMalloc(reinterpret_cast<void **>(&state->checkpoint_matrices),
                             state->num_layers * matrix_values * sizeof(float)),
                  "allocate recurrent matrix checkpoint");
    }
    if (!state->checkpoint_convolutions) {
        hip_check(hipMalloc(reinterpret_cast<void **>(&state->checkpoint_convolutions),
                             state->num_layers * conv_values * sizeof(float)),
                  "allocate recurrent convolution checkpoint");
    }
    hip_check(hipMemcpyAsync(state->checkpoint_matrices + layer_index * matrix_values,
                             gdn_state, matrix_values * sizeof(float),
                             hipMemcpyDeviceToDevice, pipeline->compute_stream),
              "checkpoint recurrent matrix on device");
    hip_check(hipMemcpyAsync(state->checkpoint_convolutions + layer_index * conv_values,
                             state->conv_states[layer_index], conv_values * sizeof(float),
                             hipMemcpyDeviceToDevice, pipeline->compute_stream),
              "checkpoint recurrent convolution on device");
}

void recurrent_restore_checkpoint_layer(DevicePipeline *pipeline, RecurrentDeviceState *state,
                                        int layer_index, void *gdn_state,
                                        int committed_tokens) {
    if (!pipeline || !state || !gdn_state || layer_index < 0 || layer_index >= state->num_layers ||
        !state->checkpoint_matrices || !state->checkpoint_convolutions ||
        committed_tokens < 0 || committed_tokens > 8) {
        throw std::invalid_argument("invalid device recurrent checkpoint restore");
    }
    const std::size_t matrix_values = static_cast<std::size_t>(state->value_heads)
                                    * state->head_size * state->head_size;
    const std::size_t conv_values = 3ULL * state->qkv_width;
    hip_check(hipMemcpyAsync(gdn_state,
                             state->checkpoint_matrices + layer_index * matrix_values,
                             matrix_values * sizeof(float), hipMemcpyDeviceToDevice,
                             pipeline->compute_stream), "restore recurrent matrix on device");
    hip_check(hipMemcpyAsync(state->conv_states[layer_index],
                             state->checkpoint_convolutions + layer_index * conv_values,
                             conv_values * sizeof(float), hipMemcpyDeviceToDevice,
                             pipeline->compute_stream), "restore recurrent convolution on device");
    if (committed_tokens != 0) {
        recurrent_commit_captured_prefix(state, layer_index, gdn_state, committed_tokens, pipeline);
    }
}

} // namespace qwen_native

#include "cpu_matvec.h"
#include "iq3s_codebook.h"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

namespace qwen_native {
namespace {

#pragma pack(push, 1)
struct Q4KBlock { std::uint16_t d, dmin; std::uint8_t scales[12], quants[128]; };
struct Q5KBlock { std::uint16_t d, dmin; std::uint8_t scales[12], high_bits[32], quants[128]; };
struct Q6KBlock { std::uint8_t low[128], high[64]; std::int8_t scales[16]; std::uint16_t d; };
struct Q8Block { std::uint16_t d; std::int8_t quants[32]; };
struct Q3KBlock { std::uint8_t hmask[32], quants[64], scales[12]; std::uint16_t d; };
struct IQ3SBlock {
    std::uint16_t d;
    std::uint8_t quants[64], high[8], signs[32], scales[4];
};
struct IQ4NLBlock { std::uint16_t d; std::uint8_t quants[16]; };
struct IQ4XSBlock { std::uint16_t d, scales_h; std::uint8_t scales_l[4], quants[128]; };
#pragma pack(pop)

static_assert(sizeof(Q4KBlock) == 144);
static_assert(sizeof(Q5KBlock) == 176);
static_assert(sizeof(Q6KBlock) == 210);
static_assert(sizeof(Q8Block) == 34);
static_assert(sizeof(Q3KBlock) == 110);
static_assert(sizeof(IQ3SBlock) == 110);
static_assert(sizeof(IQ4NLBlock) == 18);
static_assert(sizeof(IQ4XSBlock) == 136);

constexpr std::uint16_t iq3s_codebook[512] = {
    QWEN_IQ3S_CODEBOOK_VALUES
};
#undef QWEN_IQ3S_CODEBOOK_VALUES

constexpr std::int8_t iq4_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

inline float fp16(std::uint16_t bits) {
    return _cvtsh_ss(bits);
}

void scale_min(const std::uint8_t *packed, int index, std::uint8_t &scale, std::uint8_t &minimum) {
    if (index < 4) {
        scale = packed[index] & 63;
        minimum = packed[index + 4] & 63;
    } else {
        scale = (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4);
        minimum = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
    }
}

int q3_scale(const std::uint8_t *scales, int subblock) {
    const int j = subblock & 3;
    int low, high;
    if (subblock < 4) {
        low = scales[j] & 15; high = scales[8 + j] & 3;
    } else if (subblock < 8) {
        low = scales[4 + j] & 15; high = (scales[8 + j] >> 2) & 3;
    } else if (subblock < 12) {
        low = scales[j] >> 4; high = (scales[8 + j] >> 4) & 3;
    } else {
        low = scales[4 + j] >> 4; high = (scales[8 + j] >> 6) & 3;
    }
    return (low | (high << 4)) - 32;
}

inline __m256 fma_weight(__m256 accumulator, float weight, const float *input) {
    return _mm256_fmadd_ps(_mm256_set1_ps(weight), _mm256_load_ps(input), accumulator);
}

std::size_t block_values(std::uint32_t type) {
    return type == 8 || type == 20 ? 32 : 256;
}

std::size_t block_bytes(std::uint32_t type) {
    switch (type) {
    case 8: return sizeof(Q8Block);
    case 11: return sizeof(Q3KBlock);
    case 21: return sizeof(IQ3SBlock);
    case 12: return sizeof(Q4KBlock);
    case 13: return sizeof(Q5KBlock);
    case 14: return sizeof(Q6KBlock);
    case 20: return sizeof(IQ4NLBlock);
    case 23: return sizeof(IQ4XSBlock);
    default: throw std::invalid_argument("unsupported CPU quant matrix type");
    }
}

void compute_row(const std::uint8_t *row_weights, std::uint32_t type,
                 const float *transposed, std::size_t input_size, float *result) {
    __m256 accumulator = _mm256_setzero_ps();
    const std::size_t values_per_block = block_values(type);
    const std::size_t bytes_per_block = block_bytes(type);
    const std::size_t blocks = input_size / values_per_block;
    for (std::size_t block_index = 0; block_index < blocks; ++block_index) {
        const std::size_t input_base = block_index * values_per_block;
        const auto *raw = row_weights + block_index * bytes_per_block;
        if (type == 12 || type == 13) {
            const auto *q4 = reinterpret_cast<const Q4KBlock *>(raw);
            const auto *q5 = reinterpret_cast<const Q5KBlock *>(raw);
            const float d = fp16(q4->d);
            const float dmin = fp16(q4->dmin);
            const auto *scales = q4->scales;
            const auto *quants = type == 12 ? q4->quants : q5->quants;
            for (int group = 0; group < 4; ++group) {
                std::uint8_t s0, m0, s1, m1;
                scale_min(scales, 2 * group, s0, m0);
                scale_min(scales, 2 * group + 1, s1, m1);
                for (int lane = 0; lane < 32; ++lane) {
                    const std::uint8_t packed = quants[group * 32 + lane];
                    int code0 = packed & 15;
                    int code1 = packed >> 4;
                    if (type == 13) {
                        const std::uint8_t high = q5->high_bits[lane];
                        code0 += (high & (1U << (2 * group))) ? 16 : 0;
                        code1 += (high & (2U << (2 * group))) ? 16 : 0;
                    }
                    const std::size_t index0 = input_base + group * 64 + lane;
                    accumulator = fma_weight(accumulator, d * s0 * code0 - dmin * m0,
                                             transposed + index0 * 8);
                    accumulator = fma_weight(accumulator, d * s1 * code1 - dmin * m1,
                                             transposed + (index0 + 32) * 8);
                }
            }
        } else if (type == 14) {
            const auto &block = *reinterpret_cast<const Q6KBlock *>(raw);
            const float d = fp16(block.d);
            for (int half = 0; half < 2; ++half) {
                const int quant_offset = half * 64;
                const int value_offset = half * 128;
                for (int lane = 0; lane < 32; ++lane) {
                    const auto low0 = block.low[quant_offset + lane];
                    const auto low1 = block.low[quant_offset + lane + 32];
                    const auto high = block.high[half * 32 + lane];
                    const int scale = half * 8 + lane / 16;
                    const int codes[4] = {
                        static_cast<int>(low0 & 15) + ((high & 3) << 4) - 32,
                        static_cast<int>(low1 & 15) + (((high >> 2) & 3) << 4) - 32,
                        static_cast<int>(low0 >> 4) + (((high >> 4) & 3) << 4) - 32,
                        static_cast<int>(low1 >> 4) + (((high >> 6) & 3) << 4) - 32,
                    };
                    for (int stripe = 0; stripe < 4; ++stripe) {
                        const std::size_t index = input_base + value_offset + stripe * 32 + lane;
                        accumulator = fma_weight(accumulator,
                            d * block.scales[scale + 2 * stripe] * codes[stripe], transposed + index * 8);
                    }
                }
            }
        } else if (type == 21) {
            const auto &block = *reinterpret_cast<const IQ3SBlock *>(raw);
            const float d = fp16(block.d);
            for (int position = 0; position < 256; ++position) {
                const int block32 = position / 32;
                const int pair = block32 / 2;
                const int half = block32 & 1;
                const int local = position & 31;
                const int group8 = local / 8;
                const int item8 = local & 7;
                const int grid_half = item8 / 4;
                const int coordinate = item8 & 3;
                const int quant_offset = pair * 16 + half * 8 + group8 * 2 + grid_half;
                const int high_bit = (block.high[pair * 2 + half]
                                    >> (2 * group8 + grid_half)) & 1;
                const int codebook_index = block.quants[quant_offset] | (high_bit << 8);
                const auto lattice = iq3s_codebook[codebook_index];
                int magnitude = 2 * ((lattice >> (3 * coordinate)) & 7) + 1;
                if (block.signs[pair * 8 + half * 4 + group8] & (1U << item8)) {
                    magnitude = -magnitude;
                }
                const int nibble = half == 0 ? block.scales[pair] & 15
                                             : block.scales[pair] >> 4;
                const float weight = d * static_cast<float>(1 + 2 * nibble)
                                   * static_cast<float>(magnitude);
                accumulator = fma_weight(
                    accumulator, weight, transposed + (input_base + position) * 8);
            }
        } else if (type == 11) {
            const auto &block = *reinterpret_cast<const Q3KBlock *>(raw);
            const float d = fp16(block.d);
            for (int p = 0; p < 256; ++p) {
                const int lane = p & 31;
                const int stripe = p / 32;
                const int low = (block.quants[(stripe / 4) * 32 + lane] >> ((stripe & 3) * 2)) & 3;
                const int high = (block.hmask[lane] >> stripe) & 1;
                const float weight = d * q3_scale(block.scales, p / 16) * ((low | (high << 2)) - 4);
                accumulator = fma_weight(accumulator, weight, transposed + (input_base + p) * 8);
            }
        } else if (type == 8) {
            const auto &block = *reinterpret_cast<const Q8Block *>(raw);
            const float d = fp16(block.d);
            for (int lane = 0; lane < 32; ++lane) {
                accumulator = fma_weight(accumulator, d * block.quants[lane],
                                         transposed + (input_base + lane) * 8);
            }
        } else if (type == 20) {
            const auto &block = *reinterpret_cast<const IQ4NLBlock *>(raw);
            const float d = fp16(block.d);
            for (int lane = 0; lane < 32; ++lane) {
                const int code = lane < 16 ? block.quants[lane] & 15 : block.quants[lane - 16] >> 4;
                accumulator = fma_weight(accumulator, d * iq4_values[code],
                                         transposed + (input_base + lane) * 8);
            }
        } else {
            const auto &block = *reinterpret_cast<const IQ4XSBlock *>(raw);
            const float d = fp16(block.d);
            for (int subblock = 0; subblock < 8; ++subblock) {
                const int scale = ((block.scales_l[subblock / 2] >> (4 * (subblock & 1))) & 15)
                                | (((block.scales_h >> (2 * subblock)) & 3) << 4);
                for (int lane = 0; lane < 32; ++lane) {
                    const int code = lane < 16
                        ? block.quants[subblock * 16 + lane] & 15
                        : block.quants[subblock * 16 + lane - 16] >> 4;
                    accumulator = fma_weight(accumulator, d * (scale - 32) * iq4_values[code],
                        transposed + (input_base + subblock * 32 + lane) * 8);
                }
            }
        }
    }
    _mm256_storeu_ps(result, accumulator);
}

struct Q4KSoATile8 {
    alignas(16) std::uint16_t d[8];
    alignas(16) std::uint16_t dmin[8];
    std::uint8_t scales[12][8];
    std::uint8_t quants[128][8];
};
static_assert(sizeof(Q4KSoATile8) == 8 * sizeof(Q4KBlock));

template <typename Tile>
inline void soa_scale_min(const Tile &tile, int index,
                          __m256 &scale, __m256 &minimum) {
    const auto u8 = [&](int item) {
        return _mm256_cvtepu8_epi32(
            _mm_loadl_epi64(reinterpret_cast<const __m128i *>(tile.scales[item])));
    };
    __m256i s;
    __m256i m;
    if (index < 4) {
        s = _mm256_and_si256(u8(index), _mm256_set1_epi32(63));
        m = _mm256_and_si256(u8(index + 4), _mm256_set1_epi32(63));
    } else {
        s = _mm256_or_si256(
            _mm256_and_si256(u8(index + 4), _mm256_set1_epi32(15)),
            _mm256_slli_epi32(_mm256_srli_epi32(u8(index - 4), 6), 4));
        m = _mm256_or_si256(
            _mm256_srli_epi32(u8(index + 4), 4),
            _mm256_slli_epi32(_mm256_srli_epi32(u8(index), 6), 4));
    }
    scale = _mm256_cvtepi32_ps(s);
    minimum = _mm256_cvtepi32_ps(m);
}

void compute_q4k_soa_tile(const Q4KSoATile8 *tiles, std::size_t tile_index,
                          const float *input, std::size_t input_size,
                          std::size_t blocks_per_row, float *output,
                          std::size_t row_count) {
    __m256 accumulators[8] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
    };
    const Q4KSoATile8 *row_tiles = tiles + tile_index * blocks_per_row;
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto &tile = row_tiles[block_index];
        const __m256 d = _mm256_cvtph_ps(
            _mm_load_si128(reinterpret_cast<const __m128i *>(tile.d)));
        const __m256 dmin = _mm256_cvtph_ps(
            _mm_load_si128(reinterpret_cast<const __m128i *>(tile.dmin)));
        for (int group = 0; group < 4; ++group) {
            __m256 scale0, min0, scale1, min1;
            soa_scale_min(tile, 2 * group, scale0, min0);
            soa_scale_min(tile, 2 * group + 1, scale1, min1);
            const __m256 ds0 = _mm256_mul_ps(d, scale0);
            const __m256 dm0 = _mm256_mul_ps(dmin, min0);
            const __m256 ds1 = _mm256_mul_ps(d, scale1);
            const __m256 dm1 = _mm256_mul_ps(dmin, min1);
            for (int lane = 0; lane < 32; ++lane) {
                const __m128i packed8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(tile.quants[group * 32 + lane]));
                const __m256i packed = _mm256_cvtepu8_epi32(packed8);
                const __m256 code0 = _mm256_cvtepi32_ps(
                    _mm256_and_si256(packed, _mm256_set1_epi32(15)));
                const __m256 code1 = _mm256_cvtepi32_ps(_mm256_srli_epi32(packed, 4));
                const __m256 weight0 = _mm256_sub_ps(_mm256_mul_ps(ds0, code0), dm0);
                const __m256 weight1 = _mm256_sub_ps(_mm256_mul_ps(ds1, code1), dm1);
                const std::size_t index0 = block_index * 256 + group * 64 + lane;
#pragma unroll
                for (int batch = 0; batch < 8; ++batch) {
                    accumulators[batch] = _mm256_fmadd_ps(
                        weight0, _mm256_set1_ps(input[static_cast<std::size_t>(batch) * input_size + index0]),
                        accumulators[batch]);
                    accumulators[batch] = _mm256_fmadd_ps(
                        weight1, _mm256_set1_ps(input[static_cast<std::size_t>(batch) * input_size + index0 + 32]),
                        accumulators[batch]);
                }
            }
        }
    }
    const std::size_t first_row = tile_index * 8;
    const std::size_t valid_rows = std::min<std::size_t>(8, row_count - first_row);
    alignas(32) float rows[8];
    for (int batch = 0; batch < 8; ++batch) {
        _mm256_store_ps(rows, accumulators[batch]);
        std::copy_n(rows, valid_rows,
                    output + static_cast<std::size_t>(batch) * row_count + first_row);
    }
}

struct Q5KSoATile8 {
    alignas(16) std::uint16_t d[8];
    alignas(16) std::uint16_t dmin[8];
    std::uint8_t scales[12][8];
    std::uint8_t high_bits[32][8];
    std::uint8_t quants[128][8];
};
static_assert(sizeof(Q5KSoATile8) == 8 * sizeof(Q5KBlock));

void compute_q5k_soa_tile(const Q5KSoATile8 *tiles, std::size_t tile_index,
                          const float *input, std::size_t input_size,
                          std::size_t blocks_per_row, float *output,
                          std::size_t row_count) {
    __m256 accumulators[8] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
    };
    const Q5KSoATile8 *row_tiles = tiles + tile_index * blocks_per_row;
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto &tile = row_tiles[block_index];
        const __m256 d = _mm256_cvtph_ps(
            _mm_load_si128(reinterpret_cast<const __m128i *>(tile.d)));
        const __m256 dmin = _mm256_cvtph_ps(
            _mm_load_si128(reinterpret_cast<const __m128i *>(tile.dmin)));
        for (int group = 0; group < 4; ++group) {
            __m256 scale0, min0, scale1, min1;
            soa_scale_min(tile, 2 * group, scale0, min0);
            soa_scale_min(tile, 2 * group + 1, scale1, min1);
            const __m256 ds0 = _mm256_mul_ps(d, scale0);
            const __m256 dm0 = _mm256_mul_ps(dmin, min0);
            const __m256 ds1 = _mm256_mul_ps(d, scale1);
            const __m256 dm1 = _mm256_mul_ps(dmin, min1);
            for (int lane = 0; lane < 32; ++lane) {
                const __m256i packed = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(tile.quants[group * 32 + lane])));
                const __m256i high = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(tile.high_bits[lane])));
                const __m256i high0 = _mm256_slli_epi32(
                    _mm256_and_si256(_mm256_srli_epi32(high, 2 * group),
                                     _mm256_set1_epi32(1)), 4);
                const __m256i high1 = _mm256_slli_epi32(
                    _mm256_and_si256(_mm256_srli_epi32(high, 2 * group + 1),
                                     _mm256_set1_epi32(1)), 4);
                const __m256 code0 = _mm256_cvtepi32_ps(_mm256_or_si256(
                    _mm256_and_si256(packed, _mm256_set1_epi32(15)), high0));
                const __m256 code1 = _mm256_cvtepi32_ps(_mm256_or_si256(
                    _mm256_srli_epi32(packed, 4), high1));
                const __m256 weight0 = _mm256_sub_ps(_mm256_mul_ps(ds0, code0), dm0);
                const __m256 weight1 = _mm256_sub_ps(_mm256_mul_ps(ds1, code1), dm1);
                const std::size_t index0 = block_index * 256 + group * 64 + lane;
#pragma unroll
                for (int batch = 0; batch < 8; ++batch) {
                    accumulators[batch] = _mm256_fmadd_ps(
                        weight0, _mm256_set1_ps(input[static_cast<std::size_t>(batch) * input_size + index0]),
                        accumulators[batch]);
                    accumulators[batch] = _mm256_fmadd_ps(
                        weight1, _mm256_set1_ps(input[static_cast<std::size_t>(batch) * input_size + index0 + 32]),
                        accumulators[batch]);
                }
            }
        }
    }
    const std::size_t first_row = tile_index * 8;
    const std::size_t valid_rows = std::min<std::size_t>(8, row_count - first_row);
    alignas(32) float rows[8];
    for (int batch = 0; batch < 8; ++batch) {
        _mm256_store_ps(rows, accumulators[batch]);
        std::copy_n(rows, valid_rows,
                    output + static_cast<std::size_t>(batch) * row_count + first_row);
    }
}

struct Q6KSoATile8 {
    std::uint8_t low[128][8];
    std::uint8_t high[64][8];
    std::int8_t scales[16][8];
    alignas(16) std::uint16_t d[8];
};
static_assert(sizeof(Q6KSoATile8) == 8 * sizeof(Q6KBlock));

void compute_q6k_soa_tile(const Q6KSoATile8 *tiles, std::size_t tile_index,
                          const float *input, std::size_t input_size,
                          std::size_t blocks_per_row, float *output,
                          std::size_t row_count) {
    __m256 accumulators[8] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
    };
    const Q6KSoATile8 *row_tiles = tiles + tile_index * blocks_per_row;
    const __m256i nibble_mask = _mm256_set1_epi32(15);
    const __m256i two_bit_mask = _mm256_set1_epi32(3);
    const __m256i bias = _mm256_set1_epi32(32);
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto &tile = row_tiles[block_index];
        const __m256 d = _mm256_cvtph_ps(
            _mm_load_si128(reinterpret_cast<const __m128i *>(tile.d)));
        for (int half = 0; half < 2; ++half) {
            const int quant_offset = half * 64;
            const int value_offset = half * 128;
            for (int lane = 0; lane < 32; ++lane) {
                const __m256i low0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(tile.low[quant_offset + lane])));
                const __m256i low1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(tile.low[quant_offset + lane + 32])));
                const __m256i high = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(tile.high[half * 32 + lane])));
                const __m256i codes[4] = {
                    _mm256_sub_epi32(_mm256_or_si256(
                        _mm256_and_si256(low0, nibble_mask),
                        _mm256_slli_epi32(_mm256_and_si256(high, two_bit_mask), 4)), bias),
                    _mm256_sub_epi32(_mm256_or_si256(
                        _mm256_and_si256(low1, nibble_mask),
                        _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(high, 2), two_bit_mask), 4)), bias),
                    _mm256_sub_epi32(_mm256_or_si256(
                        _mm256_srli_epi32(low0, 4),
                        _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(high, 4), two_bit_mask), 4)), bias),
                    _mm256_sub_epi32(_mm256_or_si256(
                        _mm256_srli_epi32(low1, 4),
                        _mm256_slli_epi32(_mm256_srli_epi32(high, 6), 4)), bias),
                };
                const int scale_base = half * 8 + lane / 16;
#pragma unroll
                for (int stripe = 0; stripe < 4; ++stripe) {
                    const __m256 scale = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i *>(
                            tile.scales[scale_base + 2 * stripe]))));
                    const __m256 weight = _mm256_mul_ps(
                        _mm256_mul_ps(d, scale), _mm256_cvtepi32_ps(codes[stripe]));
                    const std::size_t index = block_index * 256 + value_offset + stripe * 32 + lane;
#pragma unroll
                    for (int batch = 0; batch < 8; ++batch) {
                        accumulators[batch] = _mm256_fmadd_ps(
                            weight,
                            _mm256_set1_ps(input[static_cast<std::size_t>(batch) * input_size + index]),
                            accumulators[batch]);
                    }
                }
            }
        }
    }
    const std::size_t first_row = tile_index * 8;
    const std::size_t valid_rows = std::min<std::size_t>(8, row_count - first_row);
    alignas(32) float rows[8];
    for (int batch = 0; batch < 8; ++batch) {
        _mm256_store_ps(rows, accumulators[batch]);
        std::copy_n(rows, valid_rows,
                    output + static_cast<std::size_t>(batch) * row_count + first_row);
    }
}

struct IQ4XSSoATile8 {
    alignas(16) std::uint16_t d[8];
    alignas(16) std::uint16_t scales_h[8];
    std::uint8_t scales_l[4][8];
    std::uint8_t quants[128][8];
};
static_assert(sizeof(IQ4XSSoATile8) == 8 * sizeof(IQ4XSBlock));

inline __m256 iq4_values_soa(__m128i codes) {
    const __m128i table = _mm_loadu_si128(reinterpret_cast<const __m128i *>(iq4_values));
    const __m128i looked_up = _mm_shuffle_epi8(table, codes);
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(looked_up));
}

void compute_iq4xs_soa_tile(const IQ4XSSoATile8 *tiles, std::size_t tile_index,
                            const float *input, std::size_t input_size,
                            std::size_t blocks_per_row, float *output,
                            std::size_t row_count) {
    __m256 accumulators[8] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
    };
    const IQ4XSSoATile8 *row_tiles = tiles + tile_index * blocks_per_row;
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto &tile = row_tiles[block_index];
        const __m256 d = _mm256_cvtph_ps(
            _mm_load_si128(reinterpret_cast<const __m128i *>(tile.d)));
        const __m256i scales_h = _mm256_cvtepu16_epi32(
            _mm_load_si128(reinterpret_cast<const __m128i *>(tile.scales_h)));
        for (int subblock = 0; subblock < 8; ++subblock) {
            const __m256i scales_l = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                reinterpret_cast<const __m128i *>(tile.scales_l[subblock / 2])));
            const __m256i low = _mm256_and_si256(
                _mm256_srli_epi32(scales_l, 4 * (subblock & 1)), _mm256_set1_epi32(15));
            const __m256i high = _mm256_slli_epi32(_mm256_and_si256(
                _mm256_srli_epi32(scales_h, 2 * subblock), _mm256_set1_epi32(3)), 4);
            const __m256 scale = _mm256_cvtepi32_ps(
                _mm256_sub_epi32(_mm256_or_si256(low, high), _mm256_set1_epi32(32)));
            const __m256 ds = _mm256_mul_ps(d, scale);
            for (int lane = 0; lane < 32; ++lane) {
                const __m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(
                    tile.quants[subblock * 16 + (lane & 15)]));
                const __m128i codes = lane < 16
                    ? _mm_and_si128(packed, _mm_set1_epi8(15))
                    : _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(15));
                const __m256 weight = _mm256_mul_ps(ds, iq4_values_soa(codes));
                const std::size_t index = block_index * 256 + subblock * 32 + lane;
#pragma unroll
                for (int batch = 0; batch < 8; ++batch) {
                    accumulators[batch] = _mm256_fmadd_ps(
                        weight, _mm256_set1_ps(input[static_cast<std::size_t>(batch) * input_size + index]),
                        accumulators[batch]);
                }
            }
        }
    }
    const std::size_t first_row = tile_index * 8;
    const std::size_t valid_rows = std::min<std::size_t>(8, row_count - first_row);
    alignas(32) float rows[8];
    for (int batch = 0; batch < 8; ++batch) {
        _mm256_store_ps(rows, accumulators[batch]);
        std::copy_n(rows, valid_rows,
                    output + static_cast<std::size_t>(batch) * row_count + first_row);
    }
}

class RowThreadPool {
public:
    RowThreadPool() {
        constexpr int pool_size = 24;
        workers_.reserve(pool_size);
        for (int worker = 0; worker < pool_size; ++worker) {
            workers_.emplace_back([this, worker] { worker_loop(worker); });
        }
    }

    ~RowThreadPool() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_.notify_all();
        for (auto &worker : workers_) worker.join();
    }

    void run(std::size_t rows, int requested_threads,
             std::function<void(std::size_t)> task) {
        std::lock_guard run_lock(run_mutex_);
        {
            std::lock_guard lock(mutex_);
            task_ = std::move(task);
            next_row_.store(0, std::memory_order_relaxed);
            row_count_ = rows;
            active_workers_ = std::max(1, std::min<int>(
                {requested_threads, static_cast<int>(workers_.size()), static_cast<int>(rows)}));
            remaining_workers_ = active_workers_;
            ++generation_;
        }
        start_.notify_all();
        std::unique_lock lock(mutex_);
        done_.wait(lock, [this] { return remaining_workers_ == 0; });
        task_ = {};
    }

private:
    void worker_loop(int worker_index) {
        std::size_t seen_generation = 0;
        for (;;) {
            std::unique_lock lock(mutex_);
            start_.wait(lock, [&] { return stopping_ || generation_ != seen_generation; });
            if (stopping_) return;
            seen_generation = generation_;
            if (worker_index >= active_workers_) continue;
            lock.unlock();
            constexpr std::size_t rows_per_chunk = 16;
            for (;;) {
                const std::size_t begin = next_row_.fetch_add(rows_per_chunk,
                                                               std::memory_order_relaxed);
                if (begin >= row_count_) break;
                const std::size_t end = std::min(row_count_, begin + rows_per_chunk);
                for (std::size_t row = begin; row < end; ++row) task_(row);
            }
            lock.lock();
            if (--remaining_workers_ == 0) done_.notify_one();
        }
    }

    std::mutex run_mutex_;
    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable done_;
    std::vector<std::thread> workers_;
    std::function<void(std::size_t)> task_;
    std::atomic<std::size_t> next_row_{0};
    std::size_t row_count_ = 0;
    std::size_t generation_ = 0;
    int active_workers_ = 0;
    int remaining_workers_ = 0;
    bool stopping_ = false;
};

RowThreadPool &row_thread_pool() {
    static RowThreadPool pool;
    return pool;
}

struct PackedMatrixKey {
    const void *weights = nullptr;
    std::uint32_t type = 0;
    std::size_t input_size = 0;
    std::size_t row_count = 0;

    bool operator==(const PackedMatrixKey &) const = default;
};

struct PackedMatrixKeyHash {
    std::size_t operator()(const PackedMatrixKey &key) const {
        std::size_t value = std::hash<const void *>{}(key.weights);
        value ^= std::hash<std::size_t>{}(key.input_size) + 0x9e3779b9U + (value << 6) + (value >> 2);
        value ^= std::hash<std::size_t>{}(key.row_count) + 0x9e3779b9U + (value << 6) + (value >> 2);
        value ^= static_cast<std::size_t>(key.type) * 0x85ebca6bU;
        return value;
    }
};

using PackedMatrixStorage = std::variant<std::vector<Q4KSoATile8>,
                                         std::vector<Q5KSoATile8>,
                                         std::vector<Q6KSoATile8>,
                                         std::vector<IQ4XSSoATile8>>;

struct PackedMatrixEntry {
    std::size_t blocks_per_row = 0;
    std::size_t tile_count = 0;
    std::size_t bytes = 0;
    PackedMatrixStorage storage;
};

class PackedMatrixCache {
public:
    const PackedMatrixEntry &find_or_pack(const void *weights, std::uint32_t type,
                                          std::size_t input_size, std::size_t row_count) {
        const PackedMatrixKey key{weights, type, input_size, row_count};
        std::lock_guard lock(mutex_);
        const auto found = entries_.find(key);
        if (found != entries_.end()) return found->second;
        PackedMatrixEntry entry;
        entry.blocks_per_row = input_size / 256;
        entry.tile_count = (row_count + 7) / 8;
        if (type == 12) {
            std::vector<Q4KSoATile8> tiles(entry.tile_count * entry.blocks_per_row);
            const auto *source = static_cast<const Q4KBlock *>(weights);
            for (std::size_t tile_index = 0; tile_index < entry.tile_count; ++tile_index) {
                for (std::size_t block_index = 0; block_index < entry.blocks_per_row; ++block_index) {
                    auto &destination = tiles[tile_index * entry.blocks_per_row + block_index];
                    for (std::size_t lane = 0; lane < 8; ++lane) {
                        const std::size_t row = std::min(tile_index * 8 + lane, row_count - 1);
                        const auto &block = source[row * entry.blocks_per_row + block_index];
                        destination.d[lane] = block.d;
                        destination.dmin[lane] = block.dmin;
                        for (int i = 0; i < 12; ++i) destination.scales[i][lane] = block.scales[i];
                        for (int i = 0; i < 128; ++i) destination.quants[i][lane] = block.quants[i];
                    }
                }
            }
            entry.bytes = tiles.size() * sizeof(Q4KSoATile8);
            entry.storage = std::move(tiles);
        } else if (type == 13) {
            std::vector<Q5KSoATile8> tiles(entry.tile_count * entry.blocks_per_row);
            const auto *source = static_cast<const Q5KBlock *>(weights);
            for (std::size_t tile_index = 0; tile_index < entry.tile_count; ++tile_index) {
                for (std::size_t block_index = 0; block_index < entry.blocks_per_row; ++block_index) {
                    auto &destination = tiles[tile_index * entry.blocks_per_row + block_index];
                    for (std::size_t lane = 0; lane < 8; ++lane) {
                        const std::size_t row = std::min(tile_index * 8 + lane, row_count - 1);
                        const auto &block = source[row * entry.blocks_per_row + block_index];
                        destination.d[lane] = block.d;
                        destination.dmin[lane] = block.dmin;
                        for (int i = 0; i < 12; ++i) destination.scales[i][lane] = block.scales[i];
                        for (int i = 0; i < 32; ++i) destination.high_bits[i][lane] = block.high_bits[i];
                        for (int i = 0; i < 128; ++i) destination.quants[i][lane] = block.quants[i];
                    }
                }
            }
            entry.bytes = tiles.size() * sizeof(Q5KSoATile8);
            entry.storage = std::move(tiles);
        } else if (type == 14) {
            std::vector<Q6KSoATile8> tiles(entry.tile_count * entry.blocks_per_row);
            const auto *source = static_cast<const Q6KBlock *>(weights);
            for (std::size_t tile_index = 0; tile_index < entry.tile_count; ++tile_index) {
                for (std::size_t block_index = 0; block_index < entry.blocks_per_row; ++block_index) {
                    auto &destination = tiles[tile_index * entry.blocks_per_row + block_index];
                    for (std::size_t lane = 0; lane < 8; ++lane) {
                        const std::size_t row = std::min(tile_index * 8 + lane, row_count - 1);
                        const auto &block = source[row * entry.blocks_per_row + block_index];
                        destination.d[lane] = block.d;
                        for (int i = 0; i < 128; ++i) destination.low[i][lane] = block.low[i];
                        for (int i = 0; i < 64; ++i) destination.high[i][lane] = block.high[i];
                        for (int i = 0; i < 16; ++i) destination.scales[i][lane] = block.scales[i];
                    }
                }
            }
            entry.bytes = tiles.size() * sizeof(Q6KSoATile8);
            entry.storage = std::move(tiles);
        } else if (type == 23) {
            std::vector<IQ4XSSoATile8> tiles(entry.tile_count * entry.blocks_per_row);
            const auto *source = static_cast<const IQ4XSBlock *>(weights);
            for (std::size_t tile_index = 0; tile_index < entry.tile_count; ++tile_index) {
                for (std::size_t block_index = 0; block_index < entry.blocks_per_row; ++block_index) {
                    auto &destination = tiles[tile_index * entry.blocks_per_row + block_index];
                    for (std::size_t lane = 0; lane < 8; ++lane) {
                        const std::size_t row = std::min(tile_index * 8 + lane, row_count - 1);
                        const auto &block = source[row * entry.blocks_per_row + block_index];
                        destination.d[lane] = block.d;
                        destination.scales_h[lane] = block.scales_h;
                        for (int i = 0; i < 4; ++i) destination.scales_l[i][lane] = block.scales_l[i];
                        for (int i = 0; i < 128; ++i) destination.quants[i][lane] = block.quants[i];
                    }
                }
            }
            entry.bytes = tiles.size() * sizeof(IQ4XSSoATile8);
            entry.storage = std::move(tiles);
        } else {
            throw std::invalid_argument("packed CPU cache does not support this quant type");
        }
        bytes_ += entry.bytes;
        return entries_.emplace(key, std::move(entry)).first->second;
    }

    void clear() {
        std::lock_guard lock(mutex_);
        entries_.clear();
        bytes_ = 0;
    }

    std::size_t bytes() const {
        std::lock_guard lock(mutex_);
        return bytes_;
    }

private:
    mutable std::mutex mutex_;
    std::size_t bytes_ = 0;
    std::unordered_map<PackedMatrixKey, PackedMatrixEntry, PackedMatrixKeyHash> entries_;
};

PackedMatrixCache &packed_matrix_cache() {
    static PackedMatrixCache cache;
    return cache;
}

} // namespace

void cpu_quant_matvec_batch8_rows(const void *weights, std::uint32_t type,
                                  const float *input, std::size_t input_size,
                                  float *output, std::size_t row_count,
                                  int thread_count) {
    if (!weights || !input || !output || input_size == 0 || row_count == 0) {
        throw std::invalid_argument("invalid CPU quant matvec arguments");
    }
    std::vector<float> transposed(input_size * 8 + 8);
    float *aligned = reinterpret_cast<float *>((reinterpret_cast<std::uintptr_t>(transposed.data()) + 31) & ~std::uintptr_t(31));
    for (std::size_t column = 0; column < input_size; ++column) {
        for (int batch = 0; batch < 8; ++batch) aligned[column * 8 + batch] = input[batch * input_size + column];
    }
    const std::size_t row_bytes = (input_size / block_values(type)) * block_bytes(type);
    const auto *source = static_cast<const std::uint8_t *>(weights);
    std::vector<float> row_major(row_count * 8);
    row_thread_pool().run(row_count, thread_count, [&](std::size_t row) {
        compute_row(source + row * row_bytes, type, aligned, input_size,
                    row_major.data() + row * 8);
    });
    for (int batch = 0; batch < 8; ++batch) {
        for (std::size_t row = 0; row < row_count; ++row) {
            output[batch * row_count + row] = row_major[row * 8 + batch];
        }
    }
}

void cpu_quant_matvec_batch8_rows_packed(const void *weights, std::uint32_t type,
                                         const float *input, std::size_t input_size,
                                         float *output, std::size_t row_count,
                                         int thread_count) {
    if (type != 12 && type != 13 && type != 14 && type != 23) {
        cpu_quant_matvec_batch8_rows(weights, type, input, input_size,
                                     output, row_count, thread_count);
        return;
    }
    const auto &entry = packed_matrix_cache().find_or_pack(
        weights, type, input_size, row_count);
    if (type == 12) {
        const auto &tiles = std::get<std::vector<Q4KSoATile8>>(entry.storage);
        row_thread_pool().run(entry.tile_count, thread_count, [&](std::size_t tile_index) {
            compute_q4k_soa_tile(tiles.data(), tile_index, input, input_size,
                                 entry.blocks_per_row, output, row_count);
        });
    } else if (type == 13) {
        const auto &tiles = std::get<std::vector<Q5KSoATile8>>(entry.storage);
        row_thread_pool().run(entry.tile_count, thread_count, [&](std::size_t tile_index) {
            compute_q5k_soa_tile(tiles.data(), tile_index, input, input_size,
                                 entry.blocks_per_row, output, row_count);
        });
    } else if (type == 14) {
        const auto &tiles = std::get<std::vector<Q6KSoATile8>>(entry.storage);
        row_thread_pool().run(entry.tile_count, thread_count, [&](std::size_t tile_index) {
            compute_q6k_soa_tile(tiles.data(), tile_index, input, input_size,
                                 entry.blocks_per_row, output, row_count);
        });
    } else {
        const auto &tiles = std::get<std::vector<IQ4XSSoATile8>>(entry.storage);
        row_thread_pool().run(entry.tile_count, thread_count, [&](std::size_t tile_index) {
            compute_iq4xs_soa_tile(tiles.data(), tile_index, input, input_size,
                                   entry.blocks_per_row, output, row_count);
        });
    }
}

void cpu_quant_matvec_batch8_rows_packed_dual(
    const void *gate_weights, std::uint32_t gate_type, float *gate_output,
    std::size_t gate_rows, const void *up_weights, std::uint32_t up_type,
    float *up_output, std::size_t up_rows, const float *input,
    std::size_t input_size, int thread_count) {
    if (!gate_weights || !up_weights || !input || !gate_output || !up_output ||
        input_size == 0 || gate_rows == 0 || up_rows == 0) {
        throw std::invalid_argument("invalid dual CPU quant matvec arguments");
    }
    const bool gate_soa = gate_type == 12 || gate_type == 13 || gate_type == 14 || gate_type == 23;
    const bool up_soa = up_type == 12 || up_type == 13 || up_type == 14 || up_type == 23;
    if (gate_soa && up_soa) {
        const auto &gate_entry = packed_matrix_cache().find_or_pack(
            gate_weights, gate_type, input_size, gate_rows);
        const auto &up_entry = packed_matrix_cache().find_or_pack(
            up_weights, up_type, input_size, up_rows);
        const std::size_t total_tiles = gate_entry.tile_count + up_entry.tile_count;
        // One barrier for both matrices instead of two (saves 64 barriers per decode step).
        row_thread_pool().run(total_tiles, thread_count, [&](std::size_t tile_index) {
            if (tile_index < gate_entry.tile_count) {
                if (gate_type == 12) {
                    const auto &tiles = std::get<std::vector<Q4KSoATile8>>(gate_entry.storage);
                    compute_q4k_soa_tile(tiles.data(), tile_index, input, input_size,
                                         gate_entry.blocks_per_row, gate_output, gate_rows);
                } else if (gate_type == 13) {
                    const auto &tiles = std::get<std::vector<Q5KSoATile8>>(gate_entry.storage);
                    compute_q5k_soa_tile(tiles.data(), tile_index, input, input_size,
                                         gate_entry.blocks_per_row, gate_output, gate_rows);
                } else if (gate_type == 14) {
                    const auto &tiles = std::get<std::vector<Q6KSoATile8>>(gate_entry.storage);
                    compute_q6k_soa_tile(tiles.data(), tile_index, input, input_size,
                                         gate_entry.blocks_per_row, gate_output, gate_rows);
                } else {
                    const auto &tiles = std::get<std::vector<IQ4XSSoATile8>>(gate_entry.storage);
                    compute_iq4xs_soa_tile(tiles.data(), tile_index, input, input_size,
                                           gate_entry.blocks_per_row, gate_output, gate_rows);
                }
            } else {
                const std::size_t up_tile = tile_index - gate_entry.tile_count;
                if (up_type == 12) {
                    const auto &tiles = std::get<std::vector<Q4KSoATile8>>(up_entry.storage);
                    compute_q4k_soa_tile(tiles.data(), up_tile, input, input_size,
                                         up_entry.blocks_per_row, up_output, up_rows);
                } else if (up_type == 13) {
                    const auto &tiles = std::get<std::vector<Q5KSoATile8>>(up_entry.storage);
                    compute_q5k_soa_tile(tiles.data(), up_tile, input, input_size,
                                         up_entry.blocks_per_row, up_output, up_rows);
                } else if (up_type == 14) {
                    const auto &tiles = std::get<std::vector<Q6KSoATile8>>(up_entry.storage);
                    compute_q6k_soa_tile(tiles.data(), up_tile, input, input_size,
                                         up_entry.blocks_per_row, up_output, up_rows);
                } else {
                    const auto &tiles = std::get<std::vector<IQ4XSSoATile8>>(up_entry.storage);
                    compute_iq4xs_soa_tile(tiles.data(), up_tile, input, input_size,
                                           up_entry.blocks_per_row, up_output, up_rows);
                }
            }
        });
        return;
    }
    // Mixed SoA/generic: fall back to two calls but keep single outer barrier via
    // manual thread pool splitting is non-trivial; sequential is still correct and
    // rare (e.g. Q3_K). The outer caller will handle overlap with GPU, so just run sequentially.
    cpu_quant_matvec_batch8_rows_packed(gate_weights, gate_type, input, input_size,
                                        gate_output, gate_rows, thread_count);
    cpu_quant_matvec_batch8_rows_packed(up_weights, up_type, input, input_size,
                                        up_output, up_rows, thread_count);
}

void clear_cpu_packed_matrix_cache() {
    packed_matrix_cache().clear();
}

std::size_t cpu_packed_matrix_cache_bytes() {
    return packed_matrix_cache().bytes();
}

CpuMatvecBenchmarkResult benchmark_cpu_q4k_soa_batch8(
    const void *weights, const float *input, std::size_t input_size,
    std::size_t row_count, int iterations, int thread_count,
    const float *reference_output) {
    if (!weights || !input || input_size == 0 || input_size % 256 != 0 ||
        row_count == 0 || iterations <= 0 || thread_count <= 0) {
        throw std::invalid_argument("invalid Q4_K SoA benchmark arguments");
    }
    const std::size_t blocks_per_row = input_size / 256;
    const std::size_t tile_count = (row_count + 7) / 8;
    const auto *source = static_cast<const Q4KBlock *>(weights);
    const auto packing_started = std::chrono::steady_clock::now();
    std::vector<Q4KSoATile8> tiles(tile_count * blocks_per_row);
    for (std::size_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            auto &destination = tiles[tile_index * blocks_per_row + block_index];
            for (std::size_t lane = 0; lane < 8; ++lane) {
                const std::size_t row = std::min(tile_index * 8 + lane, row_count - 1);
                const auto &block = source[row * blocks_per_row + block_index];
                destination.d[lane] = block.d;
                destination.dmin[lane] = block.dmin;
                for (int index = 0; index < 12; ++index) {
                    destination.scales[index][lane] = block.scales[index];
                }
                for (int index = 0; index < 128; ++index) {
                    destination.quants[index][lane] = block.quants[index];
                }
            }
        }
    }
    const double packing_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - packing_started).count();
    std::vector<float> output(8 * row_count);
    auto execute = [&] {
        row_thread_pool().run(tile_count, thread_count, [&](std::size_t tile_index) {
            compute_q4k_soa_tile(tiles.data(), tile_index, input, input_size,
                                 blocks_per_row, output.data(), row_count);
        });
    };
    execute();
    const auto started = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) execute();
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count() / iterations;
    CpuMatvecBenchmarkResult result;
    result.milliseconds = seconds * 1000.0;
    const std::size_t original_bytes = row_count * blocks_per_row * sizeof(Q4KBlock);
    result.bandwidth_gbps = static_cast<double>(original_bytes) / seconds / 1.0e9;
    result.packing_milliseconds = packing_ms;
    if (reference_output) {
        for (std::size_t index = 0; index < output.size(); ++index) {
            result.max_absolute_error = std::max(
                result.max_absolute_error, std::abs(output[index] - reference_output[index]));
        }
    }
    return result;
}

CpuMatvecBenchmarkResult benchmark_cpu_q5k_soa_batch8(
    const void *weights, const float *input, std::size_t input_size,
    std::size_t row_count, int iterations, int thread_count,
    const float *reference_output) {
    if (!weights || !input || input_size == 0 || input_size % 256 != 0 ||
        row_count == 0 || iterations <= 0 || thread_count <= 0) {
        throw std::invalid_argument("invalid Q5_K SoA benchmark arguments");
    }
    const std::size_t blocks_per_row = input_size / 256;
    const std::size_t tile_count = (row_count + 7) / 8;
    const auto *source = static_cast<const Q5KBlock *>(weights);
    const auto packing_started = std::chrono::steady_clock::now();
    std::vector<Q5KSoATile8> tiles(tile_count * blocks_per_row);
    for (std::size_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            auto &destination = tiles[tile_index * blocks_per_row + block_index];
            for (std::size_t lane = 0; lane < 8; ++lane) {
                const std::size_t row = std::min(tile_index * 8 + lane, row_count - 1);
                const auto &block = source[row * blocks_per_row + block_index];
                destination.d[lane] = block.d;
                destination.dmin[lane] = block.dmin;
                for (int index = 0; index < 12; ++index) destination.scales[index][lane] = block.scales[index];
                for (int index = 0; index < 32; ++index) destination.high_bits[index][lane] = block.high_bits[index];
                for (int index = 0; index < 128; ++index) destination.quants[index][lane] = block.quants[index];
            }
        }
    }
    const double packing_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - packing_started).count();
    std::vector<float> output(8 * row_count);
    auto execute = [&] {
        row_thread_pool().run(tile_count, thread_count, [&](std::size_t tile_index) {
            compute_q5k_soa_tile(tiles.data(), tile_index, input, input_size,
                                 blocks_per_row, output.data(), row_count);
        });
    };
    execute();
    const auto started = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) execute();
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count() / iterations;
    CpuMatvecBenchmarkResult result;
    result.milliseconds = seconds * 1000.0;
    result.bandwidth_gbps = static_cast<double>(row_count * blocks_per_row * sizeof(Q5KBlock))
                          / seconds / 1.0e9;
    result.packing_milliseconds = packing_ms;
    if (reference_output) {
        for (std::size_t index = 0; index < output.size(); ++index) {
            result.max_absolute_error = std::max(
                result.max_absolute_error, std::abs(output[index] - reference_output[index]));
        }
    }
    return result;
}

CpuMatvecBenchmarkResult benchmark_cpu_q6k_soa_batch8(
    const void *weights, const float *input, std::size_t input_size,
    std::size_t row_count, int iterations, int thread_count,
    const float *reference_output) {
    if (!weights || !input || input_size == 0 || input_size % 256 != 0 ||
        row_count == 0 || iterations <= 0 || thread_count <= 0) {
        throw std::invalid_argument("invalid Q6_K SoA benchmark arguments");
    }
    const std::size_t blocks_per_row = input_size / 256;
    const std::size_t tile_count = (row_count + 7) / 8;
    const auto *source = static_cast<const Q6KBlock *>(weights);
    const auto packing_started = std::chrono::steady_clock::now();
    std::vector<Q6KSoATile8> tiles(tile_count * blocks_per_row);
    for (std::size_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            auto &destination = tiles[tile_index * blocks_per_row + block_index];
            for (std::size_t lane = 0; lane < 8; ++lane) {
                const std::size_t row = std::min(tile_index * 8 + lane, row_count - 1);
                const auto &block = source[row * blocks_per_row + block_index];
                destination.d[lane] = block.d;
                for (int i = 0; i < 128; ++i) destination.low[i][lane] = block.low[i];
                for (int i = 0; i < 64; ++i) destination.high[i][lane] = block.high[i];
                for (int i = 0; i < 16; ++i) destination.scales[i][lane] = block.scales[i];
            }
        }
    }
    const double packing_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - packing_started).count();
    std::vector<float> output(8 * row_count);
    auto execute = [&] {
        row_thread_pool().run(tile_count, thread_count, [&](std::size_t tile_index) {
            compute_q6k_soa_tile(tiles.data(), tile_index, input, input_size,
                                 blocks_per_row, output.data(), row_count);
        });
    };
    execute();
    const auto started = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) execute();
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count() / iterations;
    CpuMatvecBenchmarkResult result;
    result.milliseconds = seconds * 1000.0;
    result.bandwidth_gbps = static_cast<double>(row_count * blocks_per_row * sizeof(Q6KBlock))
                          / seconds / 1.0e9;
    result.packing_milliseconds = packing_ms;
    if (reference_output) {
        for (std::size_t index = 0; index < output.size(); ++index) {
            result.max_absolute_error = std::max(
                result.max_absolute_error, std::abs(output[index] - reference_output[index]));
        }
    }
    return result;
}

CpuMatvecBenchmarkResult benchmark_cpu_iq4xs_soa_batch8(
    const void *weights, const float *input, std::size_t input_size,
    std::size_t row_count, int iterations, int thread_count,
    const float *reference_output) {
    if (!weights || !input || input_size == 0 || input_size % 256 != 0 ||
        row_count == 0 || iterations <= 0 || thread_count <= 0) {
        throw std::invalid_argument("invalid IQ4_XS SoA benchmark arguments");
    }
    const std::size_t blocks_per_row = input_size / 256;
    const std::size_t tile_count = (row_count + 7) / 8;
    const auto *source = static_cast<const IQ4XSBlock *>(weights);
    const auto packing_started = std::chrono::steady_clock::now();
    std::vector<IQ4XSSoATile8> tiles(tile_count * blocks_per_row);
    for (std::size_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            auto &destination = tiles[tile_index * blocks_per_row + block_index];
            for (std::size_t lane = 0; lane < 8; ++lane) {
                const std::size_t row = std::min(tile_index * 8 + lane, row_count - 1);
                const auto &block = source[row * blocks_per_row + block_index];
                destination.d[lane] = block.d;
                destination.scales_h[lane] = block.scales_h;
                for (int index = 0; index < 4; ++index) destination.scales_l[index][lane] = block.scales_l[index];
                for (int index = 0; index < 128; ++index) destination.quants[index][lane] = block.quants[index];
            }
        }
    }
    const double packing_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - packing_started).count();
    std::vector<float> output(8 * row_count);
    auto execute = [&] {
        row_thread_pool().run(tile_count, thread_count, [&](std::size_t tile_index) {
            compute_iq4xs_soa_tile(tiles.data(), tile_index, input, input_size,
                                   blocks_per_row, output.data(), row_count);
        });
    };
    execute();
    const auto started = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) execute();
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count() / iterations;
    CpuMatvecBenchmarkResult result;
    result.milliseconds = seconds * 1000.0;
    result.bandwidth_gbps = static_cast<double>(row_count * blocks_per_row * sizeof(IQ4XSBlock))
                          / seconds / 1.0e9;
    result.packing_milliseconds = packing_ms;
    if (reference_output) {
        for (std::size_t index = 0; index < output.size(); ++index) {
            result.max_absolute_error = std::max(
                result.max_absolute_error, std::abs(output[index] - reference_output[index]));
        }
    }
    return result;
}

void compute_dflash_noncausal_attention(
    const float *queries, const float *keys, const float *values,
    const std::uint16_t *cache_keys, const std::uint16_t *cache_values,
    std::size_t cached, float *output) {
    constexpr std::size_t block = 8;
    constexpr std::size_t heads = 32;
    constexpr std::size_t kv_heads = 8;
    constexpr std::size_t head_size = 128;
    constexpr std::size_t kv_width = kv_heads * head_size;
    const std::size_t total_items = cached + block;

    row_thread_pool().run(block * heads, 16, [&](std::size_t job) {
        const std::size_t item = job / heads;
        const std::size_t head = job % heads;
        const std::size_t kv_head = head / (heads / kv_heads);
        const float *query = queries + (item * heads + head) * head_size;
        float maximum = -std::numeric_limits<float>::infinity();

        thread_local std::vector<float> scores;
        scores.resize(total_items);

        for (std::size_t source = 0; source < total_items; ++source) {
            const std::size_t query_position = cached + item;
            const std::size_t distance = query_position > source
                ? query_position - source : source - query_position;
            if (distance >= 2048) {
                scores[source] = -std::numeric_limits<float>::infinity();
                continue;
            }
            const std::uint16_t *key = (source < cached && cache_keys)
                ? cache_keys + source * kv_width + kv_head * head_size
                : nullptr;

            float dot = 0.0F;
            if (key) {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                for (std::size_t dim = 0; dim < head_size; dim += 16) {
                    __m128i h0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(key + dim));
                    __m128i h1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(key + dim + 8));
                    __m256 kf0 = _mm256_cvtph_ps(h0);
                    __m256 kf1 = _mm256_cvtph_ps(h1);
                    __m256 qf0 = _mm256_loadu_ps(query + dim);
                    __m256 qf1 = _mm256_loadu_ps(query + dim + 8);
                    acc0 = _mm256_fmadd_ps(qf0, kf0, acc0);
                    acc1 = _mm256_fmadd_ps(qf1, kf1, acc1);
                }
                __m256 acc = _mm256_add_ps(acc0, acc1);
                __m128 hi = _mm256_extractf128_ps(acc, 1);
                __m128 lo = _mm256_castps256_ps128(acc);
                __m128 sum = _mm_add_ps(hi, lo);
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);
                dot = _mm_cvtss_f32(sum);
            } else {
                const float *kptr = keys + (source - cached) * kv_width + kv_head * head_size;
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                for (std::size_t dim = 0; dim < head_size; dim += 16) {
                    __m256 kf0 = _mm256_loadu_ps(kptr + dim);
                    __m256 kf1 = _mm256_loadu_ps(kptr + dim + 8);
                    __m256 qf0 = _mm256_loadu_ps(query + dim);
                    __m256 qf1 = _mm256_loadu_ps(query + dim + 8);
                    acc0 = _mm256_fmadd_ps(qf0, kf0, acc0);
                    acc1 = _mm256_fmadd_ps(qf1, kf1, acc1);
                }
                __m256 acc = _mm256_add_ps(acc0, acc1);
                __m128 hi = _mm256_extractf128_ps(acc, 1);
                __m128 lo = _mm256_castps256_ps128(acc);
                __m128 sum = _mm_add_ps(hi, lo);
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);
                dot = _mm_cvtss_f32(sum);
            }
            scores[source] = dot * 0.08838834764831843F;
            maximum = std::max(maximum, scores[source]);
        }

        float denominator = 0.0F;
        for (float &score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }

        float *result = output + (item * heads + head) * head_size;
        std::memset(result, 0, head_size * sizeof(float));

        for (std::size_t source = 0; source < total_items; ++source) {
            const float probability = scores[source] / denominator;
            const std::uint16_t *value = (source < cached && cache_values)
                ? cache_values + source * kv_width + kv_head * head_size
                : nullptr;
            const __m256 vprob = _mm256_set1_ps(probability);
            if (value) {
                for (std::size_t dim = 0; dim < head_size; dim += 8) {
                    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(value + dim));
                    __m256 vf = _mm256_cvtph_ps(h);
                    __m256 rf = _mm256_loadu_ps(result + dim);
                    rf = _mm256_fmadd_ps(vprob, vf, rf);
                    _mm256_storeu_ps(result + dim, rf);
                }
            } else {
                const float *vptr = values + (source - cached) * kv_width + kv_head * head_size;
                for (std::size_t dim = 0; dim < head_size; dim += 8) {
                    __m256 vf = _mm256_loadu_ps(vptr + dim);
                    __m256 rf = _mm256_loadu_ps(result + dim);
                    rf = _mm256_fmadd_ps(vprob, vf, rf);
                    _mm256_storeu_ps(result + dim, rf);
                }
            }
        }
    });
}

} // namespace qwen_native

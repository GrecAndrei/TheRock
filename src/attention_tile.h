#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#define QWEN_TILE_HD __host__ __device__
#else
#define QWEN_TILE_HD
#endif

namespace qwen_native {

// Temporary shared-memory storage only. Global Q8 history, scales, query
// ordering and softmax arithmetic remain unchanged. Callers synchronize after
// all stage() calls and before reading any key/value coefficients.
template <int Tokens, bool Packed>
struct AttentionSharedTile;

template <int Tokens>
struct AttentionSharedTile<Tokens, false> {
    float keys[Tokens * 256];
    float values[Tokens * 256];

    QWEN_TILE_HD void stage(std::size_t index, std::int8_t key_code,
                            std::int8_t value_code, float key_scale,
                            float value_scale) {
        keys[index] = static_cast<float>(key_code) * key_scale;
        values[index] = static_cast<float>(value_code) * value_scale;
    }
    QWEN_TILE_HD float key(std::size_t index) const { return keys[index]; }
    QWEN_TILE_HD float value(std::size_t index) const { return values[index]; }
};

template <int Tokens>
struct AttentionSharedTile<Tokens, true> {
    std::int8_t key_codes[Tokens * 256];
    std::int8_t value_codes[Tokens * 256];
    float key_scales[Tokens * 8];
    float value_scales[Tokens * 8];

    QWEN_TILE_HD void stage(std::size_t index, std::int8_t key_code,
                            std::int8_t value_code, float key_scale,
                            float value_scale) {
        key_codes[index] = key_code;
        value_codes[index] = value_code;
        // Exactly one cooperative writer owns each 32-element group scale.
        if (index % 32 == 0) {
            key_scales[index / 32] = key_scale;
            value_scales[index / 32] = value_scale;
        }
    }
    QWEN_TILE_HD float key(std::size_t index) const {
        return static_cast<float>(key_codes[index]) * key_scales[index / 32];
    }
    QWEN_TILE_HD float value(std::size_t index) const {
        return static_cast<float>(value_codes[index]) * value_scales[index / 32];
    }
};

static_assert(sizeof(AttentionSharedTile<16, false>) == 32768);
static_assert(sizeof(AttentionSharedTile<16, true>) == 9216);

} // namespace qwen_native

#undef QWEN_TILE_HD

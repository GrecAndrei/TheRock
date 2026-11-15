// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <string_view>

namespace rxinfer {

enum class DType : uint8_t {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    FP8_E4M3 = 3,
    I8 = 4,
    I32 = 5,
    Q4_GGUF_0 = 6,      // Standard GGUF Q4_0: block32 = {f16 d, uint8_t qs[16]} (18 B/32 elems)
    R4_GFX12_G64 = 7,   // Custom gfx1201-native W4: 256B-aligned SoA packed tiles + FP16 scales/zeros
    R4_GFX12_G128 = 8,
    FP4_E2M1_G16 = 9,   // DeepSeek-V4.1 post-RoPE KV: 16x E2M1 nibbles (8B) + 1x OCP E4M3 scale (1B) = 9B/16 elems
};

constexpr inline std::string_view dtype_name(DType dt) {
    switch (dt) {
        case DType::F32: return "f32";
        case DType::F16: return "f16";
        case DType::BF16: return "bf16";
        case DType::FP8_E4M3: return "fp8_e4m3";
        case DType::I8: return "i8";
        case DType::I32: return "i32";
        case DType::Q4_GGUF_0: return "q4_0";
        case DType::R4_GFX12_G64: return "r4_gfx12_g64";
        case DType::R4_GFX12_G128: return "r4_gfx12_g128";
        case DType::FP4_E2M1_G16: return "fp4_e2m1_g16";
    }
    return "unknown";
}

constexpr inline size_t dtype_element_bits(DType dt) {
    switch (dt) {
        case DType::F32:
        case DType::I32: return 32;
        case DType::F16:
        case DType::BF16: return 16;
        case DType::FP8_E4M3:
        case DType::I8: return 8;
        case DType::Q4_GGUF_0:
        case DType::R4_GFX12_G64:
        case DType::R4_GFX12_G128:
        case DType::FP4_E2M1_G16: return 4;
    }
    return 0;
}

struct TensorView {
    void* data = nullptr;
    void* scales = nullptr; // Optional SoA scales buffer for quantized tensors (R4_*)
    void* zeros = nullptr;  // Optional SoA zero-points (or scale*zero) buffer
    std::array<int64_t, 4> shape{0, 0, 0, 0};
    std::array<int64_t, 4> stride{0, 0, 0, 0};
    int ndim = 0;
    DType dtype = DType::F16;

    int64_t numel() const {
        if (ndim <= 0) return 0;
        int64_t n = 1;
        for (int i = 0; i < ndim; ++i) n *= shape[i];
        return n;
    }

    bool empty() const { return data == nullptr || ndim == 0; }
};

} // namespace rxinfer

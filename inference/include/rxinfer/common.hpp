// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace rxinfer {

#define RX_HIP_CHECK(expr)                                                     \
    do {                                                                       \
        hipError_t _err = (expr);                                              \
        if (_err != hipSuccess) {                                              \
            std::fprintf(stderr, "HIP error %d (%s) at %s:%d: %s\n",           \
                         static_cast<int>(_err), hipGetErrorString(_err),      \
                         __FILE__, __LINE__, #expr);                           \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

constexpr inline uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

constexpr inline uint64_t div_ceil(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
}

struct DeviceInfo {
    int device_id = 0;
    std::string name;
    std::string gcn_arch_name;
    size_t total_vram_bytes = 0;
    int multiprocessor_count = 0;
    int warp_size = 32;
    size_t shared_mem_per_block = 0;
    size_t l2_cache_size = 0;
    int memory_clock_khz = 0;
    int memory_bus_width = 0;
    bool large_bar = false;
};

DeviceInfo query_device_info(int device_id = 0);

} // namespace rxinfer

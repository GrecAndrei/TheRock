// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/arena.hpp"

#include <stdexcept>
#include <utility>

namespace rxinfer {

DeviceInfo query_device_info(int device_id) {
    hipDeviceProp_t prop{};
    RX_HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    size_t free_bytes = 0, total_bytes = 0;
    int prev_dev = 0;
    RX_HIP_CHECK(hipGetDevice(&prev_dev));
    RX_HIP_CHECK(hipSetDevice(device_id));
    RX_HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    if (prev_dev != device_id) {
        RX_HIP_CHECK(hipSetDevice(prev_dev));
    }

    DeviceInfo info;
    info.device_id = device_id;
    info.name = prop.name;
    info.gcn_arch_name = prop.gcnArchName;
    info.total_vram_bytes = total_bytes;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.warp_size = prop.warpSize;
    info.shared_mem_per_block = prop.sharedMemPerBlock;
    info.l2_cache_size = prop.l2CacheSize;
    info.memory_clock_khz = prop.memoryClockRate;
    info.memory_bus_width = prop.memoryBusWidth;
    info.large_bar = prop.isLargeBar != 0;
    return info;
}

DeviceArena::~DeviceArena() {
    release();
}

DeviceArena::DeviceArena(DeviceArena&& other) noexcept
    : base_(other.base_),
      capacity_(other.capacity_),
      offset_(other.offset_),
      label_(std::move(other.label_)) {
    other.base_ = nullptr;
    other.capacity_ = 0;
    other.offset_ = 0;
}

DeviceArena& DeviceArena::operator=(DeviceArena&& other) noexcept {
    if (this != &other) {
        release();
        base_ = other.base_;
        capacity_ = other.capacity_;
        offset_ = other.offset_;
        label_ = std::move(other.label_);
        other.base_ = nullptr;
        other.capacity_ = 0;
        other.offset_ = 0;
    }
    return *this;
}

void DeviceArena::init(size_t capacity_bytes, const std::string& label) {
    release();
    label_ = label;
    capacity_ = align_up(capacity_bytes, 256);
    if (capacity_ > 0) {
        RX_HIP_CHECK(hipMalloc(&base_, capacity_));
    }
    offset_ = 0;
}

void DeviceArena::release() {
    if (base_ != nullptr) {
        hipError_t err = hipFree(base_);
        if (err != hipSuccess) {
            std::fprintf(stderr, "Warning: hipFree failed in DeviceArena(%s): %s\n",
                         label_.c_str(), hipGetErrorString(err));
        }
        base_ = nullptr;
    }
    capacity_ = 0;
    offset_ = 0;
}

void* DeviceArena::alloc(size_t bytes, size_t alignment) {
    if (bytes == 0) return nullptr;
    const size_t aligned_off = align_up(offset_, alignment);
    if (aligned_off + bytes > capacity_) {
        throw std::runtime_error("DeviceArena(" + label_ + ") out of memory: need " +
                                 std::to_string(aligned_off + bytes) + ", capacity " +
                                 std::to_string(capacity_));
    }
    offset_ = aligned_off + bytes;
    return static_cast<uint8_t*>(base_) + aligned_off;
}

} // namespace rxinfer

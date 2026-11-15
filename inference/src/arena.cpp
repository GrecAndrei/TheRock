// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/arena.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace rxinfer {
namespace {

bool query_drm_free_vram_bytes(const hipDeviceProp_t& prop, size_t& out_free_bytes) {
    std::error_code ec;
    const std::filesystem::path drm_dir("/sys/class/drm");
    if (!std::filesystem::exists(drm_dir, ec)) return false;

    char pci_slot[32];
    std::snprintf(pci_slot, sizeof(pci_slot), "%04x:%02x:%02x.0",
                  static_cast<unsigned>(prop.pciDomainID),
                  static_cast<unsigned>(prop.pciBusID),
                  static_cast<unsigned>(prop.pciDeviceID));

    for (const auto& entry : std::filesystem::directory_iterator(drm_dir, ec)) {
        const auto name = entry.path().filename().string();
        if (name.rfind("card", 0) != 0 || name.find('-') != std::string::npos) continue;

        const auto dev_dir = entry.path() / "device";
        std::ifstream uev(dev_dir / "uevent");
        if (!uev) continue;

        bool matched = false;
        std::string line;
        while (std::getline(uev, line)) {
            if (line.rfind("PCI_SLOT_NAME=", 0) == 0 && line.substr(14) == pci_slot) {
                matched = true;
                break;
            }
        }
        if (!matched) continue;

        uint64_t vram_total = 0, vram_used = 0;
        std::ifstream ftot(dev_dir / "mem_info_vram_total");
        std::ifstream fuse(dev_dir / "mem_info_vram_used");
        if ((ftot >> vram_total) && (fuse >> vram_used) && vram_total > 0) {
            out_free_bytes = (vram_total > vram_used) ? static_cast<size_t>(vram_total - vram_used) : 0;
            return true;
        }
    }
    return false;
}

} // namespace

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

    size_t drm_free = 0;
    if (query_drm_free_vram_bytes(prop, drm_free)) {
        free_bytes = std::min(free_bytes, drm_free);
    }

    DeviceInfo info;
    info.device_id = device_id;
    info.name = prop.name;
    info.gcn_arch_name = prop.gcnArchName;
    info.total_vram_bytes = total_bytes;
    info.free_vram_bytes = free_bytes;
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

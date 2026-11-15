// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rxinfer/common.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rxinfer {

// Persistent GPU memory arena: one upfront hipMalloc per major pool (weights, state/KV, scratch),
// zero allocations on the token decode path.
class DeviceArena {
public:
    DeviceArena() = default;
    ~DeviceArena();

    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    DeviceArena(DeviceArena&& other) noexcept;
    DeviceArena& operator=(DeviceArena&& other) noexcept;

    void init(size_t capacity_bytes, const std::string& label = "arena");
    void release();

    // Bump-allocate aligned device slice (default 256B aligned for gfx1201 cachelines).
    void* alloc(size_t bytes, size_t alignment = 256);
    void reset_offset() { offset_ = 0; }

    void* base() const { return base_; }
    size_t capacity() const { return capacity_; }
    size_t used() const { return offset_; }
    size_t remaining() const { return capacity_ - offset_; }
    const std::string& label() const { return label_; }

private:
    void* base_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
    std::string label_;
};

// Device-resident per-step parameters updated once per token so hipGraph needs zero node updates.
struct alignas(64) StepParams {
    int32_t token_id = 0;
    int32_t pos = 0;         // current sequence position
    int32_t seq_len = 1;     // tokens in this step (1 for decode, >1 for prefill/MTP)
    int32_t kv_seq_len = 0;  // total valid KV cache length after this step
    int32_t mrope_pos[3] = {0, 0, 0};
};

} // namespace rxinfer

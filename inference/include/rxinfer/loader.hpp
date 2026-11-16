// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rxinfer/model.hpp"

#include <cstdint>
#include <functional>
#include <string>

namespace rxinfer {

using LoadProgressFn = std::function<void(size_t done_bytes, size_t total_bytes)>;

struct LoaderOptions {
    int num_threads = 0; // 0 -> auto hardware concurrency
    bool fast_synthetic = false; // If true in synthetic mode, fill R4 device buffers directly without FP32 packing
    LoadProgressFn progress_cb = nullptr;
};

// Compute exact device arena byte budget needed for all weights of cfg.
size_t estimate_model_weight_bytes(const ModelConfig& cfg);

// Create synthetic weights matching cfg (either realistic packed R4 weights or fast random R4 for benchmarking).
ModelWeights load_synthetic_model(const ModelConfig& cfg, uint32_t seed = 42, const LoaderOptions& opts = {});

// Load HuggingFace safetensors directory (config.json + model.safetensors[.index.json]) -> R4_GFX12_G64.
ModelWeights load_from_safetensors_dir(const std::string& model_dir,
                                       const ModelConfig* override_cfg = nullptr,
                                       const LoaderOptions& opts = {});

// Load GGUF v2/v3 file -> R4_GFX12_G64 (supports F32, F16, BF16, Q4_0, Q8_0 source tensors).
ModelWeights load_from_gguf(const std::string& gguf_path,
                            const ModelConfig* override_cfg = nullptr,
                            const LoaderOptions& opts = {});

// Native pre-packed RXR4 binary checkpoint for near-instant reload.
void save_rxr4_checkpoint(const ModelWeights& weights, const std::string& path);
ModelWeights load_from_rxr4(const std::string& path, const LoaderOptions& opts = {});

} // namespace rxinfer

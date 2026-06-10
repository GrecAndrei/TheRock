// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Static GPU descriptor table.
// Each entry is filled in from the hardware manual and empirical measurement.

#include "therock/inference/hardware.h"

#include <string.h>

// ---------------------------------------------------------------------------
// Descriptor table
// ---------------------------------------------------------------------------

static const TheRockHardwareDesc s_descriptors[] = {

    // --- RDNA2 ---

    {
        .gfx_arch               = "gfx1030",
        .marketing_name         = "RX 6800 XT / 6900 XT",
        .wavefront_size         = 32,
        .simd_per_cu            = 2,
        .cu_count               = 72,  // 6900 XT; 6800 XT = 72 too, 6800 = 60
        .matrix_instrs          = THEROCK_MATRIX_INSTR_WMMA,
        .wmma_m = 16, .wmma_n = 16, .wmma_k = 16,
        .vram_bytes             = 16ULL * 1024 * 1024 * 1024,
        .inf_cache_bytes        = 128ULL * 1024 * 1024,
        .lds_per_cu_bytes       = 64 * 1024,
        .l1_per_cu_bytes        = 16 * 1024,
        .vram_bandwidth_bps     = 512ULL * 1024 * 1024 * 1024,  // ~512 GB/s
        .inf_cache_bandwidth_bps = 1750ULL * 1024 * 1024 * 1024,
        .fp16_flops             = 23ULL * 1000 * 1000 * 1000 * 1000,
        .prefetch_lookahead_layers = 2,
    },

    {
        .gfx_arch               = "gfx1032",
        .marketing_name         = "RX 6650 XT / 6600 XT",
        .wavefront_size         = 32,
        .simd_per_cu            = 2,
        .cu_count               = 32,  // 6650 XT = 32 CU
        .matrix_instrs          = THEROCK_MATRIX_INSTR_WMMA,
        .wmma_m = 16, .wmma_n = 16, .wmma_k = 16,
        .vram_bytes             = 8ULL * 1024 * 1024 * 1024,
        .inf_cache_bytes        = 32ULL * 1024 * 1024,
        .lds_per_cu_bytes       = 64 * 1024,
        .l1_per_cu_bytes        = 16 * 1024,
        .vram_bandwidth_bps     = 280ULL * 1024 * 1024 * 1024,  // ~280 GB/s
        .inf_cache_bandwidth_bps = 1024ULL * 1024 * 1024 * 1024, // ~1 TB/s
        .fp16_flops             = 10ULL * 1000 * 1000 * 1000 * 1000,
        .prefetch_lookahead_layers = 1,
    },

    // --- RDNA3 ---

    {
        .gfx_arch               = "gfx1100",
        .marketing_name         = "RX 7900 XTX",
        .wavefront_size         = 32,
        .simd_per_cu            = 2,
        .cu_count               = 96,
        .matrix_instrs          = THEROCK_MATRIX_INSTR_WMMA,
        .wmma_m = 16, .wmma_n = 16, .wmma_k = 16,
        .vram_bytes             = 24ULL * 1024 * 1024 * 1024,
        .inf_cache_bytes        = 96ULL * 1024 * 1024,
        .lds_per_cu_bytes       = 64 * 1024,
        .l1_per_cu_bytes        = 32 * 1024,
        .vram_bandwidth_bps     = 960ULL * 1024 * 1024 * 1024,
        .inf_cache_bandwidth_bps = 2700ULL * 1024 * 1024 * 1024,
        .fp16_flops             = 61ULL * 1000 * 1000 * 1000 * 1000,
        .prefetch_lookahead_layers = 2,
    },

    // --- CDNA2 ---

    {
        .gfx_arch               = "gfx90a",
        .marketing_name         = "MI250X",
        .wavefront_size         = 64,
        .simd_per_cu            = 4,
        .cu_count               = 110,  // per GCD; MI250X has 2 GCDs
        .matrix_instrs          = THEROCK_MATRIX_INSTR_MFMA,
        .wmma_m = 0, .wmma_n = 0, .wmma_k = 0,
        .vram_bytes             = 64ULL * 1024 * 1024 * 1024,  // per GCD
        .inf_cache_bytes        = 0,
        .lds_per_cu_bytes       = 64 * 1024,
        .l1_per_cu_bytes        = 16 * 1024,
        .vram_bandwidth_bps     = 1600ULL * 1024 * 1024 * 1024,  // HBM2e per GCD
        .inf_cache_bandwidth_bps = 0,
        .fp16_flops             = 383ULL * 1000 * 1000 * 1000 * 1000,
        .prefetch_lookahead_layers = 3,
    },
};

static const size_t s_descriptor_count =
    sizeof(s_descriptors) / sizeof(s_descriptors[0]);

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------

const TheRockHardwareDesc *therock_hardware_by_arch(const char *gfx_arch) {
  for (size_t i = 0; i < s_descriptor_count; i++) {
    if (strcmp(s_descriptors[i].gfx_arch, gfx_arch) == 0)
      return &s_descriptors[i];
  }
  return NULL;
}

// Detection via HIP agent query — implemented in hardware_detect_hip.cpp
// to avoid pulling HIP headers into the C translation unit.
extern const char *therock_detect_gfx_arch(void);

const TheRockHardwareDesc *therock_hardware_detect(void) {
  const char *arch = therock_detect_gfx_arch();
  if (!arch) return NULL;
  return therock_hardware_by_arch(arch);
}

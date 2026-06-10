// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Hardware descriptor: the single source of truth about a GPU's capabilities.
// Every scheduling, memory placement, and kernel selection decision reads from this.
// Adding a new GPU = filling in one of these structs.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Matrix instruction types available on the hardware.
// Not mutually exclusive — a GPU can support multiple.
typedef enum TheRockMatrixInstr {
  THEROCK_MATRIX_INSTR_NONE   = 0,
  THEROCK_MATRIX_INSTR_WMMA   = 1 << 0,  // RDNA2/RDNA3: WMMA 16x16x16
  THEROCK_MATRIX_INSTR_MFMA   = 1 << 1,  // CDNA: MFMA family
} TheRockMatrixInstr;

// Memory tier classification.
typedef enum TheRockMemTier {
  THEROCK_MEM_VRAM        = 0,  // Main GDDR/HBM
  THEROCK_MEM_INF_CACHE   = 1,  // On-die cache (RDNA2 Infinity Cache, etc.)
  THEROCK_MEM_LDS         = 2,  // Local data share (per-CU scratchpad)
  THEROCK_MEM_GTT         = 3,  // System RAM mapped for GPU access
  THEROCK_MEM_TIER_COUNT  = 4,
} TheRockMemTier;

// Full hardware descriptor for one GPU.
// Fields are in approximate order of how frequently they're consulted.
typedef struct TheRockHardwareDesc {
  // Identity
  const char *gfx_arch;          // e.g. "gfx1032"
  const char *marketing_name;    // e.g. "RX 6650 XT"

  // Wavefront
  uint32_t wavefront_size;       // 32 or 64
  uint32_t simd_per_cu;          // SIMD units per CU
  uint32_t cu_count;             // Total compute units

  // Matrix instructions
  TheRockMatrixInstr matrix_instrs;   // bitmask of supported types
  uint32_t wmma_m, wmma_n, wmma_k;    // WMMA tile dims (0 if unsupported)

  // Memory hierarchy (sizes in bytes, 0 = tier not present)
  size_t vram_bytes;
  size_t inf_cache_bytes;        // On-die L2/Infinity Cache
  size_t lds_per_cu_bytes;       // LDS per CU
  size_t l1_per_cu_bytes;        // L1 cache per CU

  // Bandwidth (bytes/sec, 0 = unknown)
  uint64_t vram_bandwidth_bps;
  uint64_t inf_cache_bandwidth_bps;

  // Compute (FLOPS at fp16, 0 = unknown)
  uint64_t fp16_flops;

  // Tuning hints
  uint32_t prefetch_lookahead_layers;  // how many layers ahead to prefetch weights
} TheRockHardwareDesc;

// Returns the descriptor for the currently active GPU, or NULL if unrecognized.
// The returned pointer is static — do not free.
const TheRockHardwareDesc *therock_hardware_detect(void);

// Returns a descriptor by gfx arch string, or NULL if unknown.
const TheRockHardwareDesc *therock_hardware_by_arch(const char *gfx_arch);

#ifdef __cplusplus
}
#endif

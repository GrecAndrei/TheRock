// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Memory subsystem: explicit placement across the VRAM / Infinity Cache / GTT
// hierarchy. The allocator is hardware-aware — allocation size and alignment
// are chosen to maximize Infinity Cache residency for hot weight tiles.
//
// Design:
//   - Weights are allocated in pinned arenas sized to the Infinity Cache
//   - Each arena tracks a "generation" counter; the prefetch planner uses this
//     to decide whether a layer's weights are already hot
//   - Activations and KV cache live in separate pools (never compete with weights
//     for Infinity Cache space)
//   - GTT pool for CPU-visible staging (weight loading, quantized model file)

#pragma once

#include "therock/inference/hardware.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Memory tier hints — caller tells the allocator what the memory is for.
// Allocator maps these to actual HIP memory flags based on hardware descriptor.
// ---------------------------------------------------------------------------

typedef enum TheRockAllocHint {
  THEROCK_ALLOC_WEIGHTS,      // Model weights — maximize Infinity Cache residency
  THEROCK_ALLOC_ACTIVATIONS,  // Per-layer activations — short-lived, reuse aggressively
  THEROCK_ALLOC_KV_CACHE,     // KV cache — large, sequential access pattern
  THEROCK_ALLOC_STAGING,      // CPU↔GPU transfer staging — GTT mapped
} TheRockAllocHint;

// ---------------------------------------------------------------------------
// Weight arena: a contiguous VRAM region for one model's weights.
// Sized to fit within Infinity Cache when possible; otherwise spans VRAM
// in cache-line-aligned chunks.
// ---------------------------------------------------------------------------

typedef struct TheRockWeightArena TheRockWeightArena;

// Create an arena for `total_bytes` of model weights.
// `hw` determines alignment and cache geometry.
TheRockWeightArena *therock_arena_create(const TheRockHardwareDesc *hw,
                                          size_t total_bytes);
void                therock_arena_destroy(TheRockWeightArena *arena);

// Suballocate from the arena. Returns GPU pointer, aligned to cache line.
// All allocations are persistent for the arena's lifetime (no free).
void *therock_arena_alloc(TheRockWeightArena *arena, size_t bytes);

// Return the raw GPU base pointer and total size (for prefetch planning).
void  *therock_arena_base(const TheRockWeightArena *arena);
size_t therock_arena_size(const TheRockWeightArena *arena);
size_t therock_arena_used(const TheRockWeightArena *arena);

// ---------------------------------------------------------------------------
// Activation pool: a double-buffered scratchpad for per-layer activations.
// Layer N writes into buffer A while layer N-1 results sit in buffer B,
// then roles swap. Sized to hold one full activation tensor.
// ---------------------------------------------------------------------------

typedef struct TheRockActivationPool TheRockActivationPool;

TheRockActivationPool *therock_actpool_create(const TheRockHardwareDesc *hw,
                                               size_t max_tokens,
                                               size_t hidden_dim);
void                   therock_actpool_destroy(TheRockActivationPool *pool);

// Get the current write buffer (fp16, [max_tokens x hidden_dim]).
void *therock_actpool_write_buf(TheRockActivationPool *pool);
// Get the previous layer's output (read buffer).
void *therock_actpool_read_buf(TheRockActivationPool *pool);
// Swap read/write buffers (call after each layer).
void  therock_actpool_swap(TheRockActivationPool *pool);

// ---------------------------------------------------------------------------
// KV cache: paged allocation, page size tuned to Infinity Cache line geometry.
// Each page holds `page_tokens` KV pairs for all layers (interleaved).
// ---------------------------------------------------------------------------

typedef struct TheRockKVCache TheRockKVCache;

TheRockKVCache *therock_kvcache_create(const TheRockHardwareDesc *hw,
                                        uint32_t num_layers,
                                        uint32_t num_kv_heads,
                                        uint32_t head_dim,
                                        uint32_t max_pages);
void            therock_kvcache_destroy(TheRockKVCache *cache);

// Allocate a new page. Returns page index, or UINT32_MAX if full.
uint32_t therock_kvcache_alloc_page(TheRockKVCache *cache);
void     therock_kvcache_free_page(TheRockKVCache *cache, uint32_t page_idx);

// GPU pointers into the KV cache for a specific layer + page.
void *therock_kvcache_key_ptr(const TheRockKVCache *cache,
                               uint32_t layer, uint32_t page_idx);
void *therock_kvcache_val_ptr(const TheRockKVCache *cache,
                               uint32_t layer, uint32_t page_idx);

// Page size in tokens (hardware-tuned).
uint32_t therock_kvcache_page_tokens(const TheRockKVCache *cache);

#ifdef __cplusplus
}
#endif

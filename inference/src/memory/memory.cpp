// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "therock/inference/memory.h"

#include <hip/hip_runtime.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static size_t align_up(size_t v, size_t align) {
  return (v + align - 1) & ~(align - 1);
}

// Cache line size for Infinity Cache on RDNA2/RDNA3.
// The L2 cache line is 128 bytes; align weight tiles to this so a tile
// doesn't straddle two cache lines and evict its neighbor.
static size_t inf_cache_line(const TheRockHardwareDesc *hw) {
  // All known RDNA2/RDNA3 parts use 128B L2 cache lines.
  // For CDNA (no Infinity Cache) fall back to 64B.
  return (hw->inf_cache_bytes > 0) ? 128 : 64;
}

// ---------------------------------------------------------------------------
// Weight arena
// ---------------------------------------------------------------------------

struct TheRockWeightArena {
  void  *base     = nullptr;
  size_t capacity = 0;
  size_t used     = 0;
  size_t alignment = 128;
};

TheRockWeightArena *therock_arena_create(const TheRockHardwareDesc *hw,
                                          size_t total_bytes) {
  auto *a = new TheRockWeightArena{};
  a->alignment = inf_cache_line(hw);
  a->capacity  = align_up(total_bytes, a->alignment);

  // hipMalloc guarantees page alignment (4KB) which is a superset of 128B.
  hipError_t e = hipMalloc(&a->base, a->capacity);
  if (e != hipSuccess) {
    delete a;
    return nullptr;
  }
  return a;
}

void therock_arena_destroy(TheRockWeightArena *arena) {
  if (arena->base) hipFree(arena->base);
  delete arena;
}

void *therock_arena_alloc(TheRockWeightArena *arena, size_t bytes) {
  size_t off = align_up(arena->used, arena->alignment);
  size_t end = off + bytes;
  if (end > arena->capacity) return nullptr;
  arena->used = end;
  return static_cast<char *>(arena->base) + off;
}

void  *therock_arena_base(const TheRockWeightArena *a) { return a->base; }
size_t therock_arena_size(const TheRockWeightArena *a) { return a->capacity; }
size_t therock_arena_used(const TheRockWeightArena *a) { return a->used; }

// ---------------------------------------------------------------------------
// Activation pool (double-buffered)
// ---------------------------------------------------------------------------

struct TheRockActivationPool {
  void  *bufs[2] = {nullptr, nullptr};
  size_t buf_bytes = 0;
  int    write_idx = 0;  // which buffer is currently being written to
};

TheRockActivationPool *therock_actpool_create(const TheRockHardwareDesc *hw,
                                               size_t max_tokens,
                                               size_t hidden_dim) {
  auto *p = new TheRockActivationPool{};
  // fp16: 2 bytes per element
  p->buf_bytes = align_up(max_tokens * hidden_dim * 2, inf_cache_line(hw));

  for (int i = 0; i < 2; i++) {
    hipError_t e = hipMalloc(&p->bufs[i], p->buf_bytes);
    if (e != hipSuccess) {
      if (p->bufs[0]) hipFree(p->bufs[0]);
      delete p;
      return nullptr;
    }
  }
  return p;
}

void therock_actpool_destroy(TheRockActivationPool *pool) {
  for (int i = 0; i < 2; i++)
    if (pool->bufs[i]) hipFree(pool->bufs[i]);
  delete pool;
}

void *therock_actpool_write_buf(TheRockActivationPool *pool) {
  return pool->bufs[pool->write_idx];
}

void *therock_actpool_read_buf(TheRockActivationPool *pool) {
  return pool->bufs[pool->write_idx ^ 1];
}

void therock_actpool_swap(TheRockActivationPool *pool) {
  pool->write_idx ^= 1;
}

// ---------------------------------------------------------------------------
// KV cache (paged)
//
// Layout: one contiguous VRAM allocation.
// Indexed as: [page][layer][head][token_within_page][head_dim]
// This layout means a full page for one layer is contiguous — the attention
// kernel scans it sequentially, maximizing Infinity Cache line utilization.
// ---------------------------------------------------------------------------

struct TheRockKVCache {
  void    *base        = nullptr;
  size_t   total_bytes = 0;

  uint32_t num_layers   = 0;
  uint32_t num_kv_heads = 0;
  uint32_t head_dim     = 0;
  uint32_t page_tokens  = 0;  // tokens per page (hardware-tuned)
  uint32_t max_pages    = 0;

  // Bytes for one key or value tensor within a page for one layer:
  // page_tokens * num_kv_heads * head_dim * sizeof(fp16)
  size_t kv_layer_page_bytes = 0;

  // Free page bitmap (simple: one uint8 per page, 1=free 0=used)
  std::vector<uint8_t> free_map;
};

// Choose page size: fit an integer number of cache lines.
// Larger pages = better sequential bandwidth; smaller = less fragmentation.
// Target: 16 tokens per page on gfx1032 (matches a cache line's worth for
// typical head_dim=128 with 2-byte elements: 128*2=256B = 2 cache lines).
static uint32_t choose_page_tokens(const TheRockHardwareDesc *hw,
                                    uint32_t num_kv_heads, uint32_t head_dim) {
  size_t line = inf_cache_line(hw);
  // One token of KV for all heads: num_kv_heads * head_dim * 2 bytes
  size_t token_bytes = (size_t)num_kv_heads * head_dim * 2;
  // Round up to cache line, then target ~4KB per page (32 cache lines)
  size_t target_page_bytes = 32 * line;
  uint32_t tokens = (uint32_t)(target_page_bytes / token_bytes);
  return tokens < 1 ? 1 : tokens;
}

TheRockKVCache *therock_kvcache_create(const TheRockHardwareDesc *hw,
                                        uint32_t num_layers,
                                        uint32_t num_kv_heads,
                                        uint32_t head_dim,
                                        uint32_t max_pages) {
  auto *c = new TheRockKVCache{};
  c->num_layers   = num_layers;
  c->num_kv_heads = num_kv_heads;
  c->head_dim     = head_dim;
  c->max_pages    = max_pages;
  c->page_tokens  = choose_page_tokens(hw, num_kv_heads, head_dim);

  // K and V each: page_tokens * num_kv_heads * head_dim * fp16
  c->kv_layer_page_bytes = (size_t)c->page_tokens * num_kv_heads * head_dim * 2;

  // Total: max_pages * num_layers * 2 (K+V) * kv_layer_page_bytes
  c->total_bytes = (size_t)max_pages * num_layers * 2 * c->kv_layer_page_bytes;

  hipError_t e = hipMalloc(&c->base, c->total_bytes);
  if (e != hipSuccess) { delete c; return nullptr; }

  c->free_map.assign(max_pages, 1);
  return c;
}

void therock_kvcache_destroy(TheRockKVCache *cache) {
  if (cache->base) hipFree(cache->base);
  delete cache;
}

uint32_t therock_kvcache_alloc_page(TheRockKVCache *cache) {
  for (uint32_t i = 0; i < cache->max_pages; i++) {
    if (cache->free_map[i]) {
      cache->free_map[i] = 0;
      return i;
    }
  }
  return UINT32_MAX;
}

void therock_kvcache_free_page(TheRockKVCache *cache, uint32_t page_idx) {
  if (page_idx < cache->max_pages)
    cache->free_map[page_idx] = 1;
}

// Offset for key of (layer, page): page * (num_layers * 2 * kv_layer_page_bytes)
//                                  + layer * (2 * kv_layer_page_bytes)
//                                  + 0  (key comes first)
void *therock_kvcache_key_ptr(const TheRockKVCache *c,
                               uint32_t layer, uint32_t page_idx) {
  size_t off = ((size_t)page_idx * c->num_layers * 2 + layer * 2 + 0)
               * c->kv_layer_page_bytes;
  return static_cast<char *>(c->base) + off;
}

// Value immediately follows key in the same page/layer slot.
void *therock_kvcache_val_ptr(const TheRockKVCache *c,
                               uint32_t layer, uint32_t page_idx) {
  size_t off = ((size_t)page_idx * c->num_layers * 2 + layer * 2 + 1)
               * c->kv_layer_page_bytes;
  return static_cast<char *>(c->base) + off;
}

uint32_t therock_kvcache_page_tokens(const TheRockKVCache *c) {
  return c->page_tokens;
}

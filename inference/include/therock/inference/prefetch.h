// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Prefetch planner: async weight prefetch from VRAM into Infinity Cache.
//
// Strategy:
//   - Hardware descriptor says how many layers to look ahead
//   - Planner issues hipMemPrefetchAsync for the next N layers' weight arenas
//     while the current layer is computing
//   - On hardware with no Infinity Cache (CDNA), this becomes a no-op since
//     hipMemPrefetchAsync has no explicit Infinity Cache target anyway
//
// The prefetch stream runs parallel to the compute stream. Both sync at
// the executor's layer boundary via an event.

#pragma once

#include "therock/inference/hardware.h"
#include "therock/inference/memory.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TheRockPrefetcher TheRockPrefetcher;

// Layer weight descriptor: base pointer + size for a single layer's weights.
typedef struct TheRockLayerWeights {
  void   *base;
  size_t  bytes;
} TheRockLayerWeights;

TheRockPrefetcher *therock_prefetcher_create(const TheRockHardwareDesc *hw,
                                              uint32_t num_layers,
                                              const TheRockLayerWeights *layers);
void               therock_prefetcher_destroy(TheRockPrefetcher *p);

// Called before executing layer `current_layer`.
// Issues async prefetch for layers [current+1 .. current+lookahead].
// Returns immediately — compute stream can proceed in parallel.
void therock_prefetcher_advance(TheRockPrefetcher *p, uint32_t current_layer,
                                 void *compute_stream);

// Ensure prefetch for layer `layer` has completed before compute uses it.
// Inserts a stream dependency (event wait) — does not block the CPU.
void therock_prefetcher_ensure_ready(TheRockPrefetcher *p, uint32_t layer,
                                      void *compute_stream);

#ifdef __cplusplus
}
#endif

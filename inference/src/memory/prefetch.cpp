// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "therock/inference/prefetch.h"

#include <hip/hip_runtime.h>
#include <vector>
#include <cstring>

struct LayerState {
  hipEvent_t ready_event = nullptr;  // signals when prefetch for this layer is done
  bool       prefetched  = false;
};

struct TheRockPrefetcher {
  const TheRockHardwareDesc    *hw          = nullptr;
  uint32_t                      num_layers  = 0;
  uint32_t                      lookahead   = 0;
  hipStream_t                   stream      = nullptr;  // dedicated prefetch stream
  std::vector<TheRockLayerWeights> layers;
  std::vector<LayerState>          state;
};

TheRockPrefetcher *therock_prefetcher_create(const TheRockHardwareDesc *hw,
                                              uint32_t num_layers,
                                              const TheRockLayerWeights *layers) {
  auto *p = new TheRockPrefetcher{};
  p->hw         = hw;
  p->num_layers = num_layers;
  p->lookahead  = hw->prefetch_lookahead_layers;

  hipStreamCreate(&p->stream);

  p->layers.assign(layers, layers + num_layers);
  p->state.resize(num_layers);
  for (uint32_t i = 0; i < num_layers; i++)
    hipEventCreateWithFlags(&p->state[i].ready_event, hipEventDisableTiming);

  return p;
}

void therock_prefetcher_destroy(TheRockPrefetcher *p) {
  for (auto &s : p->state)
    if (s.ready_event) hipEventDestroy(s.ready_event);
  if (p->stream) hipStreamDestroy(p->stream);
  delete p;
}

void therock_prefetcher_advance(TheRockPrefetcher *p, uint32_t current_layer,
                                 void *compute_stream) {
  // No Infinity Cache → prefetch has no special target, skip
  if (p->hw->inf_cache_bytes == 0) return;

  uint32_t end = current_layer + 1 + p->lookahead;
  if (end > p->num_layers) end = p->num_layers;

  for (uint32_t l = current_layer + 1; l < end; l++) {
    if (p->state[l].prefetched) continue;

    const TheRockLayerWeights &lw = p->layers[l];
    if (!lw.base || lw.bytes == 0) continue;

    // hipMemPrefetchAsync hints the driver to bring pages into the GPU's
    // page table (and effectively L2) ahead of first use.
    hipMemPrefetchAsync(lw.base, lw.bytes, 0 /* device 0 */, p->stream);
    hipEventRecord(p->state[l].ready_event, p->stream);
    p->state[l].prefetched = true;
  }
}

void therock_prefetcher_ensure_ready(TheRockPrefetcher *p, uint32_t layer,
                                      void *compute_stream) {
  if (p->hw->inf_cache_bytes == 0) return;
  if (!p->state[layer].prefetched) return;

  // Make the compute stream wait for the prefetch event — zero CPU stall.
  hipStreamWaitEvent(static_cast<hipStream_t>(compute_stream),
                     p->state[layer].ready_event, 0);
}

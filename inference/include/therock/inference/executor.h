// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Graph executor: drives the transformer forward pass.
// Owns the compute stream, coordinates with the prefetcher, dispatches
// ops through the kernel registry.
//
// One executor per model instance. Thread-safe for reads; external
// serialization required for concurrent forward passes (use separate
// executors per thread/batch slot).

#pragma once

#include "therock/inference/hardware.h"
#include "therock/inference/kernel_registry.h"
#include "therock/inference/memory.h"
#include "therock/inference/prefetch.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Model descriptor: what the executor needs to know about a loaded model.
// Filled in by the model loader; executor treats this as read-only.
// ---------------------------------------------------------------------------

typedef struct TheRockModelDesc {
  const char *name;           // e.g. "gemma-4-12b-it-qat-q4_0"
  uint32_t    num_layers;
  uint32_t    hidden_dim;
  uint32_t    num_heads;
  uint32_t    num_kv_heads;   // < num_heads for GQA (Gemma 4 style)
  uint32_t    head_dim;
  uint32_t    intermediate_dim;  // MLP gate/up projection output dim
  uint32_t    vocab_size;

  TheRockWeightLayout weight_layout;

  // Per-layer weight pointers (length = num_layers)
  // Each sub-array is a layer's weights split by projection type.
  // Layout within each: [N x K] in whatever weight_layout specifies.
  const void **attn_q;   // [num_layers] query projection weights
  const void **attn_k;   // [num_layers] key projection weights
  const void **attn_v;   // [num_layers] value projection weights
  const void **attn_o;   // [num_layers] output projection weights
  const void **mlp_gate; // [num_layers] MLP gate projection weights
  const void **mlp_up;   // [num_layers] MLP up projection weights
  const void **mlp_down; // [num_layers] MLP down projection weights

  // RMSNorm scales per layer (fp32, length = hidden_dim each)
  const float **norm_attn; // [num_layers]
  const float **norm_mlp;  // [num_layers]
  const float  *norm_final;

  // Token embedding table (fp16, [vocab_size x hidden_dim])
  const void *embed_tokens;

  // Per-layer weight arenas for prefetch planning
  const TheRockLayerWeights *layer_weights;  // [num_layers]
} TheRockModelDesc;

// ---------------------------------------------------------------------------
// Inference request
// ---------------------------------------------------------------------------

typedef struct TheRockRequest {
  const uint32_t *token_ids;   // input token sequence
  uint32_t        num_tokens;  // sequence length (1 for decode, >1 for prefill)
  uint32_t        max_new_tokens;

  // KV cache page table for this sequence (filled by caller, updated in place)
  uint32_t       *kv_pages;    // [num_layers] page indices, UINT32_MAX = unallocated
} TheRockRequest;

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

typedef struct TheRockExecutor TheRockExecutor;

TheRockExecutor *therock_executor_create(const TheRockHardwareDesc  *hw,
                                          const TheRockModelDesc     *model,
                                          TheRockKernelRegistry      *registry,
                                          TheRockKVCache             *kv_cache);
void             therock_executor_destroy(TheRockExecutor *ex);

// Run one forward pass (prefill or decode based on request->num_tokens).
// Returns logits for the last token (fp16, [vocab_size]).
// Blocks until the forward pass is complete.
int therock_executor_forward(TheRockExecutor        *ex,
                              const TheRockRequest   *req,
                              void                   *logits_out);

#ifdef __cplusplus
}
#endif

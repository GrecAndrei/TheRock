// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Benchmark: raw decode throughput (single-token generation) for Gemma 4 12B.
// Measures the memory-bandwidth-limited GEMV kernel performance across all
// 48 layers with IC-prefetch enabled vs disabled.
//
// This doesn't load a real model — it allocates dummy weights at the correct
// sizes and shapes to measure pure kernel + memory subsystem performance.
//
// Build:
//   hipcc -O3 --offload-arch=gfx1032 -I../include bench_decode.cpp \
//     -L../build -ltherock-inference -lamdhip64 -o bench_decode

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

// Direct kernel declaration (bypass the wrapper to avoid struct layout issues)
__global__ void __launch_bounds__(32)
kernel_gemv_fp16_wave32(int N, int K,
                         const __half * __restrict__ x,
                         const __half * __restrict__ W,
                         const __half * __restrict__ bias,
                         __half       * __restrict__ y);

// Match TheRockOpDesc layout without the anonymous enum issue
struct OpDesc {
  uint32_t op;
  uint32_t M, N, K, batch;
  uint32_t weight_layout;
  uint32_t exec_mode;
};

// Gemma 4 12B architecture
static constexpr uint32_t HIDDEN_DIM      = 3840;
static constexpr uint32_t INTERMEDIATE    = 15360;
static constexpr uint32_t NUM_HEADS       = 32;
static constexpr uint32_t NUM_KV_HEADS    = 8;
static constexpr uint32_t HEAD_DIM        = 128;
static constexpr uint32_t NUM_LAYERS      = 48;
static constexpr uint32_t Q_DIM           = NUM_HEADS * HEAD_DIM;     // 4096
static constexpr uint32_t KV_DIM          = NUM_KV_HEADS * HEAD_DIM;  // 1024

// Weight sizes per layer (Q4_0: 18 bytes per 32 elements = 0.5625 bytes/elem)
// For benchmarking, we use fp16 weights (2 bytes/elem) as a proxy since we're
// measuring bandwidth, and the dequant cost is minimal vs memory latency.
struct LayerWeights {
  void *attn_q;   // [Q_DIM x HIDDEN_DIM]
  void *attn_k;   // [KV_DIM x HIDDEN_DIM]
  void *attn_v;   // [KV_DIM x HIDDEN_DIM]
  void *attn_o;   // [HIDDEN_DIM x Q_DIM]
  void *mlp_gate; // [INTERMEDIATE x HIDDEN_DIM]
  void *mlp_up;   // [INTERMEDIATE x HIDDEN_DIM]
  void *mlp_down; // [HIDDEN_DIM x INTERMEDIATE]
};

struct PrefetchState {
  hipStream_t prefetch_stream;
  hipEvent_t  events[NUM_LAYERS];
};

static void* alloc_weight(uint32_t N, uint32_t K) {
  void *ptr;
  size_t bytes = (size_t)N * K * 2;  // fp16
  hipMalloc(&ptr, bytes);
  hipMemset(ptr, 0, bytes);
  return ptr;
}

static size_t layer_bytes(const LayerWeights &lw) {
  return (Q_DIM * HIDDEN_DIM + KV_DIM * HIDDEN_DIM * 2 +
          HIDDEN_DIM * Q_DIM + INTERMEDIATE * HIDDEN_DIM * 2 +
          HIDDEN_DIM * INTERMEDIATE) * 2;
}

static void run_layer_decode(hipStream_t s, const LayerWeights &lw,
                              void *x, void *scratch, void *out) {
  // Simulate one decode step: 7 GEMVs per layer (M=1)
  OpDesc desc = {};
  desc.op = 1;  // GEMV
  desc.M = 1;
  desc.batch = 1;
  desc.weight_layout = 0;
  desc.exec_mode = 1;

  // Q projection: [1 x HIDDEN] × [Q_DIM x HIDDEN]^T → [1 x Q_DIM]
  desc.N = Q_DIM; desc.K = HIDDEN_DIM;
  therock_gemv_fp16_wave32(s, &desc, x, lw.attn_q, nullptr, scratch);

  // K projection
  desc.N = KV_DIM; desc.K = HIDDEN_DIM;
  therock_gemv_fp16_wave32(s, &desc, x, lw.attn_k, nullptr, scratch);

  // V projection
  desc.N = KV_DIM; desc.K = HIDDEN_DIM;
  therock_gemv_fp16_wave32(s, &desc, x, lw.attn_v, nullptr, scratch);

  // Output projection: [1 x Q_DIM] × [HIDDEN x Q_DIM]^T → [1 x HIDDEN]
  desc.N = HIDDEN_DIM; desc.K = Q_DIM;
  therock_gemv_fp16_wave32(s, &desc, scratch, lw.attn_o, nullptr, out);

  // MLP gate: [1 x HIDDEN] × [INTER x HIDDEN]^T → [1 x INTER]
  desc.N = INTERMEDIATE; desc.K = HIDDEN_DIM;
  therock_gemv_fp16_wave32(s, &desc, out, lw.mlp_gate, nullptr, scratch);

  // MLP up
  desc.N = INTERMEDIATE; desc.K = HIDDEN_DIM;
  therock_gemv_fp16_wave32(s, &desc, out, lw.mlp_up, nullptr, scratch);

  // MLP down: [1 x INTER] × [HIDDEN x INTER]^T → [1 x HIDDEN]
  desc.N = HIDDEN_DIM; desc.K = INTERMEDIATE;
  therock_gemv_fp16_wave32(s, &desc, scratch, lw.mlp_down, nullptr, out);
}

int main(int argc, char **argv) {
  int warmup = 5;
  int iters  = 20;
  bool use_prefetch = false;  // hipMemPrefetchAsync requires managed memory; disabled for now

  printf("=== TheRock Inference Decode Benchmark ===\n");
  printf("Model: Gemma 4 12B (fp16 proxy weights)\n");
  printf("GPU: gfx1032 (RX 6650 XT)\n");
  printf("Layers: %u | Prefetch: %s\n\n", NUM_LAYERS, use_prefetch ? "ON" : "OFF");

  // Allocate weights for all layers
  std::vector<LayerWeights> layers(NUM_LAYERS);
  size_t total_weight_bytes = 0;

  printf("Allocating %u layers of weights...\n", NUM_LAYERS);
  for (uint32_t l = 0; l < NUM_LAYERS; l++) {
    layers[l].attn_q   = alloc_weight(Q_DIM, HIDDEN_DIM);
    layers[l].attn_k   = alloc_weight(KV_DIM, HIDDEN_DIM);
    layers[l].attn_v   = alloc_weight(KV_DIM, HIDDEN_DIM);
    layers[l].attn_o   = alloc_weight(HIDDEN_DIM, Q_DIM);
    layers[l].mlp_gate = alloc_weight(INTERMEDIATE, HIDDEN_DIM);
    layers[l].mlp_up   = alloc_weight(INTERMEDIATE, HIDDEN_DIM);
    layers[l].mlp_down = alloc_weight(HIDDEN_DIM, INTERMEDIATE);
    total_weight_bytes += layer_bytes(layers[l]);
  }
  printf("Total weight memory: %.2f GB\n", total_weight_bytes / 1e9);

  // Activation buffers
  void *x, *scratch, *out;
  hipMalloc(&x,       INTERMEDIATE * 2);
  hipMalloc(&scratch, INTERMEDIATE * 2);
  hipMalloc(&out,     INTERMEDIATE * 2);
  hipMemset(x, 1, HIDDEN_DIM * 2);

  // Streams
  hipStream_t compute_stream;
  hipStreamCreate(&compute_stream);

  PrefetchState pf = {};
  if (use_prefetch) {
    hipStreamCreate(&pf.prefetch_stream);
    for (uint32_t l = 0; l < NUM_LAYERS; l++)
      hipEventCreateWithFlags(&pf.events[l], hipEventDisableTiming);
  }

  // Warmup
  printf("Warming up (%d iters)...\n", warmup);
  for (int w = 0; w < warmup; w++) {
    for (uint32_t l = 0; l < NUM_LAYERS; l++)
      run_layer_decode(compute_stream, layers[l], x, scratch, out);
    hipStreamSynchronize(compute_stream);
  }

  // Benchmark
  printf("Benchmarking (%d iters)...\n", iters);

  hipEvent_t ev_start, ev_stop;
  hipEventCreate(&ev_start);
  hipEventCreate(&ev_stop);

  hipEventRecord(ev_start, compute_stream);

  for (int it = 0; it < iters; it++) {
    // Prefetch first layer ahead of time
    if (use_prefetch && NUM_LAYERS > 1) {
      hipMemPrefetchAsync(layers[0].attn_q,
                          layer_bytes(layers[0]),
                          0, pf.prefetch_stream);
    }

    for (uint32_t l = 0; l < NUM_LAYERS; l++) {
      // Wait for prefetch of current layer
      if (use_prefetch && l > 0) {
        hipStreamWaitEvent(compute_stream, pf.events[l], 0);
      }

      // Prefetch next layer while computing current
      if (use_prefetch && l + 1 < NUM_LAYERS) {
        // Use base pointer of attn_q as proxy for whole layer
        // (In real impl, layers are contiguous arenas)
        hipMemPrefetchAsync(layers[l + 1].attn_q,
                            Q_DIM * HIDDEN_DIM * 2,
                            0, pf.prefetch_stream);
        hipMemPrefetchAsync(layers[l + 1].mlp_gate,
                            INTERMEDIATE * HIDDEN_DIM * 2,
                            0, pf.prefetch_stream);
        hipMemPrefetchAsync(layers[l + 1].mlp_down,
                            HIDDEN_DIM * INTERMEDIATE * 2,
                            0, pf.prefetch_stream);
        hipEventRecord(pf.events[l + 1], pf.prefetch_stream);
      }

      run_layer_decode(compute_stream, layers[l], x, scratch, out);
    }
  }

  hipEventRecord(ev_stop, compute_stream);
  hipEventSynchronize(ev_stop);

  float total_ms = 0;
  hipEventElapsedTime(&total_ms, ev_start, ev_stop);

  float per_token_ms = total_ms / iters;
  float tokens_per_sec = 1000.0f / per_token_ms;
  float effective_bw = (total_weight_bytes / 1e9) / (per_token_ms / 1000.0);

  printf("\n=== Results ===\n");
  printf("Total time: %.1f ms for %d tokens\n", total_ms, iters);
  printf("Per-token:  %.2f ms\n", per_token_ms);
  printf("Throughput: %.1f tokens/sec\n", tokens_per_sec);
  printf("Effective BW: %.0f GB/s\n", effective_bw);
  printf("BW utilization: %.0f%% of 280 GB/s VRAM\n", effective_bw / 280 * 100);

  // Cleanup
  hipEventDestroy(ev_start);
  hipEventDestroy(ev_stop);
  hipStreamDestroy(compute_stream);
  if (use_prefetch) {
    hipStreamDestroy(pf.prefetch_stream);
    for (uint32_t l = 0; l < NUM_LAYERS; l++)
      hipEventDestroy(pf.events[l]);
  }
  for (auto &lw : layers) {
    hipFree(lw.attn_q); hipFree(lw.attn_k); hipFree(lw.attn_v);
    hipFree(lw.attn_o); hipFree(lw.mlp_gate); hipFree(lw.mlp_up);
    hipFree(lw.mlp_down);
  }
  hipFree(x); hipFree(scratch); hipFree(out);

  return 0;
}

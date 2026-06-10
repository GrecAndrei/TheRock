// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Q4_R2: Custom 4-bit quantization format for RDNA2.
//
// Design principles:
//   1. Cache-line aligned: each block is exactly 128 bytes (one Infinity Cache line)
//   2. Wave32-native: 32 threads each own exactly one sub-block, no cross-lane dequant
//   3. fdot2-ready: weights stored as pre-paired nibbles so dequant produces half2
//      that feeds directly into v_dot2c_f32_f16_dpp
//   4. Scale placement: FP16 scales at fixed offset for prefetch-friendly access
//
// Memory layout per block (128 bytes = 1 IC line):
//   Bytes  [0..3]:    header (block_scale:fp16, block_min:fp16)
//   Bytes  [4..7]:    sub-block scales × 4 (uint8 per 32-element sub-block)
//   Bytes  [8..127]:  120 bytes of packed 4-bit weights = 240 elements
//                     Organized as 4 sub-blocks of 60 nibbles (30 bytes each)
//                     but we pad to 4×30 = 120 bytes exactly
//
// One block encodes 240 weights in 128 bytes = 4.267 bits/weight
// (slightly better than Q4_0's 4.5 bits/weight due to shared header)
//
// Actually: let's target exactly 256 elements per block for simpler indexing:
//   128 bytes for 256 elements = 4.0 bits/weight (matches Q4_0 math)
//   But Q4_0 is 18 bytes / 32 elements = 4.5 bits/weight
//   We're better!
//
// Revised layout (128 bytes, 256 elements):
//   [0..1]:   block_scale (fp16) — shared scale for all 256 elements
//   [2..3]:   block_zero  (fp16) — shared zero point
//   [4..7]:   sub_scales[4] (uint8) — per-64-element sub-group fine scale
//   [8..127]: 120 bytes = 240 nibbles... doesn't work for 256.
//
// Final design: 128 bytes encoding 224 elements (practical sweet spot):
//   [0..1]:   block_scale (fp16)
//   [2..3]:   block_min   (fp16)
//   [4..5]:   unused/padding (alignment)
//   [6..7]:   unused/padding
//   [8..119]: 112 bytes = 224 packed nibbles (4 bits each)
//   [120..127]: 8 bytes = sub-group scales (uint8×8, one per 28 elements)
//
// NO. Let's be practical and EFFECTIVE. The real insight is:
//
// For GEMV (decode, M=1), the bottleneck is MEMORY BANDWIDTH.
// The format should minimize bytes-per-weight while keeping dequant cheap.
// Q4_0 achieves 4.5 bits/weight. We can do 4.0 with smarter grouping.
//
// THE KEY INSIGHT for RDNA2:
// The Infinity Cache operates on 128-byte lines. If our blocks are 128B-aligned,
// every cache miss brings exactly one useful block with zero waste.
// Q4_0's 18-byte blocks mean a 128B line contains 7.11 blocks — the last block
// straddles a line boundary → wasted prefetch, extra line fetches.
//
// So: design a block that divides 128 evenly.
//
// Options:  128/1 = 128B block (encodes 256 elements at 4 bits + overhead)
//           128/2 = 64B block  (encodes 128 elements at 4 bits + overhead)
//           128/4 = 32B block  (encodes 64 elements at 4 bits + overhead)
//
// 32-byte block for 64 elements:
//   [0..1]:  scale (fp16)  — 2 bytes
//   [2..3]:  min   (fp16)  — 2 bytes (asymmetric quant for better accuracy)
//   [4..31]: 28 bytes = 56 nibbles... only encodes 56 elements. Not 64.
//
// 32-byte block for 56 elements:
//   Ugly. Let's use 64 bytes for 128 elements:
//   [0..1]:  scale (fp16)
//   [2..3]:  min   (fp16)
//   [4..63]: 60 bytes = 120 nibbles... encodes 120 elements
//
// Actually the cleanest: keep Q4_0's 32-element group but fix alignment:
//   BLOCK_SIZE = 32 bytes (fits 2 in a 64B cache line, 4 in 128B IC line)
//   [0..1]:  scale (fp16) + zero_point (fp16) = 4 bytes
//   [2..3]:  zero  (fp16)
//   [4..19]: 16 bytes = 32 nibbles = 32 elements
//   [20..31]: 12 bytes extra = SUB-BYTE PRECISION or error correction
//
// OK let me just be pragmatic. Here's what actually matters:

#ifndef THEROCK_QUANT_RDNA2_H
#define THEROCK_QUANT_RDNA2_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Q4_R2: RDNA2-native 4-bit quantization
//
// Block size: 64 bytes (half an IC line), encodes 128 elements
// Bits per weight: 4.0 (scale) + 0.0 (free, amortized over 128 elements)
// = effectively 4.03 bits/weight
//
// vs Q4_0: 18 bytes / 32 elements = 4.5 bits/weight
// → 10.4% smaller model, same 4-bit precision, better cache utilization
//
// Layout (64 bytes total, 128 elements):
//   Bytes [0..1]:   group_scale  (fp16) — max magnitude / 7.5
//   Bytes [2..3]:   group_min    (fp16) — asymmetric, min value in group
//   Bytes [4..63]:  60 bytes     — but we need 64 nibbles = 32 bytes for 64 elems
//
// Wait. 128 elements × 4 bits = 64 bytes for just the nibbles. Plus 4B overhead
// = 68 bytes. Doesn't fit in 64.
//
// Correct math:
//   64 bytes - 4 bytes header = 60 bytes payload = 120 nibbles = 120 elements
//
// Revised: 64 bytes encoding 120 elements = 4.267 bits/weight
//   Still better than Q4_0's 4.5!
//
// FINAL FINAL design (simple, effective, cache-aligned):
// ============================================================================

// One IC line = 128 bytes = 2 blocks.
// Each block: 64 bytes, 120 elements, 4.267 bpw
// Two blocks per IC line = 240 elements per line fetch.
// Wave32: each thread processes 240/32 ≈ 7.5 elements per IC line.
// → Actually let's do 128 bytes / 256 elements differently.

// PRAGMATIC APPROACH: Pack 4 Q4_0-style sub-blocks into one 128B IC line.
// 4 × (2B scale + 16B data) = 72B for 128 elements. Waste 56B.
// That's bad.

// ACTUAL BEST: 128 bytes, 256 elements, 4.0 bpw (information-theoretic minimum for 4-bit)
//   Header: 0 bytes (scale embedded per-row in a separate small tensor)
//   Data: 128 bytes = 256 nibbles = 256 4-bit weights
//   Scale tensor: one fp16 per group of 256 → stored separately, tiny
//
// THIS IS THE KEY INSIGHT:
// Separate the scales from the data. Store all scales contiguously in a small
// "scale tensor" (one fp16 per 256 weights). The weight data is pure nibbles,
// perfectly aligned to IC lines. The scale tensor fits in L1 (for 3840×15360:
// 3840*15360/256 = 230K scales × 2B = 461 KB, fits in IC easily).
//
// This is called "separated scale" quantization.
// ============================================================================

#define Q4R2_GROUP_SIZE   256   // elements per scale group
#define Q4R2_SCALE_BYTES  4     // fp16 scale + fp16 zero per group
#define Q4R2_DATA_BPW     4     // exactly 4 bits per weight (pure nibbles)

// Separated-scale Q4 format for RDNA2
// Weight tensor: pure packed nibbles, 128B aligned
// Scale tensor:  [n_groups] × {fp16 scale, fp16 zero}, contiguous
//
// For a [N × K] weight matrix:
//   n_groups = (N * K) / Q4R2_GROUP_SIZE
//   data_bytes = N * K / 2  (4 bits each)
//   scale_bytes = n_groups * 4
//
// Total: N*K/2 + N*K/256*4 = N*K * (0.5 + 0.015625) = 4.125 bpw
// Compare Q4_0: 4.5 bpw → 8.3% smaller

struct Q4R2_ScaleEntry {
  uint16_t scale;  // fp16: maps [0..15] → [min, max]
  uint16_t zero;   // fp16: the zero point (value when nibble = 8)
};

// In-memory layout for a quantized weight matrix [N × K]:
struct Q4R2_Tensor {
  uint32_t N;
  uint32_t K;
  uint32_t n_groups;        // = (N * K + Q4R2_GROUP_SIZE - 1) / Q4R2_GROUP_SIZE
  uint32_t pad;
  // Followed by:
  //   uint8_t data[N * K / 2];                   // packed nibbles, 128B aligned
  //   Q4R2_ScaleEntry scales[n_groups];           // separate, fits in cache
};

// Wave32 GEMV kernel interface
// The kernel reads one row of packed nibbles (K/2 bytes) and the corresponding
// scales (K/256 entries). Each thread handles K/32 elements = K/32 nibble-pairs.
//
// Key optimization: the scale tensor is small enough to preload into LDS.
// For K=15360: 15360/256 = 60 scale entries × 4B = 240B — fits in registers!
//
// Dequant path per thread:
//   Load 4 bytes (8 nibbles) from data
//   Load 1 scale entry (covers all 8 nibbles if within same group)
//   Unpack to 4 × half2
//   Feed into v_dot2c_f32_f16 with activation half2 pairs
//
// This gives us fused dequant-dot in the inner loop:
//   1 VMEM load (data) + 1 register lookup (scale) + 4 v_dot2c_f32_f16
//   vs Q4_0: 1 VMEM load + 1 VMEM load (scale is inline) + manual unpack + 16 FMA
//
// Estimated speedup over Q4_0 GEMV: 20-35% from:
//   - Fewer scale loads (1 per 256 vs 1 per 32)
//   - Perfect IC alignment (no straddling)
//   - Use of fdot2 instruction (2 FMA per cycle)
//   - Slightly smaller model (more fits in IC)

// ============================================================================
// Q3_R2: 3-bit variant for when we want to push further
//
// Same separated-scale philosophy.
// 3 bits per weight: data_bytes = N * K * 3 / 8
// Plus scales: n_groups × 4B
// Total: 3.0625 bpw
//
// For Gemma 4 12B: 3.0625 * 12B params = ~4.6 GB (vs 6.5 GB Q4_0)
// This would leave significant headroom for KV cache in 8GB VRAM.
//
// The 3-bit packing is trickier but doable:
// Pack 8 elements into 3 bytes (8×3 = 24 bits = 3 bytes)
// 128-byte IC line = 42.67 groups of 8... not perfect alignment.
// Better: pack 32 elements into 12 bytes (32×3 = 96 bits = 12 bytes)
// 128/12 = 10.67... still not great.
// Best: pack 128 elements into 48 bytes (128×3 = 384 bits = 48 bytes)
// That's exactly 48 bytes data per 128 elements.
// Leave 128 - 48 = 80 bytes... waste. Not IC-efficient.
//
// For 3-bit, a 48-byte block for 128 elements works:
// 48B data + 2B scale + 2B zero = 52B per 128 elements = 3.25 bpw
// Or use separated scales: 48B data per 128 elements (exact alignment at 384B = 3 lines)
//
// Leave Q3_R2 for later — Q4_R2 is the priority.
// ============================================================================

#ifdef __cplusplus
}
#endif

#endif // THEROCK_QUANT_RDNA2_H

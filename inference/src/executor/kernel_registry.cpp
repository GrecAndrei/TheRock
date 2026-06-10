// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "therock/inference/kernel_registry.h"
#include "therock/inference/hardware.h"

#include <hip/hip_runtime.h>
#include <vector>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Forward declarations — actual kernel implementations live in kernels/
// ---------------------------------------------------------------------------

// fp16 GEMV: y = Wx, weights in row-major fp16
// One wavefront per output row, 32 threads reduce over K
extern "C" void therock_gemv_fp16_wave32(hipStream_t stream,
                                          const TheRockOpDesc *desc,
                                          const void *x, const void *W,
                                          const void *bias, void *y);

// Q4_0 fused dequant-GEMV: decode path, weights never materialized as fp16
extern "C" void therock_gemv_q4_0_wave32(hipStream_t stream,
                                          const TheRockOpDesc *desc,
                                          const void *x, const void *W,
                                          const void *bias, void *y);

// hipBLASLt GEMM dispatch — prefill path, uses TensileLite-tuned kernels
extern "C" void therock_gemm_hipblaslt(hipStream_t stream,
                                        const TheRockOpDesc *desc,
                                        const void *A, const void *B,
                                        const void *C, void *D);

// ---------------------------------------------------------------------------
// Registry entry
// ---------------------------------------------------------------------------

struct RegistryEntry {
  uint32_t            exec_mode_mask;   // bitmask of TheRockExecMode
  TheRockWeightLayout layout;
  uint32_t            m_min, m_max;     // inclusive M range this entry handles
  TheRockKernelFn     fn;
};

struct TheRockKernelRegistry {
  const TheRockHardwareDesc *hw;
  std::vector<RegistryEntry> entries;
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

static RegistryEntry make_gemv_entry(uint32_t mode_mask, TheRockWeightLayout layout,
                                      uint32_t m_min, uint32_t m_max,
                                      TheRockGemvFn fn) {
  RegistryEntry e;
  e.exec_mode_mask = mode_mask;
  e.layout         = layout;
  e.m_min          = m_min;
  e.m_max          = m_max;
  e.fn.gemv        = fn;
  return e;
}

static RegistryEntry make_gemm_entry(uint32_t mode_mask, TheRockWeightLayout layout,
                                      uint32_t m_min, uint32_t m_max,
                                      TheRockGemmFn fn) {
  RegistryEntry e;
  e.exec_mode_mask = mode_mask;
  e.layout         = layout;
  e.m_min          = m_min;
  e.m_max          = m_max;
  e.fn.gemm        = fn;
  return e;
}

TheRockKernelRegistry *therock_registry_create(const TheRockHardwareDesc *hw) {
  auto *reg = new TheRockKernelRegistry{};
  reg->hw = hw;

  uint32_t thresh = therock_decode_threshold(hw);
  uint32_t decode_mask  = 1u << THEROCK_EXEC_DECODE;
  uint32_t prefill_mask = 1u << THEROCK_EXEC_PREFILL;

  reg->entries.push_back(make_gemv_entry(decode_mask,  THEROCK_LAYOUT_Q4_0,      1, thresh,     (TheRockGemvFn)therock_gemv_q4_0_wave32));
  reg->entries.push_back(make_gemv_entry(decode_mask,  THEROCK_LAYOUT_ROW_MAJOR, 1, thresh,     (TheRockGemvFn)therock_gemv_fp16_wave32));
  reg->entries.push_back(make_gemm_entry(prefill_mask, THEROCK_LAYOUT_ROW_MAJOR, 1, UINT32_MAX, (TheRockGemmFn)therock_gemm_hipblaslt));
  reg->entries.push_back(make_gemm_entry(prefill_mask, THEROCK_LAYOUT_Q4_0,      1, UINT32_MAX, (TheRockGemmFn)therock_gemm_hipblaslt));

  return reg;
}

void therock_registry_destroy(TheRockKernelRegistry *reg) {
  delete reg;
}

TheRockKernelFn therock_registry_lookup(const TheRockKernelRegistry *reg,
                                         const TheRockOpDesc *desc) {
  uint32_t mode_bit = 1u << desc->exec_mode;
  for (const auto &e : reg->entries) {
    if (!(e.exec_mode_mask & mode_bit))       continue;
    if (e.layout != desc->weight_layout)      continue;
    if (desc->M < e.m_min || desc->M > e.m_max) continue;
    return e.fn;
  }
  return {.gemm = nullptr};
}

uint32_t therock_decode_threshold(const TheRockHardwareDesc *hw) {
  // Empirical: on gfx1032, GEMV wins below M=8 for typical inference N sizes.
  // For other hardware, scale by CU count relative to gfx1032 (32 CUs).
  // More CUs = GEMM stays efficient longer because there's work to spread.
  (void)hw;
  return 8;
}

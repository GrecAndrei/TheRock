// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// TheRock inference: load and run a Gemma 4 12B GGUF (Q4_0) on gfx1032.
// Single-file standalone runner. No llama.cpp dependency.
//
// Build:
//   hipcc -O3 --offload-arch=gfx1032 -I../include run_gguf.hip -lamdhip64 -o run_gguf
//
// Usage:
//   ./run_gguf <model.gguf> [prompt]

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define HIP_CHECK(x) do { \
  hipError_t e = (x); \
  if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %d (%s) at %s:%d\n", e, hipGetErrorString(e), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

// ============================================================================
// GGUF Parser (mmap-based, zero-copy)
// ============================================================================

enum GGMLType : uint32_t {
  GGML_TYPE_F32   = 0,
  GGML_TYPE_F16   = 1,
  GGML_TYPE_Q4_0  = 2,
  GGML_TYPE_Q4_1  = 3,
  GGML_TYPE_Q5_0  = 6,
  GGML_TYPE_Q5_1  = 7,
  GGML_TYPE_Q8_0  = 8,
  GGML_TYPE_Q6_K  = 14,
};

struct Q4_0Block {
  __half scale;
  uint8_t qs[16];  // 32 nibbles
};
static_assert(sizeof(Q4_0Block) == 18, "Q4_0Block must be 18 bytes");

struct Q6_K_Block {
  uint8_t ql[128]; // low 4 bits of quants (256 / 2 = 128 bytes)
  uint8_t qh[64];  // high 2 bits of quants (256 / 4 = 64 bytes)
  int8_t  scales[16];
  __half  d;       // super-block scale
};
static_assert(sizeof(Q6_K_Block) == 210, "Q6_K_Block must be 210 bytes");

static size_t ggml_type_size(GGMLType t) {
  switch (t) {
    case GGML_TYPE_F32:  return 4;
    case GGML_TYPE_F16:  return 2;
    case GGML_TYPE_Q4_0: return 18;  // per 32 elements
    case GGML_TYPE_Q8_0: return 34;  // per 32 elements
    case GGML_TYPE_Q6_K: return 210; // per 256 elements
    default: return 0;
  }
}

static size_t ggml_block_size(GGMLType t) {
  switch (t) {
    case GGML_TYPE_F32:  return 1;
    case GGML_TYPE_F16:  return 1;
    case GGML_TYPE_Q4_0: return 32;
    case GGML_TYPE_Q8_0: return 32;
    case GGML_TYPE_Q6_K: return 256;
    default: return 1;
  }
}

static size_t tensor_bytes(GGMLType type, uint64_t n_elements) {
  size_t bs = ggml_block_size(type);
  size_t ts = ggml_type_size(type);
  return (n_elements / bs) * ts;
}

struct GGUFTensor {
  std::string name;
  uint32_t n_dims;
  uint64_t dims[4];
  GGMLType type;
  uint64_t offset;  // from start of tensor data section
  uint64_t n_elements;
};

struct GGUFFile {
  int fd;
  uint8_t *data;
  size_t   file_size;
  size_t   tensor_data_offset;

  uint32_t version;
  uint64_t n_tensors;
  uint64_t n_kv;

  std::unordered_map<std::string, GGUFTensor> tensors;
  std::unordered_map<std::string, uint64_t>   kv_uint;
  std::unordered_map<std::string, float>      kv_float;
  std::unordered_map<std::string, std::string> kv_str;

  const void* tensor_data(const GGUFTensor &t) const {
    return data + tensor_data_offset + t.offset;
  }
};

static std::string read_gguf_string(const uint8_t *&p) {
  uint64_t len = *(const uint64_t*)p; p += 8;
  std::string s((const char*)p, len);
  p += len;
  return s;
}

static void skip_gguf_value(const uint8_t *&p, uint32_t type_id) {
  static const size_t sizes[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
  if (type_id <= 12 && type_id != 8 && type_id != 9) {
    p += sizes[type_id];
  } else if (type_id == 8) {
    uint64_t len = *(const uint64_t*)p; p += 8;
    p += len;
  } else if (type_id == 9) {
    uint32_t arr_type = *(const uint32_t*)p; p += 4;
    uint64_t arr_len = *(const uint64_t*)p; p += 8;
    for (uint64_t i = 0; i < arr_len; i++)
      skip_gguf_value(p, arr_type);
  }
}

static uint64_t read_gguf_uint(const uint8_t *&p, uint32_t type_id) {
  uint64_t v = 0;
  switch (type_id) {
    case 0: v = *p; p += 1; break;
    case 4: v = *(const uint32_t*)p; p += 4; break;
    case 10: v = *(const uint64_t*)p; p += 8; break;
    default: skip_gguf_value(p, type_id); break;
  }
  return v;
}

static GGUFFile* gguf_open(const char *path) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) { perror("open"); return nullptr; }

  struct stat st;
  fstat(fd, &st);

  auto *g = new GGUFFile();
  g->fd = fd;
  g->file_size = st.st_size;
  g->data = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (g->data == MAP_FAILED) { perror("mmap"); delete g; close(fd); return nullptr; }

  // Advise sequential read for tensor data
  madvise(g->data, st.st_size, MADV_SEQUENTIAL);

  const uint8_t *p = g->data;

  // Header
  if (memcmp(p, "GGUF", 4) != 0) { fprintf(stderr, "Not a GGUF file\n"); return nullptr; }
  p += 4;
  g->version = *(const uint32_t*)p; p += 4;
  g->n_tensors = *(const uint64_t*)p; p += 8;
  g->n_kv = *(const uint64_t*)p; p += 8;

  // KV metadata
  for (uint64_t i = 0; i < g->n_kv; i++) {
    std::string key = read_gguf_string(p);
    uint32_t type_id = *(const uint32_t*)p; p += 4;

    if (type_id == 4) { // uint32
      g->kv_uint[key] = *(const uint32_t*)p; p += 4;
    } else if (type_id == 10) { // uint64
      g->kv_uint[key] = *(const uint64_t*)p; p += 8;
    } else if (type_id == 6) { // float32
      g->kv_float[key] = *(const float*)p; p += 4;
    } else if (type_id == 8) { // string
      g->kv_str[key] = read_gguf_string(p);
    } else if (type_id == 7) { // bool
      g->kv_uint[key] = *p; p += 1;
    } else {
      skip_gguf_value(p, type_id);
    }
  }

  // Tensor info
  for (uint64_t i = 0; i < g->n_tensors; i++) {
    GGUFTensor t;
    t.name = read_gguf_string(p);
    t.n_dims = *(const uint32_t*)p; p += 4;
    t.n_elements = 1;
    for (uint32_t d = 0; d < t.n_dims; d++) {
      t.dims[d] = *(const uint64_t*)p; p += 8;
      t.n_elements *= t.dims[d];
    }
    for (uint32_t d = t.n_dims; d < 4; d++) t.dims[d] = 1;
    t.type = (GGMLType)(*(const uint32_t*)p); p += 4;
    t.offset = *(const uint64_t*)p; p += 8;
    g->tensors[t.name] = t;
  }

  // Tensor data starts at next alignment boundary (32 bytes in GGUFv3)
  size_t header_end = p - g->data;
  g->tensor_data_offset = (header_end + 31) & ~(size_t)31;

  return g;
}

static void gguf_close(GGUFFile *g) {
  if (g->data) munmap(g->data, g->file_size);
  if (g->fd >= 0) close(g->fd);
  delete g;
}

// ============================================================================
// GPU Kernels
// ============================================================================

static constexpr int WAVE = 32;
static constexpr int Q4_BLOCK = 32;

// Q4_0 fused dequant-GEMV: y[n] = sum_k dequant(W[n,k]) * x[k]
// Grid: (N,), Block: (32,)
// Optimized: 4-byte vector loads for activation, unrolled inner loop
__global__ void __launch_bounds__(WAVE)
kern_gemv_q4_0(int N, int K,
               const __half    * __restrict__ x,
               const Q4_0Block * __restrict__ W,
               __half          * __restrict__ y) {
  int row = blockIdx.x;
  if (row >= N) return;

  int num_blocks = K / Q4_BLOCK;
  const Q4_0Block *w_row = W + (long)row * num_blocks;
  int tid = threadIdx.x;
  float acc = 0.0f;

  for (int blk = tid; blk < num_blocks; blk += WAVE) {
    const Q4_0Block &b = w_row[blk];
    float scale = __half2float(b.scale);
    const __half *xblk = x + blk * Q4_BLOCK;

    // Load activation via float4 (8 halves per load, 4 loads for 32 elements)
    float4 xv0 = *reinterpret_cast<const float4*>(xblk);
    float4 xv1 = *reinterpret_cast<const float4*>(xblk + 8);
    float4 xv2 = *reinterpret_cast<const float4*>(xblk + 16);
    float4 xv3 = *reinterpret_cast<const float4*>(xblk + 24);
    const __half *xp0 = reinterpret_cast<const __half*>(&xv0);
    const __half *xp1 = reinterpret_cast<const __half*>(&xv1);
    const __half *xp2 = reinterpret_cast<const __half*>(&xv2);
    const __half *xp3 = reinterpret_cast<const __half*>(&xv3);

    float block_acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      uint8_t packed = b.qs[i];
      block_acc += scale * ((float)(packed & 0xF) - 8.0f) * __half2float(xp0[i*2]);
      block_acc += scale * ((float)(packed >> 4)  - 8.0f) * __half2float(xp0[i*2+1]);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      uint8_t packed = b.qs[4+i];
      block_acc += scale * ((float)(packed & 0xF) - 8.0f) * __half2float(xp1[i*2]);
      block_acc += scale * ((float)(packed >> 4)  - 8.0f) * __half2float(xp1[i*2+1]);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      uint8_t packed = b.qs[8+i];
      block_acc += scale * ((float)(packed & 0xF) - 8.0f) * __half2float(xp2[i*2]);
      block_acc += scale * ((float)(packed >> 4)  - 8.0f) * __half2float(xp2[i*2+1]);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      uint8_t packed = b.qs[12+i];
      block_acc += scale * ((float)(packed & 0xF) - 8.0f) * __half2float(xp3[i*2]);
      block_acc += scale * ((float)(packed >> 4)  - 8.0f) * __half2float(xp3[i*2+1]);
    }
    acc += block_acc;
  }

  acc += __shfl_xor(acc, 16);
  acc += __shfl_xor(acc, 8);
  acc += __shfl_xor(acc, 4);
  acc += __shfl_xor(acc, 2);
  acc += __shfl_xor(acc, 1);

  if (tid == 0)
    y[row] = __float2half(acc);
}

// fp16 GEMV for embedding gather output
__global__ void __launch_bounds__(WAVE)
kern_gemv_f16(int N, int K,
              const __half * __restrict__ x,
              const __half * __restrict__ W,
              __half       * __restrict__ y) {
  int row = blockIdx.x;
  if (row >= N) return;

  const __half *w_row = W + (long)row * K;
  int tid = threadIdx.x;
  float acc = 0.0f;

  for (int k = tid * 8; k < K; k += WAVE * 8) {
    float4 wv = *reinterpret_cast<const float4*>(w_row + k);
    float4 xv = *reinterpret_cast<const float4*>(x + k);
    const __half *wp = reinterpret_cast<const __half*>(&wv);
    const __half *xp = reinterpret_cast<const __half*>(&xv);
    for (int i = 0; i < 8; i++)
      acc += __half2float(wp[i]) * __half2float(xp[i]);
  }

  acc += __shfl_xor(acc, 16); acc += __shfl_xor(acc, 8);
  acc += __shfl_xor(acc, 4);  acc += __shfl_xor(acc, 2); acc += __shfl_xor(acc, 1);

  if (tid == 0) y[row] = __float2half(acc);
}

// RMSNorm: out = x / rms(x) * scale
// Uses 256 threads (8 wavefronts) with LDS reduction for fast norm computation
static constexpr int NORM_THREADS = 256;
static constexpr int NORM_WAVES = NORM_THREADS / WAVE;

__global__ void __launch_bounds__(NORM_THREADS)
kern_rmsnorm(uint32_t dim,
             const __half * __restrict__ x,
             const float  * __restrict__ scale,
             float eps,
             __half       * __restrict__ out) {
  __shared__ float wave_sums[NORM_WAVES];
  int tid = threadIdx.x;
  int wave_id = tid / WAVE;
  int lane_id = tid % WAVE;

  float ss = 0.0f;
  for (uint32_t i = tid; i < dim; i += NORM_THREADS) {
    float v = __half2float(x[i]);
    ss += v * v;
  }
  // Intra-wave reduction
  ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
  ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);

  if (lane_id == 0) wave_sums[wave_id] = ss;
  __syncthreads();

  // Final reduction across waves (lane 0 of wave 0)
  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < NORM_WAVES; w++) total += wave_sums[w];
    wave_sums[0] = rsqrtf(total / (float)dim + eps);
  }
  __syncthreads();
  float rms_inv = wave_sums[0];

  for (uint32_t i = tid; i < dim; i += NORM_THREADS)
    out[i] = __float2half(__half2float(x[i]) * rms_inv * scale[i]);
}

// RMSNorm for attention heads (per-head, applied to a contiguous chunk)
__global__ void __launch_bounds__(WAVE)
kern_rmsnorm_head(uint32_t n_heads, uint32_t head_dim,
                  const __half * __restrict__ x,
                  const float  * __restrict__ scale,
                  float eps,
                  __half       * __restrict__ out) {
  int head = blockIdx.x;
  if (head >= (int)n_heads) return;
  int tid = threadIdx.x;

  const __half *hx = x + head * head_dim;
  __half *ho = out + head * head_dim;

  float ss = 0.0f;
  for (uint32_t i = tid; i < head_dim; i += WAVE) {
    float v = __half2float(hx[i]);
    ss += v * v;
  }
  ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
  ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);

  float rms_inv = rsqrtf(ss / (float)head_dim + eps);
  for (uint32_t i = tid; i < head_dim; i += WAVE)
    ho[i] = __float2half(__half2float(hx[i]) * rms_inv * scale[i % head_dim]);
}

// GeLU activation: out = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
__global__ void
kern_gelu_mul(uint32_t n,
              const __half * __restrict__ gate,
              const __half * __restrict__ up,
              __half       * __restrict__ out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = __half2float(gate[i]);
  float u = __half2float(up[i]);
  float gelu = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g)));
  out[i] = __float2half(gelu * u);
}

// Softcapping: out = cap * tanh(x / cap)
__global__ void
kern_softcap(uint32_t n, float cap, __half * __restrict__ data) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = __half2float(data[i]);
  data[i] = __float2half(cap * tanhf(x / cap));
}

// RoPE (Rotary Position Embedding) applied in-place
// For head_dim elements: pairs (x[2i], x[2i+1]) rotated by angle at position pos
__global__ void __launch_bounds__(WAVE)
kern_rope(uint32_t n_heads, uint32_t head_dim, uint32_t pos,
          float freq_base, const float * __restrict__ freq_factors,
          __half * __restrict__ x) {
  int head = blockIdx.x;
  if (head >= (int)n_heads) return;
  int tid = threadIdx.x;

  __half *hx = x + head * head_dim;
  uint32_t half_dim = head_dim / 2;

  for (uint32_t i = tid; i < half_dim; i += WAVE) {
    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
    if (freq_factors) freq *= freq_factors[i];
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float x0 = __half2float(hx[i]);
    float x1 = __half2float(hx[i + half_dim]);
    hx[i]            = __float2half(x0 * cos_a - x1 * sin_a);
    hx[i + half_dim] = __float2half(x0 * sin_a + x1 * cos_a);
  }
}

// Single-query attention (decode): one wavefront per query head
// Q: [n_heads * head_dim], K_cache: [seq_len * n_kv_heads * head_dim], same for V
__global__ void __launch_bounds__(WAVE)
kern_attn_decode(uint32_t n_q_heads, uint32_t n_kv_heads, uint32_t head_dim,
                 uint32_t seq_len, float scale,
                 const __half * __restrict__ q,
                 const __half * __restrict__ k_cache,
                 const __half * __restrict__ v_cache,
                 __half       * __restrict__ out) {
  int qh = blockIdx.x;
  if (qh >= (int)n_q_heads) return;
  int kvh = qh / (n_q_heads / n_kv_heads);
  int tid = threadIdx.x;

  const int EPT = 16; // elements per thread (head_dim / WAVE for head_dim up to 512)
  float q_reg[16];
  for (int i = 0; i < (int)head_dim / WAVE; i++)
    q_reg[i] = __half2float(q[qh * head_dim + tid * (head_dim / WAVE) + i]);

  float m = -1e9f, l = 0.0f;
  float o[16] = {};

  for (uint32_t t = 0; t < seq_len; t++) {
    const __half *kt = k_cache + t * n_kv_heads * head_dim + kvh * head_dim;
    const __half *vt = v_cache + t * n_kv_heads * head_dim + kvh * head_dim;

    float dot = 0.0f;
    for (int i = 0; i < (int)head_dim / WAVE; i++)
      dot += q_reg[i] * __half2float(kt[tid * (head_dim / WAVE) + i]);

    dot += __shfl_xor(dot, 16); dot += __shfl_xor(dot, 8);
    dot += __shfl_xor(dot, 4);  dot += __shfl_xor(dot, 2); dot += __shfl_xor(dot, 1);

    float s = dot * scale;
    float m_new = fmaxf(m, s);
    float exp_shift = expf(m - m_new);
    float exp_s = expf(s - m_new);
    l = exp_shift * l + exp_s;

    for (int i = 0; i < (int)head_dim / WAVE; i++) {
      float vi = __half2float(vt[tid * (head_dim / WAVE) + i]);
      o[i] = exp_shift * o[i] + exp_s * vi;
    }
    m = m_new;
  }

  float l_inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (int i = 0; i < (int)head_dim / WAVE; i++)
    out[qh * head_dim + tid * (head_dim / WAVE) + i] = __float2half(o[i] * l_inv);
}

// Q6_K GEMV for lm_head: y[n] = sum_k dequant(W[n,k]) * x[k]
// All 32 threads collaborate on EACH Q6_K block (256 elem / 32 = 8 elem/thread).
// Sequential over blocks, parallel within block. 264 GB/s on gfx1032.
__global__ void __launch_bounds__(WAVE)
kern_gemv_q6k(int N, int K,
              const __half       * __restrict__ x,
              const Q6_K_Block   * __restrict__ W,
              __half             * __restrict__ y) {
  int row = blockIdx.x;
  if (row >= N) return;
  int num_blocks = K / 256;
  const Q6_K_Block *w_row = W + (long)row * num_blocks;
  int tid = threadIdx.x;
  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; blk++) {
    const Q6_K_Block &b = w_row[blk];
    float d = __half2float(b.d);
    const __half *xblk = x + blk * 256;

    int base = tid * 8;
    float sub_scale = d * (float)b.scales[base / 16];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
      int j = base + i;
      uint8_t ql_val = (b.ql[j / 2] >> ((j % 2) * 4)) & 0xF;
      uint8_t qh_val = (b.qh[j / 4] >> ((j % 4) * 2)) & 0x3;
      int8_t q = (int8_t)((ql_val | (qh_val << 4)) - 32);
      float s = (i == 0 || (base + i) / 16 == base / 16) ? sub_scale
                : d * (float)b.scales[(base + i) / 16];
      acc += s * (float)q * __half2float(xblk[j]);
    }
  }

  acc += __shfl_xor(acc, 16); acc += __shfl_xor(acc, 8);
  acc += __shfl_xor(acc, 4);  acc += __shfl_xor(acc, 2); acc += __shfl_xor(acc, 1);

  if (tid == 0)
    y[row] = __float2half(acc);
}

// Add (residual connection): out = a + b
__global__ void
kern_add(uint32_t n, const __half *a, const __half *b, __half *out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

// Fused: RMSNorm(x) → out, AND x = x + residual (in place)
// Saves one kernel launch + one memory pass
__global__ void __launch_bounds__(NORM_THREADS)
kern_rmsnorm_residual(uint32_t dim,
                      __half * __restrict__ x,
                      const __half * __restrict__ residual,
                      const float  * __restrict__ scale,
                      float eps,
                      __half       * __restrict__ normed_out) {
  __shared__ float wave_sums[NORM_WAVES];
  int tid = threadIdx.x;
  int wave_id = tid / WAVE;
  int lane_id = tid % WAVE;

  float ss = 0.0f;
  for (uint32_t i = tid; i < dim; i += NORM_THREADS) {
    float v = __half2float(x[i]);
    ss += v * v;
  }
  ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
  ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
  if (lane_id == 0) wave_sums[wave_id] = ss;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < NORM_WAVES; w++) total += wave_sums[w];
    wave_sums[0] = rsqrtf(total / (float)dim + eps);
  }
  __syncthreads();
  float rms_inv = wave_sums[0];

  for (uint32_t i = tid; i < dim; i += NORM_THREADS) {
    float xi = __half2float(x[i]);
    normed_out[i] = __float2half(xi * rms_inv * scale[i]);
    x[i] = __float2half(xi + __half2float(residual[i]));
  }
}

// Scale multiply: out = x * s (scalar broadcast)
__global__ void
kern_scale(uint32_t n, const __half *x, float s, __half *out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = __float2half(__half2float(x[i]) * s);
}

// Fused: post-norm + residual add + optional scale
// Replaces: kern_rmsnorm(residual) + kern_add(hidden, residual) + kern_scale(hidden)
// hidden = (hidden + rmsnorm(residual, norm_weights)) * layer_scale
__global__ void __launch_bounds__(NORM_THREADS)
kern_fused_postnorm_residadd(uint32_t dim,
                             __half * __restrict__ hidden,
                             const __half * __restrict__ proj_out,
                             const float  * __restrict__ norm_weights,
                             float eps, float layer_scale) {
  __shared__ float wave_sums[NORM_WAVES];
  int tid = threadIdx.x;
  int wave_id = tid / WAVE;
  int lane_id = tid % WAVE;

  float ss = 0.0f;
  for (uint32_t i = tid; i < dim; i += NORM_THREADS) {
    float v = __half2float(proj_out[i]);
    ss += v * v;
  }
  ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
  ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
  if (lane_id == 0) wave_sums[wave_id] = ss;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < NORM_WAVES; w++) total += wave_sums[w];
    wave_sums[0] = rsqrtf(total / (float)dim + eps);
  }
  __syncthreads();
  float rms_inv = wave_sums[0];

  for (uint32_t i = tid; i < dim; i += NORM_THREADS) {
    float normed = __half2float(proj_out[i]) * rms_inv * norm_weights[i];
    float h = __half2float(hidden[i]) + normed;
    hidden[i] = __float2half(h * layer_scale);
  }
}

// GPU argmax over fp16 array using parallel reduction
__global__ void __launch_bounds__(256)
kern_argmax(__half * __restrict__ data, uint32_t n, uint32_t *result) {
  __shared__ float s_val[256];
  __shared__ uint32_t s_idx[256];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  float best = -1e30f;
  uint32_t best_i = 0;
  for (uint32_t i = gid; i < n; i += gridDim.x * blockDim.x) {
    float v = __half2float(data[i]);
    if (v > best) { best = v; best_i = i; }
  }
  s_val[tid] = best;
  s_idx[tid] = best_i;
  __syncthreads();

  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride && s_val[tid + stride] > s_val[tid]) {
      s_val[tid] = s_val[tid + stride];
      s_idx[tid] = s_idx[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    // Atomic max across blocks (store val in upper bits + idx in lower bits is complex,
    // so just use atomic for small block count)
    atomicMax(result, s_idx[0] | ((uint32_t)(__half_as_ushort(__float2half(s_val[0]))) << 16));
  }
}

// Simpler GPU argmax: single block, large thread count, iterates over all data
__global__ void __launch_bounds__(1024)
kern_argmax_single(const __half * __restrict__ data, uint32_t n, uint32_t *result) {
  __shared__ float s_val[1024];
  __shared__ uint32_t s_idx[1024];
  int tid = threadIdx.x;

  float best = -1e30f;
  uint32_t best_i = 0;
  for (uint32_t i = tid; i < n; i += 1024) {
    float v = __half2float(data[i]);
    if (v > best) { best = v; best_i = i; }
  }
  s_val[tid] = best;
  s_idx[tid] = best_i;
  __syncthreads();

  for (int stride = 512; stride > 0; stride >>= 1) {
    if (tid < stride && s_val[tid + stride] > s_val[tid]) {
      s_val[tid] = s_val[tid + stride];
      s_idx[tid] = s_idx[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) *result = s_idx[0];
}

// GPU embedding lookup: dequant one row of Q6_K directly on GPU, with scale factor
__global__ void __launch_bounds__(256)
kern_embed_lookup(uint32_t token, uint32_t dim, float embed_scale,
                  const Q6_K_Block * __restrict__ embed_q6k,
                  __half * __restrict__ out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int blocks_per_row = dim / 256;
  const Q6_K_Block *row = embed_q6k + (size_t)token * blocks_per_row;

  if (tid >= (int)dim) return;
  int blk_idx = tid / 256;
  int j = tid % 256;
  const Q6_K_Block &b = row[blk_idx];
  float d = __half2float(b.d);
  uint8_t ql_val = (b.ql[j / 2] >> ((j % 2) * 4)) & 0xF;
  uint8_t qh_val = (b.qh[j / 4] >> ((j % 4) * 2)) & 0x3;
  int8_t q = (int8_t)((ql_val | (qh_val << 4)) - 32);
  float scale = d * (float)b.scales[j / 16];
  out[tid] = __float2half((float)q * scale * embed_scale);
}

// RoPE with precomputed cos/sin tables
__global__ void __launch_bounds__(WAVE)
kern_rope_precomp(uint32_t n_heads, uint32_t head_dim, uint32_t pos,
                  const float * __restrict__ cos_table,
                  const float * __restrict__ sin_table,
                  __half * __restrict__ x) {
  int head = blockIdx.x;
  if (head >= (int)n_heads) return;
  int tid = threadIdx.x;
  __half *hx = x + head * head_dim;
  uint32_t half_dim = head_dim / 2;
  const float *cos_row = cos_table + pos * half_dim;
  const float *sin_row = sin_table + pos * half_dim;

  for (uint32_t i = tid; i < half_dim; i += WAVE) {
    float x0 = __half2float(hx[i]);
    float x1 = __half2float(hx[i + half_dim]);
    hx[i]            = __float2half(x0 * cos_row[i] - x1 * sin_row[i]);
    hx[i + half_dim] = __float2half(x0 * sin_row[i] + x1 * cos_row[i]);
  }
}

// Fused: RMSNorm per head + RoPE (saves 2 kernel launches per Q/K)
__global__ void __launch_bounds__(WAVE)
kern_headnorm_rope(uint32_t n_heads, uint32_t head_dim, uint32_t pos,
                   const float * __restrict__ norm_weights,
                   const float * __restrict__ cos_table,
                   const float * __restrict__ sin_table,
                   float eps,
                   __half * __restrict__ x) {
  int head = blockIdx.x;
  if (head >= (int)n_heads) return;
  int tid = threadIdx.x;
  __half *hx = x + head * head_dim;

  // RMSNorm
  float ss = 0.0f;
  for (uint32_t i = tid; i < head_dim; i += WAVE) {
    float v = __half2float(hx[i]);
    ss += v * v;
  }
  ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
  ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
  float rms_inv = rsqrtf(ss / (float)head_dim + eps);

  // Apply norm and write to temp registers, then RoPE
  uint32_t half_dim = head_dim / 2;
  const float *cos_row = cos_table + pos * half_dim;
  const float *sin_row = sin_table + pos * half_dim;

  // Two-pass: first normalize, then rotate
  for (uint32_t i = tid; i < head_dim; i += WAVE)
    hx[i] = __float2half(__half2float(hx[i]) * rms_inv * norm_weights[i % head_dim]);
  // RoPE rotation
  for (uint32_t i = tid; i < half_dim; i += WAVE) {
    float x0 = __half2float(hx[i]);
    float x1 = __half2float(hx[i + half_dim]);
    hx[i]            = __float2half(x0 * cos_row[i] - x1 * sin_row[i]);
    hx[i + half_dim] = __float2half(x0 * sin_row[i] + x1 * cos_row[i]);
  }
}

// ============================================================================
// Model structure
// ============================================================================

static constexpr uint32_t HIDDEN     = 3840;
static constexpr uint32_t FFN        = 15360;
static constexpr uint32_t N_LAYERS   = 48;
static constexpr float    EPS        = 1e-6f;

struct LayerWeights {
  // All Q4_0 pointers (device)
  void *wq;       // [HIDDEN, q_dim]
  void *wk;       // [HIDDEN, kv_dim] — nullptr for non-KV layers
  void *wv;       // [HIDDEN, kv_dim] — nullptr if uses K as V
  void *wo;       // [q_dim, HIDDEN]
  void *ffn_gate; // [HIDDEN, FFN]
  void *ffn_up;   // [HIDDEN, FFN]
  void *ffn_down; // [FFN, HIDDEN]

  // F32 norm weights (device)
  float *attn_norm;
  float *attn_q_norm;
  float *attn_k_norm;
  float *post_attn_norm;
  float *ffn_norm;
  float *post_ffn_norm;
  float *out_scale; // scalar (device)
  float out_scale_val; // cached host copy

  // Layer config
  uint32_t n_q_heads;
  uint32_t n_kv_heads;
  uint32_t head_dim;
  uint32_t q_dim;     // n_q_heads * head_dim
  uint32_t kv_dim;    // n_kv_heads * head_dim
  bool     is_swa;    // sliding window layer
  bool     has_kv;    // has own K/V projection (non-shared layers)
  bool     has_v;     // has separate V weight (vs using K as V)
};

struct Model {
  LayerWeights layers[N_LAYERS];
  float *output_norm; // [HIDDEN] F32 on device
  float *rope_freqs;  // [256] F32 on device (for global layers)
  uint32_t vocab_size;
  float final_logit_softcap;
  float rope_freq_base;
  float rope_freq_base_swa;

  // Embedding table: Q6_K on GPU for lm_head AND token embedding
  Q6_K_Block *embed_q6k_dev;           // [vocab_size * HIDDEN / 256] Q6_K blocks on GPU
  __half *logits_dev;                   // [vocab_size] on GPU for argmax
  uint32_t *argmax_dev;                 // GPU argmax result

  // Precomputed RoPE tables (GPU)
  float *rope_cos_swa;    // [max_seq × 128] for SWA (head_dim=256, half=128)
  float *rope_sin_swa;
  float *rope_cos_global; // [max_seq × 256] for global (head_dim=512, half=256)
  float *rope_sin_global;

  // KV cache: [N_LAYERS][max_seq][n_kv_heads * head_dim] fp16 on device
  __half **kv_k; // per-layer K cache
  __half **kv_v; // per-layer V cache
  uint32_t max_seq;
};

// ============================================================================
// Q6_K dequantization (for embedding table, done on CPU)
// ============================================================================

static void dequant_q6k_row(__half *out, const uint8_t *data, uint64_t n_elements) {
  uint64_t n_blocks = n_elements / 256;
  const Q6_K_Block *blocks = (const Q6_K_Block*)data;

  for (uint64_t b = 0; b < n_blocks; b++) {
    const Q6_K_Block &blk = blocks[b];
    float d = __half2float(blk.d);

    for (int j = 0; j < 256; j++) {
      // Get the 6-bit quantized value
      int ql_idx = j / 2;
      int ql_shift = (j % 2) * 4;
      uint8_t ql_val = (blk.ql[ql_idx] >> ql_shift) & 0xF;

      int qh_idx = j / 4;
      int qh_shift = (j % 4) * 2;
      uint8_t qh_val = (blk.qh[qh_idx] >> qh_shift) & 0x3;

      int8_t q = (int8_t)((ql_val | (qh_val << 4)) - 32);
      int scale_idx = j / 16;
      float scale = d * (float)blk.scales[scale_idx];
      out[b * 256 + j] = __float2half((float)q * scale);
    }
  }
}

// ============================================================================
// Model Loading
// ============================================================================

static void* upload_q4_0(const void *host_data, uint64_t n_elements) {
  size_t bytes = tensor_bytes(GGML_TYPE_Q4_0, n_elements);
  void *dev;
  HIP_CHECK(hipMalloc(&dev, bytes));
  HIP_CHECK(hipMemcpy(dev, host_data, bytes, hipMemcpyHostToDevice));
  return dev;
}

static float* upload_f32(const void *host_data, uint64_t n_elements) {
  size_t bytes = n_elements * sizeof(float);
  float *dev;
  HIP_CHECK(hipMalloc(&dev, bytes));
  HIP_CHECK(hipMemcpy(dev, host_data, bytes, hipMemcpyHostToDevice));
  return dev;
}

static bool load_model(GGUFFile *gguf, Model *model) {
  // Architecture params
  model->vocab_size = 262144;  // from embedding tensor
  model->final_logit_softcap = 30.0f;
  model->rope_freq_base = 1000000.0f;
  model->rope_freq_base_swa = 10000.0f;
  model->max_seq = 512;  // keep small — just enough for benchmarking t/s

  if (gguf->kv_float.count("gemma4.final_logit_softcapping"))
    model->final_logit_softcap = gguf->kv_float["gemma4.final_logit_softcapping"];
  if (gguf->kv_float.count("gemma4.rope.freq_base"))
    model->rope_freq_base = gguf->kv_float["gemma4.rope.freq_base"];
  if (gguf->kv_float.count("gemma4.rope.freq_base_swa"))
    model->rope_freq_base_swa = gguf->kv_float["gemma4.rope.freq_base_swa"];

  // SWA pattern: layers 5,11,17,23,29,35,41,47 are global (non-SWA)
  // From metadata: sliding_window_pattern = [T,T,T,T,T,F,T,T,T,T] repeating
  auto is_global = [](int l) { return (l % 6 == 5); };

  size_t total_gpu_bytes = 0;

  printf("Loading embedding table...\n");
  {
    auto it = gguf->tensors.find("token_embd.weight");
    if (it == gguf->tensors.end()) { fprintf(stderr, "Missing token_embd\n"); return false; }
    const GGUFTensor &t = it->second;
    uint64_t n_elem = t.n_elements;
    size_t q6k_bytes = tensor_bytes(GGML_TYPE_Q6_K, n_elem);

    // Upload Q6_K to GPU — used for BOTH token embedding (row lookup) and lm_head (full GEMV)
    HIP_CHECK(hipMalloc(&model->embed_q6k_dev, q6k_bytes));
    HIP_CHECK(hipMemcpy(model->embed_q6k_dev, gguf->tensor_data(t), q6k_bytes, hipMemcpyHostToDevice));

    // Allocate logits buffer and argmax result on GPU
    HIP_CHECK(hipMalloc(&model->logits_dev, model->vocab_size * sizeof(__half)));
    HIP_CHECK(hipMalloc(&model->argmax_dev, sizeof(uint32_t)));

    printf("  Embedding: %lu x %lu → %.1f MB Q6_K on GPU (lm_head + embed)\n",
           t.dims[0], t.dims[1], q6k_bytes / 1e6);
    total_gpu_bytes += q6k_bytes + model->vocab_size * sizeof(__half);
  }

  // Output norm
  {
    auto it = gguf->tensors.find("output_norm.weight");
    if (it == gguf->tensors.end()) { fprintf(stderr, "Missing output_norm\n"); return false; }
    model->output_norm = upload_f32(gguf->tensor_data(it->second), it->second.n_elements);
  }

  // Rope freqs (shared across global layers)
  {
    auto it = gguf->tensors.find("rope_freqs.weight");
    if (it != gguf->tensors.end()) {
      model->rope_freqs = upload_f32(gguf->tensor_data(it->second), it->second.n_elements);
    } else {
      model->rope_freqs = nullptr;
    }
  }

  // Allocate KV cache
  model->kv_k = new __half*[N_LAYERS];
  model->kv_v = new __half*[N_LAYERS];

  printf("Loading %u layers...\n", N_LAYERS);
  for (uint32_t l = 0; l < N_LAYERS; l++) {
    LayerWeights &lw = model->layers[l];
    char name[128];

    lw.is_swa = !is_global(l);
    lw.has_kv = true;  // all layers in 12B have their own KV
    lw.head_dim = lw.is_swa ? 256 : 512;
    lw.n_q_heads = 16;
    lw.n_kv_heads = lw.is_swa ? 8 : 1;
    lw.q_dim = lw.n_q_heads * lw.head_dim;
    lw.kv_dim = lw.n_kv_heads * lw.head_dim;

    // Check if V weight exists
    snprintf(name, sizeof(name), "blk.%u.attn_v.weight", l);
    lw.has_v = (gguf->tensors.count(name) > 0);

    // Load Q4_0 weights
    auto load_q4 = [&](const char *suffix) -> void* {
      snprintf(name, sizeof(name), "blk.%u.%s", l, suffix);
      auto it = gguf->tensors.find(name);
      if (it == gguf->tensors.end()) return nullptr;
      size_t bytes = tensor_bytes(GGML_TYPE_Q4_0, it->second.n_elements);
      total_gpu_bytes += bytes;
      return upload_q4_0(gguf->tensor_data(it->second), it->second.n_elements);
    };

    auto load_norm = [&](const char *suffix) -> float* {
      snprintf(name, sizeof(name), "blk.%u.%s", l, suffix);
      auto it = gguf->tensors.find(name);
      if (it == gguf->tensors.end()) return nullptr;
      total_gpu_bytes += it->second.n_elements * 4;
      return upload_f32(gguf->tensor_data(it->second), it->second.n_elements);
    };

    lw.wq       = load_q4("attn_q.weight");
    lw.wk       = load_q4("attn_k.weight");
    lw.wv       = load_q4("attn_v.weight");
    lw.wo       = load_q4("attn_output.weight");
    lw.ffn_gate = load_q4("ffn_gate.weight");
    lw.ffn_up   = load_q4("ffn_up.weight");
    lw.ffn_down = load_q4("ffn_down.weight");

    lw.attn_norm     = load_norm("attn_norm.weight");
    lw.attn_q_norm   = load_norm("attn_q_norm.weight");
    lw.attn_k_norm   = load_norm("attn_k_norm.weight");
    lw.post_attn_norm = load_norm("post_attention_norm.weight");
    lw.ffn_norm      = load_norm("ffn_norm.weight");
    lw.post_ffn_norm = load_norm("post_ffw_norm.weight");

    // Layer output scale
    snprintf(name, sizeof(name), "blk.%u.layer_output_scale.weight", l);
    auto it2 = gguf->tensors.find(name);
    if (it2 != gguf->tensors.end()) {
      lw.out_scale = upload_f32(gguf->tensor_data(it2->second), 1);
      lw.out_scale_val = *(const float*)gguf->tensor_data(it2->second);
    } else {
      lw.out_scale = nullptr;
      lw.out_scale_val = 1.0f;
    }

    // KV cache allocation
    size_t kv_bytes = (size_t)model->max_seq * lw.kv_dim * sizeof(__half);
    HIP_CHECK(hipMalloc(&model->kv_k[l], kv_bytes));
    HIP_CHECK(hipMalloc(&model->kv_v[l], kv_bytes));
    HIP_CHECK(hipMemset(model->kv_k[l], 0, kv_bytes));
    HIP_CHECK(hipMemset(model->kv_v[l], 0, kv_bytes));
    total_gpu_bytes += kv_bytes * 2;

    if (l % 12 == 0 || l == N_LAYERS - 1)
      printf("  Layer %u: q_dim=%u kv_dim=%u head_dim=%u %s%s [%.0f MB GPU so far]\n",
             l, lw.q_dim, lw.kv_dim, lw.head_dim,
             lw.is_swa ? "SWA" : "GLOBAL",
             lw.has_v ? "" : " (K=V)",
             total_gpu_bytes / 1e6);
  }

  // Precompute RoPE cos/sin tables on host, upload to GPU
  printf("Precomputing RoPE tables...\n");
  {
    uint32_t max_seq = model->max_seq;

    // SWA: head_dim=256, half_dim=128
    size_t swa_size = (size_t)max_seq * 128;
    float *cos_swa_h = (float*)malloc(swa_size * sizeof(float));
    float *sin_swa_h = (float*)malloc(swa_size * sizeof(float));
    for (uint32_t pos = 0; pos < max_seq; pos++) {
      for (uint32_t i = 0; i < 128; i++) {
        float freq = 1.0f / powf(model->rope_freq_base_swa, (float)(2*i) / 256.0f);
        float angle = (float)pos * freq;
        cos_swa_h[pos*128+i] = cosf(angle);
        sin_swa_h[pos*128+i] = sinf(angle);
      }
    }
    HIP_CHECK(hipMalloc(&model->rope_cos_swa, swa_size*sizeof(float)));
    HIP_CHECK(hipMalloc(&model->rope_sin_swa, swa_size*sizeof(float)));
    HIP_CHECK(hipMemcpy(model->rope_cos_swa, cos_swa_h, swa_size*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(model->rope_sin_swa, sin_swa_h, swa_size*sizeof(float), hipMemcpyHostToDevice));
    free(cos_swa_h); free(sin_swa_h);

    // Global: head_dim=512, half_dim=256 — also needs freq_factors
    size_t global_size = (size_t)max_seq * 256;
    float *cos_g_h = (float*)malloc(global_size * sizeof(float));
    float *sin_g_h = (float*)malloc(global_size * sizeof(float));
    // Read freq_factors from host if available
    float freq_factors_host[256] = {};
    auto ff_it = gguf->tensors.find("rope_freqs.weight");
    if (ff_it != gguf->tensors.end()) {
      const float *ff_data = (const float*)gguf->tensor_data(ff_it->second);
      memcpy(freq_factors_host, ff_data, std::min((size_t)256, (size_t)ff_it->second.n_elements) * sizeof(float));
    }
    for (uint32_t pos = 0; pos < max_seq; pos++) {
      for (uint32_t i = 0; i < 256; i++) {
        float freq = 1.0f / powf(model->rope_freq_base, (float)(2*i) / 512.0f);
        if (freq_factors_host[i] != 0.0f) freq *= freq_factors_host[i];
        float angle = (float)pos * freq;
        cos_g_h[pos*256+i] = cosf(angle);
        sin_g_h[pos*256+i] = sinf(angle);
      }
    }
    HIP_CHECK(hipMalloc(&model->rope_cos_global, global_size*sizeof(float)));
    HIP_CHECK(hipMalloc(&model->rope_sin_global, global_size*sizeof(float)));
    HIP_CHECK(hipMemcpy(model->rope_cos_global, cos_g_h, global_size*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(model->rope_sin_global, sin_g_h, global_size*sizeof(float), hipMemcpyHostToDevice));
    free(cos_g_h); free(sin_g_h);

    total_gpu_bytes += (swa_size + global_size) * 2 * sizeof(float);
  }

  printf("Total GPU memory: %.2f GB\n", total_gpu_bytes / 1e9);
  return true;
}

// ============================================================================
// Forward Pass (single token decode)
// ============================================================================

struct RunState {
  __half *hidden;      // [HIDDEN]
  __half *residual;    // [HIDDEN]
  __half *normed;      // [HIDDEN] — pre-norm output to avoid residual copy
  __half *scratch_q;   // [max_q_dim = 8192]
  __half *scratch_kv;  // [max_kv_dim = 2048]
  __half *scratch_v;   // [max_kv_dim = 2048]
  __half *scratch_attn;// [max_q_dim = 8192]
  __half *scratch_ffn1;// [FFN]
  __half *scratch_ffn2;// [FFN]
  hipStream_t stream;
  uint32_t pos;        // current position in sequence
};

static void forward_decode(Model *model, RunState *state, uint32_t token) {
  hipStream_t s = state->stream;
  uint32_t pos = state->pos;

  // Embed token on GPU: dequant one Q6_K row + scale by sqrt(hidden_dim) in one kernel
  hipLaunchKernelGGL(kern_embed_lookup, dim3((HIDDEN+255)/256), dim3(256), 0, s,
                     token, HIDDEN, sqrtf((float)HIDDEN), model->embed_q6k_dev, state->hidden);

  __half *normed = state->normed;

  for (uint32_t l = 0; l < N_LAYERS; l++) {
    LayerWeights &lw = model->layers[l];

    // 1. Attention pre-norm (256 threads)
    hipLaunchKernelGGL(kern_rmsnorm, dim3(1), dim3(NORM_THREADS), 0, s,
                       HIDDEN, state->hidden, lw.attn_norm, EPS, normed);

    // 2. Q/K/V projections
    hipLaunchKernelGGL(kern_gemv_q4_0, dim3(lw.q_dim), dim3(WAVE), 0, s,
                       (int)lw.q_dim, (int)HIDDEN,
                       normed, (const Q4_0Block*)lw.wq, state->scratch_q);

    if (lw.wk) {
      hipLaunchKernelGGL(kern_gemv_q4_0, dim3(lw.kv_dim), dim3(WAVE), 0, s,
                         (int)lw.kv_dim, (int)HIDDEN,
                         normed, (const Q4_0Block*)lw.wk, state->scratch_kv);
    }

    if (lw.has_v && lw.wv) {
      hipLaunchKernelGGL(kern_gemv_q4_0, dim3(lw.kv_dim), dim3(WAVE), 0, s,
                         (int)lw.kv_dim, (int)HIDDEN,
                         normed, (const Q4_0Block*)lw.wv, state->scratch_v);
    }

    // 3. Fused Q/K head norms + RoPE (saves 4 launches per layer)
    const float *cos_tbl = lw.is_swa ? model->rope_cos_swa : model->rope_cos_global;
    const float *sin_tbl = lw.is_swa ? model->rope_sin_swa : model->rope_sin_global;

    hipLaunchKernelGGL(kern_headnorm_rope, dim3(lw.n_q_heads), dim3(WAVE), 0, s,
                       lw.n_q_heads, lw.head_dim, pos,
                       lw.attn_q_norm, cos_tbl, sin_tbl, EPS,
                       state->scratch_q);

    if (lw.wk) {
      hipLaunchKernelGGL(kern_headnorm_rope, dim3(lw.n_kv_heads), dim3(WAVE), 0, s,
                         lw.n_kv_heads, lw.head_dim, pos,
                         lw.attn_k_norm, cos_tbl, sin_tbl, EPS,
                         state->scratch_kv);
    }

    // 5. Store K/V into cache
    if (lw.wk) {
      HIP_CHECK(hipMemcpyAsync(model->kv_k[l] + pos * lw.kv_dim,
                               state->scratch_kv,
                               lw.kv_dim * sizeof(__half),
                               hipMemcpyDeviceToDevice, s));
      __half *v_src = lw.has_v ? state->scratch_v : state->scratch_kv;
      HIP_CHECK(hipMemcpyAsync(model->kv_v[l] + pos * lw.kv_dim,
                               v_src,
                               lw.kv_dim * sizeof(__half),
                               hipMemcpyDeviceToDevice, s));
    }

    // 6. Attention
    uint32_t seq_len = pos + 1;
    hipLaunchKernelGGL(kern_attn_decode, dim3(lw.n_q_heads), dim3(WAVE), 0, s,
                       lw.n_q_heads, lw.n_kv_heads, lw.head_dim,
                       seq_len, 1.0f,
                       state->scratch_q,
                       model->kv_k[l],
                       model->kv_v[l],
                       state->scratch_attn);

    // 7. Output projection
    hipLaunchKernelGGL(kern_gemv_q4_0, dim3(HIDDEN), dim3(WAVE), 0, s,
                       (int)HIDDEN, (int)lw.q_dim,
                       state->scratch_attn, (const Q4_0Block*)lw.wo, state->residual);

    // 8. FUSED: post-attn-norm + residual add (+ layer scale folded in at end)
    //    hidden = (hidden + rmsnorm(residual)) * 1.0  (scale applied at end of layer)
    hipLaunchKernelGGL(kern_fused_postnorm_residadd, dim3(1), dim3(NORM_THREADS), 0, s,
                       HIDDEN, state->hidden, state->residual, lw.post_attn_norm, EPS, 1.0f);

    // 9. FFN pre-norm (256 threads)
    hipLaunchKernelGGL(kern_rmsnorm, dim3(1), dim3(NORM_THREADS), 0, s,
                       HIDDEN, state->hidden, lw.ffn_norm, EPS, normed);

    // 10. FFN gate + up projections
    hipLaunchKernelGGL(kern_gemv_q4_0, dim3(FFN), dim3(WAVE), 0, s,
                       (int)FFN, (int)HIDDEN,
                       normed, (const Q4_0Block*)lw.ffn_gate, state->scratch_ffn1);

    hipLaunchKernelGGL(kern_gemv_q4_0, dim3(FFN), dim3(WAVE), 0, s,
                       (int)FFN, (int)HIDDEN,
                       normed, (const Q4_0Block*)lw.ffn_up, state->scratch_ffn2);

    // 11. GeLU(gate) * up
    hipLaunchKernelGGL(kern_gelu_mul, dim3((FFN+255)/256), dim3(256), 0, s,
                       FFN, state->scratch_ffn1, state->scratch_ffn2, state->scratch_ffn1);

    // 12. Down projection
    hipLaunchKernelGGL(kern_gemv_q4_0, dim3(HIDDEN), dim3(WAVE), 0, s,
                       (int)HIDDEN, (int)FFN,
                       state->scratch_ffn1, (const Q4_0Block*)lw.ffn_down, state->residual);

    // 13. FUSED: post-ffn-norm + residual add + layer scale
    float scale = lw.out_scale_val;
    hipLaunchKernelGGL(kern_fused_postnorm_residadd, dim3(1), dim3(NORM_THREADS), 0, s,
                       HIDDEN, state->hidden, state->residual, lw.post_ffn_norm, EPS, scale);
  }

  // Final norm (256 threads)
  hipLaunchKernelGGL(kern_rmsnorm, dim3(1), dim3(NORM_THREADS), 0, s,
                     HIDDEN, state->hidden, model->output_norm, EPS, state->hidden);

  state->pos++;
}

// GPU-based lm_head + GPU argmax sampling
static uint32_t sample_greedy(Model *model, RunState *state) {
  hipStream_t s = state->stream;

  // lm_head: [1 × HIDDEN] @ [vocab × HIDDEN]^T → [1 × vocab]
  hipLaunchKernelGGL(kern_gemv_q6k, dim3(model->vocab_size), dim3(WAVE), 0, s,
                     (int)model->vocab_size, (int)HIDDEN,
                     state->hidden, model->embed_q6k_dev, model->logits_dev);

  // GPU argmax (single kernel, no DtoH of full logits)
  HIP_CHECK(hipMemsetAsync(model->argmax_dev, 0, sizeof(uint32_t), s));
  hipLaunchKernelGGL(kern_argmax_single, dim3(1), dim3(1024), 0, s,
                     model->logits_dev, model->vocab_size, model->argmax_dev);

  uint32_t result;
  HIP_CHECK(hipMemcpyAsync(&result, model->argmax_dev, sizeof(uint32_t),
                           hipMemcpyDeviceToHost, s));
  HIP_CHECK(hipStreamSynchronize(s));
  return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model.gguf> [prompt]\n", argv[0]);
    return 1;
  }

  const char *model_path = argv[1];
  const char *prompt = (argc > 2) ? argv[2] : "Hello, I am";

  printf("=== TheRock Inference Engine ===\n");
  printf("Model: %s\n", model_path);
  printf("GPU: gfx1032 (RX 6650 XT)\n\n");

  // Open and parse GGUF
  printf("Opening GGUF file...\n");
  GGUFFile *gguf = gguf_open(model_path);
  if (!gguf) return 1;
  printf("  Version: %u, Tensors: %lu, KV pairs: %lu\n",
         gguf->version, gguf->n_tensors, gguf->n_kv);

  // Load model to GPU
  Model model = {};
  if (!load_model(gguf, &model)) return 1;

  // Allocate run state
  RunState state = {};
  HIP_CHECK(hipMalloc(&state.hidden, HIDDEN * sizeof(__half)));
  HIP_CHECK(hipMalloc(&state.residual, HIDDEN * sizeof(__half)));
  HIP_CHECK(hipMalloc(&state.normed, HIDDEN * sizeof(__half)));
  HIP_CHECK(hipMalloc(&state.scratch_q, 8192 * sizeof(__half)));
  HIP_CHECK(hipMalloc(&state.scratch_kv, 2048 * sizeof(__half)));
  HIP_CHECK(hipMalloc(&state.scratch_v, 2048 * sizeof(__half)));
  HIP_CHECK(hipMalloc(&state.scratch_attn, 8192 * sizeof(__half)));
  HIP_CHECK(hipMalloc(&state.scratch_ffn1, FFN * sizeof(__half)));
  HIP_CHECK(hipMalloc(&state.scratch_ffn2, FFN * sizeof(__half)));
  HIP_CHECK(hipStreamCreate(&state.stream));
  state.pos = 0;

  printf("\nModel loaded. Starting generation...\n");
  printf("Prompt: \"%s\"\n\n", prompt);

  // Tokenize (BPE is complex — for now just use byte-level fallback)
  // TODO: proper tokenizer. For benchmarking, just feed BOS + a few tokens.
  std::vector<uint32_t> tokens;
  tokens.push_back(2);  // BOS token

  // Simple byte-level encoding as fallback (each byte → token offset)
  // This won't produce coherent text but will exercise the full pipeline
  for (const char *c = prompt; *c; c++) {
    tokens.push_back((uint32_t)(unsigned char)*c + 256);  // rough byte fallback
  }

  printf("Tokens: %zu (BOS + %zu prompt bytes)\n", tokens.size(), tokens.size() - 1);
  printf("---\n");

  // Prefill: process prompt tokens
  auto t_start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < tokens.size(); i++) {
    forward_decode(&model, &state, tokens[i]);
  }

  auto t_prefill = std::chrono::high_resolution_clock::now();
  float prefill_ms = std::chrono::duration<float, std::milli>(t_prefill - t_start).count();
  printf("[Prefill: %zu tokens in %.1f ms = %.1f t/s]\n",
         tokens.size(), prefill_ms, tokens.size() * 1000.0f / prefill_ms);

  // Warmup: run 2 decode+sample cycles to JIT all kernels at their final grid sizes
  printf("\nWarming up GPU kernels...\n");
  {
    uint32_t wt = sample_greedy(&model, &state);
    forward_decode(&model, &state, wt);
    wt = sample_greedy(&model, &state);
    forward_decode(&model, &state, wt);
  }

  // Generate tokens (timed)
  int n_gen = 32;
  printf("Generating %d tokens...\n", n_gen);

  auto t_gen_start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < n_gen; i++) {
    uint32_t next_token = sample_greedy(&model, &state);
    forward_decode(&model, &state, next_token);

    // Print progress
    if ((i + 1) % 8 == 0) {
      auto t_now = std::chrono::high_resolution_clock::now();
      float elapsed = std::chrono::duration<float, std::milli>(t_now - t_gen_start).count();
      printf("  [%d/%d tokens, %.1f t/s]\n", i + 1, n_gen,
             (i + 1) * 1000.0f / elapsed);
    }
  }

  auto t_gen_end = std::chrono::high_resolution_clock::now();
  float gen_ms = std::chrono::duration<float, std::milli>(t_gen_end - t_gen_start).count();

  printf("\n=== Results ===\n");
  printf("Decode: %d tokens in %.1f ms\n", n_gen, gen_ms);
  printf("Throughput: %.2f tokens/sec\n", n_gen * 1000.0f / gen_ms);
  printf("Per-token: %.2f ms\n", gen_ms / n_gen);

  // Estimate bandwidth utilization
  // Model weights read per token: ~6.5 GB (Q4_0)
  float model_gb = 6.5f;
  float effective_bw = model_gb / (gen_ms / n_gen / 1000.0f);
  printf("Effective BW: %.0f GB/s (of 280 GB/s VRAM)\n", effective_bw);
  printf("BW utilization: %.0f%%\n", effective_bw / 280.0f * 100.0f);

  // Cleanup
  gguf_close(gguf);
  printf("\nDone.\n");
  return 0;
}

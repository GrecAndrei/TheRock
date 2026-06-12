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
  std::unordered_map<std::string, std::vector<std::string>> kv_str_arr;

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

static std::vector<std::string> read_gguf_string_array(const uint8_t *&p) {
  uint32_t arr_type = *(const uint32_t*)p; p += 4;
  uint64_t arr_len = *(const uint64_t*)p; p += 8;
  std::vector<std::string> values;
  if (arr_type != 8) {
    for (uint64_t i = 0; i < arr_len; i++)
      skip_gguf_value(p, arr_type);
    return values;
  }
  values.reserve(arr_len);
  for (uint64_t i = 0; i < arr_len; i++)
    values.push_back(read_gguf_string(p));
  return values;
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
    } else if (type_id == 9) { // array
      if (key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.merges") {
        g->kv_str_arr[key] = read_gguf_string_array(p);
      } else {
        skip_gguf_value(p, type_id);
      }
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
// Tokenizer (Gemma4 BPE from GGUF metadata)
// ============================================================================

static constexpr const char *TOKEN_SPACE = "\xE2\x96\x81";

struct Model;
struct RunState;
static void forward_prefill_batch(Model *model, RunState *state,
                                  const uint32_t *tokens, int n_tokens);

struct Tokenizer {
  std::vector<std::string> id_to_piece;
  std::unordered_map<std::string, uint32_t> piece_to_id;
  std::unordered_map<std::string, uint32_t> merge_rank;
  uint32_t bos_id = 2;
  uint32_t eos_id = 1;
  uint32_t unk_id = 3;
  uint32_t turn_open_id = UINT32_MAX;
  uint32_t turn_close_id = UINT32_MAX;
  uint32_t channel_open_id = UINT32_MAX;
  uint32_t channel_close_id = UINT32_MAX;
  bool add_bos_token = true;
  bool add_space_prefix = false;
};

static std::vector<std::string> utf8_split(const std::string &text) {
  std::vector<std::string> out;
  for (size_t i = 0; i < text.size(); ) {
    unsigned char c = (unsigned char)text[i];
    size_t len = 1;
    if ((c & 0x80) == 0x00) len = 1;
    else if ((c & 0xE0) == 0xC0) len = 2;
    else if ((c & 0xF0) == 0xE0) len = 3;
    else if ((c & 0xF8) == 0xF0) len = 4;
    if (i + len > text.size()) len = 1;
    out.emplace_back(text.substr(i, len));
    i += len;
  }
  return out;
}

static std::string normalize_tokenizer_text(const std::string &text) {
  std::string out;
  out.reserve(text.size() + 16);
  for (char c : text) {
    if (c == ' ') out += TOKEN_SPACE;
    else out.push_back(c);
  }
  return out;
}

static bool load_tokenizer(GGUFFile *gguf, Tokenizer *tok) {
  auto tok_it = gguf->kv_str_arr.find("tokenizer.ggml.tokens");
  auto merge_it = gguf->kv_str_arr.find("tokenizer.ggml.merges");
  if (tok_it == gguf->kv_str_arr.end() || merge_it == gguf->kv_str_arr.end()) {
    fprintf(stderr, "Missing tokenizer metadata in GGUF\n");
    return false;
  }

  tok->id_to_piece = tok_it->second;
  tok->piece_to_id.reserve(tok->id_to_piece.size() * 2);
  for (uint32_t i = 0; i < tok->id_to_piece.size(); i++)
    tok->piece_to_id.emplace(tok->id_to_piece[i], i);

  tok->merge_rank.reserve(merge_it->second.size() * 2);
  for (uint32_t i = 0; i < merge_it->second.size(); i++)
    tok->merge_rank.emplace(merge_it->second[i], i);

  if (gguf->kv_uint.count("tokenizer.ggml.bos_token_id"))
    tok->bos_id = (uint32_t)gguf->kv_uint["tokenizer.ggml.bos_token_id"];
  if (gguf->kv_uint.count("tokenizer.ggml.eos_token_id"))
    tok->eos_id = (uint32_t)gguf->kv_uint["tokenizer.ggml.eos_token_id"];
  if (gguf->kv_uint.count("tokenizer.ggml.unknown_token_id"))
    tok->unk_id = (uint32_t)gguf->kv_uint["tokenizer.ggml.unknown_token_id"];
  if (gguf->kv_uint.count("tokenizer.ggml.add_bos_token"))
    tok->add_bos_token = (bool)gguf->kv_uint["tokenizer.ggml.add_bos_token"];
  if (gguf->kv_uint.count("tokenizer.ggml.add_space_prefix"))
    tok->add_space_prefix = (bool)gguf->kv_uint["tokenizer.ggml.add_space_prefix"];

  auto it_open = tok->piece_to_id.find("<|turn>");
  if (it_open != tok->piece_to_id.end()) tok->turn_open_id = it_open->second;
  auto it_close = tok->piece_to_id.find("<turn|>");
  if (it_close != tok->piece_to_id.end()) tok->turn_close_id = it_close->second;
  auto it_ch_open = tok->piece_to_id.find("<|channel>");
  if (it_ch_open != tok->piece_to_id.end()) tok->channel_open_id = it_ch_open->second;
  auto it_ch_close = tok->piece_to_id.find("<channel|>");
  if (it_ch_close != tok->piece_to_id.end()) tok->channel_close_id = it_ch_close->second;
  return true;
}

static void tokenizer_append_bytes(const Tokenizer &tok, const std::string &bytes,
                                   std::vector<uint32_t> *out_ids) {
  char buf[8];
  for (unsigned char c : bytes) {
    snprintf(buf, sizeof(buf), "<0x%02X>", (unsigned)c);
    auto it = tok.piece_to_id.find(buf);
    out_ids->push_back(it != tok.piece_to_id.end() ? it->second : tok.unk_id);
  }
}

static void tokenizer_append_chunk(const Tokenizer &tok, const std::string &chunk,
                                   std::vector<uint32_t> *out_ids) {
  std::vector<std::string> pieces = utf8_split(chunk);
  if (pieces.empty()) return;

  while (pieces.size() > 1) {
    uint32_t best_rank = UINT32_MAX;
    size_t best_i = pieces.size();
    for (size_t i = 0; i + 1 < pieces.size(); i++) {
      std::string key = pieces[i] + " " + pieces[i + 1];
      auto it = tok.merge_rank.find(key);
      if (it != tok.merge_rank.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
      }
    }
    if (best_i == pieces.size()) break;
    pieces[best_i] += pieces[best_i + 1];
    pieces.erase(pieces.begin() + best_i + 1);
  }

  for (const std::string &piece : pieces) {
    auto it = tok.piece_to_id.find(piece);
    if (it != tok.piece_to_id.end()) out_ids->push_back(it->second);
    else tokenizer_append_bytes(tok, piece, out_ids);
  }
}

static void tokenizer_append_text(const Tokenizer &tok, const std::string &text,
                                  std::vector<uint32_t> *out_ids) {
  std::string normalized = normalize_tokenizer_text(text);
  if (tok.add_space_prefix && !normalized.empty() && normalized.rfind(TOKEN_SPACE, 0) != 0)
    normalized = std::string(TOKEN_SPACE) + normalized;

  size_t i = 0;
  while (i < normalized.size()) {
    size_t j = i;
    if (normalized[i] == '\n') {
      while (j < normalized.size() && normalized[j] == '\n') j++;
    } else {
      while (j < normalized.size() && normalized[j] != '\n') j++;
    }
    std::string chunk = normalized.substr(i, j - i);
    auto it = tok.piece_to_id.find(chunk);
    if (it != tok.piece_to_id.end()) out_ids->push_back(it->second);
    else tokenizer_append_chunk(tok, chunk, out_ids);
    i = j;
  }
}

static std::vector<uint32_t> tokenizer_encode_prompt(const Tokenizer &tok, const std::string &text) {
  std::vector<uint32_t> ids;
  if (tok.add_bos_token) ids.push_back(tok.bos_id);
  tokenizer_append_text(tok, text, &ids);
  return ids;
}

static std::string decode_piece_for_display(const std::string &piece) {
  if (piece == "<bos>" || piece == "<eos>" || piece == "<pad>") return std::string();
  if ((piece.size() >= 4 && piece[0] == '<' && piece[1] == '|') || piece == "<turn|>" || piece == "<channel|>")
    return std::string();
  if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
    auto hex = [](char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
      if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
      return -1;
    };
    int hi = hex(piece[3]);
    int lo = hex(piece[4]);
    if (hi >= 0 && lo >= 0) return std::string(1, (char)((hi << 4) | lo));
  }
  std::string out;
  out.reserve(piece.size());
  for (size_t i = 0; i < piece.size(); ) {
    if (i + 3 <= piece.size() && memcmp(piece.data() + i, TOKEN_SPACE, 3) == 0) {
      out.push_back(' ');
      i += 3;
    } else {
      out.push_back(piece[i]);
      i++;
    }
  }
  return out;
}

static void append_model_generation_prompt(const Tokenizer &tok, std::vector<uint32_t> *ids) {
  if (tok.turn_open_id != UINT32_MAX) ids->push_back(tok.turn_open_id);
  tokenizer_append_text(tok, "model\n", ids);
  if (tok.channel_open_id != UINT32_MAX && tok.channel_close_id != UINT32_MAX) {
    ids->push_back(tok.channel_open_id);
    tokenizer_append_text(tok, "thought\n", ids);
    ids->push_back(tok.channel_close_id);
  }
}

static void append_chat_turn_tokens(const Tokenizer &tok, const char *role,
                                    const std::string &content,
                                    bool prepend_newline,
                                    bool open_next_model_turn,
                                    std::vector<uint32_t> *ids) {
  if (prepend_newline)
    tokenizer_append_text(tok, "\n", ids);
  if (tok.turn_open_id != UINT32_MAX) ids->push_back(tok.turn_open_id);
  tokenizer_append_text(tok, std::string(role) + "\n", ids);
  tokenizer_append_text(tok, content, ids);
  if (tok.turn_close_id != UINT32_MAX) ids->push_back(tok.turn_close_id);
  tokenizer_append_text(tok, "\n", ids);
  if (open_next_model_turn) append_model_generation_prompt(tok, ids);
}

static void prefill_tokens(Model *model, RunState *state, const std::vector<uint32_t> &tokens) {
  for (size_t i = 0; i < tokens.size(); ) {
    int batch = std::min((int)(tokens.size() - i), 16);
    forward_prefill_batch(model, state, tokens.data() + i, batch);
    i += batch;
  }
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

    float block_acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
      uint8_t packed = b.qs[i];
      block_acc += scale * ((float)(packed & 0xF) - 8.0f) * __half2float(xblk[i]);
      block_acc += scale * ((float)(packed >> 4)  - 8.0f) * __half2float(xblk[i + 16]);
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

__global__ void __launch_bounds__(WAVE)
kern_rmsnorm_head_noscale(uint32_t n_heads, uint32_t head_dim,
                          __half * __restrict__ x,
                          float eps) {
  int head = blockIdx.x;
  if (head >= (int)n_heads) return;
  int tid = threadIdx.x;

  __half *hx = x + head * head_dim;

  float ss = 0.0f;
  for (uint32_t i = tid; i < head_dim; i += WAVE) {
    float v = __half2float(hx[i]);
    ss += v * v;
  }
  ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
  ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);

  float rms_inv = rsqrtf(ss / (float)head_dim + eps);
  for (uint32_t i = tid; i < head_dim; i += WAVE)
    hx[i] = __float2half(__half2float(hx[i]) * rms_inv);
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

__global__ void
kern_quantize_kv_fp16_to_i8(uint32_t n,
                            const __half * __restrict__ k_src,
                            const __half * __restrict__ v_src,
                            int8_t       * __restrict__ k_dst,
                            int8_t       * __restrict__ v_dst,
                            float        * __restrict__ k_scale_out,
                            float        * __restrict__ v_scale_out) {
  __shared__ float red_k[256];
  __shared__ float red_v[256];
  __shared__ float k_scale;
  __shared__ float v_scale;
  __shared__ float k_inv_scale;
  __shared__ float v_inv_scale;

  uint32_t tid = threadIdx.x;
  float max_k = 0.0f;
  float max_v = 0.0f;
  for (uint32_t i = tid; i < n; i += blockDim.x) {
    max_k = fmaxf(max_k, fabsf(__half2float(k_src[i])));
    max_v = fmaxf(max_v, fabsf(__half2float(v_src[i])));
  }
  red_k[tid] = max_k;
  red_v[tid] = max_v;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      red_k[tid] = fmaxf(red_k[tid], red_k[tid + stride]);
      red_v[tid] = fmaxf(red_v[tid], red_v[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    k_scale = (red_k[0] > 0.0f) ? (red_k[0] / 127.0f) : (1.0f / 127.0f);
    v_scale = (red_v[0] > 0.0f) ? (red_v[0] / 127.0f) : (1.0f / 127.0f);
    k_inv_scale = 1.0f / k_scale;
    v_inv_scale = 1.0f / v_scale;
    *k_scale_out = k_scale;
    *v_scale_out = v_scale;
  }
  __syncthreads();

  for (uint32_t i = tid; i < n; i += blockDim.x) {
    float kq = __half2float(k_src[i]) * k_inv_scale;
    float vq = __half2float(v_src[i]) * v_inv_scale;
    int k_rounded = (int)lrintf(kq);
    int v_rounded = (int)lrintf(vq);
    k_rounded = (k_rounded < -127) ? -127 : ((k_rounded > 127) ? 127 : k_rounded);
    v_rounded = (v_rounded < -127) ? -127 : ((v_rounded > 127) ? 127 : v_rounded);
    k_dst[i] = (int8_t)k_rounded;
    v_dst[i] = (int8_t)v_rounded;
  }
}

__global__ void
kern_quantize_kv_fp16_to_q4(uint32_t n,
                            const __half * __restrict__ k_src,
                            const __half * __restrict__ v_src,
                            uint8_t      * __restrict__ k_dst,
                            uint8_t      * __restrict__ v_dst,
                            float        * __restrict__ k_scale_out,
                            float        * __restrict__ v_scale_out) {
  __shared__ float red_k[256];
  __shared__ float red_v[256];
  __shared__ float k_scale;
  __shared__ float v_scale;
  __shared__ float k_inv_scale;
  __shared__ float v_inv_scale;

  uint32_t tid = threadIdx.x;
  float max_k = 0.0f;
  float max_v = 0.0f;
  for (uint32_t i = tid; i < n; i += blockDim.x) {
    max_k = fmaxf(max_k, fabsf(__half2float(k_src[i])));
    max_v = fmaxf(max_v, fabsf(__half2float(v_src[i])));
  }
  red_k[tid] = max_k;
  red_v[tid] = max_v;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      red_k[tid] = fmaxf(red_k[tid], red_k[tid + stride]);
      red_v[tid] = fmaxf(red_v[tid], red_v[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    k_scale = (red_k[0] > 0.0f) ? (red_k[0] / 7.0f) : (1.0f / 7.0f);
    v_scale = (red_v[0] > 0.0f) ? (red_v[0] / 7.0f) : (1.0f / 7.0f);
    k_inv_scale = 1.0f / k_scale;
    v_inv_scale = 1.0f / v_scale;
    *k_scale_out = k_scale;
    *v_scale_out = v_scale;
  }
  __syncthreads();

  for (uint32_t pair = tid; pair < n / 2; pair += blockDim.x) {
    uint32_t idx0 = pair * 2;
    uint32_t idx1 = idx0 + 1;
    int k0 = (int)lrintf(__half2float(k_src[idx0]) * k_inv_scale);
    int k1 = (int)lrintf(__half2float(k_src[idx1]) * k_inv_scale);
    int v0 = (int)lrintf(__half2float(v_src[idx0]) * v_inv_scale);
    int v1 = (int)lrintf(__half2float(v_src[idx1]) * v_inv_scale);
    k0 = (k0 < -8) ? -8 : ((k0 > 7) ? 7 : k0);
    k1 = (k1 < -8) ? -8 : ((k1 > 7) ? 7 : k1);
    v0 = (v0 < -8) ? -8 : ((v0 > 7) ? 7 : v0);
    v1 = (v1 < -8) ? -8 : ((v1 > 7) ? 7 : v1);
    k_dst[pair] = (uint8_t)((k0 & 0xF) | ((k1 & 0xF) << 4));
    v_dst[pair] = (uint8_t)((v0 & 0xF) | ((v1 & 0xF) << 4));
  }
}

__global__ void
kern_fill_rope_row(uint32_t half_dim, uint32_t pos,
                   float freq_base,
                   const float * __restrict__ freq_factors,
                   float * __restrict__ cos_row,
                   float * __restrict__ sin_row) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= half_dim) return;
  uint32_t head_dim = half_dim * 2;
  float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
  if (freq_factors) freq /= freq_factors[i];
  float angle = (float)pos * freq;
  cos_row[i] = cosf(angle);
  sin_row[i] = sinf(angle);
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
    if (freq_factors) freq /= freq_factors[i];
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float x0 = __half2float(hx[i]);
    float x1 = __half2float(hx[i + half_dim]);
    hx[i]            = __float2half(x0 * cos_a - x1 * sin_a);
    hx[i + half_dim] = __float2half(x0 * sin_a + x1 * cos_a);
  }
}

// Flash decoding with split-K: two-phase approach
// Phase 1: N wavefronts per head, each processes a chunk of KV tokens → partial (m, l, o) to global
// Phase 2: reduce across splits
// This parallelizes the KV scan across wavefronts, letting the GPU overlap memory latency.

static constexpr int ATTN_SPLIT_SIZE = 64; // tokens per split

__device__ __forceinline__ uint32_t cache_phys_index(uint32_t cache_start,
                                                     uint32_t cache_capacity,
                                                     uint32_t t) {
  uint32_t phys = cache_start + t;
  if (phys >= cache_capacity) phys -= cache_capacity;
  return phys;
}

__device__ __forceinline__ int unpack_q4_signed(const uint8_t *src, uint32_t idx) {
  uint8_t byte = src[idx >> 1];
  uint8_t nibble = (idx & 1) ? (byte >> 4) : (byte & 0xF);
  return (nibble < 8) ? (int)nibble : ((int)nibble - 16);
}

// Phase 1: each wavefront handles one head × one chunk of tokens
// Grid: (n_q_heads, n_splits), Block: (32)
// Output: split_buf[split][head][head_dim] (partial output), split_ml[split][head][2] (m, l)
__global__ void __launch_bounds__(WAVE)
kern_attn_split(uint32_t n_q_heads, uint32_t n_kv_heads, uint32_t head_dim,
                uint32_t seq_len, uint32_t cache_capacity, uint32_t cache_start,
                float scale,
                const __half * __restrict__ q,
                const __half * __restrict__ k_cache,
                const __half * __restrict__ v_cache,
                float        * __restrict__ split_o,   // [n_splits × n_q_heads × head_dim]
                float        * __restrict__ split_ml)  // [n_splits × n_q_heads × 2]
{
  int qh = blockIdx.x;
  int split_id = blockIdx.y;
  if (qh >= (int)n_q_heads) return;
  int kvh = qh / (n_q_heads / n_kv_heads);
  int tid = threadIdx.x;
  uint32_t ept = head_dim / WAVE;

  uint32_t t_start = split_id * ATTN_SPLIT_SIZE;
  uint32_t t_end = min(t_start + ATTN_SPLIT_SIZE, seq_len);
  if (t_start >= seq_len) {
    // Empty split: write sentinel
    if (tid == 0) {
      uint32_t ml_idx = split_id * n_q_heads * 2 + qh * 2;
      split_ml[ml_idx] = -1e9f;
      split_ml[ml_idx + 1] = 0.0f;
    }
    return;
  }

  float q_reg[16];
  for (uint32_t i = 0; i < ept; i++)
    q_reg[i] = __half2float(q[qh * head_dim + tid * ept + i]);

  float m = -1e9f, l = 0.0f;
  float o[16] = {};

  for (uint32_t t = t_start; t < t_end; t++) {
    uint32_t phys_t = cache_phys_index(cache_start, cache_capacity, t);
    const __half *kt = k_cache + phys_t * n_kv_heads * head_dim + kvh * head_dim;
    const __half *vt = v_cache + phys_t * n_kv_heads * head_dim + kvh * head_dim;

    float dot = 0.0f;
    for (uint32_t i = 0; i < ept; i++)
      dot += q_reg[i] * __half2float(kt[tid * ept + i]);
    dot += __shfl_xor(dot, 16); dot += __shfl_xor(dot, 8);
    dot += __shfl_xor(dot, 4);  dot += __shfl_xor(dot, 2); dot += __shfl_xor(dot, 1);

    float s_val = dot * scale;
    float m_new = fmaxf(m, s_val);
    float exp_shift = expf(m - m_new);
    float exp_s = expf(s_val - m_new);
    l = exp_shift * l + exp_s;
    for (uint32_t i = 0; i < ept; i++) {
      float vi = __half2float(vt[tid * ept + i]);
      o[i] = exp_shift * o[i] + exp_s * vi;
    }
    m = m_new;
  }

  // Write partial results to global memory
  uint32_t o_base = (split_id * n_q_heads + qh) * head_dim + tid * ept;
  for (uint32_t i = 0; i < ept; i++)
    split_o[o_base + i] = o[i];

  if (tid == 0) {
    uint32_t ml_idx = split_id * n_q_heads * 2 + qh * 2;
    split_ml[ml_idx] = m;
    split_ml[ml_idx + 1] = l;
  }
}

// Phase 2: reduce across splits. One wavefront per head.
// Grid: (n_q_heads), Block: (32)
__global__ void __launch_bounds__(WAVE)
kern_attn_reduce(uint32_t n_q_heads, uint32_t head_dim, uint32_t n_splits,
                 const float * __restrict__ split_o,
                 const float * __restrict__ split_ml,
                 __half      * __restrict__ out) {
  int qh = blockIdx.x;
  if (qh >= (int)n_q_heads) return;
  int tid = threadIdx.x;
  uint32_t ept = head_dim / WAVE;

  // Find global max across all splits
  float global_m = -1e9f;
  for (uint32_t s = 0; s < n_splits; s++) {
    float sm = split_ml[s * n_q_heads * 2 + qh * 2];
    global_m = fmaxf(global_m, sm);
  }
  // Broadcast global_m from lane 0 (all lanes should have same value since split_ml is per-head)

  // Compute global_l and accumulate corrected outputs
  float global_l = 0.0f;
  float o[16] = {};

  for (uint32_t s = 0; s < n_splits; s++) {
    float sm = split_ml[s * n_q_heads * 2 + qh * 2];
    float sl = split_ml[s * n_q_heads * 2 + qh * 2 + 1];
    if (sl <= 0.0f) continue; // empty split

    float corr = expf(sm - global_m);
    global_l += corr * sl;

    uint32_t o_base = (s * n_q_heads + qh) * head_dim + tid * ept;
    for (uint32_t i = 0; i < ept; i++)
      o[i] += corr * split_o[o_base + i];
  }

  float l_inv = (global_l > 0.0f) ? (1.0f / global_l) : 0.0f;
  for (uint32_t i = 0; i < ept; i++)
    out[qh * head_dim + tid * ept + i] = __float2half(o[i] * l_inv);
}

// Original single-wave attention (used when seq_len is short enough)
__global__ void __launch_bounds__(WAVE)
kern_attn_decode(uint32_t n_q_heads, uint32_t n_kv_heads, uint32_t head_dim,
                 uint32_t seq_len, uint32_t cache_capacity, uint32_t cache_start,
                 float scale,
                 const __half * __restrict__ q,
                 const __half * __restrict__ k_cache,
                 const __half * __restrict__ v_cache,
                 __half       * __restrict__ out) {
  int qh = blockIdx.x;
  if (qh >= (int)n_q_heads) return;
  int kvh = qh / (n_q_heads / n_kv_heads);
  int tid = threadIdx.x;

  float q_reg[16];
  for (int i = 0; i < (int)head_dim / WAVE; i++)
    q_reg[i] = __half2float(q[qh * head_dim + tid * (head_dim / WAVE) + i]);

  float m = -1e9f, l = 0.0f;
  float o[16] = {};

  for (uint32_t t = 0; t < seq_len; t++) {
    uint32_t phys_t = cache_phys_index(cache_start, cache_capacity, t);
    const __half *kt = k_cache + phys_t * n_kv_heads * head_dim + kvh * head_dim;
    const __half *vt = v_cache + phys_t * n_kv_heads * head_dim + kvh * head_dim;

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

__global__ void __launch_bounds__(WAVE)
kern_attn_split_i8(uint32_t n_q_heads, uint32_t n_kv_heads, uint32_t head_dim,
                   uint32_t seq_len, uint32_t cache_capacity, uint32_t cache_start,
                   float scale,
                   const __half * __restrict__ q,
                   const int8_t * __restrict__ k_cache,
                   const int8_t * __restrict__ v_cache,
                   const float  * __restrict__ k_scale,
                   const float  * __restrict__ v_scale,
                   float        * __restrict__ split_o,
                   float        * __restrict__ split_ml) {
  int qh = blockIdx.x;
  int split_id = blockIdx.y;
  if (qh >= (int)n_q_heads) return;
  int kvh = qh / (n_q_heads / n_kv_heads);
  int tid = threadIdx.x;
  uint32_t ept = head_dim / WAVE;

  uint32_t t_start = split_id * ATTN_SPLIT_SIZE;
  uint32_t t_end = min(t_start + ATTN_SPLIT_SIZE, seq_len);
  if (t_start >= seq_len) {
    if (tid == 0) {
      uint32_t ml_idx = split_id * n_q_heads * 2 + qh * 2;
      split_ml[ml_idx] = -1e9f;
      split_ml[ml_idx + 1] = 0.0f;
    }
    return;
  }

  float q_reg[16];
  for (uint32_t i = 0; i < ept; i++)
    q_reg[i] = __half2float(q[qh * head_dim + tid * ept + i]);

  float m = -1e9f, l = 0.0f;
  float o[16] = {};

  for (uint32_t t = t_start; t < t_end; t++) {
    uint32_t phys_t = cache_phys_index(cache_start, cache_capacity, t);
    const int8_t *kt = k_cache + (size_t)phys_t * n_kv_heads * head_dim + kvh * head_dim;
    const int8_t *vt = v_cache + (size_t)phys_t * n_kv_heads * head_dim + kvh * head_dim;
    float ks = k_scale[phys_t];
    float vs = v_scale[phys_t];

    float dot = 0.0f;
    for (uint32_t i = 0; i < ept; i++)
      dot += q_reg[i] * ((float)kt[tid * ept + i] * ks);
    dot += __shfl_xor(dot, 16); dot += __shfl_xor(dot, 8);
    dot += __shfl_xor(dot, 4);  dot += __shfl_xor(dot, 2); dot += __shfl_xor(dot, 1);

    float s_val = dot * scale;
    float m_new = fmaxf(m, s_val);
    float exp_shift = expf(m - m_new);
    float exp_s = expf(s_val - m_new);
    l = exp_shift * l + exp_s;
    for (uint32_t i = 0; i < ept; i++)
      o[i] = exp_shift * o[i] + exp_s * ((float)vt[tid * ept + i] * vs);
    m = m_new;
  }

  uint32_t o_base = (split_id * n_q_heads + qh) * head_dim + tid * ept;
  for (uint32_t i = 0; i < ept; i++)
    split_o[o_base + i] = o[i];

  if (tid == 0) {
    uint32_t ml_idx = split_id * n_q_heads * 2 + qh * 2;
    split_ml[ml_idx] = m;
    split_ml[ml_idx + 1] = l;
  }
}

__global__ void __launch_bounds__(WAVE)
kern_attn_decode_i8(uint32_t n_q_heads, uint32_t n_kv_heads, uint32_t head_dim,
                    uint32_t seq_len, uint32_t cache_capacity, uint32_t cache_start,
                    float scale,
                    const __half * __restrict__ q,
                    const int8_t * __restrict__ k_cache,
                    const int8_t * __restrict__ v_cache,
                    const float  * __restrict__ k_scale,
                    const float  * __restrict__ v_scale,
                    __half       * __restrict__ out) {
  int qh = blockIdx.x;
  if (qh >= (int)n_q_heads) return;
  int kvh = qh / (n_q_heads / n_kv_heads);
  int tid = threadIdx.x;

  float q_reg[16];
  for (int i = 0; i < (int)head_dim / WAVE; i++)
    q_reg[i] = __half2float(q[qh * head_dim + tid * (head_dim / WAVE) + i]);

  float m = -1e9f, l = 0.0f;
  float o[16] = {};

  for (uint32_t t = 0; t < seq_len; t++) {
    uint32_t phys_t = cache_phys_index(cache_start, cache_capacity, t);
    const int8_t *kt = k_cache + (size_t)phys_t * n_kv_heads * head_dim + kvh * head_dim;
    const int8_t *vt = v_cache + (size_t)phys_t * n_kv_heads * head_dim + kvh * head_dim;
    float ks = k_scale[phys_t];
    float vs = v_scale[phys_t];

    float dot = 0.0f;
    for (int i = 0; i < (int)head_dim / WAVE; i++)
      dot += q_reg[i] * ((float)kt[tid * (head_dim / WAVE) + i] * ks);

    dot += __shfl_xor(dot, 16); dot += __shfl_xor(dot, 8);
    dot += __shfl_xor(dot, 4);  dot += __shfl_xor(dot, 2); dot += __shfl_xor(dot, 1);

    float s_val = dot * scale;
    float m_new = fmaxf(m, s_val);
    float exp_shift = expf(m - m_new);
    float exp_s = expf(s_val - m_new);
    l = exp_shift * l + exp_s;

    for (int i = 0; i < (int)head_dim / WAVE; i++)
      o[i] = exp_shift * o[i] + exp_s * ((float)vt[tid * (head_dim / WAVE) + i] * vs);
    m = m_new;
  }

  float l_inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (int i = 0; i < (int)head_dim / WAVE; i++)
    out[qh * head_dim + tid * (head_dim / WAVE) + i] = __float2half(o[i] * l_inv);
}

// Global Gemma layers use 16:1 GQA. Process all 16 query heads in a single
// block so the shared K/V head is streamed once instead of 16 times.
static constexpr int GLOBAL_GQA_HEADS = 16;
static constexpr int GLOBAL_GQA_BLOCK = GLOBAL_GQA_HEADS * WAVE;

__global__ void __launch_bounds__(GLOBAL_GQA_BLOCK)
kern_attn_split_gqa_q4(uint32_t head_dim,
                       uint32_t seq_len,
                       uint32_t cache_capacity,
                       uint32_t cache_start,
                       float scale,
                       const __half * __restrict__ q,
                       const uint8_t * __restrict__ k_cache,
                       const uint8_t * __restrict__ v_cache,
                       const float   * __restrict__ k_scale,
                       const float   * __restrict__ v_scale,
                       float         * __restrict__ split_o,
                       float         * __restrict__ split_ml) {
  int split_id = blockIdx.x;
  int tid = threadIdx.x;
  int qh = tid / WAVE;
  int lane = tid & (WAVE - 1);
  if (qh >= GLOBAL_GQA_HEADS) return;

  uint32_t ept = head_dim / WAVE;
  uint32_t t_start = split_id * ATTN_SPLIT_SIZE;
  uint32_t t_end = min(t_start + ATTN_SPLIT_SIZE, seq_len);
  if (t_start >= seq_len) {
    if (lane == 0) {
      uint32_t ml_idx = split_id * GLOBAL_GQA_HEADS * 2 + qh * 2;
      split_ml[ml_idx] = -1e9f;
      split_ml[ml_idx + 1] = 0.0f;
    }
    return;
  }

  float q_reg[16];
  for (uint32_t i = 0; i < ept; i++)
    q_reg[i] = __half2float(q[qh * head_dim + lane * ept + i]);

  float m = -1e9f;
  float l = 0.0f;
  float o[16] = {};

  for (uint32_t t = t_start; t < t_end; t++) {
    uint32_t phys_t = cache_phys_index(cache_start, cache_capacity, t);
    const uint8_t *kt = k_cache + (size_t)phys_t * (head_dim / 2);
    const uint8_t *vt = v_cache + (size_t)phys_t * (head_dim / 2);
    float ks = k_scale[phys_t];
    float vs = v_scale[phys_t];

    float dot = 0.0f;
    for (uint32_t i = 0; i < ept; i++) {
      uint32_t idx = lane * ept + i;
      dot += q_reg[i] * ((float)unpack_q4_signed(kt, idx) * ks);
    }
    dot += __shfl_xor(dot, 16); dot += __shfl_xor(dot, 8);
    dot += __shfl_xor(dot, 4);  dot += __shfl_xor(dot, 2); dot += __shfl_xor(dot, 1);

    float s_val = dot * scale;
    float m_new = fmaxf(m, s_val);
    float exp_shift = expf(m - m_new);
    float exp_s = expf(s_val - m_new);
    l = exp_shift * l + exp_s;
    for (uint32_t i = 0; i < ept; i++) {
      uint32_t idx = lane * ept + i;
      o[i] = exp_shift * o[i] + exp_s * ((float)unpack_q4_signed(vt, idx) * vs);
    }
    m = m_new;
  }

  uint32_t o_base = (split_id * GLOBAL_GQA_HEADS + qh) * head_dim + lane * ept;
  for (uint32_t i = 0; i < ept; i++)
    split_o[o_base + i] = o[i];

  if (lane == 0) {
    uint32_t ml_idx = split_id * GLOBAL_GQA_HEADS * 2 + qh * 2;
    split_ml[ml_idx] = m;
    split_ml[ml_idx + 1] = l;
  }
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

    #pragma unroll
    for (int i = 0; i < 8; i++) {
      int j = base + i;
      int group = j / 128;
      int jg = j % 128;
      int l = jg % 32;
      int quart = jg / 32;
      int ql_idx = group * 64 + l + ((quart & 1) ? 32 : 0);
      uint8_t ql_byte = b.ql[ql_idx];
      uint8_t ql_val = (quart >= 2) ? ((ql_byte >> 4) & 0xF) : (ql_byte & 0xF);
      uint8_t qh_val = (b.qh[group * 32 + l] >> (quart * 2)) & 0x3;
      int8_t q = (int8_t)((ql_val | (qh_val << 4)) - 32);
      int scale_idx = group * 8 + (l / 16) + ((quart & 2) ? 4 : 0) + ((quart & 1) ? 2 : 0);
      float s = d * (float)b.scales[scale_idx];
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
  int group = j / 128;
  int jg = j % 128;
  int l = jg % 32;
  int quart = jg / 32;
  int ql_idx = group * 64 + l + ((quart & 1) ? 32 : 0);
  uint8_t ql_byte = b.ql[ql_idx];
  uint8_t ql_val = (quart >= 2) ? ((ql_byte >> 4) & 0xF) : (ql_byte & 0xF);
  uint8_t qh_val = (b.qh[group * 32 + l] >> (quart * 2)) & 0x3;
  int8_t q = (int8_t)((ql_val | (qh_val << 4)) - 32);
  int scale_idx = group * 8 + (l / 16) + ((quart & 2) ? 4 : 0) + ((quart & 1) ? 2 : 0);
  float scale = d * (float)b.scales[scale_idx];
  out[tid] = __float2half((float)q * scale * embed_scale);
}

// Indirect embedding lookup: reads token ID from device memory (for graph/feedback loop)
__global__ void __launch_bounds__(256)
kern_embed_lookup_indirect(const uint32_t * __restrict__ token_ptr, uint32_t dim, float embed_scale,
                           const Q6_K_Block * __restrict__ embed_q6k,
                           __half * __restrict__ out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t token = *token_ptr;
  int blocks_per_row = dim / 256;
  const Q6_K_Block *row = embed_q6k + (size_t)token * blocks_per_row;

  if (tid >= (int)dim) return;
  int blk_idx = tid / 256;
  int j = tid % 256;
  const Q6_K_Block &b = row[blk_idx];
  float d = __half2float(b.d);
  int group = j / 128;
  int jg = j % 128;
  int l = jg % 32;
  int quart = jg / 32;
  int ql_idx = group * 64 + l + ((quart & 1) ? 32 : 0);
  uint8_t ql_byte = b.ql[ql_idx];
  uint8_t ql_val = (quart >= 2) ? ((ql_byte >> 4) & 0xF) : (ql_byte & 0xF);
  uint8_t qh_val = (b.qh[group * 32 + l] >> (quart * 2)) & 0x3;
  int8_t q = (int8_t)((ql_val | (qh_val << 4)) - 32);
  int scale_idx = group * 8 + (l / 16) + ((quart & 2) ? 4 : 0) + ((quart & 1) ? 2 : 0);
  float scale = d * (float)b.scales[scale_idx];
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

// Fused: RMSNorm per head + RoPE with a single precomputed row.
__global__ void __launch_bounds__(WAVE)
kern_headnorm_rope_row(uint32_t n_heads, uint32_t head_dim,
                       const float * __restrict__ norm_weights,
                       const float * __restrict__ cos_row,
                       const float * __restrict__ sin_row,
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

// Prefill path: compute RoPE on the fly so long context does not require
// max_seq-sized tables in VRAM.
__global__ void __launch_bounds__(WAVE)
kern_headnorm_rope_dynamic(uint32_t n_heads, uint32_t head_dim, uint32_t pos,
                           float freq_base,
                           const float * __restrict__ freq_factors,
                           const float * __restrict__ norm_weights,
                           float eps,
                           __half * __restrict__ x) {
  int head = blockIdx.x;
  if (head >= (int)n_heads) return;
  int tid = threadIdx.x;
  __half *hx = x + head * head_dim;

  float ss = 0.0f;
  for (uint32_t i = tid; i < head_dim; i += WAVE) {
    float v = __half2float(hx[i]);
    ss += v * v;
  }
  ss += __shfl_xor(ss, 16); ss += __shfl_xor(ss, 8);
  ss += __shfl_xor(ss, 4);  ss += __shfl_xor(ss, 2); ss += __shfl_xor(ss, 1);
  float rms_inv = rsqrtf(ss / (float)head_dim + eps);

  uint32_t half_dim = head_dim / 2;
  for (uint32_t i = tid; i < head_dim; i += WAVE)
    hx[i] = __float2half(__half2float(hx[i]) * rms_inv * norm_weights[i % head_dim]);

  for (uint32_t i = tid; i < half_dim; i += WAVE) {
    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
    if (freq_factors) freq /= freq_factors[i];
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);
    float x0 = __half2float(hx[i]);
    float x1 = __half2float(hx[i + half_dim]);
    hx[i]            = __float2half(x0 * cos_a - x1 * sin_a);
    hx[i + half_dim] = __float2half(x0 * sin_a + x1 * cos_a);
  }
}

// ============================================================================
// Software WMMA for gfx1032 (RDNA2): 16x16x16 fp16→f32
// Uses DPP8 v_dot2c_f32_f16 + permlanex16 + ds_bpermute
// ============================================================================

using half2_vec __attribute__((ext_vector_type(2))) = _Float16;

__device__ static inline float wmma_dot2_dpp8(half2_vec a, half2_vec b, float acc, uint32_t lane) {
  // Unrolled DPP8 broadcast for each of 8 lanes in the subgroup
  // The lane parameter selects which src lane's data to broadcast
  float result = acc;
  switch (lane) {
    case 0: asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[0,0,0,0,0,0,0,0]"
              : "=v"(result) : "v"(a), "v"(b), "0"(result)); break;
    case 1: asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[1,1,1,1,1,1,1,1]"
              : "=v"(result) : "v"(a), "v"(b), "0"(result)); break;
    case 2: asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[2,2,2,2,2,2,2,2]"
              : "=v"(result) : "v"(a), "v"(b), "0"(result)); break;
    case 3: asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[3,3,3,3,3,3,3,3]"
              : "=v"(result) : "v"(a), "v"(b), "0"(result)); break;
    case 4: asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[4,4,4,4,4,4,4,4]"
              : "=v"(result) : "v"(a), "v"(b), "0"(result)); break;
    case 5: asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[5,5,5,5,5,5,5,5]"
              : "=v"(result) : "v"(a), "v"(b), "0"(result)); break;
    case 6: asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[6,6,6,6,6,6,6,6]"
              : "=v"(result) : "v"(a), "v"(b), "0"(result)); break;
    case 7: asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[7,7,7,7,7,7,7,7]"
              : "=v"(result) : "v"(a), "v"(b), "0"(result)); break;
  }
  return result;
}

__device__ static inline uint32_t wmma_bpermute(uint32_t data, uint32_t srcLane) {
  return __builtin_amdgcn_ds_bpermute(srcLane << 2, data);
}

__device__ static inline uint32_t wmma_permlanex16(uint32_t data) {
  return __builtin_amdgcn_permlanex16(0u, data, 0x76543210u, 0xfedcba98u, false, true);
}

// Core 16x16x16 WMMA: C[16×16] += A[16×16] × B[16×16]
// A is row-major in LDS: a_lds[row][col], each row = 16 fp16
// B is col-major in LDS: b_lds[col][row], each col = 16 fp16
// C is distributed: each lane holds 8 f32 accumulators
// Lane mapping: lanes 0-15 = rows 0-15 of C (even cols), lanes 16-31 = rows 0-15 (odd cols)
__device__ static inline void wmma_16x16x16_f16(
    const __half *a_lds,  // [16][16] row-major
    const __half *b_lds,  // [16][16] col-major (i.e. b_lds[col*16 + row])
    float *acc            // [8] per lane
) {
  uint32_t lid = threadIdx.x & 31u;
  uint32_t lowHalf = lid < 16u;
  uint32_t laneInHalf = lid & 15u;

  // Each lane loads its row of A (16 fp16 = 8 pairs) and its col of B
  uint32_t a_row = laneInHalf;
  uint32_t b_col = laneInHalf;

  // Pack A row into 8 half2 pairs
  uint32_t aPairs[8];
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const __half *ap = a_lds + a_row * 16 + i * 2;
    aPairs[i] = *reinterpret_cast<const uint32_t*>(ap);
  }

  // Pack B col into 8 half2 pairs
  uint32_t bPairs[8];
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const __half *bp = b_lds + b_col * 16 + i * 2;
    bPairs[i] = *reinterpret_cast<const uint32_t*>(bp);
  }

  // Exchange A rows between halves: low half needs rows from high half's lanes and vice versa
  uint32_t aOther[8];
  #pragma unroll
  for (int i = 0; i < 8; i++)
    aOther[i] = wmma_permlanex16(aPairs[i]);

  // Each lane needs the A row corresponding to B's column index within its 8-lane DPP group.
  // Low half: my B col is laneInHalf, I need A rows 0-7 from low half, 8-15 from high half
  // High half: my B col is laneInHalf, I need A rows 0-7 from low half, 8-15 from high half
  // DPP8 broadcasts within groups of 8 lanes. Within each 8-lane group, lane j holds A row j (or j+8).
  // Use bpermute to rearrange so that within each 8-lane DPP subgroup, lane j has A row j.

  uint32_t myGroup = (lid & 7u); // 0-7 within 8-lane group
  // For the DPP8 pattern: lane j in the 8-group will broadcast its A, computing dot(A_row_j, B_col_mine)
  // So each lane just needs its own A row — but in the right register for bpermute to distribute
  // Actually: we need A_row_j available at lane j within each 8-lane group.
  // Low-half lanes 0-7 naturally have A rows 0-7. Lanes 8-15 have A rows 8-15.
  // The DPP8 pattern [j,j,j,j,j,j,j,j] broadcasts lane j's value to all 8 lanes in the subgroup.
  // So for computing C[myRow][col], where col iterates 0-7 within the DPP subgroup:
  // accum[col] += A[myRow][k_pair] * B[col_of_lane_j][k_pair]  via DPP8 broadcast of B

  // Wait — rethinking. The standard approach:
  // Each lane holds B column data. DPP8 broadcasts B from lane j to all lanes in the 8-group.
  // Each lane has its own A row. So: acc[j] += dot2(myA[pair], broadcast_from_j(B[pair]))
  // This gives each lane: C[myRow][col_j] for j in the 8-lane group.

  // But we have 32 lanes and 16 columns. Low half covers cols 0-7 of each row, high half covers 8-15.
  // Lane l (l<16): row = l, accumulates cols 0-7 via DPP8 within lanes 0-7
  //                        accumulates cols 8-15 via cross-half exchange + DPP8 within lanes 8-15
  // Hmm, this is getting complex. Let me use a simpler decomposition:

  // Simpler: lanes 0-15 = "group A" (rows 0-15), lanes 16-31 = "group B" (also rows 0-15 but different cols)
  // Group A lanes 0-7: each lane j holds B col j, accumulates C[row][0..7]
  // Group A lanes 8-15: each lane j holds B col j, accumulates C[row][8..15]

  // With DPP8, in lanes 0-7: broadcast lane j's B → all get B[col_j], multiply with own A → C[row][col_j]
  // Same for lanes 8-15.
  // But wait, lanes 0-7 would only produce 8 columns. We need 16 columns per row.
  // Solution: lanes 0-15 produce cols 0-7 (using DPP within their 8-lane subgroups: 0-7 and 8-15)
  //           lanes 16-31 produce cols 8-15

  // Actually this doesn't work cleanly either because DPP8 operates on fixed 8-lane groups (0-7, 8-15, 16-23, 24-31).
  // Lane 0 is in DPP group 0 (lanes 0-7), lane 8 is in DPP group 1 (lanes 8-15), etc.

  // The rocWMMA approach: use ds_bpermute to redistribute A rows so that within each DPP8 group,
  // lane j has the A row needed. Then DPP8 broadcasts A from lane j, each lane has its own B, computes dot.
  // Final: permlanex16 to exchange partial sums between halves.

  // Let me just follow rocWMMA's proven algorithm exactly:
  // 1. Low half (lanes 0-15): each lane l has A[l] and B[l]
  // 2. High half (lanes 16-31): each lane l has A[l-16] and B[l-16]  (same rows/cols)
  // 3. We use permlanex16 to get the other half's A data
  // 4. Combine: low half builds aPairs from (own low, other high), high from (other low, own high)
  //    This gives each half the full 16×16 A matrix distributed across 16 lanes
  // 5. bpermute: each lane fetches A[col_within_8group] from the appropriate lane
  // 6. DPP8: iterate over 8 columns, broadcasting A[row_j] while using own B[col]
  // 7. permlanex16: exchange results between halves, accumulate

  // Following rocWMMA's select_half2_bits: interleave low/high pairs
  uint32_t aFull[8], bFull[8];
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    uint32_t lo = aPairs[i];
    uint32_t hi = aOther[i];
    // Low half takes own low 16 bits + other's high 16 bits (column pairs interleaved)
    aFull[i] = lowHalf ? ((lo & 0xFFFF) | (hi & 0xFFFF0000u))
                       : ((hi & 0xFFFF) | (lo & 0xFFFF0000u));
    // B: same exchange
    uint32_t bOth = wmma_permlanex16(bPairs[i]);
    bFull[i] = lowHalf ? ((bPairs[i] & 0xFFFF) | (bOth & 0xFFFF0000u))
                       : ((bOth & 0xFFFF) | (bPairs[i] & 0xFFFF0000u));
  }

  // bpermute: get A rows for DPP8 subgroup matching
  // Within the 8-lane DPP subgroup, we'll broadcast A from each lane.
  // Each lane needs A data rearranged so lane j in the 8-group has A row j (or j+8).
  uint32_t srcLane = (lid & 7u) + (lowHalf ? 0u : 8u);
  uint32_t aRows[8];
  #pragma unroll
  for (int i = 0; i < 8; i++)
    aRows[i] = wmma_bpermute(aFull[i], srcLane);

  // DPP8 core: 8 pairs × 8 broadcasts = 64 FMAs (= 128 multiplies for the 16×16×16)
  float localAcc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  #pragma unroll
  for (int pair = 0; pair < 8; pair++) {
    half2_vec aP = reinterpret_cast<half2_vec&>(aRows[pair]);
    half2_vec bP = reinterpret_cast<half2_vec&>(bFull[pair]);
    // DPP8 broadcasts A from lane j, each lane multiplies with its own B
    asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[0,0,0,0,0,0,0,0]"
                 : "=v"(localAcc[0]) : "v"(aP), "v"(bP), "0"(localAcc[0]));
    asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[1,1,1,1,1,1,1,1]"
                 : "=v"(localAcc[1]) : "v"(aP), "v"(bP), "0"(localAcc[1]));
    asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[2,2,2,2,2,2,2,2]"
                 : "=v"(localAcc[2]) : "v"(aP), "v"(bP), "0"(localAcc[2]));
    asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[3,3,3,3,3,3,3,3]"
                 : "=v"(localAcc[3]) : "v"(aP), "v"(bP), "0"(localAcc[3]));
    asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[4,4,4,4,4,4,4,4]"
                 : "=v"(localAcc[4]) : "v"(aP), "v"(bP), "0"(localAcc[4]));
    asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[5,5,5,5,5,5,5,5]"
                 : "=v"(localAcc[5]) : "v"(aP), "v"(bP), "0"(localAcc[5]));
    asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[6,6,6,6,6,6,6,6]"
                 : "=v"(localAcc[6]) : "v"(aP), "v"(bP), "0"(localAcc[6]));
    asm volatile("v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[7,7,7,7,7,7,7,7]"
                 : "=v"(localAcc[7]) : "v"(aP), "v"(bP), "0"(localAcc[7]));
  }

  // Exchange partial results between halves
  uint32_t partnerAcc[8];
  #pragma unroll
  for (int i = 0; i < 8; i++)
    partnerAcc[i] = wmma_permlanex16(reinterpret_cast<uint32_t&>(localAcc[i]));

  // Accumulate: low half gets even-indexed results, high half gets odd-indexed
  if (lowHalf) {
    acc[0] += localAcc[0];
    acc[1] += localAcc[2];
    acc[2] += localAcc[4];
    acc[3] += localAcc[6];
    acc[4] += reinterpret_cast<float&>(partnerAcc[0]);
    acc[5] += reinterpret_cast<float&>(partnerAcc[2]);
    acc[6] += reinterpret_cast<float&>(partnerAcc[4]);
    acc[7] += reinterpret_cast<float&>(partnerAcc[6]);
  } else {
    acc[0] += reinterpret_cast<float&>(partnerAcc[1]);
    acc[1] += reinterpret_cast<float&>(partnerAcc[3]);
    acc[2] += reinterpret_cast<float&>(partnerAcc[5]);
    acc[3] += reinterpret_cast<float&>(partnerAcc[7]);
    acc[4] += localAcc[1];
    acc[5] += localAcc[3];
    acc[6] += localAcc[5];
    acc[7] += localAcc[7];
  }
}

// ============================================================================
// GEMM kernel using software WMMA for batched prefill
// Y[M×N] = X[M×K] × W[N×K]^T   (W stored row-major as Q4_0 blocks, transposed for GEMM)
// Each wavefront computes one 16×16 output tile. M must be multiple of 16.
// Grid: (N/16, M/16), Block: (32,)
// ============================================================================

// LDS tile: 16×16 fp16 = 512 bytes. Two tiles (A + B) = 1 KB per wavefront.
// With 4 waves per CU: 4 KB LDS — easily fits within 32 KB LDS budget.
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

__global__ void __launch_bounds__(WAVE)
kern_gemm_q4_0_wmma(int M, int N, int K,
                    const __half    * __restrict__ X,     // [M × K] row-major fp16
                    const Q4_0Block * __restrict__ W,     // [N × K/32] Q4_0 blocks (row per output)
                    __half          * __restrict__ Y) {   // [M × N] row-major fp16
  // Output tile position
  int tile_n = blockIdx.x;  // which 16-col tile of output
  int tile_m = blockIdx.y;  // which 16-row tile of output
  int out_row_base = tile_m * WMMA_M;
  int out_col_base = tile_n * WMMA_N;

  if (out_col_base + WMMA_N > N || out_row_base + WMMA_M > M) return;

  int tid = threadIdx.x;

  // LDS for staging dequanted tiles
  __shared__ __half a_tile[WMMA_M * WMMA_K];  // X tile: 16 rows × 16 cols
  __shared__ __half b_tile[WMMA_N * WMMA_K];  // W tile: 16 cols × 16 rows (col-major for WMMA)

  // Accumulators: 8 f32 per lane
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  int num_k_tiles = K / WMMA_K;

  for (int kt = 0; kt < num_k_tiles; kt++) {
    int k_base = kt * WMMA_K;

    // Load A tile: X[out_row_base + row][k_base + col] → a_tile[row][col]
    // 32 threads, 256 elements to load = 8 per thread
    #pragma unroll
    for (int i = tid; i < WMMA_M * WMMA_K; i += WAVE) {
      int row = i / WMMA_K;
      int col = i % WMMA_K;
      a_tile[i] = X[(out_row_base + row) * K + k_base + col];
    }

    // Load B tile: dequant W[out_col_base + col][k_base .. k_base+15] → b_tile[col * 16 + k_offset]
    // Each Q4_0 block has 32 elements. k_base might not align to block boundary.
    // k_base/32 = which block, k_base%32 = offset within block
    // For 16 elements starting at k_base: might span 1 or 2 Q4_0 blocks
    #pragma unroll
    for (int i = tid; i < WMMA_N * WMMA_K; i += WAVE) {
      int col = i / WMMA_K;  // which output row of W (0-15)
      int k_off = i % WMMA_K;  // which element within the 16-wide K chunk

      int w_row = out_col_base + col;
      int k_idx = k_base + k_off;
      int blk_idx = k_idx / Q4_BLOCK;
      int elem_in_blk = k_idx % Q4_BLOCK;

      const Q4_0Block &blk = W[(long)w_row * (K / Q4_BLOCK) + blk_idx];
      float scale = __half2float(blk.scale);
      uint8_t packed = blk.qs[elem_in_blk & 15];
      float val = (elem_in_blk < 16) ? scale * ((float)(packed & 0xF) - 8.0f)
                                     : scale * ((float)(packed >> 4) - 8.0f);
      // Store col-major for WMMA B: b_tile[col][k_off]
      b_tile[col * WMMA_K + k_off] = __float2half(val);
    }

    __syncthreads();

    // Software WMMA: acc += a_tile × b_tile
    wmma_16x16x16_f16(a_tile, b_tile, acc);

    __syncthreads();
  }

  // Store results: each lane has 8 elements of C
  // Lane mapping: lid < 16 → row = lid, cols depend on acc index
  // lid >= 16 → row = lid - 16, cols offset by 8
  uint32_t lid = tid & 31u;
  uint32_t row = lid & 15u;
  uint32_t col_base_out = (lid < 16u) ? 0u : 8u;

  #pragma unroll
  for (int i = 0; i < 8; i++) {
    int out_r = out_row_base + row;
    int out_c = out_col_base + col_base_out + i;
    if (out_r < M && out_c < N)
      Y[out_r * N + out_c] = __float2half(acc[i]);
  }
}

// Batched GEMV using WMMA: for when M < 16, pad to 16 and use GEMM kernel
// For small batch (1-15 tokens), this still outperforms N separate GEMVs
// because the weight read is amortized.
__global__ void __launch_bounds__(WAVE)
kern_gemm_q4_0_wmma_padded(int M_real, int N, int K,
                           const __half    * __restrict__ X,
                           const Q4_0Block * __restrict__ W,
                           __half          * __restrict__ Y) {
  int tile_n = blockIdx.x;
  int out_col_base = tile_n * WMMA_N;
  if (out_col_base + WMMA_N > N) return;

  int tid = threadIdx.x;

  __shared__ __half a_tile[WMMA_M * WMMA_K];
  __shared__ __half b_tile[WMMA_N * WMMA_K];

  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  int num_k_tiles = K / WMMA_K;

  for (int kt = 0; kt < num_k_tiles; kt++) {
    int k_base = kt * WMMA_K;

    #pragma unroll
    for (int i = tid; i < WMMA_M * WMMA_K; i += WAVE) {
      int row = i / WMMA_K;
      int col = i % WMMA_K;
      a_tile[i] = (row < M_real) ? X[row * K + k_base + col] : __float2half(0.0f);
    }

    #pragma unroll
    for (int i = tid; i < WMMA_N * WMMA_K; i += WAVE) {
      int col = i / WMMA_K;
      int k_off = i % WMMA_K;
      int w_row = out_col_base + col;
      int k_idx = k_base + k_off;
      int blk_idx = k_idx / Q4_BLOCK;
      int elem_in_blk = k_idx % Q4_BLOCK;
      const Q4_0Block &blk = W[(long)w_row * (K / Q4_BLOCK) + blk_idx];
      float scale = __half2float(blk.scale);
      uint8_t packed = blk.qs[elem_in_blk & 15];
      float val = (elem_in_blk < 16) ? scale * ((float)(packed & 0xF) - 8.0f)
                                     : scale * ((float)(packed >> 4) - 8.0f);
      b_tile[col * WMMA_K + k_off] = __float2half(val);
    }

    __syncthreads();
    wmma_16x16x16_f16(a_tile, b_tile, acc);
    __syncthreads();
  }

  uint32_t lid = tid & 31u;
  uint32_t row = lid & 15u;
  uint32_t col_base_out = (lid < 16u) ? 0u : 8u;

  if ((int)row < M_real) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      int out_c = out_col_base + col_base_out + i;
      if (out_c < N)
        Y[row * N + out_c] = __float2half(acc[i]);
    }
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
  uint32_t attn_window;
  uint32_t kv_capacity;
  bool     has_kv;    // has own K/V projection (non-shared layers)
  bool     has_v;     // has separate V weight (vs using K as V)
  bool     kv_quant_i8;
  bool     kv_quant_q4;

  // Layer-local KV storage
  __half *kv_k;
  __half *kv_v;
  int8_t *kv_k_i8;
  int8_t *kv_v_i8;
  uint8_t *kv_k_q4;
  uint8_t *kv_v_q4;
  float  *kv_k_scale;
  float  *kv_v_scale;
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

  // Per-token RoPE rows (GPU) reused across all layers of the same type
  float *rope_cos_swa_row;    // [128]
  float *rope_sin_swa_row;    // [128]
  float *rope_cos_global_row; // [256]
  float *rope_sin_global_row; // [256]
  uint32_t max_seq;
  uint32_t sliding_window;

  // Split attention scratch buffers
  float *attn_split_o;   // [max_splits × max_q_heads × max_head_dim]
  float *attn_split_ml;  // [max_splits × max_q_heads × 2]
};

static uint32_t getenv_u32(const char *name, uint32_t default_value) {
  const char *v = getenv(name);
  if (!v || !*v) return default_value;
  char *end = nullptr;
  unsigned long parsed = strtoul(v, &end, 10);
  if (!end || *end != '\0' || parsed == 0 || parsed > 0xffffffffUL)
    return default_value;
  return (uint32_t)parsed;
}

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
      int group = j / 128;
      int jg = j % 128;
      int l = jg % 32;
      int quart = jg / 32;
      int ql_idx = group * 64 + l + ((quart & 1) ? 32 : 0);
      uint8_t ql_byte = blk.ql[ql_idx];
      uint8_t ql_val = (quart >= 2) ? ((ql_byte >> 4) & 0xF) : (ql_byte & 0xF);
      uint8_t qh_val = (blk.qh[group * 32 + l] >> (quart * 2)) & 0x3;
      int8_t q = (int8_t)((ql_val | (qh_val << 4)) - 32);
      int scale_idx = group * 8 + (l / 16) + ((quart & 2) ? 4 : 0) + ((quart & 1) ? 2 : 0);
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
  model->max_seq = 65536;
  model->sliding_window = 1024;

  if (gguf->kv_float.count("gemma4.final_logit_softcapping"))
    model->final_logit_softcap = gguf->kv_float["gemma4.final_logit_softcapping"];
  if (gguf->kv_float.count("gemma4.rope.freq_base"))
    model->rope_freq_base = gguf->kv_float["gemma4.rope.freq_base"];
  if (gguf->kv_float.count("gemma4.rope.freq_base_swa"))
    model->rope_freq_base_swa = gguf->kv_float["gemma4.rope.freq_base_swa"];
  if (gguf->kv_uint.count("gemma4.attention.sliding_window"))
    model->sliding_window = (uint32_t)gguf->kv_uint["gemma4.attention.sliding_window"];
  if (gguf->kv_uint.count("gemma4.context_length"))
    model->max_seq = (uint32_t)std::min<uint64_t>(gguf->kv_uint["gemma4.context_length"], model->max_seq);
  model->max_seq = getenv_u32("THEROCK_MAX_SEQ", model->max_seq);

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

  printf("Loading %u layers...\n", N_LAYERS);
  bool force_fp16_kv = getenv("THEROCK_FORCE_FP16_KV") != nullptr;
  for (uint32_t l = 0; l < N_LAYERS; l++) {
    LayerWeights &lw = model->layers[l];
    char name[128];

    lw.is_swa = !is_global(l);
    lw.attn_window = lw.is_swa ? model->sliding_window : model->max_seq;
    lw.has_kv = true;  // all layers in 12B have their own KV
    lw.head_dim = lw.is_swa ? 256 : 512;
    lw.n_q_heads = 16;
    lw.n_kv_heads = lw.is_swa ? 8 : 1;
    lw.q_dim = lw.n_q_heads * lw.head_dim;
    lw.kv_dim = lw.n_kv_heads * lw.head_dim;
    lw.kv_capacity = lw.is_swa ? std::min(model->max_seq, lw.attn_window) : model->max_seq;
    lw.kv_quant_i8 = lw.is_swa && !force_fp16_kv;
    lw.kv_quant_q4 = !lw.is_swa && !force_fp16_kv;
    lw.kv_k = nullptr;
    lw.kv_v = nullptr;
    lw.kv_k_i8 = nullptr;
    lw.kv_v_i8 = nullptr;
    lw.kv_k_q4 = nullptr;
    lw.kv_v_q4 = nullptr;
    lw.kv_k_scale = nullptr;
    lw.kv_v_scale = nullptr;

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
    size_t kv_elems = (size_t)lw.kv_capacity * lw.kv_dim;
    if (lw.kv_quant_q4) {
      size_t kv_bytes = (kv_elems / 2) * sizeof(uint8_t);
      size_t scale_bytes = (size_t)lw.kv_capacity * sizeof(float);
      HIP_CHECK(hipMalloc(&lw.kv_k_q4, kv_bytes));
      HIP_CHECK(hipMalloc(&lw.kv_v_q4, kv_bytes));
      HIP_CHECK(hipMalloc(&lw.kv_k_scale, scale_bytes));
      HIP_CHECK(hipMalloc(&lw.kv_v_scale, scale_bytes));
      HIP_CHECK(hipMemset(lw.kv_k_q4, 0, kv_bytes));
      HIP_CHECK(hipMemset(lw.kv_v_q4, 0, kv_bytes));
      HIP_CHECK(hipMemset(lw.kv_k_scale, 0, scale_bytes));
      HIP_CHECK(hipMemset(lw.kv_v_scale, 0, scale_bytes));
      total_gpu_bytes += kv_bytes * 2 + scale_bytes * 2;
    } else if (lw.kv_quant_i8) {
      size_t kv_bytes = kv_elems * sizeof(int8_t);
      size_t scale_bytes = (size_t)lw.kv_capacity * sizeof(float);
      HIP_CHECK(hipMalloc(&lw.kv_k_i8, kv_bytes));
      HIP_CHECK(hipMalloc(&lw.kv_v_i8, kv_bytes));
      HIP_CHECK(hipMalloc(&lw.kv_k_scale, scale_bytes));
      HIP_CHECK(hipMalloc(&lw.kv_v_scale, scale_bytes));
      HIP_CHECK(hipMemset(lw.kv_k_i8, 0, kv_bytes));
      HIP_CHECK(hipMemset(lw.kv_v_i8, 0, kv_bytes));
      HIP_CHECK(hipMemset(lw.kv_k_scale, 0, scale_bytes));
      HIP_CHECK(hipMemset(lw.kv_v_scale, 0, scale_bytes));
      total_gpu_bytes += kv_bytes * 2 + scale_bytes * 2;
    } else {
      size_t kv_bytes = kv_elems * sizeof(__half);
      HIP_CHECK(hipMalloc(&lw.kv_k, kv_bytes));
      HIP_CHECK(hipMalloc(&lw.kv_v, kv_bytes));
      HIP_CHECK(hipMemset(lw.kv_k, 0, kv_bytes));
      HIP_CHECK(hipMemset(lw.kv_v, 0, kv_bytes));
      total_gpu_bytes += kv_bytes * 2;
    }

    if (l % 12 == 0 || l == N_LAYERS - 1)
      printf("  Layer %u: q_dim=%u kv_dim=%u head_dim=%u %s%s cap=%u %s [%.0f MB GPU so far]\n",
             l, lw.q_dim, lw.kv_dim, lw.head_dim,
             lw.is_swa ? "SWA" : "GLOBAL",
             lw.has_v ? "" : " (K=V)",
             lw.kv_capacity,
             lw.kv_quant_q4 ? "q4-kv" : (lw.kv_quant_i8 ? "int8-kv" : "fp16-kv"),
             total_gpu_bytes / 1e6);
  }

  // Allocate per-token RoPE rows. Decode fills these once per token and reuses
  // them across all layers of the same type.
  printf("Allocating RoPE row buffers...\n");
  {
    HIP_CHECK(hipMalloc(&model->rope_cos_swa_row, 128 * sizeof(float)));
    HIP_CHECK(hipMalloc(&model->rope_sin_swa_row, 128 * sizeof(float)));
    HIP_CHECK(hipMalloc(&model->rope_cos_global_row, 256 * sizeof(float)));
    HIP_CHECK(hipMalloc(&model->rope_sin_global_row, 256 * sizeof(float)));
    total_gpu_bytes += (128 + 128 + 256 + 256) * sizeof(float);
  }

  // Split attention scratch: max_splits = max_seq / ATTN_SPLIT_SIZE + 1
  {
    uint32_t max_splits = (model->max_seq + ATTN_SPLIT_SIZE - 1) / ATTN_SPLIT_SIZE;
    uint32_t max_q_heads = 16;
    uint32_t max_head_dim = 512;
    size_t split_o_bytes = (size_t)max_splits * max_q_heads * max_head_dim * sizeof(float);
    size_t split_ml_bytes = (size_t)max_splits * max_q_heads * 2 * sizeof(float);
    HIP_CHECK(hipMalloc(&model->attn_split_o, split_o_bytes));
    HIP_CHECK(hipMalloc(&model->attn_split_ml, split_ml_bytes));
    total_gpu_bytes += split_o_bytes + split_ml_bytes;
  }

  printf("Total GPU memory: %.2f GB\n", total_gpu_bytes / 1e9);
  return true;
}

// Dispatch attention: uses split-K for long sequences, single wave for short
static constexpr uint32_t ATTN_SPLIT_THRESHOLD = 64;

static inline uint32_t layer_seq_len(const LayerWeights &lw, uint32_t pos) {
  return lw.is_swa ? std::min(pos + 1, lw.attn_window) : (pos + 1);
}

static inline uint32_t layer_cache_start(const LayerWeights &lw, uint32_t pos, uint32_t seq_len) {
  if (!lw.is_swa || lw.kv_capacity == 0)
    return 0;
  return (pos + 1 - seq_len) % lw.kv_capacity;
}

static inline uint32_t layer_store_index(const LayerWeights &lw, uint32_t pos) {
  return lw.is_swa ? (pos % lw.kv_capacity) : pos;
}

static void dispatch_attention_fp16(hipStream_t s, Model *model,
                                    uint32_t n_q_heads, uint32_t n_kv_heads, uint32_t head_dim,
                                    uint32_t seq_len, uint32_t cache_capacity, uint32_t cache_start,
                                    float scale,
                                    const __half *q, const __half *k_cache, const __half *v_cache,
                                    __half *out) {
  if (seq_len <= ATTN_SPLIT_THRESHOLD) {
    hipLaunchKernelGGL(kern_attn_decode, dim3(n_q_heads), dim3(WAVE), 0, s,
                       n_q_heads, n_kv_heads, head_dim, seq_len, cache_capacity, cache_start, scale,
                       q, k_cache, v_cache, out);
  } else {
    uint32_t n_splits = (seq_len + ATTN_SPLIT_SIZE - 1) / ATTN_SPLIT_SIZE;
    hipLaunchKernelGGL(kern_attn_split, dim3(n_q_heads, n_splits), dim3(WAVE), 0, s,
                       n_q_heads, n_kv_heads, head_dim, seq_len, cache_capacity, cache_start, scale,
                       q, k_cache, v_cache, model->attn_split_o, model->attn_split_ml);
    hipLaunchKernelGGL(kern_attn_reduce, dim3(n_q_heads), dim3(WAVE), 0, s,
                       n_q_heads, head_dim, n_splits,
                       model->attn_split_o, model->attn_split_ml, out);
  }
}

static void dispatch_attention_i8(hipStream_t s, Model *model,
                                  uint32_t n_q_heads, uint32_t n_kv_heads, uint32_t head_dim,
                                  uint32_t seq_len, uint32_t cache_capacity, uint32_t cache_start,
                                  float scale,
                                  const __half *q,
                                  const int8_t *k_cache,
                                  const int8_t *v_cache,
                                  const float *k_scale,
                                  const float *v_scale,
                                  __half *out) {
  if (seq_len <= ATTN_SPLIT_THRESHOLD) {
    hipLaunchKernelGGL(kern_attn_decode_i8, dim3(n_q_heads), dim3(WAVE), 0, s,
                       n_q_heads, n_kv_heads, head_dim, seq_len, cache_capacity, cache_start, scale,
                       q, k_cache, v_cache, k_scale, v_scale, out);
  } else {
    uint32_t n_splits = (seq_len + ATTN_SPLIT_SIZE - 1) / ATTN_SPLIT_SIZE;
    hipLaunchKernelGGL(kern_attn_split_i8, dim3(n_q_heads, n_splits), dim3(WAVE), 0, s,
                       n_q_heads, n_kv_heads, head_dim, seq_len, cache_capacity, cache_start, scale,
                       q, k_cache, v_cache, k_scale, v_scale,
                       model->attn_split_o, model->attn_split_ml);
    hipLaunchKernelGGL(kern_attn_reduce, dim3(n_q_heads), dim3(WAVE), 0, s,
                       n_q_heads, head_dim, n_splits,
                       model->attn_split_o, model->attn_split_ml, out);
  }
}

static void dispatch_attention_gqa_q4(hipStream_t s, Model *model,
                                      uint32_t seq_len, uint32_t cache_capacity, uint32_t cache_start,
                                      float scale,
                                      const __half *q,
                                      const uint8_t *k_cache,
                                      const uint8_t *v_cache,
                                      const float *k_scale,
                                      const float *v_scale,
                                      __half *out) {
  uint32_t n_splits = (seq_len + ATTN_SPLIT_SIZE - 1) / ATTN_SPLIT_SIZE;
  hipLaunchKernelGGL(kern_attn_split_gqa_q4, dim3(n_splits), dim3(GLOBAL_GQA_BLOCK), 0, s,
                     512, seq_len, cache_capacity, cache_start, scale,
                     q, k_cache, v_cache, k_scale, v_scale,
                     model->attn_split_o, model->attn_split_ml);
  hipLaunchKernelGGL(kern_attn_reduce, dim3(GLOBAL_GQA_HEADS), dim3(WAVE), 0, s,
                     GLOBAL_GQA_HEADS, 512, n_splits,
                     model->attn_split_o, model->attn_split_ml, out);
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

  // Batch prefill buffers (WMMA path): sized for up to 16 tokens
  __half *batch_x;     // [16 × HIDDEN] — batched activations
  __half *batch_normed;// [16 × HIDDEN]
  __half *batch_q;     // [16 × max_q_dim]
  __half *batch_kv;    // [16 × max_kv_dim]
  __half *batch_v;     // [16 × max_kv_dim]
  __half *batch_ffn1;  // [16 × FFN]
  __half *batch_ffn2;  // [16 × FFN]
  __half *batch_out;   // [16 × HIDDEN] — projection output
};

static void forward_decode(Model *model, RunState *state, uint32_t token) {
  hipStream_t s = state->stream;
  uint32_t pos = state->pos;

  // Embed token on GPU: dequant one Q6_K row + scale by sqrt(hidden_dim) in one kernel
  hipLaunchKernelGGL(kern_embed_lookup, dim3((HIDDEN+255)/256), dim3(256), 0, s,
                     token, HIDDEN, sqrtf((float)HIDDEN), model->embed_q6k_dev, state->hidden);

  hipLaunchKernelGGL(kern_fill_rope_row, dim3(1), dim3(128), 0, s,
                     128, pos, model->rope_freq_base_swa, nullptr,
                     model->rope_cos_swa_row, model->rope_sin_swa_row);
  hipLaunchKernelGGL(kern_fill_rope_row, dim3(1), dim3(256), 0, s,
                     256, pos, model->rope_freq_base, model->rope_freqs,
                     model->rope_cos_global_row, model->rope_sin_global_row);

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
      if (!lw.has_v) {
        HIP_CHECK(hipMemcpyAsync(state->scratch_v, state->scratch_kv,
                                 lw.kv_dim * sizeof(__half),
                                 hipMemcpyDeviceToDevice, s));
      }
    }

    if (lw.has_v && lw.wv) {
      hipLaunchKernelGGL(kern_gemv_q4_0, dim3(lw.kv_dim), dim3(WAVE), 0, s,
                         (int)lw.kv_dim, (int)HIDDEN,
                         normed, (const Q4_0Block*)lw.wv, state->scratch_v);
    }

    if (lw.wk) {
      hipLaunchKernelGGL(kern_rmsnorm_head_noscale, dim3(lw.n_kv_heads), dim3(WAVE), 0, s,
                         lw.n_kv_heads, lw.head_dim,
                         state->scratch_v, EPS);
    }

    // 3. Fused Q/K head norms + RoPE using per-token rows.
    const float *cos_row = lw.is_swa ? model->rope_cos_swa_row : model->rope_cos_global_row;
    const float *sin_row = lw.is_swa ? model->rope_sin_swa_row : model->rope_sin_global_row;

    hipLaunchKernelGGL(kern_headnorm_rope_row, dim3(lw.n_q_heads), dim3(WAVE), 0, s,
                       lw.n_q_heads, lw.head_dim,
                       lw.attn_q_norm, cos_row, sin_row, EPS,
                       state->scratch_q);

    if (lw.wk) {
      hipLaunchKernelGGL(kern_headnorm_rope_row, dim3(lw.n_kv_heads), dim3(WAVE), 0, s,
                         lw.n_kv_heads, lw.head_dim,
                         lw.attn_k_norm, cos_row, sin_row, EPS,
                         state->scratch_kv);
    }

    // 5. Store K/V into cache
    if (lw.wk) {
      uint32_t store_idx = layer_store_index(lw, pos);
      __half *v_src = lw.has_v ? state->scratch_v : state->scratch_kv;
      if (lw.kv_quant_q4) {
        hipLaunchKernelGGL(kern_quantize_kv_fp16_to_q4, dim3(1), dim3(256), 0, s,
                           lw.kv_dim, state->scratch_kv, v_src,
                           lw.kv_k_q4 + (size_t)store_idx * (lw.kv_dim / 2),
                           lw.kv_v_q4 + (size_t)store_idx * (lw.kv_dim / 2),
                           lw.kv_k_scale + store_idx,
                           lw.kv_v_scale + store_idx);
      } else if (lw.kv_quant_i8) {
        hipLaunchKernelGGL(kern_quantize_kv_fp16_to_i8, dim3(1), dim3(256), 0, s,
                           lw.kv_dim, state->scratch_kv, v_src,
                           lw.kv_k_i8 + (size_t)store_idx * lw.kv_dim,
                           lw.kv_v_i8 + (size_t)store_idx * lw.kv_dim,
                           lw.kv_k_scale + store_idx,
                           lw.kv_v_scale + store_idx);
      } else {
        HIP_CHECK(hipMemcpyAsync(lw.kv_k + (size_t)store_idx * lw.kv_dim,
                                 state->scratch_kv,
                                 lw.kv_dim * sizeof(__half),
                                 hipMemcpyDeviceToDevice, s));
        HIP_CHECK(hipMemcpyAsync(lw.kv_v + (size_t)store_idx * lw.kv_dim,
                                 v_src,
                                 lw.kv_dim * sizeof(__half),
                                 hipMemcpyDeviceToDevice, s));
      }
    }

    // 6. Attention (split-K for long context)
    uint32_t seq_len = layer_seq_len(lw, pos);
    uint32_t cache_start = layer_cache_start(lw, pos, seq_len);
    if (lw.kv_quant_q4) {
      dispatch_attention_gqa_q4(s, model, seq_len, lw.kv_capacity, cache_start,
                                1.0f, state->scratch_q,
                                lw.kv_k_q4, lw.kv_v_q4,
                                lw.kv_k_scale, lw.kv_v_scale,
                                state->scratch_attn);
    } else if (lw.kv_quant_i8) {
      dispatch_attention_i8(s, model, lw.n_q_heads, lw.n_kv_heads, lw.head_dim,
                            seq_len, lw.kv_capacity, cache_start,
                            1.0f, state->scratch_q,
                            lw.kv_k_i8, lw.kv_v_i8,
                            lw.kv_k_scale, lw.kv_v_scale,
                            state->scratch_attn);
    } else {
      dispatch_attention_fp16(s, model, lw.n_q_heads, lw.n_kv_heads, lw.head_dim,
                              seq_len, lw.kv_capacity, cache_start,
                              1.0f, state->scratch_q,
                              lw.kv_k, lw.kv_v, state->scratch_attn);
    }

    // 7. Output projection
    hipLaunchKernelGGL(kern_gemv_q4_0, dim3(HIDDEN), dim3(WAVE), 0, s,
                       (int)HIDDEN, (int)lw.q_dim,
                       state->scratch_attn, (const Q4_0Block*)lw.wo, state->residual);

    // 8. FUSED: post-attn-norm + residual add (+ layer scale folded in at end)
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

// Batched prefill using WMMA GEMM: process up to 16 tokens simultaneously
// All linear projections use WMMA GEMM, norms and activations remain per-token.
static void forward_prefill_batch(Model *model, RunState *state,
                                  const uint32_t *tokens, int n_tokens) {
  hipStream_t s = state->stream;
  int M = n_tokens;  // 1..16

  // Embed all tokens into batch_x[tok][HIDDEN]
  for (int t = 0; t < M; t++) {
    hipLaunchKernelGGL(kern_embed_lookup, dim3((HIDDEN+255)/256), dim3(256), 0, s,
                       tokens[t], HIDDEN, sqrtf((float)HIDDEN), model->embed_q6k_dev,
                       state->batch_x + t * HIDDEN);
  }

  for (uint32_t l = 0; l < N_LAYERS; l++) {
    LayerWeights &lw = model->layers[l];

    // 1. Per-token attention pre-norm
    for (int t = 0; t < M; t++) {
      hipLaunchKernelGGL(kern_rmsnorm, dim3(1), dim3(NORM_THREADS), 0, s,
                         HIDDEN, state->batch_x + t * HIDDEN, lw.attn_norm, EPS,
                         state->batch_normed + t * HIDDEN);
    }

    // 2. Q projection: [M × HIDDEN] × [q_dim × HIDDEN]^T → [M × q_dim]
    // Use WMMA GEMM for batch, GEMV for single token
    if (M >= 2) {
      int grid_n = (lw.q_dim + WMMA_N - 1) / WMMA_N;
      int grid_m = (M + WMMA_M - 1) / WMMA_M;
      if (M < WMMA_M) {
        hipLaunchKernelGGL(kern_gemm_q4_0_wmma_padded, dim3(grid_n), dim3(WAVE), 0, s,
                           M, (int)lw.q_dim, (int)HIDDEN,
                           state->batch_normed, (const Q4_0Block*)lw.wq, state->batch_q);
      } else {
        hipLaunchKernelGGL(kern_gemm_q4_0_wmma, dim3(grid_n, grid_m), dim3(WAVE), 0, s,
                           M, (int)lw.q_dim, (int)HIDDEN,
                           state->batch_normed, (const Q4_0Block*)lw.wq, state->batch_q);
      }
    } else {
      hipLaunchKernelGGL(kern_gemv_q4_0, dim3(lw.q_dim), dim3(WAVE), 0, s,
                         (int)lw.q_dim, (int)HIDDEN,
                         state->batch_normed, (const Q4_0Block*)lw.wq, state->batch_q);
    }

    // K projection
    if (lw.wk) {
      if (M >= 2) {
        int grid_n = (lw.kv_dim + WMMA_N - 1) / WMMA_N;
        if (M < WMMA_M) {
          hipLaunchKernelGGL(kern_gemm_q4_0_wmma_padded, dim3(grid_n), dim3(WAVE), 0, s,
                             M, (int)lw.kv_dim, (int)HIDDEN,
                             state->batch_normed, (const Q4_0Block*)lw.wk, state->batch_kv);
        } else {
          int grid_m = (M + WMMA_M - 1) / WMMA_M;
          hipLaunchKernelGGL(kern_gemm_q4_0_wmma, dim3(grid_n, grid_m), dim3(WAVE), 0, s,
                             M, (int)lw.kv_dim, (int)HIDDEN,
                             state->batch_normed, (const Q4_0Block*)lw.wk, state->batch_kv);
        }
      } else {
        hipLaunchKernelGGL(kern_gemv_q4_0, dim3(lw.kv_dim), dim3(WAVE), 0, s,
                           (int)lw.kv_dim, (int)HIDDEN,
                           state->batch_normed, (const Q4_0Block*)lw.wk, state->batch_kv);
      }
      if (!lw.has_v) {
        HIP_CHECK(hipMemcpyAsync(state->batch_v, state->batch_kv,
                                 (size_t)M * lw.kv_dim * sizeof(__half),
                                 hipMemcpyDeviceToDevice, s));
      }
    }

    // V projection
    if (lw.has_v && lw.wv) {
      if (M >= 2) {
        int grid_n = (lw.kv_dim + WMMA_N - 1) / WMMA_N;
        if (M < WMMA_M) {
          hipLaunchKernelGGL(kern_gemm_q4_0_wmma_padded, dim3(grid_n), dim3(WAVE), 0, s,
                             M, (int)lw.kv_dim, (int)HIDDEN,
                             state->batch_normed, (const Q4_0Block*)lw.wv, state->batch_v);
        } else {
          int grid_m = (M + WMMA_M - 1) / WMMA_M;
          hipLaunchKernelGGL(kern_gemm_q4_0_wmma, dim3(grid_n, grid_m), dim3(WAVE), 0, s,
                             M, (int)lw.kv_dim, (int)HIDDEN,
                             state->batch_normed, (const Q4_0Block*)lw.wv, state->batch_v);
        }
      } else {
        hipLaunchKernelGGL(kern_gemv_q4_0, dim3(lw.kv_dim), dim3(WAVE), 0, s,
                           (int)lw.kv_dim, (int)HIDDEN,
                           state->batch_normed, (const Q4_0Block*)lw.wv, state->batch_v);
      }
    }

    if (lw.wk) {
      for (int t = 0; t < M; t++) {
        hipLaunchKernelGGL(kern_rmsnorm_head_noscale, dim3(lw.n_kv_heads), dim3(WAVE), 0, s,
                           lw.n_kv_heads, lw.head_dim,
                           state->batch_v + t * lw.kv_dim, EPS);
      }
    }

    // 3. Per-token: head norm + RoPE, store KV, attention
    for (int t = 0; t < M; t++) {
      uint32_t pos = state->pos + t;
      const float *freq_factors = lw.is_swa ? nullptr : model->rope_freqs;
      float freq_base = lw.is_swa ? model->rope_freq_base_swa : model->rope_freq_base;

      // Q head norm + RoPE
      hipLaunchKernelGGL(kern_headnorm_rope_dynamic, dim3(lw.n_q_heads), dim3(WAVE), 0, s,
                         lw.n_q_heads, lw.head_dim, pos,
                         freq_base, freq_factors, lw.attn_q_norm, EPS,
                         state->batch_q + t * lw.q_dim);

      // K head norm + RoPE
      if (lw.wk) {
        hipLaunchKernelGGL(kern_headnorm_rope_dynamic, dim3(lw.n_kv_heads), dim3(WAVE), 0, s,
                           lw.n_kv_heads, lw.head_dim, pos,
                           freq_base, freq_factors, lw.attn_k_norm, EPS,
                           state->batch_kv + t * lw.kv_dim);
      }

      // Store K/V to cache
      if (lw.wk) {
        uint32_t store_idx = layer_store_index(lw, pos);
        __half *v_src = lw.has_v ? (state->batch_v + t * lw.kv_dim)
                                 : (state->batch_kv + t * lw.kv_dim);
        if (lw.kv_quant_q4) {
          hipLaunchKernelGGL(kern_quantize_kv_fp16_to_q4, dim3(1), dim3(256), 0, s,
                             lw.kv_dim,
                             state->batch_kv + t * lw.kv_dim,
                             v_src,
                             lw.kv_k_q4 + (size_t)store_idx * (lw.kv_dim / 2),
                             lw.kv_v_q4 + (size_t)store_idx * (lw.kv_dim / 2),
                             lw.kv_k_scale + store_idx,
                             lw.kv_v_scale + store_idx);
        } else if (lw.kv_quant_i8) {
          hipLaunchKernelGGL(kern_quantize_kv_fp16_to_i8, dim3(1), dim3(256), 0, s,
                             lw.kv_dim,
                             state->batch_kv + t * lw.kv_dim,
                             v_src,
                             lw.kv_k_i8 + (size_t)store_idx * lw.kv_dim,
                             lw.kv_v_i8 + (size_t)store_idx * lw.kv_dim,
                             lw.kv_k_scale + store_idx,
                             lw.kv_v_scale + store_idx);
        } else {
          HIP_CHECK(hipMemcpyAsync(lw.kv_k + (size_t)store_idx * lw.kv_dim,
                                   state->batch_kv + t * lw.kv_dim,
                                   lw.kv_dim * sizeof(__half), hipMemcpyDeviceToDevice, s));
          HIP_CHECK(hipMemcpyAsync(lw.kv_v + (size_t)store_idx * lw.kv_dim,
                                   v_src, lw.kv_dim * sizeof(__half), hipMemcpyDeviceToDevice, s));
        }
      }

      // Attention (causal: token t sees positions 0..pos+t)
      uint32_t seq_len = layer_seq_len(lw, pos);
      uint32_t cache_start = layer_cache_start(lw, pos, seq_len);
      if (lw.kv_quant_q4) {
        dispatch_attention_gqa_q4(s, model, seq_len, lw.kv_capacity, cache_start,
                                  1.0f, state->batch_q + t * lw.q_dim,
                                  lw.kv_k_q4, lw.kv_v_q4,
                                  lw.kv_k_scale, lw.kv_v_scale,
                                  state->scratch_attn);
      } else if (lw.kv_quant_i8) {
        dispatch_attention_i8(s, model, lw.n_q_heads, lw.n_kv_heads, lw.head_dim,
                              seq_len, lw.kv_capacity, cache_start,
                              1.0f, state->batch_q + t * lw.q_dim,
                              lw.kv_k_i8, lw.kv_v_i8,
                              lw.kv_k_scale, lw.kv_v_scale,
                              state->scratch_attn);
      } else {
        dispatch_attention_fp16(s, model, lw.n_q_heads, lw.n_kv_heads, lw.head_dim,
                                seq_len, lw.kv_capacity, cache_start,
                                1.0f, state->batch_q + t * lw.q_dim,
                                lw.kv_k, lw.kv_v, state->scratch_attn);
      }

      // Output projection for this token: [1 × q_dim] × [HIDDEN × q_dim]^T → [1 × HIDDEN]
      hipLaunchKernelGGL(kern_gemv_q4_0, dim3(HIDDEN), dim3(WAVE), 0, s,
                         (int)HIDDEN, (int)lw.q_dim,
                         state->scratch_attn, (const Q4_0Block*)lw.wo,
                         state->batch_out + t * HIDDEN);

      // Fused post-attn-norm + residual add
      hipLaunchKernelGGL(kern_fused_postnorm_residadd, dim3(1), dim3(NORM_THREADS), 0, s,
                         HIDDEN, state->batch_x + t * HIDDEN,
                         state->batch_out + t * HIDDEN, lw.post_attn_norm, EPS, 1.0f);
    }

    // 4. FFN pre-norm (per token)
    for (int t = 0; t < M; t++) {
      hipLaunchKernelGGL(kern_rmsnorm, dim3(1), dim3(NORM_THREADS), 0, s,
                         HIDDEN, state->batch_x + t * HIDDEN, lw.ffn_norm, EPS,
                         state->batch_normed + t * HIDDEN);
    }

    // 5. FFN gate + up projections: WMMA GEMM
    if (M >= 2) {
      int grid_n = (FFN + WMMA_N - 1) / WMMA_N;
      if (M < WMMA_M) {
        hipLaunchKernelGGL(kern_gemm_q4_0_wmma_padded, dim3(grid_n), dim3(WAVE), 0, s,
                           M, (int)FFN, (int)HIDDEN,
                           state->batch_normed, (const Q4_0Block*)lw.ffn_gate, state->batch_ffn1);
        hipLaunchKernelGGL(kern_gemm_q4_0_wmma_padded, dim3(grid_n), dim3(WAVE), 0, s,
                           M, (int)FFN, (int)HIDDEN,
                           state->batch_normed, (const Q4_0Block*)lw.ffn_up, state->batch_ffn2);
      } else {
        int grid_m = (M + WMMA_M - 1) / WMMA_M;
        hipLaunchKernelGGL(kern_gemm_q4_0_wmma, dim3(grid_n, grid_m), dim3(WAVE), 0, s,
                           M, (int)FFN, (int)HIDDEN,
                           state->batch_normed, (const Q4_0Block*)lw.ffn_gate, state->batch_ffn1);
        hipLaunchKernelGGL(kern_gemm_q4_0_wmma, dim3(grid_n, grid_m), dim3(WAVE), 0, s,
                           M, (int)FFN, (int)HIDDEN,
                           state->batch_normed, (const Q4_0Block*)lw.ffn_up, state->batch_ffn2);
      }
    } else {
      hipLaunchKernelGGL(kern_gemv_q4_0, dim3(FFN), dim3(WAVE), 0, s,
                         (int)FFN, (int)HIDDEN,
                         state->batch_normed, (const Q4_0Block*)lw.ffn_gate, state->batch_ffn1);
      hipLaunchKernelGGL(kern_gemv_q4_0, dim3(FFN), dim3(WAVE), 0, s,
                         (int)FFN, (int)HIDDEN,
                         state->batch_normed, (const Q4_0Block*)lw.ffn_up, state->batch_ffn2);
    }

    // 6. GeLU + down projection + post-FFN norm (per token)
    for (int t = 0; t < M; t++) {
      hipLaunchKernelGGL(kern_gelu_mul, dim3((FFN+255)/256), dim3(256), 0, s,
                         FFN, state->batch_ffn1 + t * FFN, state->batch_ffn2 + t * FFN,
                         state->batch_ffn1 + t * FFN);

      hipLaunchKernelGGL(kern_gemv_q4_0, dim3(HIDDEN), dim3(WAVE), 0, s,
                         (int)HIDDEN, (int)FFN,
                         state->batch_ffn1 + t * FFN, (const Q4_0Block*)lw.ffn_down,
                         state->batch_out + t * HIDDEN);

      float scale = lw.out_scale_val;
      hipLaunchKernelGGL(kern_fused_postnorm_residadd, dim3(1), dim3(NORM_THREADS), 0, s,
                         HIDDEN, state->batch_x + t * HIDDEN,
                         state->batch_out + t * HIDDEN, lw.post_ffn_norm, EPS, scale);
    }
  }

  // Final norm on last token → state->hidden (for sampling)
  int last = M - 1;
  hipLaunchKernelGGL(kern_rmsnorm, dim3(1), dim3(NORM_THREADS), 0, s,
                     HIDDEN, state->batch_x + last * HIDDEN, model->output_norm, EPS,
                     state->hidden);

  state->pos += M;
}

// GPU-based lm_head + GPU argmax sampling
static uint32_t sample_greedy(Model *model, RunState *state) {
  hipStream_t s = state->stream;

  // lm_head: [1 × HIDDEN] @ [vocab × HIDDEN]^T → [1 × vocab]
  hipLaunchKernelGGL(kern_gemv_q6k, dim3(model->vocab_size), dim3(WAVE), 0, s,
                     (int)model->vocab_size, (int)HIDDEN,
                     state->hidden, model->embed_q6k_dev, model->logits_dev);

  if (model->final_logit_softcap > 0.0f) {
    hipLaunchKernelGGL(kern_softcap, dim3((model->vocab_size + 255) / 256), dim3(256), 0, s,
                       model->vocab_size, model->final_logit_softcap, model->logits_dev);
  }

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

static int run_chat(Model *model, RunState *state, const Tokenizer &tok, const char *system_prompt) {
  std::vector<uint32_t> init_tokens;
  if (tok.add_bos_token) init_tokens.push_back(tok.bos_id);
  if (system_prompt && *system_prompt)
    append_chat_turn_tokens(tok, "system", system_prompt, false, false, &init_tokens);
  if (!init_tokens.empty())
    prefill_tokens(model, state, init_tokens);

  bool prepend_newline = false;
  const uint32_t max_new_tokens = getenv_u32("THEROCK_CHAT_MAX_NEW", 256);
  char line[8192];

  while (true) {
    fprintf(stdout, "user> ");
    fflush(stdout);
    if (!fgets(line, sizeof(line), stdin)) break;
    std::string user_text(line);
    while (!user_text.empty() && (user_text.back() == '\n' || user_text.back() == '\r'))
      user_text.pop_back();
    if (user_text.empty()) continue;
    if (user_text == "/quit" || user_text == "/exit") break;

    std::vector<uint32_t> turn_tokens;
    append_chat_turn_tokens(tok, "user", user_text, prepend_newline, true, &turn_tokens);
    if (state->pos + turn_tokens.size() + max_new_tokens >= model->max_seq) {
      fprintf(stderr, "Context full at %u tokens; restart chat or raise THEROCK_MAX_SEQ\n", state->pos);
      return 1;
    }
    prefill_tokens(model, state, turn_tokens);

    fprintf(stdout, "assistant> ");
    fflush(stdout);
    auto t0 = std::chrono::high_resolution_clock::now();
    uint32_t generated = 0;
    bool in_hidden_channel = false;
    for (; generated < max_new_tokens; generated++) {
      uint32_t tok_id = sample_greedy(model, state);
      forward_decode(model, state, tok_id);
      if (tok_id == tok.eos_id || tok_id == tok.turn_close_id) break;
      if (tok_id == tok.channel_open_id) {
        in_hidden_channel = true;
        continue;
      }
      if (tok_id == tok.channel_close_id) {
        in_hidden_channel = false;
        continue;
      }
      if (in_hidden_channel) continue;
      if (tok_id < tok.id_to_piece.size()) {
        std::string piece = decode_piece_for_display(tok.id_to_piece[tok_id]);
        if (!piece.empty()) {
          fwrite(piece.data(), 1, piece.size(), stdout);
          fflush(stdout);
        }
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    fprintf(stdout, "\n[turn: %u tokens in %.1f ms = %.2f t/s]\n", generated, ms,
            generated > 0 ? generated * 1000.0f / ms : 0.0f);
    prepend_newline = (generated < max_new_tokens);
  }
  return 0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model.gguf> [--chat] [--system TEXT] [prompt]\n", argv[0]);
    return 1;
  }

  const char *model_path = argv[1];
  bool chat_mode = false;
  const char *prompt = nullptr;
  const char *system_prompt = getenv("THEROCK_SYSTEM_PROMPT");
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--chat") == 0) {
      chat_mode = true;
    } else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc) {
      system_prompt = argv[++i];
    } else if (!prompt) {
      prompt = argv[i];
    }
  }
  if (!prompt && !chat_mode)
    prompt = "Hello, I am";

  printf("=== TheRock Inference Engine ===\n");
  printf("Model: %s\n", model_path);
  printf("GPU: gfx1032 (RX 6650 XT)\n\n");

  // Open and parse GGUF
  printf("Opening GGUF file...\n");
  GGUFFile *gguf = gguf_open(model_path);
  if (!gguf) return 1;
  printf("  Version: %u, Tensors: %lu, KV pairs: %lu\n",
         gguf->version, gguf->n_tensors, gguf->n_kv);

  Tokenizer tokenizer;
  if (!load_tokenizer(gguf, &tokenizer)) return 1;

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
  // Batch prefill buffers (16 tokens max)
  HIP_CHECK(hipMalloc(&state.batch_x, 16 * HIDDEN * sizeof(__half)));
  HIP_CHECK(hipMalloc(&state.batch_normed, 16 * HIDDEN * sizeof(__half)));
  HIP_CHECK(hipMalloc(&state.batch_q, 16 * 8192 * sizeof(__half)));
  HIP_CHECK(hipMalloc(&state.batch_kv, 16 * 2048 * sizeof(__half)));
  HIP_CHECK(hipMalloc(&state.batch_v, 16 * 2048 * sizeof(__half)));
  HIP_CHECK(hipMalloc(&state.batch_ffn1, 16 * FFN * sizeof(__half)));
  HIP_CHECK(hipMalloc(&state.batch_ffn2, 16 * FFN * sizeof(__half)));
  HIP_CHECK(hipMalloc(&state.batch_out, 16 * HIDDEN * sizeof(__half)));
  HIP_CHECK(hipStreamCreate(&state.stream));
  state.pos = 0;

  // GPU telemetry helpers
  auto read_sysfs = [](const char *path) -> int64_t {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int64_t val = -1;
    fscanf(f, "%ld", &val);
    fclose(f);
    return val;
  };

  auto print_gpu_stats = [&](const char *label) {
    int64_t temp = read_sysfs("/sys/class/hwmon/hwmon6/temp1_input");
    int64_t power = read_sysfs("/sys/class/hwmon/hwmon6/power1_average");
    int64_t freq = read_sysfs("/sys/class/hwmon/hwmon6/freq1_input");
    int64_t busy = read_sysfs("/sys/class/drm/card2/device/gpu_busy_percent");
    printf("[GPU %s] freq=%ldMHz temp=%ld°C power=%.1fW busy=%ld%%\n",
           label, freq / 1000000, temp / 1000, power / 1e6, busy);
  };

  printf("\nModel loaded. Starting generation...\n");
  if (!chat_mode)
    printf("Prompt: \"%s\"\n\n", prompt);
  print_gpu_stats("idle");

  if (chat_mode) {
    int rc = run_chat(&model, &state, tokenizer, system_prompt ? system_prompt : "");
    gguf_close(gguf);
    return rc;
  }

  std::vector<uint32_t> tokens = tokenizer_encode_prompt(tokenizer, prompt ? prompt : "");

  int n_gen = (int)getenv_u32("THEROCK_N_GEN", 32);
  if (tokens.size() + 2 + (size_t)n_gen > model.max_seq) {
    fprintf(stderr, "Prompt + warmup + generation exceed max_seq=%u\n", model.max_seq);
    return 1;
  }

  printf("Tokens: %zu\n", tokens.size());
  printf("---\n");

  // Prefill: process prompt tokens in batches of 16 using WMMA GEMM
  auto t_start = std::chrono::high_resolution_clock::now();

  prefill_tokens(&model, &state, tokens);

  auto t_prefill = std::chrono::high_resolution_clock::now();
  float prefill_ms = std::chrono::duration<float, std::milli>(t_prefill - t_start).count();
  printf("[Prefill: %zu tokens in %.1f ms = %.1f t/s (WMMA batched)]\n",
         tokens.size(), prefill_ms, tokens.size() * 1000.0f / prefill_ms);

  // Warmup: 2 cycles to JIT, then reset KV cache for clean benchmark
  printf("\nWarming up GPU kernels...\n");
  {
    uint32_t wt = sample_greedy(&model, &state);
    forward_decode(&model, &state, wt);
    wt = sample_greedy(&model, &state);
    forward_decode(&model, &state, wt);
    print_gpu_stats("post-warmup");
  }

  // Generate tokens (timed) — pipelined: forward reads token from device memory,
  // GPU does embed+forward+lm_head+argmax → writes next token to argmax_dev,
  // host only syncs to read token for logging (not for feedback).
  printf("Generating %d tokens...\n", n_gen);

  auto t_gen_start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < n_gen; i++) {
    // Launch lm_head + argmax → writes to model.argmax_dev
    hipStream_t s = state.stream;
    hipLaunchKernelGGL(kern_gemv_q6k, dim3(model.vocab_size), dim3(WAVE), 0, s,
                       (int)model.vocab_size, (int)HIDDEN,
                       state.hidden, model.embed_q6k_dev, model.logits_dev);
    if (model.final_logit_softcap > 0.0f) {
      hipLaunchKernelGGL(kern_softcap, dim3((model.vocab_size + 255) / 256), dim3(256), 0, s,
                         model.vocab_size, model.final_logit_softcap, model.logits_dev);
    }
    HIP_CHECK(hipMemsetAsync(model.argmax_dev, 0, sizeof(uint32_t), s));
    hipLaunchKernelGGL(kern_argmax_single, dim3(1), dim3(1024), 0, s,
                       model.logits_dev, model.vocab_size, model.argmax_dev);

    // Forward pass reads token indirectly from argmax_dev (no host sync needed)
    hipLaunchKernelGGL(kern_embed_lookup_indirect, dim3((HIDDEN+255)/256), dim3(256), 0, s,
                       model.argmax_dev, HIDDEN, sqrtf((float)HIDDEN),
                       model.embed_q6k_dev, state.hidden);

    // Run all layers (same as forward_decode minus embedding)
    __half *normed = state.normed;
    uint32_t pos = state.pos;

    hipLaunchKernelGGL(kern_fill_rope_row, dim3(1), dim3(128), 0, s,
                       128, pos, model.rope_freq_base_swa, nullptr,
                       model.rope_cos_swa_row, model.rope_sin_swa_row);
    hipLaunchKernelGGL(kern_fill_rope_row, dim3(1), dim3(256), 0, s,
                       256, pos, model.rope_freq_base, model.rope_freqs,
                       model.rope_cos_global_row, model.rope_sin_global_row);

    for (uint32_t l = 0; l < N_LAYERS; l++) {
      LayerWeights &lw = model.layers[l];

      hipLaunchKernelGGL(kern_rmsnorm, dim3(1), dim3(NORM_THREADS), 0, s,
                         HIDDEN, state.hidden, lw.attn_norm, EPS, normed);

      hipLaunchKernelGGL(kern_gemv_q4_0, dim3(lw.q_dim), dim3(WAVE), 0, s,
                         (int)lw.q_dim, (int)HIDDEN,
                         normed, (const Q4_0Block*)lw.wq, state.scratch_q);

      if (lw.wk) {
        hipLaunchKernelGGL(kern_gemv_q4_0, dim3(lw.kv_dim), dim3(WAVE), 0, s,
                           (int)lw.kv_dim, (int)HIDDEN,
                           normed, (const Q4_0Block*)lw.wk, state.scratch_kv);
        if (!lw.has_v) {
          HIP_CHECK(hipMemcpyAsync(state.scratch_v, state.scratch_kv,
                                   lw.kv_dim * sizeof(__half),
                                   hipMemcpyDeviceToDevice, s));
        }
      }
      if (lw.has_v && lw.wv) {
        hipLaunchKernelGGL(kern_gemv_q4_0, dim3(lw.kv_dim), dim3(WAVE), 0, s,
                           (int)lw.kv_dim, (int)HIDDEN,
                           normed, (const Q4_0Block*)lw.wv, state.scratch_v);
      }
      if (lw.wk) {
        hipLaunchKernelGGL(kern_rmsnorm_head_noscale, dim3(lw.n_kv_heads), dim3(WAVE), 0, s,
                           lw.n_kv_heads, lw.head_dim,
                           state.scratch_v, EPS);
      }

      const float *cos_row = lw.is_swa ? model.rope_cos_swa_row : model.rope_cos_global_row;
      const float *sin_row = lw.is_swa ? model.rope_sin_swa_row : model.rope_sin_global_row;

      hipLaunchKernelGGL(kern_headnorm_rope_row, dim3(lw.n_q_heads), dim3(WAVE), 0, s,
                         lw.n_q_heads, lw.head_dim,
                         lw.attn_q_norm, cos_row, sin_row, EPS, state.scratch_q);
      if (lw.wk) {
        hipLaunchKernelGGL(kern_headnorm_rope_row, dim3(lw.n_kv_heads), dim3(WAVE), 0, s,
                           lw.n_kv_heads, lw.head_dim,
                           lw.attn_k_norm, cos_row, sin_row, EPS, state.scratch_kv);
      }

      if (lw.wk) {
        uint32_t store_idx = layer_store_index(lw, pos);
        __half *v_src = lw.has_v ? state.scratch_v : state.scratch_kv;
        if (lw.kv_quant_q4) {
          hipLaunchKernelGGL(kern_quantize_kv_fp16_to_q4, dim3(1), dim3(256), 0, s,
                             lw.kv_dim, state.scratch_kv, v_src,
                             lw.kv_k_q4 + (size_t)store_idx * (lw.kv_dim / 2),
                             lw.kv_v_q4 + (size_t)store_idx * (lw.kv_dim / 2),
                             lw.kv_k_scale + store_idx,
                             lw.kv_v_scale + store_idx);
        } else if (lw.kv_quant_i8) {
          hipLaunchKernelGGL(kern_quantize_kv_fp16_to_i8, dim3(1), dim3(256), 0, s,
                             lw.kv_dim, state.scratch_kv, v_src,
                             lw.kv_k_i8 + (size_t)store_idx * lw.kv_dim,
                             lw.kv_v_i8 + (size_t)store_idx * lw.kv_dim,
                             lw.kv_k_scale + store_idx,
                             lw.kv_v_scale + store_idx);
        } else {
          HIP_CHECK(hipMemcpyAsync(lw.kv_k + (size_t)store_idx * lw.kv_dim,
                                   state.scratch_kv, lw.kv_dim * sizeof(__half),
                                   hipMemcpyDeviceToDevice, s));
          HIP_CHECK(hipMemcpyAsync(lw.kv_v + (size_t)store_idx * lw.kv_dim,
                                   v_src, lw.kv_dim * sizeof(__half),
                                   hipMemcpyDeviceToDevice, s));
        }
      }

      uint32_t seq_len = layer_seq_len(lw, pos);
      uint32_t cache_start = layer_cache_start(lw, pos, seq_len);
      if (lw.kv_quant_q4) {
        dispatch_attention_gqa_q4(s, &model, seq_len, lw.kv_capacity, cache_start,
                                  1.0f, state.scratch_q,
                                  lw.kv_k_q4, lw.kv_v_q4,
                                  lw.kv_k_scale, lw.kv_v_scale,
                                  state.scratch_attn);
      } else if (lw.kv_quant_i8) {
        dispatch_attention_i8(s, &model, lw.n_q_heads, lw.n_kv_heads, lw.head_dim,
                              seq_len, lw.kv_capacity, cache_start,
                              1.0f, state.scratch_q,
                              lw.kv_k_i8, lw.kv_v_i8,
                              lw.kv_k_scale, lw.kv_v_scale,
                              state.scratch_attn);
      } else {
        dispatch_attention_fp16(s, &model, lw.n_q_heads, lw.n_kv_heads, lw.head_dim,
                                seq_len, lw.kv_capacity, cache_start,
                                1.0f, state.scratch_q,
                                lw.kv_k, lw.kv_v, state.scratch_attn);
      }

      hipLaunchKernelGGL(kern_gemv_q4_0, dim3(HIDDEN), dim3(WAVE), 0, s,
                         (int)HIDDEN, (int)lw.q_dim,
                         state.scratch_attn, (const Q4_0Block*)lw.wo, state.residual);

      hipLaunchKernelGGL(kern_fused_postnorm_residadd, dim3(1), dim3(NORM_THREADS), 0, s,
                         HIDDEN, state.hidden, state.residual, lw.post_attn_norm, EPS, 1.0f);

      hipLaunchKernelGGL(kern_rmsnorm, dim3(1), dim3(NORM_THREADS), 0, s,
                         HIDDEN, state.hidden, lw.ffn_norm, EPS, normed);

      hipLaunchKernelGGL(kern_gemv_q4_0, dim3(FFN), dim3(WAVE), 0, s,
                         (int)FFN, (int)HIDDEN,
                         normed, (const Q4_0Block*)lw.ffn_gate, state.scratch_ffn1);
      hipLaunchKernelGGL(kern_gemv_q4_0, dim3(FFN), dim3(WAVE), 0, s,
                         (int)FFN, (int)HIDDEN,
                         normed, (const Q4_0Block*)lw.ffn_up, state.scratch_ffn2);

      hipLaunchKernelGGL(kern_gelu_mul, dim3((FFN+255)/256), dim3(256), 0, s,
                         FFN, state.scratch_ffn1, state.scratch_ffn2, state.scratch_ffn1);

      hipLaunchKernelGGL(kern_gemv_q4_0, dim3(HIDDEN), dim3(WAVE), 0, s,
                         (int)HIDDEN, (int)FFN,
                         state.scratch_ffn1, (const Q4_0Block*)lw.ffn_down, state.residual);

      float scale = lw.out_scale_val;
      hipLaunchKernelGGL(kern_fused_postnorm_residadd, dim3(1), dim3(NORM_THREADS), 0, s,
                         HIDDEN, state.hidden, state.residual, lw.post_ffn_norm, EPS, scale);
    }

    // Final norm
    hipLaunchKernelGGL(kern_rmsnorm, dim3(1), dim3(NORM_THREADS), 0, s,
                       HIDDEN, state.hidden, model.output_norm, EPS, state.hidden);

    state.pos++;

    // Progress reporting: use events to avoid pipeline stalls
    if ((i + 1) == n_gen) {
      // Only sync at the very end (handled after loop)
    }
  }

  HIP_CHECK(hipStreamSynchronize(state.stream));
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

  print_gpu_stats("post-gen");

  // Profile: separate host dispatch time from GPU execution time
  printf("\n--- Per-kernel profiling (1 token) ---\n");
  {
    hipEvent_t ev_start, ev_end;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_end));

    // Measure host dispatch time (time to submit all kernels, no GPU wait)
    HIP_CHECK(hipStreamSynchronize(state.stream)); // drain queue first
    auto h_start = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipEventRecord(ev_start, state.stream));
    forward_decode(&model, &state, 1);
    auto h_end = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipEventRecord(ev_end, state.stream));
    HIP_CHECK(hipEventSynchronize(ev_end));

    float host_ms = std::chrono::duration<float, std::milli>(h_end - h_start).count();
    float gpu_ms;
    HIP_CHECK(hipEventElapsedTime(&gpu_ms, ev_start, ev_end));
    printf("Host dispatch: %.2f ms\n", host_ms);
    printf("GPU execution: %.2f ms (= %.1f t/s)\n", gpu_ms, 1000.0f/gpu_ms);
    printf("Overlap: %.2f ms hidden by GPU\n", host_ms - (gpu_ms > host_ms ? 0 : gpu_ms - host_ms));

    // Time sample
    HIP_CHECK(hipEventRecord(ev_start, state.stream));
    sample_greedy(&model, &state);
    HIP_CHECK(hipEventRecord(ev_end, state.stream));
    HIP_CHECK(hipEventSynchronize(ev_end));
    float sample_ms;
    HIP_CHECK(hipEventElapsedTime(&sample_ms, ev_start, ev_end));
    printf("Sample (lm_head+argmax): %.2f ms\n", sample_ms);
    printf("Total GPU/token: %.2f ms (= %.1f t/s potential)\n",
           gpu_ms + sample_ms, 1000.0f/(gpu_ms + sample_ms));

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_end));
  }

  // Cleanup
  gguf_close(gguf);
  printf("\nDone.\n");
  return 0;
}

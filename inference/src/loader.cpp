// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/loader.hpp"
#include "rxinfer/common.hpp"
#include "rxinfer/quant.hpp"
#include "config_json.hpp"
#include "gguf_iq_tables.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace rxinfer {
namespace {

inline float bf16_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline float fp16_to_f32(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t f32_bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            f32_bits = sign;
        } else {
            // Subnormal
            float val = std::ldexp(static_cast<float>(mant), -24);
            return (sign != 0) ? -val : val;
        }
    } else if (exp == 31) {
        f32_bits = sign | 0x7F800000u | (mant << 13);
    } else {
        f32_bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &f32_bits, sizeof(f));
    return f;
}

[[maybe_unused]] inline uint16_t f32_to_fp16_bits(float f) {
    __half h = __float2half(f);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// Parallel row-tile packer: calls `fill_row(int row, float* dst_k)` for row in [0..N-1],
// packs directly into R4_GFX12_G64 host buffers (qs, scales) across threads.
template <typename RowFn>
void pack_r4_parallel(int N, int K, void* dst_qs, __half* dst_scales, int num_threads, RowFn&& fill_row) {
    const int n_tiles = static_cast<int>(div_ceil(N, 16));
    const int k_tiles = static_cast<int>(div_ceil(K, 64));
    auto* qs = static_cast<uint4*>(dst_qs);

    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads <= 0) num_threads = 4;
    }
    num_threads = std::min(num_threads, std::max(1, n_tiles));

    auto worker = [&](int tid) {
        std::vector<float> rows(16ULL * K, 0.0f);
        for (int nt = tid; nt < n_tiles; nt += num_threads) {
            const int n0 = nt * 16;
            const int valid_rows = std::min(16, N - n0);
            for (int nl = 0; nl < valid_rows; ++nl) {
                fill_row(n0 + nl, rows.data() + static_cast<size_t>(nl) * K);
            }
            for (int kt = 0; kt < k_tiles; ++kt) {
                const size_t tile_idx = static_cast<size_t>(nt) * k_tiles + kt;
                const int k0 = kt * 64;
                const int k_lim = std::min(64, K - k0);

                float d_row[16]{};
                float id_row[16]{};
                for (int nl = 0; nl < valid_rows; ++nl) {
                    const float* x = rows.data() + static_cast<size_t>(nl) * K + k0;
                    float amax = 0.0f;
                    float vmax = 0.0f;
                    for (int i = 0; i < k_lim; ++i) {
                        const float ax = std::fabs(x[i]);
                        if (amax < ax) {
                            amax = ax;
                            vmax = x[i];
                        }
                    }
                    d_row[nl] = vmax / -8.0f;
                    id_row[nl] = (d_row[nl] != 0.0f) ? (1.0f / d_row[nl]) : 0.0f;
                    dst_scales[tile_idx * 16 + nl] = __float2half(d_row[nl]);
                }
                for (int nl = valid_rows; nl < 16; ++nl) {
                    dst_scales[tile_idx * 16 + nl] = __float2half(0.0f);
                }

                for (int lane = 0; lane < 32; ++lane) {
                    const int nl = lane & 15;
                    const int k_half = (lane >= 16) ? 8 : 0;
                    const float id = id_row[nl];
                    const bool ok = nl < valid_rows;
                    const float* x = ok ? (rows.data() + static_cast<size_t>(nl) * K + k0) : nullptr;

                    uint32_t words[4];
                    for (int s = 0; s < 4; ++s) {
                        const int kb = s * 16 + k_half;
                        uint32_t u[8];
                        for (int i = 0; i < 8; ++i) {
                            const float xv = (ok && (kb + i) < k_lim) ? x[kb + i] : 0.0f;
                            u[i] = static_cast<uint32_t>(std::min(15, std::max(0,
                                    static_cast<int>(std::trunc(xv * id + 8.5f)))));
                        }
                        words[s] = (u[0]) | (u[2] << 4) | (u[4] << 8) | (u[6] << 12) |
                                   (u[1] << 16) | (u[3] << 20) | (u[5] << 24) | (u[7] << 28);
                    }
                    qs[tile_idx * 32 + lane] = make_uint4(words[0], words[1], words[2], words[3]);
                }
            }
        }
    };

    if (num_threads == 1) {
        worker(0);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) threads.emplace_back(worker, t);
        for (auto& th : threads) th.join();
    }
}

// Deterministic layout allocator shared across estimation, loaders, and RXR4 checkpointing.
template <typename AllocFn>
void walk_model_layout(const ModelConfig& cfg, ModelWeights& w, AllocFn&& alloc_fn) {
    const int32_t H = cfg.hidden_size;
    const int32_t I = cfg.intermediate_size;
    const int32_t V = cfg.vocab_size;

    auto alloc_r4 = [&](R4Matrix& mat, int32_t N, int32_t K) {
        mat.N = N;
        mat.K = K;
        mat.qs = alloc_fn(quant::r4_g64_qs_bytes(N, K));
        mat.scales = static_cast<__half*>(alloc_fn(quant::r4_g64_scales_bytes(N, K)));
    };

    alloc_r4(w.embed_tokens, V, H);
    w.final_norm = static_cast<const __half*>(alloc_fn(static_cast<size_t>(H) * sizeof(__half)));
    if (cfg.tie_word_embeddings) {
        w.lm_head = w.embed_tokens;
    } else {
        alloc_r4(w.lm_head, V, H);
    }

    w.layers.resize(cfg.num_hidden_layers);
    for (int32_t l = 0; l < cfg.num_hidden_layers; ++l) {
        LayerWeights& lw = w.layers[l];
        lw.kind = (l < static_cast<int32_t>(cfg.layer_types.size()))
                      ? cfg.layer_types[l]
                      : LayerKind::FullAttention;

        lw.input_norm = static_cast<const __half*>(alloc_fn(static_cast<size_t>(H) * sizeof(__half)));
        lw.post_attn_norm = static_cast<const __half*>(alloc_fn(static_cast<size_t>(H) * sizeof(__half)));

        if (lw.kind == LayerKind::FullAttention) {
            const int32_t D = cfg.head_dim;
            const int32_t q_out = cfg.num_attention_heads * D * (cfg.attn_output_gate ? 2 : 1);
            const int32_t kv_out = 2 * cfg.num_key_value_heads * D;
            const int32_t q_dim = cfg.num_attention_heads * D;
            alloc_r4(lw.q_proj, q_out, H);
            alloc_r4(lw.kv_proj, kv_out, H);
            alloc_r4(lw.o_proj, H, q_dim);
            if (cfg.qk_norm) {
                lw.q_norm = static_cast<const __half*>(alloc_fn(static_cast<size_t>(D) * sizeof(__half)));
                lw.k_norm = static_cast<const __half*>(alloc_fn(static_cast<size_t>(D) * sizeof(__half)));
            }
        } else {
            const int32_t key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
            const int32_t value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
            const int32_t conv_dim = 2 * key_dim + value_dim;
            const int32_t conv_k = cfg.linear_conv_kernel_dim;
            alloc_r4(lw.in_proj_qkvz, conv_dim + value_dim, H);
            lw.in_proj_ba = static_cast<const __half*>(
                alloc_fn(static_cast<size_t>(2 * cfg.linear_num_value_heads) * H * sizeof(__half)));
            lw.conv_w = static_cast<const __half*>(
                alloc_fn(static_cast<size_t>(conv_k) * conv_dim * sizeof(__half)));
            lw.A_log = static_cast<const float*>(
                alloc_fn(static_cast<size_t>(cfg.linear_num_value_heads) * sizeof(float)));
            lw.dt_bias = static_cast<const float*>(
                alloc_fn(static_cast<size_t>(cfg.linear_num_value_heads) * sizeof(float)));
            lw.gdn_norm_w = static_cast<const __half*>(
                alloc_fn(static_cast<size_t>(cfg.linear_value_head_dim) * sizeof(__half)));
            alloc_r4(lw.out_proj, H, value_dim);
        }

        alloc_r4(lw.gate_up_proj, 2 * I, H);
        alloc_r4(lw.down_proj, H, I);
    }
}

void allocate_model_arena(const ModelConfig& cfg, ModelWeights& w) {
    w.config = cfg;
    const size_t total = estimate_model_weight_bytes(cfg);
    w.weight_arena.init(total, "weights");
    walk_model_layout(cfg, w, [&](size_t bytes) -> void* {
        return w.weight_arena.alloc(bytes, 256);
    });
}

void upload_r4_host_to_device(const R4Matrix& dst, const void* h_qs, const __half* h_sc) {
    RX_HIP_CHECK(hipMemcpy(dst.qs, h_qs, quant::r4_g64_qs_bytes(dst.N, dst.K), hipMemcpyHostToDevice));
    RX_HIP_CHECK(hipMemcpy(dst.scales, h_sc, quant::r4_g64_scales_bytes(dst.N, dst.K), hipMemcpyHostToDevice));
}

void upload_f16_host_to_device(const __half* dst, const void* src, size_t elems) {
    RX_HIP_CHECK(hipMemcpy(const_cast<__half*>(dst), src, elems * sizeof(__half), hipMemcpyHostToDevice));
}

void upload_f32_host_to_device(const float* dst, const void* src, size_t elems) {
    RX_HIP_CHECK(hipMemcpy(const_cast<float*>(dst), src, elems * sizeof(float), hipMemcpyHostToDevice));
}

// Mmap RAII helper for safetensors / gguf files
struct MappedFile {
    int fd = -1;
    size_t size = 0;
    const uint8_t* data = nullptr;

    explicit MappedFile(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Failed to open file: " + path);
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error("fstat failed: " + path);
        }
        size = static_cast<size_t>(st.st_size);
        if (size > 0) {
            void* p = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (p == MAP_FAILED) {
                ::close(fd);
                throw std::runtime_error("mmap failed: " + path);
            }
            data = static_cast<const uint8_t*>(p);
        }
    }

    ~MappedFile() {
        if (data && size > 0) ::munmap(const_cast<uint8_t*>(data), size);
        if (fd >= 0) ::close(fd);
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
};

struct SafetensorEntry {
    const MappedFile* file = nullptr;
    std::string dtype;
    std::vector<int64_t> shape;
    size_t byte_offset = 0;
    size_t byte_length = 0;

    const uint8_t* ptr() const { return file->data + byte_offset; }
};

void decode_row_to_f32(const SafetensorEntry& t, int row, float* dst, int cols) {
    const uint8_t* base = t.ptr();
    if (t.dtype == "BF16") {
        const auto* src = reinterpret_cast<const uint16_t*>(base) + static_cast<size_t>(row) * cols;
        for (int c = 0; c < cols; ++c) dst[c] = bf16_to_f32(src[c]);
    } else if (t.dtype == "F16") {
        const auto* src = reinterpret_cast<const uint16_t*>(base) + static_cast<size_t>(row) * cols;
        for (int c = 0; c < cols; ++c) dst[c] = fp16_to_f32(src[c]);
    } else if (t.dtype == "F32") {
        const auto* src = reinterpret_cast<const float*>(base) + static_cast<size_t>(row) * cols;
        std::memcpy(dst, src, static_cast<size_t>(cols) * sizeof(float));
    } else {
        throw std::runtime_error("Unsupported safetensors dtype: " + t.dtype);
    }
}

std::vector<float> load_tensor_f32(const SafetensorEntry& t) {
    size_t n = 1;
    for (int64_t d : t.shape) n *= static_cast<size_t>(d);
    std::vector<float> out(n);
    if (t.dtype == "BF16") {
        const auto* src = reinterpret_cast<const uint16_t*>(t.ptr());
        for (size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(src[i]);
    } else if (t.dtype == "F16") {
        const auto* src = reinterpret_cast<const uint16_t*>(t.ptr());
        for (size_t i = 0; i < n; ++i) out[i] = fp16_to_f32(src[i]);
    } else if (t.dtype == "F32") {
        std::memcpy(out.data(), t.ptr(), n * sizeof(float));
    } else {
        throw std::runtime_error("Unsupported safetensors dtype: " + t.dtype);
    }
    return out;
}

std::vector<__half> load_tensor_f16(const SafetensorEntry& t) {
    auto f32 = load_tensor_f32(t);
    std::vector<__half> out(f32.size());
    for (size_t i = 0; i < f32.size(); ++i) out[i] = __float2half(f32[i]);
    return out;
}

std::string serialize_config_json(const ModelConfig& c) {
    std::ostringstream os;
    os << "{\n"
       << "  \"model_type\": \"" << c.model_type << "\",\n"
       << "  \"architectures\": [\"" << c.arch_name << "\"],\n"
       << "  \"vocab_size\": " << c.vocab_size << ",\n"
       << "  \"hidden_size\": " << c.hidden_size << ",\n"
       << "  \"intermediate_size\": " << c.intermediate_size << ",\n"
       << "  \"num_hidden_layers\": " << c.num_hidden_layers << ",\n"
       << "  \"max_position_embeddings\": " << c.max_position_embeddings << ",\n"
       << "  \"tie_word_embeddings\": " << (c.tie_word_embeddings ? "true" : "false") << ",\n"
       << "  \"rms_norm_eps\": " << c.rms_norm_eps << ",\n"
       << "  \"num_attention_heads\": " << c.num_attention_heads << ",\n"
       << "  \"num_key_value_heads\": " << c.num_key_value_heads << ",\n"
       << "  \"head_dim\": " << c.head_dim << ",\n"
       << "  \"attention_bias\": " << (c.attention_bias ? "true" : "false") << ",\n"
       << "  \"attn_output_gate\": " << (c.attn_output_gate ? "true" : "false") << ",\n"
       << "  \"qk_norm\": " << (c.qk_norm ? "true" : "false") << ",\n"
       << "  \"bos_token_id\": " << c.bos_token_id << ",\n"
       << "  \"eos_token_id\": [";
    for (size_t i = 0; i < c.eos_token_ids.size(); ++i) {
        if (i) os << ", ";
        os << c.eos_token_ids[i];
    }
    os << "],\n"
       << "  \"rope_theta\": " << c.rope_theta << ",\n"
       << "  \"rope_kind\": \"" << (c.rope_kind == RopeKind::Interleaved ? "interleaved" : (c.rope_kind == RopeKind::None ? "none" : "neox")) << "\",\n"
       << "  \"norm_kind\": \"" << (c.norm_kind == NormKind::RMSNormPlusOne ? "rmsnorm_plus_one" : (c.norm_kind == NormKind::LayerNorm ? "layernorm" : "rmsnorm")) << "\",\n"
       << "  \"partial_rotary_factor\": " << c.partial_rotary_factor << ",\n"
       << "  \"mrope_interleaved\": " << (c.mrope_interleaved ? "true" : "false") << ",\n"
       << "  \"mrope_section\": [";
    for (size_t i = 0; i < c.mrope_section.size(); ++i) {
        if (i) os << ", ";
        os << c.mrope_section[i];
    }
    os << "],\n"
       << "  \"linear_num_key_heads\": " << c.linear_num_key_heads << ",\n"
       << "  \"linear_key_head_dim\": " << c.linear_key_head_dim << ",\n"
       << "  \"linear_num_value_heads\": " << c.linear_num_value_heads << ",\n"
       << "  \"linear_value_head_dim\": " << c.linear_value_head_dim << ",\n"
       << "  \"linear_conv_kernel_dim\": " << c.linear_conv_kernel_dim << ",\n"
       << "  \"gdn_tiled_v_heads\": " << (c.gdn_tiled_v_heads ? "true" : "false") << ",\n"
       << "  \"mtp_num_hidden_layers\": " << c.mtp_num_hidden_layers << ",\n"
       << "  \"layer_types\": [";
    for (size_t i = 0; i < c.layer_types.size(); ++i) {
        if (i) os << ", ";
        os << "\"" << (c.layer_types[i] == LayerKind::LinearAttentionGDN ? "linear_attention" : "full_attention") << "\"";
    }
    os << "]\n}\n";
    return os.str();
}

} // namespace

size_t estimate_model_weight_bytes(const ModelConfig& cfg) {
    size_t bytes = 0;
    ModelWeights dummy;
    walk_model_layout(cfg, dummy, [&](size_t sz) -> void* {
        bytes = static_cast<size_t>(align_up(bytes, 256)) + sz;
        return nullptr;
    });
    return align_up(bytes + 65536ULL, 256);
}

ModelWeights load_synthetic_model(const ModelConfig& cfg, uint32_t seed, const LoaderOptions& opts) {
    ModelWeights w;
    allocate_model_arena(cfg, w);

    const int num_threads = (opts.num_threads > 0)
                                ? opts.num_threads
                                : std::max(1, static_cast<int>(std::thread::hardware_concurrency()));

    if (opts.fast_synthetic) {
        // Ultra-fast direct R4/F16 fill for benchmarking full 27B scale without waiting on CPU quantization.
        std::vector<uint8_t> h_qs;
        std::vector<__half> h_sc;
        std::vector<__half> h_f16;
        std::vector<float> h_f32;

        auto fill_r4_fast = [&](const R4Matrix& m, uint32_t tag) {
            const size_t qb = quant::r4_g64_qs_bytes(m.N, m.K);
            const size_t sb = quant::r4_g64_scales_bytes(m.N, m.K);
            h_qs.resize(qb);
            h_sc.resize(sb / sizeof(__half));
            const uint32_t pat = 0x79867986u ^ (tag * 0x11111111u);
            auto* q32 = reinterpret_cast<uint32_t*>(h_qs.data());
            for (size_t i = 0; i < qb / 4; ++i) q32[i] = pat + static_cast<uint32_t>(i & 0xF);
            const __half sc_val = __float2half(0.015f);
            std::fill(h_sc.begin(), h_sc.end(), sc_val);
            upload_r4_host_to_device(m, h_qs.data(), h_sc.data());
        };
        auto fill_f16 = [&](const __half* ptr, size_t n, float val) {
            if (!ptr) return;
            h_f16.assign(n, __float2half(val));
            upload_f16_host_to_device(ptr, h_f16.data(), n);
        };
        auto fill_f32 = [&](const float* ptr, size_t n, float val) {
            if (!ptr) return;
            h_f32.assign(n, val);
            upload_f32_host_to_device(ptr, h_f32.data(), n);
        };

        fill_r4_fast(w.embed_tokens, seed + 1);
        fill_f16(w.final_norm, cfg.hidden_size, cfg.norm_kind == NormKind::RMSNormPlusOne ? 0.0f : 1.0f);
        if (!cfg.tie_word_embeddings) fill_r4_fast(w.lm_head, seed + 2);

        const float norm_val = (cfg.norm_kind == NormKind::RMSNormPlusOne) ? 0.0f : 1.0f;
        for (size_t l = 0; l < w.layers.size(); ++l) {
            auto& lw = w.layers[l];
            fill_f16(lw.input_norm, cfg.hidden_size, norm_val);
            fill_f16(lw.post_attn_norm, cfg.hidden_size, norm_val);
            if (lw.kind == LayerKind::FullAttention) {
                fill_r4_fast(lw.q_proj, seed + 10 + static_cast<uint32_t>(l));
                fill_r4_fast(lw.kv_proj, seed + 20 + static_cast<uint32_t>(l));
                fill_r4_fast(lw.o_proj, seed + 30 + static_cast<uint32_t>(l));
                fill_f16(lw.q_norm, cfg.head_dim, norm_val);
                fill_f16(lw.k_norm, cfg.head_dim, norm_val);
            } else {
                const int conv_dim = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim +
                                     cfg.linear_num_value_heads * cfg.linear_value_head_dim;
                fill_r4_fast(lw.in_proj_qkvz, seed + 40 + static_cast<uint32_t>(l));
                fill_f16(lw.in_proj_ba, 2 * cfg.linear_num_value_heads * cfg.hidden_size, 0.01f);
                fill_f16(lw.conv_w, cfg.linear_conv_kernel_dim * conv_dim, 0.05f);
                fill_f32(lw.A_log, cfg.linear_num_value_heads, -1.0f);
                fill_f32(lw.dt_bias, cfg.linear_num_value_heads, 0.0f);
                fill_f16(lw.gdn_norm_w, cfg.linear_value_head_dim, 1.0f);
                fill_r4_fast(lw.out_proj, seed + 50 + static_cast<uint32_t>(l));
            }
            fill_r4_fast(lw.gate_up_proj, seed + 60 + static_cast<uint32_t>(l));
            fill_r4_fast(lw.down_proj, seed + 70 + static_cast<uint32_t>(l));
        }
        return w;
    }

    // Deterministic Gaussian packed R4 weights for accuracy/unit tests
    std::vector<uint8_t> h_qs;
    std::vector<__half> h_sc;
    auto pack_rand_r4 = [&](const R4Matrix& m, uint32_t tag, float std_dev) {
        h_qs.resize(quant::r4_g64_qs_bytes(m.N, m.K));
        h_sc.resize(quant::r4_g64_scales_bytes(m.N, m.K) / sizeof(__half));
        pack_r4_parallel(m.N, m.K, h_qs.data(), h_sc.data(), num_threads, [&](int r, float* row_out) {
            uint64_t state = (static_cast<uint64_t>(tag) << 32) ^ (static_cast<uint64_t>(r) * 2654435761ULL) ^ seed;
            for (int c = 0; c < m.K; ++c) {
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                const int32_t s = static_cast<int32_t>((state >> 32) & 0xFFFF) - 32768;
                row_out[c] = (static_cast<float>(s) / 32768.0f) * std_dev;
            }
        });
        upload_r4_host_to_device(m, h_qs.data(), h_sc.data());
    };

    auto fill_f16_const = [&](const __half* ptr, size_t n, float val) {
        if (!ptr) return;
        std::vector<__half> v(n, __float2half(val));
        upload_f16_host_to_device(ptr, v.data(), n);
    };
    auto fill_f32_const = [&](const float* ptr, size_t n, float val) {
        if (!ptr) return;
        std::vector<float> v(n, val);
        upload_f32_host_to_device(ptr, v.data(), n);
    };

    const float std_w = 0.02f;
    const float norm_val = (cfg.norm_kind == NormKind::RMSNormPlusOne) ? 0.0f : 1.0f;
    pack_rand_r4(w.embed_tokens, 1, 0.05f);
    fill_f16_const(w.final_norm, cfg.hidden_size, norm_val);
    if (!cfg.tie_word_embeddings) pack_rand_r4(w.lm_head, 2, std_w);

    for (size_t l = 0; l < w.layers.size(); ++l) {
        auto& lw = w.layers[l];
        const uint32_t base = static_cast<uint32_t>(100 + l * 16);
        fill_f16_const(lw.input_norm, cfg.hidden_size, norm_val);
        fill_f16_const(lw.post_attn_norm, cfg.hidden_size, norm_val);
        if (lw.kind == LayerKind::FullAttention) {
            pack_rand_r4(lw.q_proj, base + 0, std_w);
            pack_rand_r4(lw.kv_proj, base + 1, std_w);
            pack_rand_r4(lw.o_proj, base + 2, std_w);
            fill_f16_const(lw.q_norm, cfg.head_dim, norm_val);
            fill_f16_const(lw.k_norm, cfg.head_dim, norm_val);
        } else {
            const int conv_dim = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim +
                                 cfg.linear_num_value_heads * cfg.linear_value_head_dim;
            pack_rand_r4(lw.in_proj_qkvz, base + 3, std_w);
            fill_f16_const(lw.in_proj_ba, 2 * cfg.linear_num_value_heads * cfg.hidden_size, 0.01f);
            fill_f16_const(lw.conv_w, cfg.linear_conv_kernel_dim * conv_dim, 0.05f);
            fill_f32_const(lw.A_log, cfg.linear_num_value_heads, -1.0f);
            fill_f32_const(lw.dt_bias, cfg.linear_num_value_heads, 0.0f);
            fill_f16_const(lw.gdn_norm_w, cfg.linear_value_head_dim, 1.0f);
            pack_rand_r4(lw.out_proj, base + 4, std_w);
        }
        pack_rand_r4(lw.gate_up_proj, base + 5, std_w);
        pack_rand_r4(lw.down_proj, base + 6, std_w);
    }
    return w;
}

ModelWeights load_from_safetensors_dir(const std::string& model_dir,
                                       const ModelConfig* override_cfg,
                                       const LoaderOptions& opts) {
    ModelConfig cfg = override_cfg ? *override_cfg : ModelConfig::from_hf_json_file(model_dir + "/config.json");

    // Discover safetensors shard files
    std::vector<std::string> shard_files;
    const std::string index_path = model_dir + "/model.safetensors.index.json";
    std::ifstream idx_ifs(index_path);
    if (idx_ifs) {
        std::ostringstream oss;
        oss << idx_ifs.rdbuf();
        const json::Value idx_json = json::parse(oss.str());
        const json::Value* wmap = idx_json.get("weight_map");
        if (!wmap || !wmap->is_object()) throw std::runtime_error("Invalid safetensors index JSON");
        std::unordered_map<std::string, bool> seen;
        for (const auto& kv : std::get<json::Object>(wmap->data)) {
            const std::string fname = kv.second.as_str();
            if (!seen[fname]) {
                seen[fname] = true;
                shard_files.push_back(model_dir + "/" + fname);
            }
        }
    } else {
        shard_files.push_back(model_dir + "/model.safetensors");
    }

    std::vector<std::unique_ptr<MappedFile>> mapped;
    std::unordered_map<std::string, SafetensorEntry> tensors;
    for (const auto& sf : shard_files) {
        auto mf = std::make_unique<MappedFile>(sf);
        if (mf->size < 8) throw std::runtime_error("Corrupt safetensors file: " + sf);
        uint64_t hdr_len = 0;
        std::memcpy(&hdr_len, mf->data, 8);
        if (8 + hdr_len > mf->size) throw std::runtime_error("Invalid safetensors header length: " + sf);
        const std::string hdr_str(reinterpret_cast<const char*>(mf->data + 8), hdr_len);
        const json::Value hdr = json::parse(hdr_str);
        if (!hdr.is_object()) throw std::runtime_error("Safetensors header not object: " + sf);
        const size_t data_base = 8 + hdr_len;

        for (const auto& kv : std::get<json::Object>(hdr.data)) {
            if (kv.first == "__metadata__") continue;
            const auto& obj = kv.second;
            SafetensorEntry entry;
            entry.file = mf.get();
            entry.dtype = obj.get("dtype")->as_str();
            if (const auto* shp = obj.get("shape"); shp && shp->is_array()) {
                for (const auto& d : std::get<json::Array>(shp->data)) {
                    entry.shape.push_back(d.as_i64());
                }
            }
            const auto* offs = obj.get("data_offsets");
            const auto& arr = std::get<json::Array>(offs->data);
            const size_t b0 = static_cast<size_t>(arr[0].as_i64());
            const size_t b1 = static_cast<size_t>(arr[1].as_i64());
            entry.byte_offset = data_base + b0;
            entry.byte_length = b1 - b0;
            tensors.emplace(kv.first, std::move(entry));
        }
        mapped.push_back(std::move(mf));
    }

    auto find_tensor = [&](const std::string& suffix) -> const SafetensorEntry& {
        for (const char* prefix : {"model.language_model.", "model.", ""}) {
            auto it = tensors.find(std::string(prefix) + suffix);
            if (it != tensors.end()) return it->second;
        }
        throw std::runtime_error("Missing required tensor in safetensors: " + suffix);
    };
    auto try_find = [&](const std::string& suffix) -> const SafetensorEntry* {
        for (const char* prefix : {"model.language_model.", "model.", ""}) {
            auto it = tensors.find(std::string(prefix) + suffix);
            if (it != tensors.end()) return &it->second;
        }
        return nullptr;
    };

    if (!override_cfg) {
        if (const auto* emb = try_find("embed_tokens.weight"); emb && emb->shape.size() >= 2 && emb->shape[0] > 0) {
            cfg.vocab_size = static_cast<int32_t>(emb->shape[0]);
        }
        if (!cfg.tie_word_embeddings && try_find("lm_head.weight") == nullptr) {
            cfg.tie_word_embeddings = true;
        }
        for (int32_t l = 0; l < cfg.num_hidden_layers; ++l) {
            const LayerKind lk = (l < static_cast<int32_t>(cfg.layer_types.size()))
                                     ? cfg.layer_types[l]
                                     : LayerKind::FullAttention;
            if (lk == LayerKind::FullAttention) {
                const std::string pfx = "layers." + std::to_string(l) + ".";
                if (const auto* qp = try_find(pfx + "self_attn.q_proj.weight"); qp && qp->shape.size() >= 1) {
                    const int64_t q_dim = static_cast<int64_t>(cfg.num_attention_heads) * cfg.head_dim;
                    if (qp->shape[0] == 2 * q_dim) cfg.attn_output_gate = true;
                    else if (qp->shape[0] == q_dim) cfg.attn_output_gate = false;
                }
                if (try_find(pfx + "self_attn.q_norm.weight") != nullptr) {
                    cfg.qk_norm = true;
                } else {
                    cfg.qk_norm = false;
                }
                break;
            }
        }
    }

    ModelWeights w;
    allocate_model_arena(cfg, w);

    const int num_threads = (opts.num_threads > 0)
                                ? opts.num_threads
                                : std::max(1, static_cast<int>(std::thread::hardware_concurrency()));

    std::vector<uint8_t> h_qs;
    std::vector<__half> h_sc;

    auto pack_1_r4 = [&](const R4Matrix& dst, const SafetensorEntry& src) {
        h_qs.resize(quant::r4_g64_qs_bytes(dst.N, dst.K));
        h_sc.resize(quant::r4_g64_scales_bytes(dst.N, dst.K) / sizeof(__half));
        pack_r4_parallel(dst.N, dst.K, h_qs.data(), h_sc.data(), num_threads, [&](int r, float* row_out) {
            decode_row_to_f32(src, r, row_out, dst.K);
        });
        upload_r4_host_to_device(dst, h_qs.data(), h_sc.data());
    };

    auto pack_2_r4 = [&](const R4Matrix& dst, const SafetensorEntry& top, int top_rows, const SafetensorEntry& bot) {
        h_qs.resize(quant::r4_g64_qs_bytes(dst.N, dst.K));
        h_sc.resize(quant::r4_g64_scales_bytes(dst.N, dst.K) / sizeof(__half));
        pack_r4_parallel(dst.N, dst.K, h_qs.data(), h_sc.data(), num_threads, [&](int r, float* row_out) {
            if (r < top_rows) decode_row_to_f32(top, r, row_out, dst.K);
            else decode_row_to_f32(bot, r - top_rows, row_out, dst.K);
        });
        upload_r4_host_to_device(dst, h_qs.data(), h_sc.data());
    };

    pack_1_r4(w.embed_tokens, find_tensor("embed_tokens.weight"));
    {
        auto fn = load_tensor_f16(find_tensor("norm.weight"));
        upload_f16_host_to_device(w.final_norm, fn.data(), fn.size());
    }
    if (!cfg.tie_word_embeddings) {
        const SafetensorEntry* lm = try_find("lm_head.weight");
        if (lm) pack_1_r4(w.lm_head, *lm);
        else pack_1_r4(w.lm_head, find_tensor("embed_tokens.weight"));
    }

    for (int32_t l = 0; l < cfg.num_hidden_layers; ++l) {
        LayerWeights& lw = w.layers[l];
        const std::string pfx = "layers." + std::to_string(l) + ".";

        auto in_norm = load_tensor_f16(find_tensor(pfx + "input_layernorm.weight"));
        auto post_norm = load_tensor_f16(find_tensor(pfx + "post_attention_layernorm.weight"));
        upload_f16_host_to_device(lw.input_norm, in_norm.data(), in_norm.size());
        upload_f16_host_to_device(lw.post_attn_norm, post_norm.data(), post_norm.size());

        if (lw.kind == LayerKind::FullAttention) {
            pack_1_r4(lw.q_proj, find_tensor(pfx + "self_attn.q_proj.weight"));
            const int kv_rows = cfg.num_key_value_heads * cfg.head_dim;
            pack_2_r4(lw.kv_proj,
                      find_tensor(pfx + "self_attn.k_proj.weight"), kv_rows,
                      find_tensor(pfx + "self_attn.v_proj.weight"));
            pack_1_r4(lw.o_proj, find_tensor(pfx + "self_attn.o_proj.weight"));
            if (cfg.qk_norm) {
                auto qn = load_tensor_f16(find_tensor(pfx + "self_attn.q_norm.weight"));
                auto kn = load_tensor_f16(find_tensor(pfx + "self_attn.k_norm.weight"));
                upload_f16_host_to_device(lw.q_norm, qn.data(), qn.size());
                upload_f16_host_to_device(lw.k_norm, kn.data(), kn.size());
            }
        } else {
            const int32_t key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
            const int32_t value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
            const int32_t conv_dim = 2 * key_dim + value_dim;
            const int32_t conv_k = cfg.linear_conv_kernel_dim;

            if (const auto* qkvz = try_find(pfx + "linear_attn.in_proj_qkvz.weight")) {
                if (cfg.model_type.find("qwen3_next") != std::string::npos &&
                    cfg.linear_num_key_heads > 0 &&
                    (cfg.linear_num_value_heads % cfg.linear_num_key_heads) == 0) {
                    const int Kh = cfg.linear_num_key_heads;
                    const int Kd = cfg.linear_key_head_dim;
                    const int Vh = cfg.linear_num_value_heads;
                    const int Vd = cfg.linear_value_head_dim;
                    const int r = Vh / Kh;
                    const int grp_stride = 2 * Kd + 2 * r * Vd;
                    h_qs.resize(quant::r4_g64_qs_bytes(lw.in_proj_qkvz.N, lw.in_proj_qkvz.K));
                    h_sc.resize(quant::r4_g64_scales_bytes(lw.in_proj_qkvz.N, lw.in_proj_qkvz.K) / sizeof(__half));
                    pack_r4_parallel(lw.in_proj_qkvz.N, lw.in_proj_qkvz.K, h_qs.data(), h_sc.data(), num_threads,
                                     [&](int row, float* row_out) {
                                         int src_row = row;
                                         if (row < key_dim) {
                                             const int g = row / Kd;
                                             const int d = row % Kd;
                                             src_row = g * grp_stride + d;
                                         } else if (row < 2 * key_dim) {
                                             const int k_off = row - key_dim;
                                             const int g = k_off / Kd;
                                             const int d = k_off % Kd;
                                             src_row = g * grp_stride + Kd + d;
                                         } else if (row < 2 * key_dim + value_dim) {
                                             const int v_off = row - 2 * key_dim;
                                             const int g = v_off / (r * Vd);
                                             const int rv = v_off % (r * Vd);
                                             src_row = g * grp_stride + 2 * Kd + rv;
                                         } else {
                                             const int z_off = row - (2 * key_dim + value_dim);
                                             const int g = z_off / (r * Vd);
                                             const int rv = z_off % (r * Vd);
                                             src_row = g * grp_stride + 2 * Kd + r * Vd + rv;
                                         }
                                         decode_row_to_f32(*qkvz, src_row, row_out, lw.in_proj_qkvz.K);
                                     });
                    upload_r4_host_to_device(lw.in_proj_qkvz, h_qs.data(), h_sc.data());
                } else {
                    pack_1_r4(lw.in_proj_qkvz, *qkvz);
                }
            } else {
                pack_2_r4(lw.in_proj_qkvz,
                          find_tensor(pfx + "linear_attn.in_proj_qkv.weight"), conv_dim,
                          find_tensor(pfx + "linear_attn.in_proj_z.weight"));
            }

            if (const auto* ba = try_find(pfx + "linear_attn.in_proj_ba.weight")) {
                auto ba_raw = load_tensor_f16(*ba);
                if (cfg.model_type.find("qwen3_next") != std::string::npos &&
                    cfg.linear_num_key_heads > 0 &&
                    (cfg.linear_num_value_heads % cfg.linear_num_key_heads) == 0) {
                    const int Kh = cfg.linear_num_key_heads;
                    const int Vh = cfg.linear_num_value_heads;
                    const int r = Vh / Kh;
                    const int H = cfg.hidden_size;
                    std::vector<__half> ba_f16(ba_raw.size());
                    for (int g = 0; g < Kh; ++g) {
                        for (int i = 0; i < r; ++i) {
                            const int b_dst = g * r + i;
                            const int a_dst = Vh + g * r + i;
                            const int b_src = g * (2 * r) + i;
                            const int a_src = g * (2 * r) + r + i;
                            std::memcpy(ba_f16.data() + static_cast<size_t>(b_dst) * H,
                                        ba_raw.data() + static_cast<size_t>(b_src) * H,
                                        static_cast<size_t>(H) * sizeof(__half));
                            std::memcpy(ba_f16.data() + static_cast<size_t>(a_dst) * H,
                                        ba_raw.data() + static_cast<size_t>(a_src) * H,
                                        static_cast<size_t>(H) * sizeof(__half));
                        }
                    }
                    upload_f16_host_to_device(lw.in_proj_ba, ba_f16.data(), ba_f16.size());
                } else {
                    upload_f16_host_to_device(lw.in_proj_ba, ba_raw.data(), ba_raw.size());
                }
            } else {
                auto b_f16 = load_tensor_f16(find_tensor(pfx + "linear_attn.in_proj_b.weight"));
                auto a_f16 = load_tensor_f16(find_tensor(pfx + "linear_attn.in_proj_a.weight"));
                std::vector<__half> ba_f16;
                ba_f16.reserve(b_f16.size() + a_f16.size());
                ba_f16.insert(ba_f16.end(), b_f16.begin(), b_f16.end());
                ba_f16.insert(ba_f16.end(), a_f16.begin(), a_f16.end());
                upload_f16_host_to_device(lw.in_proj_ba, ba_f16.data(), ba_f16.size());
            }

            // Transpose conv1d weight [conv_dim, 1, conv_k] -> [conv_k, conv_dim] for coalesced access
            auto cw_src = load_tensor_f16(find_tensor(pfx + "linear_attn.conv1d.weight"));
            std::vector<__half> cw_t(static_cast<size_t>(conv_k) * conv_dim);
            for (int c = 0; c < conv_dim; ++c) {
                for (int k = 0; k < conv_k; ++k) {
                    cw_t[static_cast<size_t>(k) * conv_dim + c] = cw_src[static_cast<size_t>(c) * conv_k + k];
                }
            }
            upload_f16_host_to_device(lw.conv_w, cw_t.data(), cw_t.size());

            auto alog = load_tensor_f32(find_tensor(pfx + "linear_attn.A_log"));
            auto dtb = load_tensor_f32(find_tensor(pfx + "linear_attn.dt_bias"));
            auto gn = load_tensor_f16(find_tensor(pfx + "linear_attn.norm.weight"));
            upload_f32_host_to_device(lw.A_log, alog.data(), alog.size());
            upload_f32_host_to_device(lw.dt_bias, dtb.data(), dtb.size());
            upload_f16_host_to_device(lw.gdn_norm_w, gn.data(), gn.size());
            pack_1_r4(lw.out_proj, find_tensor(pfx + "linear_attn.out_proj.weight"));
        }

        pack_2_r4(lw.gate_up_proj,
                  find_tensor(pfx + "mlp.gate_proj.weight"), cfg.intermediate_size,
                  find_tensor(pfx + "mlp.up_proj.weight"));
        pack_1_r4(lw.down_proj, find_tensor(pfx + "mlp.down_proj.weight"));
    }

    return w;
}

// Minimal GGUF reader supporting F32, F16, BF16, Q4_0/1, Q5_0/1, Q8_0, Q2_K..Q6_K, IQ1..IQ4 repacking -> R4_GFX12_G64.
namespace gguf_detail {
struct Reader {
    const uint8_t* base;
    size_t size;
    size_t pos = 0;

    template <typename T>
    T read() {
        if (pos + sizeof(T) > size) throw std::runtime_error("GGUF unexpected EOF");
        T v;
        std::memcpy(&v, base + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }
    std::string read_str() {
        uint64_t len = read<uint64_t>();
        if (pos + len > size) throw std::runtime_error("GGUF string EOF");
        std::string s(reinterpret_cast<const char*>(base + pos), len);
        pos += len;
        return s;
    }
    void skip_value(uint32_t type) {
        switch (type) {
            case 0: case 1: case 7: pos += 1; break;
            case 2: case 3: pos += 2; break;
            case 4: case 5: case 6: pos += 4; break;
            case 10: case 11: case 12: pos += 8; break;
            case 8: (void)read_str(); break;
            case 9: {
                uint32_t et = read<uint32_t>();
                uint64_t n = read<uint64_t>();
                for (uint64_t i = 0; i < n; ++i) skip_value(et);
                break;
            }
            default: throw std::runtime_error("Unknown GGUF metadata type");
        }
    }
};

struct MetaVal {
    uint32_t type = 0;
    uint64_t u64 = 0;
    int64_t i64 = 0;
    double f64 = 0.0;
    bool bval = false;
    std::string sval;
    std::vector<uint32_t> u32_arr;
    std::vector<bool> bool_arr;

    uint32_t as_u32(uint32_t def = 0) const {
        if (type == 4 || type == 10 || type == 0 || type == 2) return static_cast<uint32_t>(u64);
        if (type == 5 || type == 11 || type == 1 || type == 3) return static_cast<uint32_t>(i64);
        return def;
    }
    float as_f32(float def = 0.0f) const {
        if (type == 6 || type == 12) return static_cast<float>(f64);
        return def;
    }
};

inline uint16_t load_u16(const void* p) {
    uint16_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline uint32_t load_u32(const void* p) {
    uint32_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

struct GGUFTensor {
    uint32_t n_dims = 0;
    uint64_t dims[4]{};
    uint32_t type = 0;
    uint64_t offset = 0;
    const uint8_t* ptr = nullptr;
};

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

void decode_gguf_row(const GGUFTensor& t, int row, float* dst, int cols) {
    if (t.type == 0) { // F32
        std::memcpy(dst, reinterpret_cast<const float*>(t.ptr) + static_cast<size_t>(row) * cols, cols * sizeof(float));
    } else if (t.type == 1) { // F16
        const auto* src = reinterpret_cast<const uint16_t*>(t.ptr) + static_cast<size_t>(row) * cols;
        for (int c = 0; c < cols; ++c) dst[c] = fp16_to_f32(src[c]);
    } else if (t.type == 30) { // BF16
        const auto* src = reinterpret_cast<const uint16_t*>(t.ptr) + static_cast<size_t>(row) * cols;
        for (int c = 0; c < cols; ++c) dst[c] = bf16_to_f32(src[c]);
    } else if (t.type == 2) { // Q4_0: 18 bytes / 32 weights
        const int blocks_per_row = cols / 32;
        const auto* blk = reinterpret_cast<const quant::BlockQ4_0*>(t.ptr) + static_cast<size_t>(row) * blocks_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const float d = fp16_to_f32(blk[b].d);
            for (int i = 0; i < 16; ++i) {
                const uint8_t q = blk[b].qs[i];
                dst[b * 32 + i] = static_cast<float>(static_cast<int>(q & 0x0F) - 8) * d;
                dst[b * 32 + 16 + i] = static_cast<float>(static_cast<int>(q >> 4) - 8) * d;
            }
        }
    } else if (t.type == 3) { // Q4_1: 20 bytes / 32 weights
        const int blocks_per_row = cols / 32;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 20;
        for (int b = 0; b < blocks_per_row; ++b) {
            uint16_t d_bits, m_bits;
            std::memcpy(&d_bits, row_p + b * 20, 2);
            std::memcpy(&m_bits, row_p + b * 20 + 2, 2);
            const float d = fp16_to_f32(d_bits);
            const float m = fp16_to_f32(m_bits);
            const uint8_t* qs = row_p + b * 20 + 4;
            for (int i = 0; i < 16; ++i) {
                dst[b * 32 + i] = static_cast<float>(qs[i] & 0x0F) * d + m;
                dst[b * 32 + 16 + i] = static_cast<float>(qs[i] >> 4) * d + m;
            }
        }
    } else if (t.type == 6) { // Q5_0: 22 bytes / 32 weights
        const int blocks_per_row = cols / 32;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 22;
        for (int b = 0; b < blocks_per_row; ++b) {
            uint16_t d_bits;
            uint32_t qh;
            std::memcpy(&d_bits, row_p + b * 22, 2);
            std::memcpy(&qh, row_p + b * 22 + 2, 4);
            const float d = fp16_to_f32(d_bits);
            const uint8_t* qs = row_p + b * 22 + 6;
            for (int i = 0; i < 16; ++i) {
                const uint32_t xh_0 = ((qh >> i) & 1u) << 4;
                const uint32_t xh_1 = ((qh >> (i + 16)) & 1u) << 4;
                const int x0 = static_cast<int>((qs[i] & 0x0Fu) | xh_0) - 16;
                const int x1 = static_cast<int>((qs[i] >> 4) | xh_1) - 16;
                dst[b * 32 + i] = static_cast<float>(x0) * d;
                dst[b * 32 + 16 + i] = static_cast<float>(x1) * d;
            }
        }
    } else if (t.type == 7) { // Q5_1: 24 bytes / 32 weights
        const int blocks_per_row = cols / 32;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 24;
        for (int b = 0; b < blocks_per_row; ++b) {
            uint16_t d_bits, m_bits;
            uint32_t qh;
            std::memcpy(&d_bits, row_p + b * 24, 2);
            std::memcpy(&m_bits, row_p + b * 24 + 2, 2);
            std::memcpy(&qh, row_p + b * 24 + 4, 4);
            const float d = fp16_to_f32(d_bits);
            const float m = fp16_to_f32(m_bits);
            const uint8_t* qs = row_p + b * 24 + 8;
            for (int i = 0; i < 16; ++i) {
                const uint32_t xh_0 = ((qh >> i) & 1u) << 4;
                const uint32_t xh_1 = ((qh >> (i + 16)) & 1u) << 4;
                const int x0 = static_cast<int>((qs[i] & 0x0Fu) | xh_0);
                const int x1 = static_cast<int>((qs[i] >> 4) | xh_1);
                dst[b * 32 + i] = static_cast<float>(x0) * d + m;
                dst[b * 32 + 16 + i] = static_cast<float>(x1) * d + m;
            }
        }
    } else if (t.type == 8) { // Q8_0: 34 bytes / 32 weights
        const int blocks_per_row = cols / 32;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 34;
        for (int b = 0; b < blocks_per_row; ++b) {
            uint16_t d_bits;
            std::memcpy(&d_bits, row_p + b * 34, 2);
            const float d = fp16_to_f32(d_bits);
            const int8_t* qs = reinterpret_cast<const int8_t*>(row_p + b * 34 + 2);
            for (int i = 0; i < 32; ++i) dst[b * 32 + i] = static_cast<float>(qs[i]) * d;
        }
    } else if (t.type == 12) { // Q4_K: 144 bytes / 256 weights
        const int blocks_per_row = cols / 256;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 144;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_p + b * 144;
            uint16_t d_bits, dmin_bits;
            std::memcpy(&d_bits, blk, 2);
            std::memcpy(&dmin_bits, blk + 2, 2);
            const float d = fp16_to_f32(d_bits);
            const float dmin = fp16_to_f32(dmin_bits);
            const uint8_t* scales = blk + 4;
            const uint8_t* qs = blk + 16;
            for (int chunk = 0; chunk < 4; ++chunk) {
                uint8_t sc0, mn0, sc1, mn1;
                get_scale_min_k4(chunk * 2 + 0, scales, sc0, mn0);
                get_scale_min_k4(chunk * 2 + 1, scales, sc1, mn1);
                const float d1 = d * sc0;
                const float m1 = dmin * mn0;
                const float d2 = d * sc1;
                const float m2 = dmin * mn1;
                const uint8_t* q = qs + chunk * 32;
                float* out = dst + b * 256 + chunk * 64;
                for (int l = 0; l < 32; ++l) {
                    out[l] = d1 * static_cast<float>(q[l] & 0x0F) - m1;
                    out[32 + l] = d2 * static_cast<float>(q[l] >> 4) - m2;
                }
            }
        }
    } else if (t.type == 13) { // Q5_K: 176 bytes / 256 weights
        const int blocks_per_row = cols / 256;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 176;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_p + b * 176;
            uint16_t d_bits, dmin_bits;
            std::memcpy(&d_bits, blk, 2);
            std::memcpy(&dmin_bits, blk + 2, 2);
            const float d = fp16_to_f32(d_bits);
            const float dmin = fp16_to_f32(dmin_bits);
            const uint8_t* scales = blk + 4;
            const uint8_t* qh = blk + 16;
            const uint8_t* qs = blk + 48;
            for (int chunk = 0; chunk < 4; ++chunk) {
                uint8_t sc0, mn0, sc1, mn1;
                get_scale_min_k4(chunk * 2 + 0, scales, sc0, mn0);
                get_scale_min_k4(chunk * 2 + 1, scales, sc1, mn1);
                const float d1 = d * sc0;
                const float m1 = dmin * mn0;
                const float d2 = d * sc1;
                const float m2 = dmin * mn1;
                const uint8_t* ql = qs + chunk * 32;
                const uint8_t u1 = static_cast<uint8_t>(1u << (2 * chunk));
                const uint8_t u2 = static_cast<uint8_t>(1u << (2 * chunk + 1));
                float* out = dst + b * 256 + chunk * 64;
                for (int l = 0; l < 32; ++l) {
                    out[l] = d1 * static_cast<float>((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
                    out[32 + l] = d2 * static_cast<float>((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
                }
            }
        }
    } else if (t.type == 14) { // Q6_K: 210 bytes / 256 weights
        const int blocks_per_row = cols / 256;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 210;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_p + b * 210;
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* scales = reinterpret_cast<const int8_t*>(blk + 192);
            uint16_t d_bits;
            std::memcpy(&d_bits, blk + 208, 2);
            const float d = fp16_to_f32(d_bits);
            for (int half = 0; half < 2; ++half) {
                const uint8_t* ql_p = ql + half * 64;
                const uint8_t* qh_p = qh + half * 32;
                const int8_t* sc_p = scales + half * 8;
                float* out = dst + b * 256 + half * 128;
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int q1 = static_cast<int>((ql_p[l] & 0x0F) | (((qh_p[l] >> 0) & 3) << 4)) - 32;
                    const int q2 = static_cast<int>((ql_p[l + 32] & 0x0F) | (((qh_p[l] >> 2) & 3) << 4)) - 32;
                    const int q3 = static_cast<int>((ql_p[l] >> 4) | (((qh_p[l] >> 4) & 3) << 4)) - 32;
                    const int q4 = static_cast<int>((ql_p[l + 32] >> 4) | (((qh_p[l] >> 6) & 3) << 4)) - 32;
                    out[l + 0] = d * sc_p[is + 0] * q1;
                    out[l + 32] = d * sc_p[is + 2] * q2;
                    out[l + 64] = d * sc_p[is + 4] * q3;
                    out[l + 96] = d * sc_p[is + 6] * q4;
                }
            }
        }
    } else if (t.type == 10) { // Q2_K: 84 bytes / 256 weights
        if (cols % 256 != 0) throw std::runtime_error("Q2_K requires cols % 256 == 0");
        const int blocks_per_row = cols / 256;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 84;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_p + b * 84;
            const uint8_t* scales = blk;
            const uint8_t* q = blk + 16;
            const float d = fp16_to_f32(load_u16(blk + 80));
            const float dmin = fp16_to_f32(load_u16(blk + 82));
            float* y = dst + b * 256;
            int is = 0;
            for (int n = 0; n < 256; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    uint8_t sc = scales[is++];
                    float dl = d * static_cast<float>(sc & 0x0F);
                    float ml = dmin * static_cast<float>(sc >> 4);
                    for (int l = 0; l < 16; ++l) {
                        *y++ = dl * static_cast<float>((q[l] >> shift) & 3) - ml;
                    }
                    sc = scales[is++];
                    dl = d * static_cast<float>(sc & 0x0F);
                    ml = dmin * static_cast<float>(sc >> 4);
                    for (int l = 0; l < 16; ++l) {
                        *y++ = dl * static_cast<float>((q[l + 16] >> shift) & 3) - ml;
                    }
                    shift += 2;
                }
                q += 32;
            }
        }
    } else if (t.type == 11) { // Q3_K: 110 bytes / 256 weights
        if (cols % 256 != 0) throw std::runtime_error("Q3_K requires cols % 256 == 0");
        const int blocks_per_row = cols / 256;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 110;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_p + b * 110;
            const uint8_t* hm = blk;
            const uint8_t* q = blk + 32;
            uint32_t aux[4]{};
            std::memcpy(aux, blk + 96, 12);
            const uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
            aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
            aux[0] = (aux[0] & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);
            aux[1] = (aux[1] & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);
            const float d_all = fp16_to_f32(load_u16(blk + 108));
            float* y = dst + b * 256;
            uint8_t m = 1;
            int is = 0;
            auto get_sc = [&](int idx) -> int {
                return static_cast<int>(static_cast<int8_t>(static_cast<uint8_t>((aux[idx >> 2] >> (8 * (idx & 3))) & 0xFFu)));
            };
            for (int n = 0; n < 256; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    const float dl0 = d_all * static_cast<float>(get_sc(is++) - 32);
                    for (int l = 0; l < 16; ++l) {
                        *y++ = dl0 * static_cast<float>(static_cast<int>((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                    }
                    const float dl1 = d_all * static_cast<float>(get_sc(is++) - 32);
                    for (int l = 0; l < 16; ++l) {
                        *y++ = dl1 * static_cast<float>(static_cast<int>((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                    }
                    shift += 2;
                    m = static_cast<uint8_t>(m << 1);
                }
                q += 32;
            }
        }
    } else if (t.type == 16) { // IQ2_XXS: 66 bytes / 256 weights
        if (cols % 256 != 0) throw std::runtime_error("IQ2_XXS requires cols % 256 == 0");
        const int blocks_per_row = cols / 256;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 66;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_p + b * 66;
            const float d = fp16_to_f32(load_u16(blk));
            float* y = dst + b * 256;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const uint32_t aux0 = load_u32(blk + 2 + 8 * ib32);
                const uint32_t aux1 = load_u32(blk + 2 + 8 * ib32 + 4);
                const float db = d * (0.5f + static_cast<float>(aux1 >> 28)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const uint64_t grid = iq2xxs_grid[(aux0 >> (8 * l)) & 0xFFu];
                    const uint8_t signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127u];
                    for (int j = 0; j < 8; ++j) {
                        const float gval = static_cast<float>((grid >> (8 * j)) & 0xFFu);
                        y[j] = db * gval * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    }
                    y += 8;
                }
            }
        }
    } else if (t.type == 17) { // IQ2_XS: 74 bytes / 256 weights
        if (cols % 256 != 0) throw std::runtime_error("IQ2_XS requires cols % 256 == 0");
        const int blocks_per_row = cols / 256;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 74;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_p + b * 74;
            const float d = fp16_to_f32(load_u16(blk));
            const uint8_t* qs = blk + 2;
            const uint8_t* scales = blk + 66;
            float* y = dst + b * 256;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const float db[2] = {
                    d * (0.5f + static_cast<float>(scales[ib32] & 0x0Fu)) * 0.25f,
                    d * (0.5f + static_cast<float>(scales[ib32] >> 4)) * 0.25f,
                };
                for (int l = 0; l < 4; ++l) {
                    const uint16_t qval = load_u16(qs + (4 * ib32 + l) * 2);
                    const uint64_t grid = iq2xs_grid[qval & 511u];
                    const uint8_t signs = ksigns_iq2xs[qval >> 9];
                    const float dl = db[l / 2];
                    for (int j = 0; j < 8; ++j) {
                        const float gval = static_cast<float>((grid >> (8 * j)) & 0xFFu);
                        y[j] = dl * gval * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    }
                    y += 8;
                }
            }
        }
    } else if (t.type == 18) { // IQ3_XXS: 98 bytes / 256 weights
        if (cols % 256 != 0) throw std::runtime_error("IQ3_XXS requires cols % 256 == 0");
        const int blocks_per_row = cols / 256;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 98;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_p + b * 98;
            const float d = fp16_to_f32(load_u16(blk));
            const uint8_t* qs = blk + 2;
            const uint8_t* scales_and_signs = blk + 66;
            float* y = dst + b * 256;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const uint32_t aux32 = load_u32(scales_and_signs + 4 * ib32);
                const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127u];
                    const uint32_t grid1 = iq3xxs_grid[qs[2 * l + 0]];
                    const uint32_t grid2 = iq3xxs_grid[qs[2 * l + 1]];
                    for (int j = 0; j < 4; ++j) {
                        y[j + 0] = db * static_cast<float>((grid1 >> (8 * j)) & 0xFFu) *
                                   ((signs & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                        y[j + 4] = db * static_cast<float>((grid2 >> (8 * j)) & 0xFFu) *
                                   ((signs & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                    }
                    y += 8;
                }
                qs += 8;
            }
        }
    } else if (t.type == 19) { // IQ1_S: 50 bytes / 256 weights
        if (cols % 256 != 0) throw std::runtime_error("IQ1_S requires cols % 256 == 0");
        const int blocks_per_row = cols / 256;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 50;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_p + b * 50;
            const float d = fp16_to_f32(load_u16(blk));
            const uint8_t* qs = blk + 2;
            const uint8_t* qh = blk + 34;
            float* y = dst + b * 256;
            for (int ib = 0; ib < 8; ++ib) {
                const uint16_t qhv = load_u16(qh + 2 * ib);
                const float dl = d * static_cast<float>(2 * ((qhv >> 12) & 7) + 1);
                const float delta = (qhv & 0x8000u) ? -kIQ1SDelta : kIQ1SDelta;
                for (int l = 0; l < 4; ++l) {
                    const uint64_t grid = iq1s_grid[qs[l] | (((static_cast<uint32_t>(qhv) >> (3 * l)) & 7u) << 8)];
                    for (int j = 0; j < 8; ++j) {
                        const int8_t gv = static_cast<int8_t>(static_cast<uint8_t>((grid >> (8 * j)) & 0xFFu));
                        y[j] = dl * (static_cast<float>(gv) + delta);
                    }
                    y += 8;
                }
                qs += 4;
            }
        }
    } else if (t.type == 20) { // IQ4_NL: 18 bytes / 32 weights
        if (cols % 32 != 0) throw std::runtime_error("IQ4_NL requires cols % 32 == 0");
        const int blocks_per_row = cols / 32;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 18;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_p + b * 18;
            const float d = fp16_to_f32(load_u16(blk));
            const uint8_t* qs = blk + 2;
            float* y = dst + b * 32;
            for (int j = 0; j < 16; ++j) {
                y[j + 0] = d * static_cast<float>(kvalues_iq4nl[qs[j] & 0x0Fu]);
                y[j + 16] = d * static_cast<float>(kvalues_iq4nl[qs[j] >> 4]);
            }
        }
    } else if (t.type == 21) { // IQ3_S: 110 bytes / 256 weights
        if (cols % 256 != 0) throw std::runtime_error("IQ3_S requires cols % 256 == 0");
        const int blocks_per_row = cols / 256;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 110;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_p + b * 110;
            const float d = fp16_to_f32(load_u16(blk));
            const uint8_t* qs = blk + 2;
            const uint8_t* qh = blk + 66;
            const uint8_t* signs = blk + 74;
            const uint8_t* scales = blk + 106;
            float* y = dst + b * 256;
            for (int ib32 = 0; ib32 < 8; ib32 += 2) {
                const float db1 = d * static_cast<float>(1 + 2 * (scales[ib32 / 2] & 0x0Fu));
                const float db2 = d * static_cast<float>(1 + 2 * (scales[ib32 / 2] >> 4));
                for (int l = 0; l < 4; ++l) {
                    const uint32_t grid1 = iq3s_grid[qs[2 * l + 0] | ((static_cast<uint32_t>(qh[0]) << (8 - 2 * l)) & 256u)];
                    const uint32_t grid2 = iq3s_grid[qs[2 * l + 1] | ((static_cast<uint32_t>(qh[0]) << (7 - 2 * l)) & 256u)];
                    for (int j = 0; j < 4; ++j) {
                        y[j + 0] = db1 * static_cast<float>((grid1 >> (8 * j)) & 0xFFu) *
                                   ((signs[l] & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                        y[j + 4] = db1 * static_cast<float>((grid2 >> (8 * j)) & 0xFFu) *
                                   ((signs[l] & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                    }
                    y += 8;
                }
                qs += 8;
                signs += 4;
                for (int l = 0; l < 4; ++l) {
                    const uint32_t grid1 = iq3s_grid[qs[2 * l + 0] | ((static_cast<uint32_t>(qh[1]) << (8 - 2 * l)) & 256u)];
                    const uint32_t grid2 = iq3s_grid[qs[2 * l + 1] | ((static_cast<uint32_t>(qh[1]) << (7 - 2 * l)) & 256u)];
                    for (int j = 0; j < 4; ++j) {
                        y[j + 0] = db2 * static_cast<float>((grid1 >> (8 * j)) & 0xFFu) *
                                   ((signs[l] & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                        y[j + 4] = db2 * static_cast<float>((grid2 >> (8 * j)) & 0xFFu) *
                                   ((signs[l] & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                    }
                    y += 8;
                }
                qh += 2;
                qs += 8;
                signs += 4;
            }
        }
    } else if (t.type == 22) { // IQ2_S: 82 bytes / 256 weights
        if (cols % 256 != 0) throw std::runtime_error("IQ2_S requires cols % 256 == 0");
        const int blocks_per_row = cols / 256;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 82;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_p + b * 82;
            const float d = fp16_to_f32(load_u16(blk));
            const uint8_t* qs = blk + 2;
            const uint8_t* signs = qs + 32;
            const uint8_t* qh = blk + 66;
            const uint8_t* scales = blk + 74;
            float* y = dst + b * 256;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const float db[2] = {
                    d * (0.5f + static_cast<float>(scales[ib32] & 0x0Fu)) * 0.25f,
                    d * (0.5f + static_cast<float>(scales[ib32] >> 4)) * 0.25f,
                };
                for (int l = 0; l < 4; ++l) {
                    const float dl = db[l / 2];
                    const uint64_t grid = iq2s_grid[qs[l] | ((static_cast<uint32_t>(qh[ib32]) << (8 - 2 * l)) & 0x300u)];
                    for (int j = 0; j < 8; ++j) {
                        y[j] = dl * static_cast<float>((grid >> (8 * j)) & 0xFFu) *
                               ((signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    }
                    y += 8;
                }
                qs += 4;
                signs += 4;
            }
        }
    } else if (t.type == 23) { // IQ4_XS: 136 bytes / 256 weights
        if (cols % 256 != 0) throw std::runtime_error("IQ4_XS requires cols % 256 == 0");
        const int blocks_per_row = cols / 256;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 136;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_p + b * 136;
            const float d = fp16_to_f32(load_u16(blk));
            const uint16_t scales_h = load_u16(blk + 2);
            const uint8_t* scales_l = blk + 4;
            const uint8_t* qs = blk + 8;
            float* y = dst + b * 256;
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) |
                               (((scales_h >> (2 * ib)) & 3) << 4);
                const float dl = d * static_cast<float>(ls - 32);
                for (int j = 0; j < 16; ++j) {
                    y[j + 0] = dl * static_cast<float>(kvalues_iq4nl[qs[j] & 0x0Fu]);
                    y[j + 16] = dl * static_cast<float>(kvalues_iq4nl[qs[j] >> 4]);
                }
                y += 32;
                qs += 16;
            }
        }
    } else if (t.type == 29) { // IQ1_M: 56 bytes / 256 weights
        if (cols % 256 != 0) throw std::runtime_error("IQ1_M requires cols % 256 == 0");
        const int blocks_per_row = cols / 256;
        const uint8_t* row_p = t.ptr + static_cast<size_t>(row) * blocks_per_row * 56;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row_p + b * 56;
            const uint8_t* qs = blk;
            const uint8_t* qh = blk + 32;
            const uint16_t sc[4] = {
                load_u16(blk + 48), load_u16(blk + 50), load_u16(blk + 52), load_u16(blk + 54),
            };
            const uint16_t d_bits = static_cast<uint16_t>(
                (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0u) | ((sc[2] >> 4) & 0x0F00u) | (sc[3] & 0xF000u));
            const float d = fp16_to_f32(d_bits);
            float* y = dst + b * 256;
            for (int ib = 0; ib < 8; ++ib) {
                const float dl1 = d * static_cast<float>(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
                const float dl2 = d * static_cast<float>(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
                const uint16_t idx[4] = {
                    static_cast<uint16_t>(qs[0] | ((static_cast<uint16_t>(qh[0]) << 8) & 0x700u)),
                    static_cast<uint16_t>(qs[1] | ((static_cast<uint16_t>(qh[0]) << 4) & 0x700u)),
                    static_cast<uint16_t>(qs[2] | ((static_cast<uint16_t>(qh[1]) << 8) & 0x700u)),
                    static_cast<uint16_t>(qs[3] | ((static_cast<uint16_t>(qh[1]) << 4) & 0x700u)),
                };
                const float delta[4] = {
                    (qh[0] & 0x08u) ? -kIQ1SDelta : kIQ1SDelta,
                    (qh[0] & 0x80u) ? -kIQ1SDelta : kIQ1SDelta,
                    (qh[1] & 0x08u) ? -kIQ1SDelta : kIQ1SDelta,
                    (qh[1] & 0x80u) ? -kIQ1SDelta : kIQ1SDelta,
                };
                for (int l = 0; l < 2; ++l) {
                    const uint64_t grid = iq1s_grid[idx[l]];
                    for (int j = 0; j < 8; ++j) {
                        const int8_t gv = static_cast<int8_t>(static_cast<uint8_t>((grid >> (8 * j)) & 0xFFu));
                        y[j] = dl1 * (static_cast<float>(gv) + delta[l]);
                    }
                    y += 8;
                }
                for (int l = 2; l < 4; ++l) {
                    const uint64_t grid = iq1s_grid[idx[l]];
                    for (int j = 0; j < 8; ++j) {
                        const int8_t gv = static_cast<int8_t>(static_cast<uint8_t>((grid >> (8 * j)) & 0xFFu));
                        y[j] = dl2 * (static_cast<float>(gv) + delta[l]);
                    }
                    y += 8;
                }
                qs += 4;
                qh += 2;
            }
        }
    } else {
        throw std::runtime_error("Unsupported GGUF tensor type: " + std::to_string(t.type));
    }
}
} // namespace gguf_detail

ModelWeights load_from_gguf(const std::string& gguf_path,
                            const ModelConfig* override_cfg,
                            const LoaderOptions& opts) {
    MappedFile mf(gguf_path);
    gguf_detail::Reader rd{mf.data, mf.size, 0};
    if (rd.read<uint32_t>() != 0x46554747u) throw std::runtime_error("Not a GGUF file: " + gguf_path);
    const uint32_t ver = rd.read<uint32_t>();
    if (ver < 2 || ver > 3) throw std::runtime_error("Unsupported GGUF version");
    const uint64_t n_tensors = rd.read<uint64_t>();
    const uint64_t n_kv = rd.read<uint64_t>();

    ModelConfig cfg{};
    uint64_t alignment = 32;
    std::string arch = "llama";
    std::unordered_map<std::string, gguf_detail::MetaVal> meta;

    for (uint64_t i = 0; i < n_kv; ++i) {
        const std::string key = rd.read_str();
        const uint32_t vt = rd.read<uint32_t>();

        // Skip huge tokenizer tables here (parsed in Tokenizer::from_gguf)
        if (key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.merges" ||
            key == "tokenizer.ggml.token_type" || key == "tokenizer.ggml.scores" ||
            key == "tokenizer.chat_template") {
            rd.skip_value(vt);
            continue;
        }

        gguf_detail::MetaVal mv;
        mv.type = vt;
        switch (vt) {
            case 0: mv.u64 = rd.read<uint8_t>(); break;
            case 1: mv.i64 = rd.read<int8_t>(); break;
            case 2: mv.u64 = rd.read<uint16_t>(); break;
            case 3: mv.i64 = rd.read<int16_t>(); break;
            case 4: mv.u64 = rd.read<uint32_t>(); break;
            case 5: mv.i64 = rd.read<int32_t>(); break;
            case 6: mv.f64 = rd.read<float>(); break;
            case 7: mv.bval = (rd.read<uint8_t>() != 0); break;
            case 8: mv.sval = rd.read_str(); break;
            case 10: mv.u64 = rd.read<uint64_t>(); break;
            case 11: mv.i64 = rd.read<int64_t>(); break;
            case 12: mv.f64 = rd.read<double>(); break;
            case 9: {
                const uint32_t et = rd.read<uint32_t>();
                const uint64_t n = rd.read<uint64_t>();
                if (n <= 4096 && (et == 4 || et == 5 || et == 10 || et == 11 || et == 7)) {
                    for (uint64_t j = 0; j < n; ++j) {
                        if (et == 7) mv.bool_arr.push_back(rd.read<uint8_t>() != 0);
                        else if (et == 4) mv.u32_arr.push_back(rd.read<uint32_t>());
                        else if (et == 5) mv.u32_arr.push_back(static_cast<uint32_t>(rd.read<int32_t>()));
                        else if (et == 10) mv.u32_arr.push_back(static_cast<uint32_t>(rd.read<uint64_t>()));
                        else if (et == 11) mv.u32_arr.push_back(static_cast<uint32_t>(rd.read<int64_t>()));
                    }
                } else {
                    for (uint64_t j = 0; j < n; ++j) rd.skip_value(et);
                }
                break;
            }
            default:
                throw std::runtime_error("Unknown GGUF metadata type");
        }
        meta.emplace(key, std::move(mv));
    }

    auto find_meta = [&](const std::string& k) -> const gguf_detail::MetaVal* {
        auto it = meta.find(k);
        return (it != meta.end()) ? &it->second : nullptr;
    };
    if (const auto* mv = find_meta("general.alignment")) alignment = std::max<uint64_t>(1, mv->as_u32(32));
    if (const auto* mv = find_meta("general.architecture"); mv && mv->type == 8) arch = mv->sval;
    cfg.model_type = arch;

    uint32_t rope_dim_count = 0;
    uint32_t nextn_predict_layers = 0;
    uint32_t full_attn_interval = 0;
    uint32_t ssm_state_size = 0;
    uint32_t ssm_inner_size = 0;
    std::vector<bool> recurrent_layers;

    if (const auto* mv = find_meta(arch + ".block_count")) cfg.num_hidden_layers = static_cast<int32_t>(mv->as_u32());
    if (const auto* mv = find_meta(arch + ".embedding_length")) cfg.hidden_size = static_cast<int32_t>(mv->as_u32());
    if (const auto* mv = find_meta(arch + ".feed_forward_length")) cfg.intermediate_size = static_cast<int32_t>(mv->as_u32());
    if (const auto* mv = find_meta(arch + ".attention.head_count")) cfg.num_attention_heads = static_cast<int32_t>(mv->as_u32());
    if (const auto* mv = find_meta(arch + ".attention.head_count_kv")) cfg.num_key_value_heads = static_cast<int32_t>(mv->as_u32());
    if (const auto* mv = find_meta(arch + ".attention.key_length")) cfg.head_dim = static_cast<int32_t>(mv->as_u32());
    if (const auto* mv = find_meta(arch + ".context_length")) cfg.max_position_embeddings = static_cast<int32_t>(mv->as_u32(32768));
    if (const auto* mv = find_meta(arch + ".attention.layer_norm_rms_epsilon")) cfg.rms_norm_eps = mv->as_f32(1e-6f);
    if (const auto* mv = find_meta(arch + ".rope.freq_base")) cfg.rope_theta = mv->as_f32(10000.0f);
    if (const auto* mv = find_meta(arch + ".rope.dimension_count")) rope_dim_count = mv->as_u32();
    if (const auto* mv = find_meta(arch + ".rope.dimension_sections")) {
        for (size_t j = 0; j < 3 && j < mv->u32_arr.size(); ++j) {
            cfg.mrope_section.push_back(static_cast<int32_t>(mv->u32_arr[j]));
        }
    }
    if (const auto* mv = find_meta(arch + ".nextn_predict_layers")) nextn_predict_layers = mv->as_u32();
    if (const auto* mv = find_meta(arch + ".full_attention_interval")) full_attn_interval = mv->as_u32();
    if (const auto* mv = find_meta(arch + ".attention.recurrent_layers")) recurrent_layers = mv->bool_arr;
    if (const auto* mv = find_meta(arch + ".ssm.conv_kernel")) cfg.linear_conv_kernel_dim = static_cast<int32_t>(mv->as_u32(4));
    if (const auto* mv = find_meta(arch + ".ssm.state_size")) ssm_state_size = mv->as_u32();
    if (const auto* mv = find_meta(arch + ".ssm.group_count")) cfg.linear_num_key_heads = static_cast<int32_t>(mv->as_u32());
    if (const auto* mv = find_meta(arch + ".ssm.time_step_rank")) cfg.linear_num_value_heads = static_cast<int32_t>(mv->as_u32());
    if (const auto* mv = find_meta(arch + ".ssm.inner_size")) ssm_inner_size = mv->as_u32();
    if (const auto* mv = find_meta("tokenizer.ggml.bos_token_id")) cfg.bos_token_id = static_cast<int32_t>(mv->as_u32());
    auto append_eos_meta = [&](const gguf_detail::MetaVal* mv) {
        if (!mv) return;
        if (!mv->u32_arr.empty()) {
            for (uint32_t eid : mv->u32_arr) cfg.eos_token_ids.push_back(static_cast<int32_t>(eid));
        } else {
            cfg.eos_token_ids.push_back(static_cast<int32_t>(mv->as_u32()));
        }
    };
    append_eos_meta(find_meta("tokenizer.ggml.eos_token_id"));
    append_eos_meta(find_meta("tokenizer.ggml.eot_token_id"));

    std::unordered_map<std::string, gguf_detail::GGUFTensor> tmap;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        const std::string name = rd.read_str();
        gguf_detail::GGUFTensor gt;
        gt.n_dims = rd.read<uint32_t>();
        for (uint32_t d = 0; d < gt.n_dims; ++d) gt.dims[d] = rd.read<uint64_t>();
        gt.type = rd.read<uint32_t>();
        gt.offset = rd.read<uint64_t>();
        tmap.emplace(name, gt);
    }

    const size_t data_base = static_cast<size_t>(align_up(rd.pos, alignment));
    for (auto& kv : tmap) {
        kv.second.ptr = mf.data + data_base + kv.second.offset;
    }

    const bool is_qwen35 = (arch.find("qwen35") != std::string::npos || arch.find("qwen3_5") != std::string::npos);
    const bool is_qwen3next = (arch.find("qwen3next") != std::string::npos || arch.find("qwen3_next") != std::string::npos);

    if (override_cfg) {
        cfg = *override_cfg;
    } else {
        if (auto it = tmap.find("token_embd.weight"); it != tmap.end() && it->second.n_dims >= 2) {
            cfg.vocab_size = static_cast<int32_t>(it->second.dims[1]);
        }
        if (cfg.head_dim == 0 && cfg.num_attention_heads > 0) {
            cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
        }
        if (cfg.num_key_value_heads == 0) cfg.num_key_value_heads = cfg.num_attention_heads;
        if (rope_dim_count > 0 && cfg.head_dim > 0) {
            cfg.partial_rotary_factor = static_cast<float>(rope_dim_count) / static_cast<float>(cfg.head_dim);
        }

        if (nextn_predict_layers > 0 && cfg.num_hidden_layers > static_cast<int32_t>(nextn_predict_layers)) {
            cfg.mtp_num_hidden_layers = static_cast<int32_t>(nextn_predict_layers);
            cfg.num_hidden_layers -= static_cast<int32_t>(nextn_predict_layers);
        }

        if (ssm_state_size > 0) {
            cfg.linear_key_head_dim = static_cast<int32_t>(ssm_state_size);
            cfg.linear_value_head_dim =
                (ssm_inner_size > 0 && cfg.linear_num_value_heads > 0)
                    ? static_cast<int32_t>(ssm_inner_size / static_cast<uint32_t>(cfg.linear_num_value_heads))
                    : static_cast<int32_t>(ssm_state_size);
        }

        if (cfg.num_hidden_layers > 0) {
            if (!recurrent_layers.empty() && static_cast<int32_t>(recurrent_layers.size()) >= cfg.num_hidden_layers) {
                cfg.layer_types.resize(cfg.num_hidden_layers);
                for (int32_t l = 0; l < cfg.num_hidden_layers; ++l) {
                    cfg.layer_types[l] = recurrent_layers[l] ? LayerKind::LinearAttentionGDN
                                                             : LayerKind::FullAttention;
                }
            } else if (full_attn_interval > 0 && cfg.linear_num_value_heads > 0) {
                cfg.layer_types.resize(cfg.num_hidden_layers);
                for (int32_t l = 0; l < cfg.num_hidden_layers; ++l) {
                    cfg.layer_types[l] = (((l + 1) % static_cast<int32_t>(full_attn_interval)) == 0)
                                             ? LayerKind::FullAttention
                                             : LayerKind::LinearAttentionGDN;
                }
            } else {
                cfg.layer_types.assign(cfg.num_hidden_layers, LayerKind::FullAttention);
            }
        }

        // GGUF norm weights already include any +1.0 offset from HF conversion.
        cfg.norm_kind = NormKind::RMSNorm;
        cfg.rope_kind = (arch == "llama") ? RopeKind::Interleaved : RopeKind::NeoX;

        if (is_qwen35) {
            int32_t mrope_sum = 0;
            for (int32_t s : cfg.mrope_section) mrope_sum += s;
            if (mrope_sum > 0) cfg.mrope_interleaved = true;
            if (cfg.linear_num_key_heads > 0 && cfg.linear_num_value_heads > cfg.linear_num_key_heads) {
                const auto* mv_hg = find_meta("prism.hadamard.gdn_v_grouped");
                const bool has_hadamard_grouped = (mv_hg && mv_hg->type == 7 && mv_hg->bval);
                cfg.gdn_tiled_v_heads = !has_hadamard_grouped;
            }
        }

        for (int32_t l = 0; l < cfg.num_hidden_layers; ++l) {
            if (cfg.layer_types[l] == LayerKind::FullAttention) {
                const std::string pfx = "blk." + std::to_string(l) + ".";
                if (tmap.count(pfx + "attn_q_norm.weight")) cfg.qk_norm = true;
                if (auto it = tmap.find(pfx + "attn_q.weight"); it != tmap.end() && it->second.n_dims >= 2) {
                    const uint64_t q_dim = static_cast<uint64_t>(cfg.num_attention_heads) * cfg.head_dim;
                    if (it->second.dims[1] == 2 * q_dim) cfg.attn_output_gate = true;
                    else if (it->second.dims[1] == q_dim) cfg.attn_output_gate = false;
                }
                break;
            }
        }
        cfg.tie_word_embeddings = (tmap.count("output.weight") == 0);
    }

    ModelWeights w;
    allocate_model_arena(cfg, w);
    const int num_threads = (opts.num_threads > 0)
                                ? opts.num_threads
                                : std::max(1, static_cast<int>(std::thread::hardware_concurrency()));

    std::vector<uint8_t> h_qs;
    std::vector<__half> h_sc;

    auto require = [&](const std::string& name) -> const gguf_detail::GGUFTensor& {
        auto it = tmap.find(name);
        if (it == tmap.end()) throw std::runtime_error("Missing GGUF tensor: " + name);
        return it->second;
    };
    auto try_get = [&](const std::string& name) -> const gguf_detail::GGUFTensor* {
        auto it = tmap.find(name);
        return (it != tmap.end()) ? &it->second : nullptr;
    };

    auto pack_g1 = [&](const R4Matrix& dst, const gguf_detail::GGUFTensor& src) {
        h_qs.resize(quant::r4_g64_qs_bytes(dst.N, dst.K));
        h_sc.resize(quant::r4_g64_scales_bytes(dst.N, dst.K) / sizeof(__half));
        pack_r4_parallel(dst.N, dst.K, h_qs.data(), h_sc.data(), num_threads, [&](int r, float* row_out) {
            gguf_detail::decode_gguf_row(src, r, row_out, dst.K);
        });
        upload_r4_host_to_device(dst, h_qs.data(), h_sc.data());
    };
    auto pack_g2 = [&](const R4Matrix& dst, const gguf_detail::GGUFTensor& top, int top_rows,
                       const gguf_detail::GGUFTensor& bot) {
        h_qs.resize(quant::r4_g64_qs_bytes(dst.N, dst.K));
        h_sc.resize(quant::r4_g64_scales_bytes(dst.N, dst.K) / sizeof(__half));
        pack_r4_parallel(dst.N, dst.K, h_qs.data(), h_sc.data(), num_threads, [&](int r, float* row_out) {
            if (r < top_rows) gguf_detail::decode_gguf_row(top, r, row_out, dst.K);
            else gguf_detail::decode_gguf_row(bot, r - top_rows, row_out, dst.K);
        });
        upload_r4_host_to_device(dst, h_qs.data(), h_sc.data());
    };
    auto load_g_f16 = [&](const __half* dst, const gguf_detail::GGUFTensor& src, int n) {
        std::vector<float> tmp(n);
        gguf_detail::decode_gguf_row(src, 0, tmp.data(), n);
        std::vector<__half> h(n);
        for (int i = 0; i < n; ++i) h[i] = __float2half(tmp[i]);
        upload_f16_host_to_device(dst, h.data(), n);
    };
    auto load_g_f16_rows = [&](const gguf_detail::GGUFTensor& src, int rows, int cols, std::vector<__half>& out) {
        out.resize(static_cast<size_t>(rows) * cols);
        std::vector<float> tmp(cols);
        for (int r = 0; r < rows; ++r) {
            gguf_detail::decode_gguf_row(src, r, tmp.data(), cols);
            __half* dst_row = out.data() + static_cast<size_t>(r) * cols;
            for (int c = 0; c < cols; ++c) dst_row[c] = __float2half(tmp[c]);
        }
    };

    pack_g1(w.embed_tokens, require("token_embd.weight"));
    load_g_f16(w.final_norm, require("output_norm.weight"), cfg.hidden_size);
    if (!cfg.tie_word_embeddings) {
        pack_g1(w.lm_head, require("output.weight"));
    }

    for (int32_t l = 0; l < cfg.num_hidden_layers; ++l) {
        LayerWeights& lw = w.layers[l];
        const std::string pfx = "blk." + std::to_string(l) + ".";
        load_g_f16(lw.input_norm, require(pfx + "attn_norm.weight"), cfg.hidden_size);
        if (const auto* pn = try_get(pfx + "post_attention_norm.weight")) {
            load_g_f16(lw.post_attn_norm, *pn, cfg.hidden_size);
        } else {
            load_g_f16(lw.post_attn_norm, require(pfx + "ffn_norm.weight"), cfg.hidden_size);
        }
        if (lw.kind == LayerKind::FullAttention) {
            pack_g1(lw.q_proj, require(pfx + "attn_q.weight"));
            pack_g2(lw.kv_proj, require(pfx + "attn_k.weight"),
                    cfg.num_key_value_heads * cfg.head_dim,
                    require(pfx + "attn_v.weight"));
            pack_g1(lw.o_proj, require(pfx + "attn_output.weight"));
            if (cfg.qk_norm) {
                load_g_f16(lw.q_norm, require(pfx + "attn_q_norm.weight"), cfg.head_dim);
                load_g_f16(lw.k_norm, require(pfx + "attn_k_norm.weight"), cfg.head_dim);
            }
        } else {
            const int32_t Kh = cfg.linear_num_key_heads;
            const int32_t Kd = cfg.linear_key_head_dim;
            const int32_t Vh = cfg.linear_num_value_heads;
            const int32_t Vd = cfg.linear_value_head_dim;
            const int32_t key_dim = Kh * Kd;
            const int32_t value_dim = Vh * Vd;
            const int32_t conv_dim = 2 * key_dim + value_dim;
            const int32_t conv_k = cfg.linear_conv_kernel_dim;
            const int32_t H = cfg.hidden_size;

            if (const auto* qkvz = try_get(pfx + "attn_qkvz.weight")) {
                if (is_qwen3next && Kh > 0 && (Vh % Kh) == 0) {
                    const int r_grp = Vh / Kh;
                    const int grp_stride = 2 * Kd + 2 * r_grp * Vd;
                    h_qs.resize(quant::r4_g64_qs_bytes(lw.in_proj_qkvz.N, lw.in_proj_qkvz.K));
                    h_sc.resize(quant::r4_g64_scales_bytes(lw.in_proj_qkvz.N, lw.in_proj_qkvz.K) / sizeof(__half));
                    pack_r4_parallel(lw.in_proj_qkvz.N, lw.in_proj_qkvz.K, h_qs.data(), h_sc.data(), num_threads,
                                     [&](int row, float* row_out) {
                                         int src_row = row;
                                         if (row < key_dim) {
                                             src_row = (row / Kd) * grp_stride + (row % Kd);
                                         } else if (row < 2 * key_dim) {
                                             const int ko = row - key_dim;
                                             src_row = (ko / Kd) * grp_stride + Kd + (ko % Kd);
                                         } else if (row < 2 * key_dim + value_dim) {
                                             const int vo = row - 2 * key_dim;
                                             src_row = (vo / (r_grp * Vd)) * grp_stride + 2 * Kd + (vo % (r_grp * Vd));
                                         } else {
                                             const int zo = row - (2 * key_dim + value_dim);
                                             src_row = (zo / (r_grp * Vd)) * grp_stride + 2 * Kd + r_grp * Vd +
                                                       (zo % (r_grp * Vd));
                                         }
                                         gguf_detail::decode_gguf_row(*qkvz, src_row, row_out, lw.in_proj_qkvz.K);
                                     });
                    upload_r4_host_to_device(lw.in_proj_qkvz, h_qs.data(), h_sc.data());
                } else {
                    pack_g1(lw.in_proj_qkvz, *qkvz);
                }
            } else if (const auto* qkv_gate = try_get(pfx + "attn_qkv_gate.weight")) {
                pack_g1(lw.in_proj_qkvz, *qkv_gate);
            } else {
                pack_g2(lw.in_proj_qkvz,
                        require(pfx + "attn_qkv.weight"), conv_dim,
                        require(pfx + "attn_gate.weight"));
            }

            const auto* ba_merged = try_get(pfx + "ssm_beta_alpha.weight");
            if (!ba_merged) ba_merged = try_get(pfx + "ssm_ba.weight");
            if (ba_merged != nullptr) {
                std::vector<__half> ba_raw;
                load_g_f16_rows(*ba_merged, 2 * Vh, H, ba_raw);
                if (is_qwen3next && Kh > 0 && (Vh % Kh) == 0) {
                    const int r_grp = Vh / Kh;
                    std::vector<__half> ba_f16(ba_raw.size());
                    for (int g = 0; g < Kh; ++g) {
                        for (int i = 0; i < r_grp; ++i) {
                            const int b_dst = g * r_grp + i;
                            const int a_dst = Vh + g * r_grp + i;
                            const int b_src = g * (2 * r_grp) + i;
                            const int a_src = g * (2 * r_grp) + r_grp + i;
                            std::memcpy(ba_f16.data() + static_cast<size_t>(b_dst) * H,
                                        ba_raw.data() + static_cast<size_t>(b_src) * H,
                                        static_cast<size_t>(H) * sizeof(__half));
                            std::memcpy(ba_f16.data() + static_cast<size_t>(a_dst) * H,
                                        ba_raw.data() + static_cast<size_t>(a_src) * H,
                                        static_cast<size_t>(H) * sizeof(__half));
                        }
                    }
                    upload_f16_host_to_device(lw.in_proj_ba, ba_f16.data(), ba_f16.size());
                } else {
                    upload_f16_host_to_device(lw.in_proj_ba, ba_raw.data(), ba_raw.size());
                }
            } else {
                std::vector<__half> b_f16, a_f16;
                load_g_f16_rows(require(pfx + "ssm_beta.weight"), Vh, H, b_f16);
                load_g_f16_rows(require(pfx + "ssm_alpha.weight"), Vh, H, a_f16);
                std::vector<__half> ba_f16;
                ba_f16.reserve(b_f16.size() + a_f16.size());
                ba_f16.insert(ba_f16.end(), b_f16.begin(), b_f16.end());
                ba_f16.insert(ba_f16.end(), a_f16.begin(), a_f16.end());
                upload_f16_host_to_device(lw.in_proj_ba, ba_f16.data(), ba_f16.size());
            }

            // Transpose conv1d weight [conv_dim, conv_k] -> [conv_k, conv_dim] for coalesced access
            const auto& conv_t = require(pfx + "ssm_conv1d.weight");
            std::vector<__half> cw_t(static_cast<size_t>(conv_k) * conv_dim);
            std::vector<float> crow(conv_k);
            for (int c = 0; c < conv_dim; ++c) {
                gguf_detail::decode_gguf_row(conv_t, c, crow.data(), conv_k);
                for (int k = 0; k < conv_k; ++k) {
                    cw_t[static_cast<size_t>(k) * conv_dim + c] = __float2half(crow[k]);
                }
            }
            upload_f16_host_to_device(lw.conv_w, cw_t.data(), cw_t.size());

            // GGUF ssm_a stores -exp(A_log); convert back to A_log = log(-ssm_a).
            std::vector<float> ssm_a(Vh);
            gguf_detail::decode_gguf_row(require(pfx + "ssm_a"), 0, ssm_a.data(), Vh);
            for (int h = 0; h < Vh; ++h) {
                if (!(ssm_a[h] < 0.0f)) {
                    throw std::runtime_error("Invalid non-negative ssm_a in GGUF layer " + std::to_string(l));
                }
                ssm_a[h] = std::log(-ssm_a[h]);
            }
            upload_f32_host_to_device(lw.A_log, ssm_a.data(), Vh);

            std::vector<float> dtb(Vh);
            gguf_detail::decode_gguf_row(require(pfx + "ssm_dt.bias"), 0, dtb.data(), Vh);
            upload_f32_host_to_device(lw.dt_bias, dtb.data(), Vh);

            load_g_f16(lw.gdn_norm_w, require(pfx + "ssm_norm.weight"), Vd);
            pack_g1(lw.out_proj, require(pfx + "ssm_out.weight"));
        }
        pack_g2(lw.gate_up_proj, require(pfx + "ffn_gate.weight"), cfg.intermediate_size,
                require(pfx + "ffn_up.weight"));
        pack_g1(lw.down_proj, require(pfx + "ffn_down.weight"));
    }
    return w;
}

void save_rxr4_checkpoint(const ModelWeights& weights, const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Failed to open RXR4 output: " + path);

    const char magic[4] = {'R', 'X', 'R', '4'};
    const uint32_t version = 1;
    const std::string cfg_str = serialize_config_json(weights.config);
    const uint64_t cfg_len = cfg_str.size();
    const uint64_t arena_bytes = weights.weight_arena.used();

    ofs.write(magic, 4);
    ofs.write(reinterpret_cast<const char*>(&version), 4);
    ofs.write(reinterpret_cast<const char*>(&cfg_len), 8);
    ofs.write(reinterpret_cast<const char*>(&arena_bytes), 8);
    ofs.write(cfg_str.data(), cfg_len);

    // Stream device arena in 64 MB chunks
    constexpr size_t CHUNK = 64ULL << 20;
    std::vector<uint8_t> buf(std::min(CHUNK, static_cast<size_t>(arena_bytes)));
    size_t off = 0;
    while (off < arena_bytes) {
        const size_t n = std::min(CHUNK, static_cast<size_t>(arena_bytes - off));
        RX_HIP_CHECK(hipMemcpy(buf.data(), static_cast<const uint8_t*>(weights.weight_arena.base()) + off,
                               n, hipMemcpyDeviceToHost));
        ofs.write(reinterpret_cast<const char*>(buf.data()), n);
        off += n;
    }
}

ModelWeights load_from_rxr4(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open RXR4 file: " + path);

    char magic[4]{};
    uint32_t version = 0;
    uint64_t cfg_len = 0, arena_bytes = 0;
    ifs.read(magic, 4);
    ifs.read(reinterpret_cast<char*>(&version), 4);
    ifs.read(reinterpret_cast<char*>(&cfg_len), 8);
    ifs.read(reinterpret_cast<char*>(&arena_bytes), 8);
    if (std::memcmp(magic, "RXR4", 4) != 0 || version != 1) {
        throw std::runtime_error("Invalid RXR4 header: " + path);
    }
    std::string cfg_str(cfg_len, '\0');
    ifs.read(cfg_str.data(), cfg_len);

    ModelConfig cfg = ModelConfig::from_hf_json_string(cfg_str);
    ModelWeights w;
    allocate_model_arena(cfg, w);
    if (w.weight_arena.used() != arena_bytes) {
        throw std::runtime_error("RXR4 arena size mismatch");
    }

    constexpr size_t CHUNK = 64ULL << 20;
    std::vector<uint8_t> buf(std::min(CHUNK, static_cast<size_t>(arena_bytes)));
    size_t off = 0;
    while (off < arena_bytes) {
        const size_t n = std::min(CHUNK, static_cast<size_t>(arena_bytes - off));
        ifs.read(reinterpret_cast<char*>(buf.data()), n);
        RX_HIP_CHECK(hipMemcpy(static_cast<uint8_t*>(w.weight_arena.base()) + off, buf.data(),
                               n, hipMemcpyHostToDevice));
        off += n;
    }
    return w;
}

} // namespace rxinfer

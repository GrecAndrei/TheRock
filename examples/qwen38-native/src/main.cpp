#include "gguf.h"
#include "kernels.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fcntl.h>
#include <map>
#include <numeric>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void hip_check(hipError_t status, const char *operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

void show_device() {
    int runtime_version = 0;
    int device_count = 0;
    hip_check(hipRuntimeGetVersion(&runtime_version), "hipRuntimeGetVersion");
    hip_check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");

    std::cout << "TheRock HIP runtime " << runtime_version << '\n';
    std::cout << "devices: " << device_count << '\n';
    for (int index = 0; index < device_count; ++index) {
        hipDeviceProp_t properties{};
        hip_check(hipGetDeviceProperties(&properties, index), "hipGetDeviceProperties");
        std::cout << "[" << index << "] " << properties.name << '\n'
                  << "    arch:       " << properties.gcnArchName << '\n'
                  << "    VRAM:       " << std::fixed << std::setprecision(2)
                  << static_cast<double>(properties.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0) << " GiB\n"
                  << "    CUs:        " << properties.multiProcessorCount << '\n'
                  << "    max clock:  " << properties.clockRate / 1000 << " MHz\n"
                  << "    warp size:  " << properties.warpSize << '\n';
    }
}

void inspect(const std::string &path, bool list_tensors) {
    const qwen_native::GgufFile model(path);
    std::cout << model.path() << '\n'
              << "  GGUF version: " << model.version() << '\n'
              << "  file size:    " << std::fixed << std::setprecision(2)
              << static_cast<double>(model.file_size()) / (1024.0 * 1024.0 * 1024.0) << " GiB\n"
              << "  data offset:  " << model.data_offset() << '\n'
              << "  metadata:     " << model.metadata().size() << '\n'
              << "  tensors:      " << model.tensors().size() << '\n';

    static constexpr const char *important_keys[] = {
        "general.architecture", "general.name", "general.file_type",
        "qwen35.block_count", "qwen35.embedding_length", "qwen35.feed_forward_length",
        "qwen35.attention.head_count", "qwen35.attention.head_count_kv",
        "qwen35.context_length", "tokenizer.ggml.model", "dflash.block_size",
        "dflash.conv_kernel_size", "dflash.conv_group_size", "dflash.selector_rank",
        "dflash.selector_top_k", "dflash.target_layers"
    };
    for (const auto *key : important_keys) {
        if (const auto *value = model.find_metadata(key)) {
            std::cout << "  " << key << ": " << qwen_native::metadata_to_string(*value) << '\n';
        }
    }

    std::map<std::string, std::size_t> types;
    for (const auto &tensor : model.tensors()) {
        ++types[qwen_native::ggml_type_name(tensor.type)];
    }
    std::cout << "  tensor types:";
    for (const auto &[name, count] : types) {
        std::cout << ' ' << name << '=' << count;
    }
    std::cout << '\n';

    if (list_tensors) {
        for (const auto &tensor : model.tensors()) {
            std::cout << "  " << tensor.name << " [";
            for (std::size_t i = 0; i < tensor.dimensions.size(); ++i) {
                std::cout << (i == 0 ? "" : " x ") << tensor.dimensions[i];
            }
            std::cout << "] " << qwen_native::ggml_type_name(tensor.type)
                      << " @ " << model.data_offset() + tensor.offset << '\n';
        }
    }
}

void show_metadata(const std::string &path) {
    const qwen_native::GgufFile model(path);
    std::map<std::string, qwen_native::MetadataValue> sorted(model.metadata().begin(), model.metadata().end());
    for (const auto &[key, value] : sorted) {
        std::cout << key << ": " << qwen_native::metadata_to_string(value) << '\n';
    }
}

void show_token(const std::string &path, std::size_t token) {
    const qwen_native::GgufFile model(path);
    const auto *metadata = model.find_metadata("tokenizer.ggml.tokens");
    if (!metadata || !std::holds_alternative<std::vector<std::string>>(*metadata)) {
        throw std::runtime_error("GGUF does not contain a string tokenizer vocabulary");
    }
    const auto &tokens = std::get<std::vector<std::string>>(*metadata);
    if (token >= tokens.size()) {
        throw std::out_of_range("token ID is outside the tokenizer vocabulary");
    }
    std::cout << "token " << token << ": " << std::quoted(tokens[token]) << '\n';
}

class Mapping {
public:
    explicit Mapping(const std::string &path) {
        descriptor_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor_ < 0) {
            throw std::runtime_error("cannot open model for mapping");
        }
        struct stat status {};
        if (fstat(descriptor_, &status) != 0) {
            close(descriptor_);
            throw std::runtime_error("cannot stat model");
        }
        size_ = static_cast<std::size_t>(status.st_size);
        data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor_, 0);
        if (data_ == MAP_FAILED) {
            close(descriptor_);
            throw std::runtime_error("cannot mmap model");
        }
    }

    ~Mapping() {
        munmap(data_, size_);
        close(descriptor_);
    }

    Mapping(const Mapping &) = delete;
    Mapping &operator=(const Mapping &) = delete;

    const void *at(std::size_t offset) const {
        if (offset >= size_) {
            throw std::out_of_range("tensor offset is outside model");
        }
        return static_cast<const char *>(data_) + offset;
    }

private:
    int descriptor_ = -1;
    void *data_ = nullptr;
    std::size_t size_ = 0;
};

float fp16_to_float(std::uint16_t bits) {
    __half value;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return __half2float(value);
}

void unpack_q4k_scale_min(const std::uint8_t *packed, int index,
                          std::uint8_t &scale, std::uint8_t &minimum) {
    if (index < 4) {
        scale = packed[index] & 63;
        minimum = packed[index + 4] & 63;
    } else {
        scale = (packed[index + 4] & 0x0F) | ((packed[index - 4] >> 6) << 4);
        minimum = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
    }
}

std::vector<float> dequantize_q4k_row(const void *data, std::size_t values) {
    constexpr std::size_t block_values = 256;
    constexpr std::size_t block_bytes = 144;
    if (values == 0 || values % block_values != 0) {
        throw std::runtime_error("Q4_K row has an invalid width");
    }
    const auto *source = static_cast<const std::uint8_t *>(data);
    std::vector<float> result(values);
    for (std::size_t block_index = 0; block_index < values / block_values; ++block_index) {
        const auto *block = source + block_index * block_bytes;
        const float d = fp16_to_float(*reinterpret_cast<const std::uint16_t *>(block));
        const float dmin = fp16_to_float(*reinterpret_cast<const std::uint16_t *>(block + 2));
        const auto *scales = block + 4;
        const auto *quants = block + 16;
        for (int group = 0; group < 4; ++group) {
            std::uint8_t scale0 = 0;
            std::uint8_t min0 = 0;
            std::uint8_t scale1 = 0;
            std::uint8_t min1 = 0;
            unpack_q4k_scale_min(scales, group * 2, scale0, min0);
            unpack_q4k_scale_min(scales, group * 2 + 1, scale1, min1);
            for (int lane = 0; lane < 32; ++lane) {
                const std::uint8_t quant = quants[group * 32 + lane];
                const std::size_t offset = block_index * block_values + group * 64 + lane;
                result[offset] = d * static_cast<float>(scale0) * static_cast<float>(quant & 0x0F)
                               - dmin * static_cast<float>(min0);
                result[offset + 32] = d * static_cast<float>(scale1) * static_cast<float>(quant >> 4)
                                    - dmin * static_cast<float>(min1);
            }
        }
    }
    return result;
}

double metadata_number_or(const qwen_native::GgufFile &model, const std::string &key,
                          double fallback) {
    const auto *value = model.find_metadata(key);
    if (!value) {
        return fallback;
    }
    if (const auto *number = std::get_if<double>(value)) {
        return *number;
    }
    if (const auto *number = std::get_if<std::uint64_t>(value)) {
        return static_cast<double>(*number);
    }
    if (const auto *number = std::get_if<std::int64_t>(value)) {
        return static_cast<double>(*number);
    }
    throw std::runtime_error("GGUF metadata is not numeric: " + key);
}

const qwen_native::TensorInfo &require_tensor(const qwen_native::GgufFile &model,
                                              const std::string &name) {
    const auto *tensor = model.find_tensor(name);
    if (!tensor) {
        throw std::runtime_error("missing required tensor: " + name);
    }
    return *tensor;
}

const void *tensor_data(const Mapping &mapping, const qwen_native::GgufFile &model,
                        const qwen_native::TensorInfo &tensor) {
    return mapping.at(model.data_offset() + tensor.offset);
}

std::vector<float> rms_norm(const std::vector<float> &input, const float *weights,
                            std::size_t width, float epsilon) {
    if (width == 0 || input.size() % width != 0) {
        throw std::runtime_error("RMS norm has an inconsistent width");
    }
    std::vector<float> output(input.size());
    for (std::size_t row = 0; row < input.size() / width; ++row) {
        const float *source = input.data() + row * width;
        float sum = 0.0F;
        for (std::size_t index = 0; index < width; ++index) {
            sum = std::fma(source[index], source[index], sum);
        }
        const float scale = 1.0F / std::sqrt(sum / static_cast<float>(width) + epsilon);
        for (std::size_t index = 0; index < width; ++index) {
            output[row * width + index] = source[index] * scale * weights[index];
        }
    }
    return output;
}

void l2_norm_heads(std::vector<float> &values, std::size_t head_size, float epsilon) {
    if (head_size == 0 || values.size() % head_size != 0) {
        throw std::runtime_error("L2 norm has an inconsistent head width");
    }
    for (std::size_t head = 0; head < values.size() / head_size; ++head) {
        float sum = 0.0F;
        for (std::size_t index = 0; index < head_size; ++index) {
            const float value = values[head * head_size + index];
            sum = std::fma(value, value, sum);
        }
        const float scale = 1.0F / std::sqrt(sum + epsilon);
        for (std::size_t index = 0; index < head_size; ++index) {
            values[head * head_size + index] *= scale;
        }
    }
}

float silu(float value) {
    return value / (1.0F + std::exp(-value));
}

float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

std::vector<float> add_vectors(const std::vector<float> &left, const std::vector<float> &right) {
    if (left.size() != right.size()) {
        throw std::runtime_error("residual operands have different sizes");
    }
    std::vector<float> output(left.size());
    for (std::size_t index = 0; index < left.size(); ++index) {
        output[index] = left[index] + right[index];
    }
    return output;
}

void target_first_token(const std::string &path, std::uint32_t token) {
    constexpr std::size_t embedding_size = 5120;
    constexpr std::size_t ffn_size = 17408;
    constexpr std::size_t recurrent_key_heads = 4;
    constexpr std::size_t recurrent_value_heads = 48;
    constexpr std::size_t recurrent_head_size = 128;
    constexpr std::size_t attention_heads = 24;
    constexpr std::size_t attention_kv_heads = 4;
    constexpr std::size_t attention_head_size = 256;

    const qwen_native::GgufFile model(path);
    const auto *architecture = model.find_metadata("general.architecture");
    if (!architecture || !std::holds_alternative<std::string>(*architecture) ||
        std::get<std::string>(*architecture) != "qwen35") {
        throw std::runtime_error("target-first-token requires a qwen35 GGUF target model");
    }
    const auto &embedding = require_tensor(model, "token_embd.weight");
    if (embedding.type != 12 || embedding.dimensions.size() != 2 ||
        embedding.dimensions[0] != embedding_size || token >= embedding.dimensions[1]) {
        throw std::runtime_error("target embedding is not the expected Q4_K vocabulary matrix");
    }
    const auto live_layers = static_cast<int>(metadata_number_or(model, "qwen35.block_count", 0.0)
                                            - metadata_number_or(model, "qwen35.nextn_predict_layers", 0.0));
    if (live_layers != 64 || metadata_number_or(model, "qwen35.embedding_length", 0.0) != embedding_size ||
        metadata_number_or(model, "qwen35.feed_forward_length", 0.0) != ffn_size) {
        throw std::runtime_error("target-first-token only supports the supplied Qwen3.8-27B topology");
    }
    const float epsilon = static_cast<float>(metadata_number_or(
        model, "qwen35.attention.layer_norm_rms_epsilon", 1.0e-6));
    const Mapping mapping(path);

    const auto f32_data = [&](const qwen_native::TensorInfo &tensor, std::size_t values) -> const float * {
        if (tensor.type != 0 || tensor.dimensions.empty() ||
            std::accumulate(tensor.dimensions.begin(), tensor.dimensions.end(), std::size_t{1},
                            [](std::size_t left, std::uint64_t right) { return left * static_cast<std::size_t>(right); }) != values) {
            throw std::runtime_error("expected F32 tensor with a different shape: " + tensor.name);
        }
        return static_cast<const float *>(tensor_data(mapping, model, tensor));
    };
    const auto projection = [&](const qwen_native::TensorInfo &tensor,
                                const std::vector<float> &input) -> std::vector<float> {
        if (tensor.dimensions.size() != 2 || tensor.dimensions[0] != input.size()) {
            throw std::runtime_error("projection shape does not match activation: " + tensor.name);
        }
        if (tensor.type != 12 && tensor.type != 13 && tensor.type != 14) {
            throw std::runtime_error("projection is not Q4_K, Q5_K, or Q6_K: " + tensor.name);
        }
        std::vector<float> output(static_cast<std::size_t>(tensor.dimensions[1]));
        qwen_native::execute_k_matvec(tensor_data(mapping, model, tensor), tensor.type,
                                      input.data(), input.size(), output.data(), output.size());
        return output;
    };
    const auto f32_projection = [&](const qwen_native::TensorInfo &tensor,
                                    const std::vector<float> &input) -> std::vector<float> {
        if (tensor.type != 0 || tensor.dimensions.size() != 2 || tensor.dimensions[0] != input.size()) {
            throw std::runtime_error("F32 projection shape does not match activation: " + tensor.name);
        }
        const std::size_t outputs = static_cast<std::size_t>(tensor.dimensions[1]);
        const auto *weights = static_cast<const float *>(tensor_data(mapping, model, tensor));
        std::vector<float> output(outputs);
        for (std::size_t output_index = 0; output_index < outputs; ++output_index) {
            const float *row = weights + output_index * input.size();
            float sum = 0.0F;
            for (std::size_t input_index = 0; input_index < input.size(); ++input_index) {
                sum = std::fma(row[input_index], input[input_index], sum);
            }
            output[output_index] = sum;
        }
        return output;
    };
    const auto vector_weights = [&](const std::string &name, std::size_t values) -> const float * {
        return f32_data(require_tensor(model, name), values);
    };

    constexpr std::size_t embedding_row_bytes = embedding_size / 256 * 144;
    auto hidden = dequantize_q4k_row(static_cast<const char *>(tensor_data(mapping, model, embedding))
                                         + static_cast<std::size_t>(token) * embedding_row_bytes,
                                     embedding_size);
    const auto started = std::chrono::steady_clock::now();
    for (int layer = 0; layer < live_layers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const auto &attn_norm = require_tensor(model, prefix + "attn_norm.weight");
        const auto &post_norm = require_tensor(model, prefix + "post_attention_norm.weight");
        const auto attention_input = rms_norm(hidden, f32_data(attn_norm, embedding_size), embedding_size, epsilon);
        std::vector<float> attention_output;

        if ((layer + 1) % 4 != 0) {
            const auto qkv = projection(require_tensor(model, prefix + "attn_qkv.weight"), attention_input);
            const auto z = projection(require_tensor(model, prefix + "attn_gate.weight"), attention_input);
            const auto beta_unscaled = f32_projection(require_tensor(model, prefix + "ssm_beta.weight"), attention_input);
            const auto conv_weights = f32_data(require_tensor(model, prefix + "ssm_conv1d.weight"), 4 * 10240);
            std::vector<float> convolved(10240);
            for (std::size_t index = 0; index < convolved.size(); ++index) {
                // Position zero has three zero history slots; the fourth tap consumes qkv.
                convolved[index] = silu(qkv[index] * conv_weights[index * 4 + 3]);
            }
            std::vector<float> queries(convolved.begin(), convolved.begin() + recurrent_key_heads * recurrent_head_size);
            std::vector<float> keys(convolved.begin() + recurrent_key_heads * recurrent_head_size,
                                    convolved.begin() + 2 * recurrent_key_heads * recurrent_head_size);
            std::vector<float> values(convolved.begin() + 2 * recurrent_key_heads * recurrent_head_size, convolved.end());
            l2_norm_heads(queries, recurrent_head_size, epsilon);
            l2_norm_heads(keys, recurrent_head_size, epsilon);
            std::vector<float> recurrent_output(recurrent_value_heads * recurrent_head_size);
            for (std::size_t head = 0; head < recurrent_value_heads; ++head) {
                const std::size_t key_head = head % recurrent_key_heads;
                float dot = 0.0F;
                for (std::size_t dimension = 0; dimension < recurrent_head_size; ++dimension) {
                    dot = std::fma(queries[key_head * recurrent_head_size + dimension],
                                   keys[key_head * recurrent_head_size + dimension], dot);
                }
                const float scale = sigmoid(beta_unscaled[head]) * dot / std::sqrt(static_cast<float>(recurrent_head_size));
                for (std::size_t dimension = 0; dimension < recurrent_head_size; ++dimension) {
                    recurrent_output[head * recurrent_head_size + dimension]
                        = values[head * recurrent_head_size + dimension] * scale;
                }
            }
            auto normalized_output = rms_norm(recurrent_output,
                                              vector_weights(prefix + "ssm_norm.weight", recurrent_head_size),
                                              recurrent_head_size, epsilon);
            for (std::size_t index = 0; index < normalized_output.size(); ++index) {
                normalized_output[index] *= silu(z[index]);
            }
            attention_output = projection(require_tensor(model, prefix + "ssm_out.weight"), normalized_output);
        } else {
            // At the first decoded token the causal attention distribution contains
            // only the current V, so Q/K normalization and MRoPE cannot change its
            // value. They are populated by the multi-token cache executor next.
            const auto q_and_gate = projection(require_tensor(model, prefix + "attn_q.weight"), attention_input);
            const auto values = projection(require_tensor(model, prefix + "attn_v.weight"), attention_input);
            std::vector<float> gated_values(attention_heads * attention_head_size);
            for (std::size_t head = 0; head < attention_heads; ++head) {
                const std::size_t kv_head = head / (attention_heads / attention_kv_heads);
                for (std::size_t dimension = 0; dimension < attention_head_size; ++dimension) {
                    const float gate = sigmoid(q_and_gate[head * attention_head_size * 2 + attention_head_size + dimension]);
                    gated_values[head * attention_head_size + dimension]
                        = values[kv_head * attention_head_size + dimension] * gate;
                }
            }
            attention_output = projection(require_tensor(model, prefix + "attn_output.weight"), gated_values);
        }

        hidden = add_vectors(hidden, attention_output);
        const auto ffn_input = rms_norm(hidden, f32_data(post_norm, embedding_size), embedding_size, epsilon);
        auto ffn_gate = projection(require_tensor(model, prefix + "ffn_gate.weight"), ffn_input);
        const auto ffn_up = projection(require_tensor(model, prefix + "ffn_up.weight"), ffn_input);
        for (std::size_t index = 0; index < ffn_gate.size(); ++index) {
            ffn_gate[index] = silu(ffn_gate[index]) * ffn_up[index];
        }
        hidden = add_vectors(hidden, projection(require_tensor(model, prefix + "ffn_down.weight"), ffn_gate));
    }
    const auto final_hidden = rms_norm(hidden,
                                       vector_weights("output_norm.weight", embedding_size),
                                       embedding_size, epsilon);
    const auto logits = projection(require_tensor(model, "output.weight"), final_hidden);
    const auto best = std::max_element(logits.begin(), logits.end());
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cout << "native target first-token trunk\n"
              << "  input token:  " << token << '\n'
              << "  live layers:  " << live_layers << " (48 recurrent, 16 causal attention)\n"
              << "  position:     0 (causal attention is the current V; no cache read)\n"
              << "  elapsed:      " << std::fixed << std::setprecision(3) << elapsed << " s\n"
              << "  top token:    " << std::distance(logits.begin(), best) << '\n'
              << "  top logit:    " << std::setprecision(6) << *best << '\n';
}

void bench_matvec(const std::string &path, const std::string &tensor_name, int iterations) {
    const qwen_native::GgufFile model(path);
    const auto *tensor = model.find_tensor(tensor_name);
    if (!tensor) {
        throw std::runtime_error("tensor not found: " + tensor_name);
    }
    if ((tensor->type != 12 && tensor->type != 13 && tensor->type != 14) || tensor->dimensions.size() != 2) {
        throw std::runtime_error("benchmark requires a rank-2 Q4_K, Q5_K, or Q6_K tensor");
    }
    const Mapping mapping(path);
    const auto input_size = static_cast<std::size_t>(tensor->dimensions[0]);
    const auto output_size = static_cast<std::size_t>(tensor->dimensions[1]);
    std::cout << qwen_native::ggml_type_name(tensor->type) << " real-weight benchmark: " << tensor_name << '\n'
              << "  shape: " << input_size << " x " << output_size << '\n';
    const void *weights = mapping.at(model.data_offset() + tensor->offset);
    qwen_native::Q4KBenchmarkResult result;
    qwen_native::Q4KBenchmarkResult batch8;
    if (tensor->type == 12) {
        result = qwen_native::benchmark_q4k(weights, input_size, output_size, iterations);
        batch8 = qwen_native::benchmark_q4k_batch8(weights, input_size, output_size, iterations);
    } else if (tensor->type == 13) {
        result = qwen_native::benchmark_q5k(weights, input_size, output_size, iterations);
        batch8 = qwen_native::benchmark_q5k_batch8(weights, input_size, output_size, iterations);
    } else {
        result = qwen_native::benchmark_q6k(weights, input_size, output_size, iterations);
        batch8 = qwen_native::benchmark_q6k_batch8(weights, input_size, output_size, iterations);
    }
    std::cout << "  native direct:\n"
              << "    latency:    " << std::fixed << std::setprecision(3) << result.milliseconds << " ms\n"
              << "    bandwidth:  " << std::setprecision(2) << result.bandwidth_gbps << " GB/s\n"
              << "    max error:  " << std::scientific << result.max_absolute_error << '\n';
    std::cout << "  eight-token verifier:\n"
              << "    latency:    " << std::fixed << std::setprecision(3) << batch8.milliseconds << " ms\n"
              << "    weight BW:  " << std::setprecision(2) << batch8.bandwidth_gbps << " GB/s\n"
              << "    max error:  " << std::scientific << batch8.max_absolute_error << '\n';
    if (tensor->type == 12) {
        const auto n23 = qwen_native::benchmark_n23q4_from_q4k(weights, input_size, output_size, iterations);
        std::cout << "  N23Q4 wave32 + DP4A:\n"
                  << "    latency:    " << std::fixed << std::setprecision(3) << n23.milliseconds << " ms\n"
                  << "    bandwidth:  " << std::setprecision(2) << n23.bandwidth_gbps << " GB/s\n"
                  << "    max delta:  " << std::scientific << n23.max_absolute_error << '\n'
                  << "    rel. RMSE:  " << n23.relative_rmse << '\n'
                  << "    kernel err: " << n23.kernel_max_absolute_error << '\n';
        const auto n23_batch8 = qwen_native::benchmark_n23q4_batch8_from_q4k(weights, input_size, output_size, iterations);
        std::cout << "  N23Q4 eight-token verifier:\n"
                  << "    latency:    " << std::fixed << std::setprecision(3) << n23_batch8.milliseconds << " ms\n"
                  << "    weight BW:  " << std::setprecision(2) << n23_batch8.bandwidth_gbps << " GB/s\n"
                  << "    max delta:  " << std::scientific << n23_batch8.max_absolute_error << '\n'
                  << "    rel. RMSE:  " << n23_batch8.relative_rmse << '\n';
    }
}

void bench_ffn(const std::string &path, int layer, int iterations) {
    if (layer < 0) {
        throw std::invalid_argument("FFN layer index must be non-negative");
    }
    const qwen_native::GgufFile model(path);
    const std::string prefix = "blk." + std::to_string(layer) + ".ffn_";
    const auto *gate = model.find_tensor(prefix + "gate.weight");
    const auto *up = model.find_tensor(prefix + "up.weight");
    const auto *down = model.find_tensor(prefix + "down.weight");
    if (!gate || !up || !down) {
        throw std::runtime_error("layer " + std::to_string(layer) + " does not contain a complete FFN");
    }
    if (gate->dimensions.size() != 2 || up->dimensions.size() != 2 || down->dimensions.size() != 2 ||
        gate->dimensions[0] != up->dimensions[0] || gate->dimensions[1] != up->dimensions[1] ||
        down->dimensions[0] != gate->dimensions[1]) {
        throw std::runtime_error("FFN tensor shapes are inconsistent in layer " + std::to_string(layer));
    }
    const Mapping mapping(path);
    const auto result = qwen_native::benchmark_ffn_batch8(
        mapping.at(model.data_offset() + gate->offset), gate->type,
        mapping.at(model.data_offset() + up->offset), up->type,
        mapping.at(model.data_offset() + down->offset), down->type,
        static_cast<std::size_t>(gate->dimensions[0]),
        static_cast<std::size_t>(gate->dimensions[1]),
        static_cast<std::size_t>(down->dimensions[1]), iterations);
    std::cout << "native eight-token FFN verifier · layer " << layer << '\n'
              << "  projections: " << qwen_native::ggml_type_name(gate->type) << " gate, "
              << qwen_native::ggml_type_name(up->type) << " up, "
              << qwen_native::ggml_type_name(down->type) << " down\n"
              << "  shape:       " << gate->dimensions[0] << " -> " << gate->dimensions[1]
              << " -> " << down->dimensions[1] << '\n'
              << "  GPU latency: " << std::fixed << std::setprecision(3) << result.milliseconds << " ms/pass\n"
              << "  weight BW:   " << std::setprecision(2) << result.weight_bandwidth_gbps << " GB/s\n"
              << "  verifier:    " << result.verifier_tokens_per_second << " proposed tok/s\n"
              << "  max error:   " << std::scientific << result.max_absolute_error << '\n';
    if (gate->type == 12 && up->type == 12) {
        const auto n23 = qwen_native::benchmark_ffn_batch8_n23q4(
            mapping.at(model.data_offset() + gate->offset),
            mapping.at(model.data_offset() + up->offset),
            mapping.at(model.data_offset() + down->offset), down->type,
            static_cast<std::size_t>(gate->dimensions[0]),
            static_cast<std::size_t>(gate->dimensions[1]),
            static_cast<std::size_t>(down->dimensions[1]), iterations);
        std::cout << "  N23Q4 gate/up:\n"
                  << "    GPU latency: " << std::fixed << std::setprecision(3) << n23.milliseconds << " ms/pass\n"
                  << "    weight BW:   " << std::setprecision(2) << n23.weight_bandwidth_gbps << " GB/s\n"
                  << "    verifier:    " << n23.verifier_tokens_per_second << " proposed tok/s\n"
                  << "    max error:   " << std::scientific << n23.max_absolute_error << '\n';
    }
}

std::uint64_t metadata_u64(const qwen_native::GgufFile &file, const std::string &key) {
    const auto *value = file.find_metadata(key);
    if (!value) {
        throw std::runtime_error("missing GGUF metadata: " + key);
    }
    if (const auto *number = std::get_if<std::uint64_t>(value)) {
        return *number;
    }
    throw std::runtime_error("GGUF metadata is not unsigned: " + key);
}

double gib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

void plan_128k(const std::string &target_path, const std::string &draft_path) {
    constexpr std::uint64_t context = 131072;
    constexpr std::uint64_t n23kv4_block_bytes = 18;
    constexpr std::uint64_t n23kv4_block_values = 32;
    constexpr std::uint64_t q8_block_bytes = 34;
    constexpr std::uint64_t q8_block_values = 32;
    constexpr std::uint64_t runtime_reserve = 512ULL * 1024 * 1024;

    const qwen_native::GgufFile target(target_path);
    const qwen_native::GgufFile draft(draft_path);
    const auto all_layers = metadata_u64(target, "qwen35.block_count");
    const auto nextn_layers = metadata_u64(target, "qwen35.nextn_predict_layers");
    const auto layers = all_layers - nextn_layers;
    const auto interval = metadata_u64(target, "qwen35.full_attention_interval");
    const auto full_layers = layers / interval;
    const auto recurrent_layers = layers - full_layers;
    const auto target_kv_heads = metadata_u64(target, "qwen35.attention.head_count_kv");
    const auto target_key_size = metadata_u64(target, "qwen35.attention.key_length");
    const auto target_value_size = metadata_u64(target, "qwen35.attention.value_length");
    const auto target_state_size = metadata_u64(target, "qwen35.ssm.state_size");
    const auto target_value_heads = metadata_u64(target, "qwen35.ssm.time_step_rank");
    const auto conv_kernel = metadata_u64(target, "qwen35.ssm.conv_kernel");
    const auto inner_size = metadata_u64(target, "qwen35.ssm.inner_size");
    const auto key_heads = metadata_u64(target, "qwen35.ssm.group_count");

    const std::uint64_t target_kv_values = context * full_layers * target_kv_heads
        * (target_key_size + target_value_size);
    const std::uint64_t target_kv_n23 = target_kv_values / n23kv4_block_values * n23kv4_block_bytes;
    const std::uint64_t target_kv_q8 = target_kv_values / q8_block_values * q8_block_bytes;
    const std::uint64_t recurrent_state = recurrent_layers * target_value_heads
        * target_state_size * target_state_size * sizeof(std::uint16_t);
    const std::uint64_t conv_channels = 2 * key_heads * target_state_size + inner_size;
    const std::uint64_t conv_state = recurrent_layers * (conv_kernel - 1) * conv_channels
        * sizeof(std::uint16_t);

    const auto draft_layers = metadata_u64(draft, "dflash.block_count");
    const auto draft_window = metadata_u64(draft, "dflash.attention.sliding_window");
    const auto draft_kv_heads = metadata_u64(draft, "dflash.attention.head_count_kv");
    const auto draft_key_size = metadata_u64(draft, "dflash.attention.key_length");
    const auto draft_value_size = metadata_u64(draft, "dflash.attention.value_length");
    const std::uint64_t draft_kv_values = draft_layers * draft_window * draft_kv_heads
        * (draft_key_size + draft_value_size);
    const std::uint64_t draft_kv_n23 = draft_kv_values / n23kv4_block_values * n23kv4_block_bytes;

    int device = 0;
    hipDeviceProp_t properties{};
    hip_check(hipGetDevice(&device), "hipGetDevice");
    hip_check(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
    const std::uint64_t vram = properties.totalGlobalMem;
    const std::uint64_t fixed_native = target_kv_n23 + recurrent_state + conv_state
        + draft.file_size() + draft_kv_n23 + runtime_reserve;
    const std::uint64_t fixed_q8 = target_kv_q8 + recurrent_state + conv_state
        + draft.file_size() + draft_kv_n23 + runtime_reserve;
    const std::uint64_t native_weight_budget = fixed_native < vram ? vram - fixed_native : 0;
    const std::uint64_t q8_weight_budget = fixed_q8 < vram ? vram - fixed_q8 : 0;

    std::cout << "128K native memory plan · " << properties.name << '\n'
              << "  live target layers: " << layers << " (" << full_layers << " full attention, "
              << recurrent_layers << " recurrent)\n"
              << "  target KV Q8_0:     " << std::fixed << std::setprecision(2) << gib(target_kv_q8) << " GiB\n"
              << "  target KV N23KV4:   " << gib(target_kv_n23) << " GiB\n"
              << "  recurrent state:    " << gib(recurrent_state + conv_state) << " GiB\n"
              << "  DFlash2 weights:     " << gib(draft.file_size()) << " GiB\n"
              << "  DFlash2 SWA KV4:     " << gib(draft_kv_n23) << " GiB\n"
              << "  runtime reserve:     " << gib(runtime_reserve) << " GiB\n"
              << "  target-weight room:\n"
              << "    with Q8 target KV: " << gib(q8_weight_budget) << " GiB\n"
              << "    with N23KV4:       " << gib(native_weight_budget) << " GiB\n"
              << "  reclaimed for weights: " << gib(native_weight_budget - q8_weight_budget) << " GiB\n";
}

void bench_kv4(std::size_t context_size, int iterations, int partitions) {
    std::cout << "N23KV4 24Q/4KV-head score benchmark\n"
              << "  context: " << context_size << " tokens\n";
    const auto result = qwen_native::benchmark_n23kv4(context_size, iterations);
    std::cout << "  latency:      " << std::fixed << std::setprecision(3)
              << result.milliseconds << " ms per layer\n"
              << "  effective BW: " << std::setprecision(2) << result.bandwidth_gbps << " GB/s\n"
              << "  max error:    " << std::scientific << result.max_absolute_error << '\n';
    const auto batch8 = qwen_native::benchmark_n23kv4_batch8(context_size, iterations);
    std::cout << "  eight-query verifier:\n"
              << "    latency:      " << std::fixed << std::setprecision(3)
              << batch8.milliseconds << " ms per layer\n"
              << "    reused-key BW:" << std::setprecision(2) << ' ' << batch8.bandwidth_gbps << " GB/s\n"
              << "    max error:    " << std::scientific << batch8.max_absolute_error << '\n';
    const auto flash = qwen_native::benchmark_n23kv4_flash_batch8(context_size, partitions, iterations);
    std::cout << "  fused online-softmax + V verifier:\n"
              << "    partitions:   " << partitions << '\n'
              << "    latency:      " << std::fixed << std::setprecision(3)
              << flash.milliseconds << " ms per layer\n"
              << "    effective BW: " << std::setprecision(2) << flash.bandwidth_gbps << " GB/s\n"
              << "    max error:    " << std::scientific << flash.max_absolute_error << '\n';
}

void bench_gdn(int iterations) {
    const auto result = qwen_native::benchmark_gated_delta_step(iterations);
    std::cout << "native gated-delta recurrent step · 48 x 128x128 FP16 state\n"
              << "  latency:      " << std::fixed << std::setprecision(3)
              << result.milliseconds << " ms/layer\n"
              << "  state BW:     " << std::setprecision(2) << result.bandwidth_gbps << " GB/s\n"
              << "  max error:    " << std::scientific << result.max_absolute_error << '\n';
}

void usage(const char *program) {
    std::cout << "Usage:\n"
              << "  " << program << " device\n"
              << "  " << program << " inspect MODEL.gguf [--tensors]\n"
              << "  " << program << " metadata MODEL.gguf\n"
              << "  " << program << " show-token MODEL.gguf TOKEN_ID\n"
              << "  " << program << " bench-matvec MODEL.gguf TENSOR [ITERATIONS]\n"
              << "  " << program << " bench-ffn MODEL.gguf LAYER [ITERATIONS]\n"
              << "  " << program << " bench-gdn [ITERATIONS]\n"
              << "  " << program << " target-first-token TARGET.gguf TOKEN_ID\n"
              << "  " << program << " bench-kv4 [CONTEXT] [ITERATIONS] [PARTITIONS]\n"
              << "  " << program << " plan-128k TARGET.gguf DFLASH.gguf\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "device") {
            show_device();
            return EXIT_SUCCESS;
        }
        if ((argc == 3 || argc == 4) && std::string(argv[1]) == "inspect") {
            const bool list_tensors = argc == 4 && std::string(argv[3]) == "--tensors";
            if (argc == 4 && !list_tensors) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            inspect(argv[2], list_tensors);
            return EXIT_SUCCESS;
        }
        if (argc == 3 && std::string(argv[1]) == "metadata") {
            show_metadata(argv[2]);
            return EXIT_SUCCESS;
        }
        if (argc == 4 && std::string(argv[1]) == "show-token") {
            show_token(argv[2], static_cast<std::size_t>(std::stoull(argv[3])));
            return EXIT_SUCCESS;
        }
        if ((argc == 4 || argc == 5) && (std::string(argv[1]) == "bench-matvec" || std::string(argv[1]) == "bench-q4")) {
            const int iterations = argc == 5 ? std::stoi(argv[4]) : 100;
            bench_matvec(argv[2], argv[3], iterations);
            return EXIT_SUCCESS;
        }
        if ((argc == 4 || argc == 5) && std::string(argv[1]) == "bench-ffn") {
            const int iterations = argc == 5 ? std::stoi(argv[4]) : 20;
            bench_ffn(argv[2], std::stoi(argv[3]), iterations);
            return EXIT_SUCCESS;
        }
        if (argc == 4 && std::string(argv[1]) == "plan-128k") {
            plan_128k(argv[2], argv[3]);
            return EXIT_SUCCESS;
        }
        if ((argc == 2 || argc == 3) && std::string(argv[1]) == "bench-gdn") {
            const int iterations = argc == 3 ? std::stoi(argv[2]) : 100;
            bench_gdn(iterations);
            return EXIT_SUCCESS;
        }
        if (argc == 4 && std::string(argv[1]) == "target-first-token") {
            target_first_token(argv[2], static_cast<std::uint32_t>(std::stoul(argv[3])));
            return EXIT_SUCCESS;
        }
        if (argc >= 2 && argc <= 5 && std::string(argv[1]) == "bench-kv4") {
            const auto context = argc >= 3 ? static_cast<std::size_t>(std::stoull(argv[2])) : 131072;
            const int iterations = argc >= 4 ? std::stoi(argv[3]) : 20;
            const int partitions = argc >= 5 ? std::stoi(argv[4]) : 64;
            bench_kv4(context, iterations, partitions);
            return EXIT_SUCCESS;
        }
        usage(argv[0]);
        return EXIT_FAILURE;
    } catch (const std::exception &error) {
        std::cerr << "qwen38-native: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

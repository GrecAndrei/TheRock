#include "gguf.h"
#include "kernels.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <fcntl.h>
#include <map>
#include <numeric>
#include <sstream>
#include <unordered_map>
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

class Tokenizer {
public:
    explicit Tokenizer(const qwen_native::GgufFile &model) {
        const auto *tokens_meta = model.find_metadata("tokenizer.ggml.tokens");
        if (!tokens_meta || !std::holds_alternative<std::vector<std::string>>(*tokens_meta)) {
            throw std::runtime_error("GGUF missing tokenizer.ggml.tokens");
        }
        vocab_ = std::get<std::vector<std::string>>(*tokens_meta);

        const auto *merges_meta = model.find_metadata("tokenizer.ggml.merges");
        if (!merges_meta || !std::holds_alternative<std::vector<std::string>>(*merges_meta)) {
            throw std::runtime_error("GGUF missing tokenizer.ggml.merges");
        }
        const auto &merges = std::get<std::vector<std::string>>(*merges_meta);

        for (std::size_t i = 0; i < vocab_.size(); ++i) {
            token_to_id_[vocab_[i]] = static_cast<std::uint32_t>(i);
        }

        for (std::size_t rank = 0; rank < merges.size(); ++rank) {
            merge_ranks_[merges[rank]] = static_cast<int>(rank);
        }
    }

    std::vector<std::uint32_t> encode_full(const std::string &text) const {
        auto byte_encoded = gpt2_byte_encode(text);
        auto symbols = split_into_symbols(byte_encoded);
        auto merged = bpe_on_symbols(symbols);
        std::vector<std::uint32_t> result;
        for (const auto &piece : merged) {
            auto it = token_to_id_.find(piece);
            if (it != token_to_id_.end()) {
                result.push_back(it->second);
            }
        }
        return result;
    }

    std::string decode(const std::vector<std::uint32_t> &tokens) const {
        std::string encoded;
        for (const auto id : tokens) {
            if (id < vocab_.size()) {
                encoded += vocab_[id];
            }
        }
        return gpt2_byte_decode(encoded);
    }

    std::string decode_token(std::uint32_t id) const {
        if (id >= vocab_.size()) return "";
        return gpt2_byte_decode(vocab_[id]);
    }

    [[nodiscard]] std::uint32_t bos_token() const { return 248044; }
    [[nodiscard]] std::uint32_t eos_token() const { return 248046; }
    [[nodiscard]] std::uint32_t im_start() const { return 248045; }
    [[nodiscard]] std::uint32_t im_end() const { return 248046; }
    [[nodiscard]] std::uint32_t newline_token() const { return 198; }
    [[nodiscard]] std::size_t vocab_size() const { return vocab_.size(); }

    std::uint32_t find_token(const std::string &text) const {
        auto it = token_to_id_.find(text);
        if (it == token_to_id_.end()) {
            throw std::runtime_error("token not found: " + text);
        }
        return it->second;
    }

private:
    static std::string gpt2_byte_encode(const std::string &text) {
        std::string result;
        for (unsigned char byte : text) {
            result += gpt2_byte_to_unicode(byte);
        }
        return result;
    }

    static std::string gpt2_byte_decode(const std::string &encoded) {
        std::string result;
        std::size_t pos = 0;
        while (pos < encoded.size()) {
            unsigned char first = static_cast<unsigned char>(encoded[pos]);
            int len = 1;
            if ((first & 0xE0) == 0xC0) len = 2;
            else if ((first & 0xF0) == 0xE0) len = 3;
            else if ((first & 0xF8) == 0xF0) len = 4;

            std::string utf8_char = encoded.substr(pos, len);
            pos += len;

            int byte = gpt2_unicode_to_byte(utf8_char);
            if (byte >= 0) {
                result += static_cast<char>(byte);
            } else {
                result += utf8_char;
            }
        }
        return result;
    }

    static std::string gpt2_byte_to_unicode(unsigned char byte) {
        // GPT-2 maps bytes to unicode codepoints:
        // 33-126 -> same, 161-172 -> same, 174-255 -> same
        // 0-32, 127-160, 173 -> 256+offset
        char32_t codepoint;
        if ((byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || (byte >= 174)) {
            codepoint = byte;
        } else {
            static const unsigned char remap[] = {
                0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
                127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,
                150,151,152,153,154,155,156,157,158,159,160,173
            };
            int index = -1;
            for (int i = 0; i < 68; ++i) {
                if (remap[i] == byte) { index = i; break; }
            }
            codepoint = 256 + index;
        }
        std::string utf8;
        if (codepoint < 0x80) {
            utf8 += static_cast<char>(codepoint);
        } else if (codepoint < 0x800) {
            utf8 += static_cast<char>(0xC0 | (codepoint >> 6));
            utf8 += static_cast<char>(0x80 | (codepoint & 0x3F));
        } else {
            utf8 += static_cast<char>(0xE0 | (codepoint >> 12));
            utf8 += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            utf8 += static_cast<char>(0x80 | (codepoint & 0x3F));
        }
        return utf8;
    }

    static int gpt2_unicode_to_byte(const std::string &utf8_char) {
        char32_t codepoint = 0;
        unsigned char first = static_cast<unsigned char>(utf8_char[0]);
        if ((first & 0x80) == 0) {
            codepoint = first;
        } else if ((first & 0xE0) == 0xC0 && utf8_char.size() >= 2) {
            codepoint = (first & 0x1F) << 6;
            codepoint |= (static_cast<unsigned char>(utf8_char[1]) & 0x3F);
        } else if ((first & 0xF0) == 0xE0 && utf8_char.size() >= 3) {
            codepoint = (first & 0x0F) << 12;
            codepoint |= (static_cast<unsigned char>(utf8_char[1]) & 0x3F) << 6;
            codepoint |= (static_cast<unsigned char>(utf8_char[2]) & 0x3F);
        } else {
            return -1;
        }

        if ((codepoint >= 33 && codepoint <= 126) ||
            (codepoint >= 161 && codepoint <= 172) ||
            (codepoint >= 174 && codepoint <= 255)) {
            return static_cast<int>(codepoint);
        }
        if (codepoint >= 256 && codepoint < 256 + 68) {
            static const unsigned char remap[] = {
                0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
                127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,
                150,151,152,153,154,155,156,157,158,159,160,173
            };
            return remap[codepoint - 256];
        }
        return -1;
    }

    std::vector<std::string> split_into_symbols(const std::string &encoded) const {
        std::vector<std::string> symbols;
        std::size_t pos = 0;
        while (pos < encoded.size()) {
            unsigned char c = static_cast<unsigned char>(encoded[pos]);
            int len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            symbols.push_back(encoded.substr(pos, len));
            pos += len;
        }
        return symbols;
    }

    std::vector<std::string> bpe_on_symbols(std::vector<std::string> symbols) const {
        if (symbols.size() <= 1) return symbols;
        while (true) {
            int best_rank = std::numeric_limits<int>::max();
            std::size_t best_pos = symbols.size();
            for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
                std::string pair = symbols[i] + " " + symbols[i + 1];
                auto it = merge_ranks_.find(pair);
                if (it != merge_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pos = i;
                }
            }
            if (best_pos >= symbols.size()) break;
            symbols[best_pos] = symbols[best_pos] + symbols[best_pos + 1];
            symbols.erase(symbols.begin() + best_pos + 1);
            if (symbols.size() <= 1) break;
        }
        return symbols;
    }

    std::vector<std::string> bpe(const std::string &token) const {
        auto symbols = split_into_symbols(token);
        return bpe_on_symbols(symbols);
    }

    std::vector<std::string> vocab_;
    std::unordered_map<std::string, std::uint32_t> token_to_id_;
    std::unordered_map<std::string, int> merge_ranks_;
};

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

void find_token(const std::string &path, const std::string &needle) {
    const qwen_native::GgufFile model(path);
    const auto *metadata = model.find_metadata("tokenizer.ggml.tokens");
    if (!metadata || !std::holds_alternative<std::vector<std::string>>(*metadata)) {
        throw std::runtime_error("GGUF does not contain a string tokenizer vocabulary");
    }
    const auto &tokens = std::get<std::vector<std::string>>(*metadata);
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (tokens[index] == needle) {
            std::cout << index << '\n';
        }
    }
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

void l2_norm_heads_inplace(float *values, std::size_t total_size, std::size_t head_size, float epsilon) {
    for (std::size_t head = 0; head < total_size / head_size; ++head) {
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

// Retained only as a structural projection smoke test while comparing historical
// measurements.  It omits persistent recurrent state and must never be used as
// a decoder; target-generate below is the semantic execution path.
[[maybe_unused]] void legacy_position_zero_smoke(const std::string &path, std::uint32_t token) {
    constexpr std::size_t embedding_size = 5120;
    constexpr std::size_t ffn_size = 17408;
    constexpr std::size_t recurrent_key_heads = 16;
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

// A correctness-first, single-sequence decoder.  All model projections use the
// native HIP K-quant matvecs; the relatively small recurrent and attention
// state transitions remain on the host while the full target is streamed.
// Keeping this executor explicit makes it a useful reference path for the
// device-resident verifier/target executor rather than hiding semantics in a
// second runtime.
class NativeTargetSession {
public:
    static constexpr std::size_t kEmbedding = 5120;
    static constexpr std::size_t kFfn = 17408;
    // Qwen3.8's recurrent delta-net has 16 key groups (the full-attention
    // branch separately has four KV heads).  Keeping these distinct matters:
    // Q/K/V is 2 * 16 * 128 + 48 * 128 = 10240 channels.
    static constexpr std::size_t kRecurrentKeyHeads = 16;
    static constexpr std::size_t kRecurrentValueHeads = 48;
    static constexpr std::size_t kRecurrentHeadSize = 128;
    static constexpr std::size_t kAttentionHeads = 24;
    static constexpr std::size_t kAttentionKvHeads = 4;
    static constexpr std::size_t kAttentionHeadSize = 256;
    static constexpr std::size_t kKvWidth = kAttentionKvHeads * kAttentionHeadSize;

    struct StepResult {
        std::uint32_t next_token = 0;
        float top_logit = 0.0F;
        double seconds = 0.0;
    };

    struct BatchStepResult {
        std::array<std::uint32_t, 8> next_tokens{};
        std::array<float, 8> top_logits{};
        double seconds = 0.0;
        double recurrent_seconds = 0.0;
        double attention_seconds = 0.0;
        double ffn_seconds = 0.0;
        double output_seconds = 0.0;
    };

    NativeTargetSession(const std::string &path, std::size_t context_capacity,
                        std::size_t resident_weight_budget)
        : model_(path), mapping_(path), context_capacity_(context_capacity) {
        if (context_capacity_ == 0 || context_capacity_ > 262144) {
            throw std::invalid_argument("context capacity must be in [1, 262144]");
        }
        const auto *architecture = model_.find_metadata("general.architecture");
        if (!architecture || !std::holds_alternative<std::string>(*architecture) ||
            std::get<std::string>(*architecture) != "qwen35") {
            throw std::runtime_error("native target decoder requires a qwen35 GGUF target model");
        }
        const auto live_layers = metadata_number_or(model_, "qwen35.block_count", 0.0)
            - metadata_number_or(model_, "qwen35.nextn_predict_layers", 0.0);
        if (live_layers != 64 || metadata_number_or(model_, "qwen35.embedding_length", 0.0) != kEmbedding ||
            metadata_number_or(model_, "qwen35.feed_forward_length", 0.0) != kFfn ||
            metadata_number_or(model_, "qwen35.attention.head_count", 0.0) != kAttentionHeads ||
            metadata_number_or(model_, "qwen35.attention.head_count_kv", 0.0) != kAttentionKvHeads ||
            metadata_number_or(model_, "qwen35.ssm.group_count", 0.0) != kRecurrentKeyHeads) {
            throw std::runtime_error("native target decoder only supports the supplied Qwen3.8-27B topology");
        }
        epsilon_ = static_cast<float>(metadata_number_or(
            model_, "qwen35.attention.layer_norm_rms_epsilon", 1.0e-6));
        rope_frequency_base_ = static_cast<float>(metadata_number_or(
            model_, "qwen35.rope.freq_base", 1.0e7));
        const auto &embedding = require_tensor(model_, "token_embd.weight");
        if (embedding.type != 12 || embedding.dimensions.size() != 2 ||
            embedding.dimensions[0] != kEmbedding) {
            throw std::runtime_error("target embedding is not the expected Q4_K matrix");
        }
        vocabulary_size_ = static_cast<std::size_t>(embedding.dimensions[1]);
        qwen_native::set_resident_weight_cache_budget(resident_weight_budget);

        recurrent_.resize(48);
        for (auto &state : recurrent_) {
            state.matrix.assign(kRecurrentValueHeads * kRecurrentHeadSize * kRecurrentHeadSize, 0.0F);
            state.convolution.assign(3 * (2 * kRecurrentKeyHeads * kRecurrentHeadSize +
                                          kRecurrentValueHeads * kRecurrentHeadSize), 0.0F);
            state.device_matrix = qwen_native::create_gated_delta_state();
        }
        attention_.resize(16);
        // Reserve a modest working set.  The cache grows on demand, so declaring
        // a 128K context does not fault in gigabytes before the prompt exists.
        const std::size_t initial_tokens = std::min<std::size_t>(context_capacity_, 512);
        for (auto &cache : attention_) {
            cache.keys.reserve(initial_tokens * kKvWidth);
            cache.values.reserve(initial_tokens * kKvWidth);
        }

        for (const auto &t : model_.tensors()) {
            if (t.type == 21 && t.dimensions.size() == 2) {
                std::cerr << "  converting IQ3_S -> Q8_0: " << t.name << "\n";
                iq3s_converted_[t.name] = convert_iq3s_to_q8_0(t);
            }
        }

        // Preload deterministic high-priority matrices before optional entries
        // compete for the remaining cache.  This is a model-load cost, not a
        // decode-timing cost.  Priority: ffn_down (avoids 3-way transfer_stream
        // stall), then recurrent weights that block compute with no prefetch.
        if (resident_weight_budget != 0) {
            preload_projection("output.weight");
            for (int layer = 0; layer < 64; ++layer) {
                const std::string prefix = "blk." + std::to_string(layer) + ".";
                preload_projection(prefix + "ffn_down.weight");
            }
            for (int layer = 0; layer < 64; ++layer) {
                const std::string prefix = "blk." + std::to_string(layer) + ".";
                if ((layer + 1) % 4 != 0) {
                    preload_projection(prefix + "attn_qkv.weight");
                    preload_projection(prefix + "attn_gate.weight");
                } else {
                    preload_projection(prefix + "attn_k.weight");
                    preload_projection(prefix + "attn_v.weight");
                    preload_projection(prefix + "attn_output.weight");
                }
            }
        }

        constexpr std::size_t max_ffn_weight = (kFfn / 32) * kEmbedding * 34;
        pipeline_ = qwen_native::create_device_pipeline(kEmbedding, kFfn, max_ffn_weight);

        constexpr int kQkvWidth = 2 * kRecurrentKeyHeads * kRecurrentHeadSize +
                                  kRecurrentValueHeads * kRecurrentHeadSize;
        constexpr int kRecurrentWidth = kRecurrentValueHeads * kRecurrentHeadSize;
        recurrent_device_ = qwen_native::create_recurrent_device_state(
            48, kQkvWidth, kRecurrentWidth, kEmbedding, kRecurrentValueHeads, kRecurrentHeadSize);
        for (int i = 0; i < 48; ++i) {
            const int layer = i + (i / 3);  // skip every 4th (attention) layer
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            const auto &beta_t = tensor(prefix + "ssm_beta.weight");
            const auto &alpha_t = tensor(prefix + "ssm_alpha.weight");
            const auto beta_f32 = dequantize_q8_projection(beta_t);
            const auto alpha_f32 = dequantize_q8_projection(alpha_t);
            qwen_native::recurrent_upload_weights(recurrent_device_, i,
                f32_vector(prefix + "ssm_conv1d.weight", 4 * kQkvWidth),
                beta_f32.data(), alpha_f32.data(),
                f32_vector(prefix + "ssm_dt.bias", kRecurrentValueHeads),
                f32_vector(prefix + "ssm_a", kRecurrentValueHeads),
                f32_vector(prefix + "ssm_norm.weight", kRecurrentHeadSize));
        }

        attn_device_ = qwen_native::create_attention_device_state(
            16, 4096, rope_frequency_base_);
    }

    [[nodiscard]] std::size_t context_capacity() const { return context_capacity_; }
    [[nodiscard]] std::size_t active_tokens() const { return active_tokens_; }
    [[nodiscard]] std::size_t vocabulary_size() const { return vocabulary_size_; }
    [[nodiscard]] qwen_native::ResidentWeightCacheStats resident_weight_stats() const {
        return qwen_native::resident_weight_cache_stats();
    }
    [[nodiscard]] const std::array<std::vector<float>, 5> &dflash_features() const {
        return dflash_features_;
    }
    void set_resident_weight_budget(std::size_t bytes) {
        qwen_native::set_resident_weight_cache_budget(bytes);
    }
    void freeze_target_weight_cache() { allow_dynamic_target_cache_ = false; }
    void preload_mtp_weights() {
        static const char *names[] = {
            "blk.64.attn_k.weight", "blk.64.attn_v.weight",
            "blk.64.attn_output.weight", "blk.64.nextn.eh_proj.weight",
            "blk.64.attn_q.weight", "blk.64.ffn_down.weight",
            "blk.64.ffn_gate.weight", "blk.64.ffn_up.weight",
        };
        for (const char *name : names) {
            preload_projection(name);
        }
    }
    void simulate_full_attention_window() {
        qwen_native::attention_advance(attn_device_, 4096 - 8);
    }
    void rewind(std::size_t count) {
        active_tokens_ -= count;
        qwen_native::attention_advance(attn_device_, -static_cast<int>(count));
    }
    std::vector<float> dflash_token_embedding(std::uint32_t token) const {
        if (token >= vocabulary_size_) {
            throw std::out_of_range("input token ID is outside the target vocabulary");
        }
        return embedding(token);
    }
    std::vector<float> dflash_output_logits_batch8(const std::vector<float> &hidden) const {
        return projection_batch8("output.weight", hidden);
    }

    ~NativeTargetSession() {
        if (pipeline_) {
            qwen_native::destroy_device_pipeline(pipeline_);
            pipeline_ = nullptr;
        }
        if (attn_device_) {
            qwen_native::destroy_attention_device_state(attn_device_);
            attn_device_ = nullptr;
        }
        for (auto &state : recurrent_) {
            qwen_native::destroy_gated_delta_state(state.device_matrix);
            state.device_matrix = nullptr;
        }
    }

    StepResult step(std::uint32_t token) {
        if (token >= vocabulary_size_) {
            throw std::out_of_range("input token ID is outside the target vocabulary");
        }
        if (active_tokens_ >= context_capacity_) {
            throw std::runtime_error("native target context is full");
        }
        const auto started = std::chrono::steady_clock::now();
        auto hidden = embedding(token);
        for (auto &feature : dflash_features_) {
            feature.clear();
        }
        std::size_t recurrent_index = 0;
        std::size_t attention_index = 0;
        for (int layer = 0; layer < 64; ++layer) {
            switch (layer) {
            case 6: dflash_features_[0] = hidden; break;
            case 20: dflash_features_[1] = hidden; break;
            case 34: dflash_features_[2] = hidden; break;
            case 48: dflash_features_[3] = hidden; break;
            case 62: dflash_features_[4] = hidden; break;
            default: break;
            }
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            const auto attention_input = rms_norm(hidden, f32_vector(prefix + "attn_norm.weight", kEmbedding),
                                                  kEmbedding, epsilon_);
            std::vector<float> attention_output;
            if ((layer + 1) % 4 != 0) {
                attention_output = recurrent_layer(prefix, attention_input, recurrent_.at(recurrent_index++));
            } else {
                attention_output = full_attention_layer(prefix, attention_input, attention_.at(attention_index++));
            }
            hidden = add_vectors(hidden, attention_output);
            const auto ffn_input = rms_norm(hidden, f32_vector(prefix + "post_attention_norm.weight", kEmbedding),
                                            kEmbedding, epsilon_);
            auto gate = projection(prefix + "ffn_gate.weight", ffn_input);
            const auto up = projection(prefix + "ffn_up.weight", ffn_input);
            for (std::size_t index = 0; index < gate.size(); ++index) {
                gate[index] = silu(gate[index]) * up[index];
            }
            hidden = add_vectors(hidden, projection(prefix + "ffn_down.weight", gate));
        }
        trunk_hidden_ = hidden;
        const auto final_hidden = rms_norm(hidden, f32_vector("output_norm.weight", kEmbedding),
                                           kEmbedding, epsilon_);
        const auto logits = projection("output.weight", final_hidden);
        const auto best = std::max_element(logits.begin(), logits.end());
        ++active_tokens_;
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return {static_cast<std::uint32_t>(std::distance(logits.begin(), best)), *best, elapsed};
    }

    [[nodiscard]] const std::vector<float> &trunk_hidden() const { return trunk_hidden_; }

    std::vector<float> step_logits(std::uint32_t token) {
        if (token >= vocabulary_size_) {
            throw std::out_of_range("input token ID is outside the target vocabulary");
        }
        if (active_tokens_ >= context_capacity_) {
            throw std::runtime_error("native target context is full");
        }
        auto hidden = embedding(token);
        for (auto &feature : dflash_features_) {
            feature.clear();
        }
        std::size_t recurrent_index = 0;
        std::size_t attention_index = 0;
        for (int layer = 0; layer < 64; ++layer) {
            switch (layer) {
            case 6: dflash_features_[0] = hidden; break;
            case 20: dflash_features_[1] = hidden; break;
            case 34: dflash_features_[2] = hidden; break;
            case 48: dflash_features_[3] = hidden; break;
            case 62: dflash_features_[4] = hidden; break;
            default: break;
            }
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            const auto attention_input = rms_norm(hidden, f32_vector(prefix + "attn_norm.weight", kEmbedding),
                                                  kEmbedding, epsilon_);
            std::vector<float> attention_output;
            if ((layer + 1) % 4 != 0) {
                attention_output = recurrent_layer(prefix, attention_input, recurrent_.at(recurrent_index++));
            } else {
                attention_output = full_attention_layer(prefix, attention_input, attention_.at(attention_index++));
            }
            hidden = add_vectors(hidden, attention_output);
            const auto ffn_input = rms_norm(hidden, f32_vector(prefix + "post_attention_norm.weight", kEmbedding),
                                            kEmbedding, epsilon_);
            auto gate = projection(prefix + "ffn_gate.weight", ffn_input);
            const auto up = projection(prefix + "ffn_up.weight", ffn_input);
            for (std::size_t index = 0; index < gate.size(); ++index) {
                gate[index] = silu(gate[index]) * up[index];
            }
            hidden = add_vectors(hidden, projection(prefix + "ffn_down.weight", gate));
        }
        const auto final_hidden = rms_norm(hidden, f32_vector("output_norm.weight", kEmbedding),
                                           kEmbedding, epsilon_);
        ++active_tokens_;
        return projection("output.weight", final_hidden);
    }

    std::uint32_t step_mtp(const std::vector<float> &trunk_hidden, std::uint32_t predicted_token) {
        const auto token_emb = embedding(predicted_token);
        const auto enorm = rms_norm(token_emb, f32_vector("blk.64.nextn.enorm.weight", kEmbedding),
                                    kEmbedding, epsilon_);
        const auto hnorm = rms_norm(trunk_hidden, f32_vector("blk.64.nextn.hnorm.weight", kEmbedding),
                                    kEmbedding, epsilon_);
        std::vector<float> concat(2 * kEmbedding);
        std::copy(enorm.begin(), enorm.end(), concat.begin());
        std::copy(hnorm.begin(), hnorm.end(), concat.begin() + kEmbedding);
        auto hidden = projection("blk.64.nextn.eh_proj.weight", concat);

        const std::string prefix = "blk.64.";
        const auto attention_input = rms_norm(hidden, f32_vector(prefix + "attn_norm.weight", kEmbedding),
                                              kEmbedding, epsilon_);
        const auto q_and_gate = projection(prefix + "attn_q.weight", attention_input);
        const auto keys = projection(prefix + "attn_k.weight", attention_input);
        const auto values = projection(prefix + "attn_v.weight", attention_input);
        const auto attention_output = full_attention_from_projected(
            prefix, q_and_gate, keys, values, mtp_attention_cache_, active_tokens_ - 1);
        hidden = add_vectors(hidden, attention_output);
        const auto ffn_input = rms_norm(hidden, f32_vector(prefix + "post_attention_norm.weight", kEmbedding),
                                        kEmbedding, epsilon_);
        auto gate = projection(prefix + "ffn_gate.weight", ffn_input);
        const auto up = projection(prefix + "ffn_up.weight", ffn_input);
        for (std::size_t index = 0; index < gate.size(); ++index) {
            gate[index] = silu(gate[index]) * up[index];
        }
        hidden = add_vectors(hidden, projection(prefix + "ffn_down.weight", gate));
        const auto final_hidden = rms_norm(hidden, f32_vector("blk.64.nextn.shared_head_norm.weight", kEmbedding),
                                           kEmbedding, epsilon_);
        const auto logits = projection("output.weight", final_hidden);
        const auto best = std::max_element(logits.begin(), logits.end());
        return static_cast<std::uint32_t>(std::distance(logits.begin(), best));
    }

    std::uint32_t step_mtp_from_batch(std::size_t batch_index, std::uint32_t predicted_token) {
        std::vector<float> hidden_at(batch_trunk_hidden_.begin() + batch_index * kEmbedding,
                                     batch_trunk_hidden_.begin() + (batch_index + 1) * kEmbedding);
        return step_mtp(hidden_at, predicted_token);
    }

    // This is the target side of speculative verification.  The recurrent
    // state is still advanced in token order, but every quantized projection
    // is issued as one native batch-8 HIP kernel.  Consequently cold target
    // weights cross PCIe once per proposed block rather than eight times.
    BatchStepResult step_batch8(const std::array<std::uint32_t, 8> &tokens) {
        if (active_tokens_ + tokens.size() > context_capacity_) {
            throw std::runtime_error("native target context is full");
        }
        for (const auto token : tokens) {
            if (token >= vocabulary_size_) {
                throw std::out_of_range("input token ID is outside the target vocabulary");
            }
        }
        const auto started = std::chrono::steady_clock::now();
        BatchStepResult result;
        prepare_batch_recurrent_states();
        auto hidden = embeddings_batch8(tokens);
        std::size_t recurrent_index = 0;
        std::size_t attention_index = 0;
        for (int layer = 0; layer < 64; ++layer) {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            const auto attention_input = rms_norm(hidden, f32_vector(prefix + "attn_norm.weight", kEmbedding),
                                                  kEmbedding, epsilon_);
            std::vector<float> attention_output;
            const auto attention_started = std::chrono::steady_clock::now();
            if ((layer + 1) % 4 != 0) {
                attention_output = recurrent_layer_batch8(prefix, attention_input,
                                                          recurrent_.at(recurrent_index++));
            } else {
                attention_output = full_attention_layer_batch8(prefix, attention_input,
                                                               attention_.at(attention_index++));
            }
            const double attention_elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - attention_started).count();
            if ((layer + 1) % 4 != 0) {
                result.recurrent_seconds += attention_elapsed;
            } else {
                result.attention_seconds += attention_elapsed;
            }
            hidden = add_vectors(hidden, attention_output);
            const auto ffn_input = rms_norm(hidden, f32_vector(prefix + "post_attention_norm.weight", kEmbedding),
                                            kEmbedding, epsilon_);
            const auto ffn_started = std::chrono::steady_clock::now();
            auto gate = projection_batch8(prefix + "ffn_gate.weight", ffn_input);
            const auto up = projection_batch8(prefix + "ffn_up.weight", ffn_input);
            for (std::size_t index = 0; index < gate.size(); ++index) {
                gate[index] = silu(gate[index]) * up[index];
            }
            hidden = add_vectors(hidden, projection_batch8(prefix + "ffn_down.weight", gate));
            result.ffn_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - ffn_started).count();
        }
        const auto final_hidden = rms_norm(hidden, f32_vector("output_norm.weight", kEmbedding),
                                           kEmbedding, epsilon_);
        const auto output_started = std::chrono::steady_clock::now();
        const auto logits = projection_batch8("output.weight", final_hidden);
        result.output_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - output_started).count();
        for (std::size_t batch = 0; batch < tokens.size(); ++batch) {
            const float *begin = logits.data() + batch * vocabulary_size_;
            const float *best = std::max_element(begin, begin + vocabulary_size_);
            result.next_tokens[batch] = static_cast<std::uint32_t>(best - begin);
            result.top_logits[batch] = *best;
        }
        active_tokens_ += tokens.size();
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return result;
    }

    BatchStepResult step_batch8_pipelined(const std::array<std::uint32_t, 8> &tokens) {
        if (active_tokens_ + tokens.size() > context_capacity_) {
            throw std::runtime_error("native target context is full");
        }
        for (const auto token : tokens) {
            if (token >= vocabulary_size_) {
                throw std::out_of_range("input token ID is outside the target vocabulary");
            }
        }
        const auto started = std::chrono::steady_clock::now();
        BatchStepResult result;
        prepare_batch_recurrent_states();

        auto hidden = embeddings_batch8(tokens);
        constexpr std::size_t hidden_count = 8 * kEmbedding;
        qwen_native::pipeline_upload_hidden(pipeline_, hidden.data(), hidden_count);

        std::size_t recurrent_index = 0;
        std::size_t attention_index = 0;

        for (int layer = 0; layer < 64; ++layer) {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            const auto &gate_tensor = tensor(prefix + "ffn_gate.weight");
            const auto &up_tensor = tensor(prefix + "ffn_up.weight");
            const auto &down_tensor = tensor(prefix + "ffn_down.weight");

            {
                const void *gp_data = gate_tensor.type == 21
                    ? static_cast<const void *>(iq3s_converted_.at(prefix + "ffn_gate.weight").data())
                    : tensor_data(mapping_, model_, gate_tensor);
                const std::uint32_t gp_type = gate_tensor.type == 21 ? 8u : gate_tensor.type;
                qwen_native::pipeline_prefetch_weights(pipeline_, gp_data, gp_type, kEmbedding, kFfn);
            }

            const auto attention_started = std::chrono::steady_clock::now();
            if ((layer + 1) % 4 != 0) {
                auto &state = recurrent_.at(recurrent_index);
                constexpr std::size_t qkv_width = 2 * kRecurrentKeyHeads * kRecurrentHeadSize +
                                                  kRecurrentValueHeads * kRecurrentHeadSize;
                constexpr std::size_t recurrent_width = kRecurrentValueHeads * kRecurrentHeadSize;

                qwen_native::pipeline_save_hidden(pipeline_, 3, hidden_count);
                qwen_native::pipeline_rms_norm(pipeline_,
                    f32_vector(prefix + "attn_norm.weight", kEmbedding), kEmbedding, epsilon_);

                const auto &qkv_tensor = tensor(prefix + "attn_qkv.weight");
                const auto &gate_t = tensor(prefix + "attn_gate.weight");
                const auto &ssm_out_tensor = tensor(prefix + "ssm_out.weight");
                qwen_native::pipeline_project_batch8(pipeline_,
                    tensor_data(mapping_, model_, qkv_tensor), qkv_tensor.type,
                    kEmbedding, qkv_width, true, 0);
                qwen_native::pipeline_project_batch8(pipeline_,
                    tensor_data(mapping_, model_, gate_t), gate_t.type,
                    kEmbedding, recurrent_width, true, 1);

                qwen_native::pipeline_prefetch_inline(pipeline_,
                    tensor_data(mapping_, model_, ssm_out_tensor), ssm_out_tensor.type,
                    recurrent_width, kEmbedding);

                qwen_native::pipeline_recurrent_batch8(pipeline_, recurrent_device_,
                    static_cast<int>(recurrent_index), state.device_matrix, epsilon_);
                state.host_matrix_current = false;
                state.device_matrix_current = true;

                qwen_native::pipeline_project_batch8(pipeline_,
                    tensor_data(mapping_, model_, ssm_out_tensor), ssm_out_tensor.type,
                    recurrent_width, kEmbedding, true, 0);
                qwen_native::pipeline_slot_to_hidden(pipeline_, 3, hidden_count);
                qwen_native::pipeline_residual_add(pipeline_, 0);

                recurrent_index++;
                result.recurrent_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - attention_started).count();
            } else {
                constexpr std::size_t q_width = 2 * kAttentionHeads * kAttentionHeadSize;
                constexpr std::size_t attn_result_width = kAttentionHeads * kAttentionHeadSize;

                qwen_native::pipeline_save_hidden(pipeline_, 3, hidden_count);
                qwen_native::pipeline_rms_norm(pipeline_,
                    f32_vector(prefix + "attn_norm.weight", kEmbedding), kEmbedding, epsilon_);

                const auto &q_tensor = tensor(prefix + "attn_q.weight");
                const auto &k_tensor = tensor(prefix + "attn_k.weight");
                const auto &v_tensor = tensor(prefix + "attn_v.weight");
                qwen_native::pipeline_project_batch8(pipeline_,
                    tensor_data(mapping_, model_, q_tensor), q_tensor.type,
                    kEmbedding, q_width, true, 0);
                qwen_native::pipeline_project_batch8(pipeline_,
                    tensor_data(mapping_, model_, k_tensor), k_tensor.type,
                    kEmbedding, kKvWidth, true, 1);
                qwen_native::pipeline_project_batch8(pipeline_,
                    tensor_data(mapping_, model_, v_tensor), v_tensor.type,
                    kEmbedding, kKvWidth, true, 2);


                qwen_native::pipeline_attention_batch8(pipeline_, attn_device_,
                    static_cast<int>(attention_index),
                    f32_vector(prefix + "attn_q_norm.weight", kAttentionHeadSize),
                    f32_vector(prefix + "attn_k_norm.weight", kAttentionHeadSize),
                    epsilon_);

                qwen_native::pipeline_slot_to_hidden(pipeline_, 0, 8 * attn_result_width);

                const auto &out_tensor = tensor(prefix + "attn_output.weight");
                qwen_native::pipeline_project_batch8(pipeline_,
                    tensor_data(mapping_, model_, out_tensor), out_tensor.type,
                    attn_result_width, kEmbedding, true, 0);
                qwen_native::pipeline_slot_to_hidden(pipeline_, 3, hidden_count);
                qwen_native::pipeline_residual_add(pipeline_, 0);

                attention_index++;
                result.attention_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - attention_started).count();
            }


            const void *gate_data = gate_tensor.type == 21
                ? static_cast<const void *>(iq3s_converted_.at(prefix + "ffn_gate.weight").data())
                : tensor_data(mapping_, model_, gate_tensor);
            const std::uint32_t gate_type = gate_tensor.type == 21 ? 8u : gate_tensor.type;
            const void *up_data = up_tensor.type == 21
                ? static_cast<const void *>(iq3s_converted_.at(prefix + "ffn_up.weight").data())
                : tensor_data(mapping_, model_, up_tensor);
            const std::uint32_t up_type = up_tensor.type == 21 ? 8u : up_tensor.type;
            const void *down_data = down_tensor.type == 21
                ? static_cast<const void *>(iq3s_converted_.at(prefix + "ffn_down.weight").data())
                : tensor_data(mapping_, model_, down_tensor);
            const std::uint32_t down_type = down_tensor.type == 21 ? 8u : down_tensor.type;

            qwen_native::pipeline_prefetch_weights(pipeline_, up_data, up_type, kEmbedding, kFfn);
            const auto ffn_started = std::chrono::steady_clock::now();
            const bool keep_gate_up = allow_dynamic_target_cache_;
            qwen_native::pipeline_fused_ffn_batch8(pipeline_,
                f32_vector(prefix + "post_attention_norm.weight", kEmbedding),
                kEmbedding, kFfn, epsilon_,
                gate_data, gate_type, up_data, up_type, down_data, down_type,
                keep_gate_up);
            result.ffn_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - ffn_started).count();

        }

        qwen_native::pipeline_download_hidden(pipeline_, hidden.data(), hidden_count);
        batch_trunk_hidden_ = hidden;
        const auto final_hidden = rms_norm(hidden, f32_vector("output_norm.weight", kEmbedding),
                                           kEmbedding, epsilon_);
        const auto output_started = std::chrono::steady_clock::now();
        const auto logits = projection_batch8("output.weight", final_hidden);
        result.output_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - output_started).count();

        for (std::size_t batch = 0; batch < tokens.size(); ++batch) {
            const float *begin = logits.data() + batch * vocabulary_size_;
            const float *best = std::max_element(begin, begin + vocabulary_size_);
            result.next_tokens[batch] = static_cast<std::uint32_t>(best - begin);
            result.top_logits[batch] = *best;
        }
        active_tokens_ += tokens.size();
        qwen_native::attention_advance(attn_device_, static_cast<int>(tokens.size()));
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return result;
    }

private:
    struct RecurrentState {
        std::vector<float> matrix;
        // Three previous q/k/v vectors, stored channel-major.
        std::vector<float> convolution;
        void *device_matrix = nullptr;
        bool host_matrix_current = true;
        bool device_matrix_current = true;
    };

    struct AttentionCache {
        // IEEE fp16 cache: 4 KV heads x 256 dimensions for each active token.
        // It is deliberately host-owned in this reference executor; the N23KV4
        // device cache is the follow-on hot-path representation.
        std::vector<std::uint16_t> keys;
        std::vector<std::uint16_t> values;
    };

    void prepare_batch_recurrent_states() {
        for (auto &state : recurrent_) {
            if (!state.device_matrix_current) {
                qwen_native::upload_gated_delta_state(state.device_matrix, state.matrix.data());
                state.device_matrix_current = true;
            }
        }
    }

    static void prepare_host_recurrent_state(RecurrentState &state) {
        if (!state.host_matrix_current) {
            qwen_native::download_gated_delta_state(state.device_matrix, state.matrix.data());
            state.host_matrix_current = true;
        }
    }

    const qwen_native::TensorInfo &tensor(const std::string &name) const {
        return require_tensor(model_, name);
    }

    const float *f32_vector(const std::string &name, std::size_t values) const {
        const auto &item = tensor(name);
        if (item.type != 0 || item.dimensions.empty() ||
            std::accumulate(item.dimensions.begin(), item.dimensions.end(), std::size_t{1},
                            [](std::size_t left, std::uint64_t right) {
                                return left * static_cast<std::size_t>(right);
                            }) != values) {
            throw std::runtime_error("expected F32 tensor with a different shape: " + name);
        }
        return static_cast<const float *>(tensor_data(mapping_, model_, item));
    }

    std::vector<float> f32_projection(const std::string &name, const std::vector<float> &input) const {
        const auto &item = tensor(name);
        if (item.dimensions.size() != 2 || item.dimensions[0] != input.size()) {
            throw std::runtime_error("F32/Q8 projection shape does not match activation: " + name);
        }
        const std::size_t outputs = static_cast<std::size_t>(item.dimensions[1]);
        const auto *raw = static_cast<const char *>(tensor_data(mapping_, model_, item));
        std::vector<float> output(outputs);
        if (item.type == 0) {
            const auto *weights = reinterpret_cast<const float *>(raw);
            for (std::size_t oi = 0; oi < outputs; ++oi) {
                const float *row = weights + oi * input.size();
                float sum = 0.0F;
                for (std::size_t ii = 0; ii < input.size(); ++ii) {
                    sum = std::fma(row[ii], input[ii], sum);
                }
                output[oi] = sum;
            }
        } else if (item.type == 8) {
            constexpr std::size_t block_size = 32;
            const std::size_t blocks_per_row = input.size() / block_size;
            constexpr std::size_t block_bytes = 34;
            for (std::size_t oi = 0; oi < outputs; ++oi) {
                const char *row = raw + oi * blocks_per_row * block_bytes;
                float sum = 0.0F;
                for (std::size_t bi = 0; bi < blocks_per_row; ++bi) {
                    const auto *block = row + bi * block_bytes;
                    std::uint16_t d_bits;
                    std::memcpy(&d_bits, block, 2);
                    const float d = __half2float(*reinterpret_cast<const __half *>(&d_bits));
                    const auto *qs = reinterpret_cast<const std::int8_t *>(block + 2);
                    for (std::size_t j = 0; j < block_size; ++j) {
                        sum = std::fma(d * static_cast<float>(qs[j]),
                                       input[bi * block_size + j], sum);
                    }
                }
                output[oi] = sum;
            }
        } else {
            throw std::runtime_error("f32_projection: unsupported type for " + name);
        }
        return output;
    }

    std::vector<float> f32_projection_raw(const std::string &name,
                                            const float *input, std::size_t input_size) const {
        const auto &item = tensor(name);
        if (item.dimensions.size() != 2 || item.dimensions[0] != input_size) {
            throw std::runtime_error("F32/Q8 projection shape does not match activation: " + name);
        }
        const std::size_t outputs = static_cast<std::size_t>(item.dimensions[1]);
        const auto *raw = static_cast<const char *>(tensor_data(mapping_, model_, item));
        std::vector<float> output(outputs);
        if (item.type == 0) {
            const auto *weights = reinterpret_cast<const float *>(raw);
            for (std::size_t oi = 0; oi < outputs; ++oi) {
                const float *row = weights + oi * input_size;
                float sum = 0.0F;
                for (std::size_t ii = 0; ii < input_size; ++ii) {
                    sum = std::fma(row[ii], input[ii], sum);
                }
                output[oi] = sum;
            }
        } else if (item.type == 8) {
            constexpr std::size_t block_size = 32;
            const std::size_t blocks_per_row = input_size / block_size;
            constexpr std::size_t block_bytes = 34;
            for (std::size_t oi = 0; oi < outputs; ++oi) {
                const char *row_ptr = raw + oi * blocks_per_row * block_bytes;
                float sum = 0.0F;
                for (std::size_t bi = 0; bi < blocks_per_row; ++bi) {
                    const auto *block = row_ptr + bi * block_bytes;
                    std::uint16_t d_bits;
                    std::memcpy(&d_bits, block, 2);
                    const float d = __half2float(*reinterpret_cast<const __half *>(&d_bits));
                    const auto *qs = reinterpret_cast<const std::int8_t *>(block + 2);
                    for (std::size_t j = 0; j < block_size; ++j) {
                        sum = std::fma(d * static_cast<float>(qs[j]),
                                       input[bi * block_size + j], sum);
                    }
                }
                output[oi] = sum;
            }
        } else {
            throw std::runtime_error("f32_projection_raw: unsupported type for " + name);
        }
        return output;
    }

    static bool is_supported_quant(std::uint32_t type) {
        return type == 8 || type == 11 || type == 12 || type == 13 || type == 14 ||
               type == 20 || type == 21 || type == 23;
    }

    static constexpr std::uint32_t kIQ3SGridTable[512] = {
        0x01010101,0x01010103,0x01010105,0x0101010b,0x0101010f,0x01010301,0x01010303,0x01010305,
        0x01010309,0x0101030d,0x01010501,0x01010503,0x0101050b,0x01010707,0x01010901,0x01010905,
        0x0101090b,0x0101090f,0x01010b03,0x01010b07,0x01010d01,0x01010d05,0x01010f03,0x01010f09,
        0x01010f0f,0x01030101,0x01030103,0x01030105,0x01030109,0x01030301,0x01030303,0x0103030b,
        0x01030501,0x01030507,0x0103050f,0x01030703,0x0103070b,0x01030909,0x01030d03,0x01030d0b,
        0x01030f05,0x01050101,0x01050103,0x0105010b,0x0105010f,0x01050301,0x01050307,0x0105030d,
        0x01050503,0x0105050b,0x01050701,0x01050709,0x01050905,0x0105090b,0x0105090f,0x01050b03,
        0x01050b07,0x01050f01,0x01050f07,0x01070107,0x01070303,0x0107030b,0x01070501,0x01070505,
        0x01070703,0x01070707,0x0107070d,0x01070909,0x01070b01,0x01070b05,0x01070d0f,0x01070f03,
        0x01070f0b,0x01090101,0x01090307,0x0109030f,0x01090503,0x01090509,0x01090705,0x01090901,
        0x01090907,0x01090b03,0x01090f01,0x010b0105,0x010b0109,0x010b0501,0x010b0505,0x010b050d,
        0x010b0707,0x010b0903,0x010b090b,0x010b090f,0x010b0d0d,0x010b0f07,0x010d010d,0x010d0303,
        0x010d0307,0x010d0703,0x010d0b05,0x010d0f03,0x010f0101,0x010f0105,0x010f0109,0x010f0501,
        0x010f0505,0x010f050d,0x010f0707,0x010f0b01,0x010f0b09,0x03010101,0x03010103,0x03010105,
        0x03010109,0x03010301,0x03010303,0x03010307,0x0301030b,0x0301030f,0x03010501,0x03010505,
        0x03010703,0x03010709,0x0301070d,0x03010b09,0x03010b0d,0x03010d03,0x03010f05,0x03030101,
        0x03030103,0x03030107,0x0303010d,0x03030301,0x03030309,0x03030503,0x03030701,0x03030707,
        0x03030903,0x03030b01,0x03030b05,0x03030f01,0x03030f0d,0x03050101,0x03050305,0x0305030b,
        0x0305030f,0x03050501,0x03050509,0x03050705,0x03050901,0x03050907,0x03050b0b,0x03050d01,
        0x03050f05,0x03070103,0x03070109,0x0307010f,0x03070301,0x03070307,0x03070503,0x0307050f,
        0x03070701,0x03070709,0x03070903,0x03070d05,0x03070f01,0x03090107,0x0309010b,0x03090305,
        0x03090309,0x03090703,0x03090707,0x03090905,0x0309090d,0x03090b01,0x03090b09,0x030b0103,
        0x030b0301,0x030b0307,0x030b0503,0x030b0701,0x030b0705,0x030b0b03,0x030d0501,0x030d0509,
        0x030d050f,0x030d0909,0x030d090d,0x030f0103,0x030f0107,0x030f0301,0x030f0305,0x030f0503,
        0x030f070b,0x030f0903,0x030f0d05,0x030f0f01,0x05010101,0x05010103,0x05010107,0x0501010b,
        0x0501010f,0x05010301,0x05010305,0x05010309,0x0501030d,0x05010503,0x05010507,0x0501050f,
        0x05010701,0x05010705,0x05010903,0x05010907,0x0501090b,0x05010b01,0x05010b05,0x05010d0f,
        0x05010f01,0x05010f07,0x05010f0b,0x05030101,0x05030105,0x05030301,0x05030307,0x0503030f,
        0x05030505,0x0503050b,0x05030703,0x05030709,0x05030905,0x05030b03,0x05050103,0x05050109,
        0x0505010f,0x05050503,0x05050507,0x05050701,0x0505070f,0x05050903,0x05050b07,0x05050b0f,
        0x05050f03,0x05050f09,0x05070101,0x05070105,0x0507010b,0x05070303,0x05070505,0x05070509,
        0x05070703,0x05070707,0x05070905,0x05070b01,0x05070d0d,0x05090103,0x0509010f,0x05090501,
        0x05090507,0x05090705,0x0509070b,0x05090903,0x05090f05,0x05090f0b,0x050b0109,0x050b0303,
        0x050b0505,0x050b070f,0x050b0901,0x050b0b07,0x050b0f01,0x050d0101,0x050d0105,0x050d010f,
        0x050d0503,0x050d0b0b,0x050d0d03,0x050f010b,0x050f0303,0x050f050d,0x050f0701,0x050f0907,
        0x050f0b01,0x07010105,0x07010303,0x07010307,0x0701030b,0x0701030f,0x07010505,0x07010703,
        0x07010707,0x0701070b,0x07010905,0x07010909,0x0701090f,0x07010b03,0x07010d07,0x07010f03,
        0x07030103,0x07030107,0x0703010b,0x07030309,0x07030503,0x07030507,0x07030901,0x07030d01,
        0x07030f05,0x07030f0d,0x07050101,0x07050305,0x07050501,0x07050705,0x07050709,0x07050b01,
        0x07070103,0x07070301,0x07070309,0x07070503,0x07070507,0x0707050f,0x07070701,0x07070903,
        0x07070907,0x0707090f,0x07070b0b,0x07070f07,0x07090107,0x07090303,0x0709030d,0x07090505,
        0x07090703,0x07090b05,0x07090d01,0x07090d09,0x070b0103,0x070b0301,0x070b0305,0x070b050b,
        0x070b0705,0x070b0909,0x070b0b0d,0x070b0f07,0x070d030d,0x070d0903,0x070f0103,0x070f0107,
        0x070f0501,0x070f0505,0x070f070b,0x09010101,0x09010109,0x09010305,0x09010501,0x09010509,
        0x0901050f,0x09010705,0x09010903,0x09010b01,0x09010f01,0x09030105,0x0903010f,0x09030303,
        0x09030307,0x09030505,0x09030701,0x0903070b,0x09030907,0x09030b03,0x09030b0b,0x09050103,
        0x09050107,0x09050301,0x0905030b,0x09050503,0x09050707,0x09050901,0x09050b0f,0x09050d05,
        0x09050f01,0x09070109,0x09070303,0x09070307,0x09070501,0x09070505,0x09070703,0x0907070b,
        0x09090101,0x09090105,0x09090509,0x0909070f,0x09090901,0x09090f03,0x090b010b,0x090b010f,
        0x090b0503,0x090b0d05,0x090d0307,0x090d0709,0x090d0d01,0x090f0301,0x090f030b,0x090f0701,
        0x090f0907,0x090f0b03,0x0b010105,0x0b010301,0x0b010309,0x0b010505,0x0b010901,0x0b010909,
        0x0b01090f,0x0b010b05,0x0b010d0d,0x0b010f09,0x0b030103,0x0b030107,0x0b03010b,0x0b030305,
        0x0b030503,0x0b030705,0x0b030f05,0x0b050101,0x0b050303,0x0b050507,0x0b050701,0x0b05070d,
        0x0b050b07,0x0b070105,0x0b07010f,0x0b070301,0x0b07050f,0x0b070909,0x0b070b03,0x0b070d0b,
        0x0b070f07,0x0b090103,0x0b090109,0x0b090501,0x0b090705,0x0b09090d,0x0b0b0305,0x0b0b050d,
        0x0b0b0b03,0x0b0b0b07,0x0b0d0905,0x0b0f0105,0x0b0f0109,0x0b0f0505,0x0d010303,0x0d010307,
        0x0d01030b,0x0d010703,0x0d010707,0x0d010d01,0x0d030101,0x0d030501,0x0d03050f,0x0d030d09,
        0x0d050305,0x0d050709,0x0d050905,0x0d050b0b,0x0d050d05,0x0d050f01,0x0d070101,0x0d070309,
        0x0d070503,0x0d070901,0x0d09050b,0x0d090907,0x0d090d05,0x0d0b0101,0x0d0b0107,0x0d0b0709,
        0x0d0b0d01,0x0d0d010b,0x0d0d0901,0x0d0f0303,0x0d0f0307,0x0f010101,0x0f010109,0x0f01010f,
        0x0f010501,0x0f010505,0x0f01070d,0x0f010901,0x0f010b09,0x0f010d05,0x0f030105,0x0f030303,
        0x0f030509,0x0f030907,0x0f03090b,0x0f050103,0x0f050109,0x0f050301,0x0f05030d,0x0f050503,
        0x0f050701,0x0f050b03,0x0f070105,0x0f070705,0x0f07070b,0x0f070b07,0x0f090103,0x0f09010b,
        0x0f090307,0x0f090501,0x0f090b01,0x0f0b0505,0x0f0b0905,0x0f0d0105,0x0f0d0703,0x0f0f0101,
    };

    std::vector<char> convert_iq3s_to_q8_0(const qwen_native::TensorInfo &item) const {
        const std::size_t input_size = static_cast<std::size_t>(item.dimensions[0]);
        const std::size_t output_size = static_cast<std::size_t>(item.dimensions[1]);
        const auto *raw = static_cast<const std::uint8_t *>(tensor_data(mapping_, model_, item));
        constexpr std::size_t iq3s_block_size = 256;
        constexpr std::size_t iq3s_block_bytes = 110;
        constexpr std::size_t q8_block_size = 32;
        constexpr std::size_t q8_block_bytes = 34;
        const std::size_t iq3s_blocks_per_row = input_size / iq3s_block_size;
        const std::size_t q8_blocks_per_row = input_size / q8_block_size;
        std::vector<char> q8_data(output_size * q8_blocks_per_row * q8_block_bytes);
        std::vector<float> row_f32(input_size);
        for (std::size_t row = 0; row < output_size; ++row) {
            const auto *src = raw + row * iq3s_blocks_per_row * iq3s_block_bytes;
            for (std::size_t bi = 0; bi < iq3s_blocks_per_row; ++bi) {
                const auto *block = src + bi * iq3s_block_bytes;
                std::uint16_t d_bits;
                std::memcpy(&d_bits, block, 2);
                const float d = __half2float(*reinterpret_cast<const __half *>(&d_bits));
                const auto *qs = block + 2;
                const auto *qh = block + 2 + 64;
                const auto *signs = block + 2 + 64 + 8;
                const auto *scales = block + 2 + 64 + 8 + 32;
                float *out = row_f32.data() + bi * iq3s_block_size;
                std::size_t out_idx = 0;
                for (int ib32 = 0; ib32 < 8; ib32 += 2) {
                    const float db1 = d * static_cast<float>(1 + 2 * (scales[ib32 / 2] & 0xf));
                    const float db2 = d * static_cast<float>(1 + 2 * (scales[ib32 / 2] >> 4));
                    for (int l = 0; l < 4; ++l) {
                        const auto *grid1 = reinterpret_cast<const std::uint8_t *>(
                            &kIQ3SGridTable[qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)]);
                        const auto *grid2 = reinterpret_cast<const std::uint8_t *>(
                            &kIQ3SGridTable[qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)]);
                        for (int j = 0; j < 4; ++j) {
                            out[out_idx + j + 0] = db1 * grid1[j] * ((signs[l] & (1 << j)) ? -1.0F : 1.0F);
                            out[out_idx + j + 4] = db1 * grid2[j] * ((signs[l] & (1 << (j + 4))) ? -1.0F : 1.0F);
                        }
                        out_idx += 8;
                    }
                    qs += 8;
                    signs += 4;
                    for (int l = 0; l < 4; ++l) {
                        const auto *grid1 = reinterpret_cast<const std::uint8_t *>(
                            &kIQ3SGridTable[qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)]);
                        const auto *grid2 = reinterpret_cast<const std::uint8_t *>(
                            &kIQ3SGridTable[qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)]);
                        for (int j = 0; j < 4; ++j) {
                            out[out_idx + j + 0] = db2 * grid1[j] * ((signs[l] & (1 << j)) ? -1.0F : 1.0F);
                            out[out_idx + j + 4] = db2 * grid2[j] * ((signs[l] & (1 << (j + 4))) ? -1.0F : 1.0F);
                        }
                        out_idx += 8;
                    }
                    qh += 2;
                    qs += 8;
                    signs += 4;
                }
            }
            auto *dst = q8_data.data() + row * q8_blocks_per_row * q8_block_bytes;
            for (std::size_t qi = 0; qi < q8_blocks_per_row; ++qi) {
                float amax = 0.0F;
                for (std::size_t j = 0; j < q8_block_size; ++j) {
                    float v = std::abs(row_f32[qi * q8_block_size + j]);
                    if (v > amax) amax = v;
                }
                const float scale = amax / 127.0F;
                const float inv_scale = (scale != 0.0F) ? 127.0F / amax : 0.0F;
                __half h_scale = __float2half(scale);
                std::memcpy(dst + qi * q8_block_bytes, &h_scale, 2);
                auto *q8_qs = reinterpret_cast<std::int8_t *>(dst + qi * q8_block_bytes + 2);
                for (std::size_t j = 0; j < q8_block_size; ++j) {
                    float v = row_f32[qi * q8_block_size + j] * inv_scale;
                    q8_qs[j] = static_cast<std::int8_t>(std::round(std::clamp(v, -127.0F, 127.0F)));
                }
            }
        }
        return q8_data;
    }

    std::vector<float> dequantize_q8_projection(const qwen_native::TensorInfo &item) const {
        const std::size_t input_size = static_cast<std::size_t>(item.dimensions[0]);
        const std::size_t outputs = static_cast<std::size_t>(item.dimensions[1]);
        const auto *raw = static_cast<const char *>(tensor_data(mapping_, model_, item));
        constexpr std::size_t block_size = 32;
        constexpr std::size_t block_bytes = 34;
        const std::size_t blocks_per_row = input_size / block_size;
        std::vector<float> result(outputs * input_size);
        for (std::size_t oi = 0; oi < outputs; ++oi) {
            const char *row_ptr = raw + oi * blocks_per_row * block_bytes;
            for (std::size_t bi = 0; bi < blocks_per_row; ++bi) {
                const auto *block = row_ptr + bi * block_bytes;
                std::uint16_t d_bits;
                std::memcpy(&d_bits, block, 2);
                const float d = __half2float(*reinterpret_cast<const __half *>(&d_bits));
                const auto *qs = reinterpret_cast<const std::int8_t *>(block + 2);
                for (std::size_t j = 0; j < block_size; ++j) {
                    result[oi * input_size + bi * block_size + j] = d * static_cast<float>(qs[j]);
                }
            }
        }
        return result;
    }

    std::vector<float> projection(const std::string &name, const std::vector<float> &input) const {
        const auto &item = tensor(name);
        if (item.dimensions.size() != 2 || item.dimensions[0] != input.size() ||
            !is_supported_quant(item.type)) {
            throw std::runtime_error("projection shape/type does not match activation: " + name);
        }
        std::vector<float> output(static_cast<std::size_t>(item.dimensions[1]));
        const bool keep_resident = allow_dynamic_target_cache_ && (name == "output.weight" ||
            (name.size() >= 6 && name.compare(0, 6, "blk.64") == 0) ||
            (name.size() >= 16 && (name.compare(name.size() - 16, 16, ".ffn_down.weight") == 0 ||
                                   name.compare(name.size() - 16, 16, ".attn_qkv.weight") == 0)));
        if (item.type == 21) {
            auto it = iq3s_converted_.find(name);
            if (it == iq3s_converted_.end()) {
                throw std::runtime_error("IQ3_S tensor not pre-converted: " + name);
            }
            qwen_native::execute_k_matvec_cached(it->second.data(), 8,
                                                 input.data(), input.size(), output.data(), output.size(), keep_resident);
        } else {
            qwen_native::execute_k_matvec_cached(tensor_data(mapping_, model_, item), item.type,
                                                 input.data(), input.size(), output.data(), output.size(), keep_resident);
        }
        return output;
    }

    std::vector<float> projection_batch8(const std::string &name, const std::vector<float> &input) const {
        constexpr std::size_t batch_size = 8;
        const auto &item = tensor(name);
        if (item.dimensions.size() != 2 || input.size() != batch_size * item.dimensions[0] ||
            !is_supported_quant(item.type)) {
            throw std::runtime_error("batch projection shape/type does not match activation: " + name);
        }
        const std::size_t outputs = static_cast<std::size_t>(item.dimensions[1]);
        std::vector<float> output(batch_size * outputs);
        const bool keep_resident = allow_dynamic_target_cache_ && (name == "output.weight" ||
            (name.size() >= 16 && (name.compare(name.size() - 16, 16, ".ffn_down.weight") == 0 ||
                                   name.compare(name.size() - 16, 16, ".attn_qkv.weight") == 0)));
        if (item.type == 21) {
            auto it = iq3s_converted_.find(name);
            if (it == iq3s_converted_.end()) {
                throw std::runtime_error("IQ3_S tensor not pre-converted: " + name);
            }
            qwen_native::execute_k_matvec_cached_batch8(
                it->second.data(), 8, input.data(),
                static_cast<std::size_t>(item.dimensions[0]), output.data(), outputs, keep_resident);
        } else {
            qwen_native::execute_k_matvec_cached_batch8(
                tensor_data(mapping_, model_, item), item.type, input.data(),
                static_cast<std::size_t>(item.dimensions[0]), output.data(), outputs, keep_resident);
        }
        return output;
    }

    void preload_projection(const std::string &name) const {
        const auto &item = tensor(name);
        if (item.dimensions.size() != 2 || !is_supported_quant(item.type)) {
            return;
        }
        if (item.type == 21) {
            auto it = iq3s_converted_.find(name);
            if (it == iq3s_converted_.end()) return;
            (void) qwen_native::preload_resident_k_matrix(
                it->second.data(), 8,
                static_cast<std::size_t>(item.dimensions[0]), static_cast<std::size_t>(item.dimensions[1]));
            return;
        }
        (void) qwen_native::preload_resident_k_matrix(
            tensor_data(mapping_, model_, item), item.type,
            static_cast<std::size_t>(item.dimensions[0]), static_cast<std::size_t>(item.dimensions[1]));
    }

    std::vector<float> embedding(std::uint32_t token) const {
        const auto &item = tensor("token_embd.weight");
        constexpr std::size_t row_bytes = kEmbedding / 256 * 144;
        return dequantize_q4k_row(static_cast<const char *>(tensor_data(mapping_, model_, item)) +
                                      static_cast<std::size_t>(token) * row_bytes,
                                  kEmbedding);
    }

    std::vector<float> embeddings_batch8(const std::array<std::uint32_t, 8> &tokens) const {
        std::vector<float> output(tokens.size() * kEmbedding);
        for (std::size_t batch = 0; batch < tokens.size(); ++batch) {
            const auto row = embedding(tokens[batch]);
            std::copy(row.begin(), row.end(), output.begin() + batch * kEmbedding);
        }
        return output;
    }

    static float softplus(float value) {
        // Avoid overflow while preserving the values that affect the decay.
        return value > 20.0F ? value : std::log1p(std::exp(value));
    }

    static std::uint16_t float_to_fp16(float value) {
        const __half half = __float2half(value);
        std::uint16_t bits = 0;
        static_assert(sizeof(bits) == sizeof(half));
        std::memcpy(&bits, &half, sizeof(bits));
        return bits;
    }

    void apply_rope(std::vector<float> &values, std::size_t heads, std::size_t position) const {
        constexpr std::size_t rotated = 64;
        constexpr std::size_t pairs = rotated / 2;
        const float theta_scale = std::pow(rope_frequency_base_, -2.0F / static_cast<float>(rotated));
        float theta = static_cast<float>(position);
        std::array<float, pairs> cosine{};
        std::array<float, pairs> sine{};
        for (std::size_t pair = 0; pair < pairs; ++pair) {
            cosine[pair] = std::cos(theta);
            sine[pair] = std::sin(theta);
            theta *= theta_scale;
        }
        for (std::size_t head = 0; head < heads; ++head) {
            float *item = values.data() + head * kAttentionHeadSize;
            for (std::size_t pair = 0; pair < pairs; ++pair) {
                const float left = item[pair];
                const float right = item[pair + pairs];
                item[pair] = left * cosine[pair] - right * sine[pair];
                item[pair + pairs] = left * sine[pair] + right * cosine[pair];
            }
        }
    }

    std::vector<float> recurrent_layer_from_projected(const std::string &prefix,
                                                      const std::vector<float> &input,
                                                      const std::vector<float> &qkv,
                                                      const std::vector<float> &z,
                                                      RecurrentState &state) const {
        const auto beta_unscaled = f32_projection(prefix + "ssm_beta.weight", input);
        const auto alpha = f32_projection(prefix + "ssm_alpha.weight", input);
        const auto *conv_weights = f32_vector(prefix + "ssm_conv1d.weight", 4 * qkv.size());
        const auto *dt_bias = f32_vector(prefix + "ssm_dt.bias", kRecurrentValueHeads);
        const auto *a = f32_vector(prefix + "ssm_a", kRecurrentValueHeads);

        std::vector<float> convolved(qkv.size());
        for (std::size_t channel = 0; channel < qkv.size(); ++channel) {
            float sum = qkv[channel] * conv_weights[channel * 4 + 3];
            for (std::size_t tap = 0; tap < 3; ++tap) {
                sum = std::fma(state.convolution[channel * 3 + tap],
                               conv_weights[channel * 4 + tap], sum);
            }
            convolved[channel] = silu(sum);
            state.convolution[channel * 3] = state.convolution[channel * 3 + 1];
            state.convolution[channel * 3 + 1] = state.convolution[channel * 3 + 2];
            state.convolution[channel * 3 + 2] = qkv[channel];
        }

        std::vector<float> queries(convolved.begin(), convolved.begin() + kRecurrentKeyHeads * kRecurrentHeadSize);
        std::vector<float> keys(convolved.begin() + kRecurrentKeyHeads * kRecurrentHeadSize,
                                convolved.begin() + 2 * kRecurrentKeyHeads * kRecurrentHeadSize);
        std::vector<float> values(convolved.begin() + 2 * kRecurrentKeyHeads * kRecurrentHeadSize, convolved.end());
        l2_norm_heads(queries, kRecurrentHeadSize, epsilon_);
        l2_norm_heads(keys, kRecurrentHeadSize, epsilon_);

        std::vector<float> output(kRecurrentValueHeads * kRecurrentHeadSize);
        constexpr std::size_t state_stride = kRecurrentHeadSize * kRecurrentHeadSize;
        const float q_scale = 1.0F / std::sqrt(static_cast<float>(kRecurrentHeadSize));
        for (std::size_t head = 0; head < kRecurrentValueHeads; ++head) {
            const std::size_t key_head = head % kRecurrentKeyHeads;
            const float decay = std::exp(softplus(alpha[head] + dt_bias[head]) * a[head]);
            const float beta = sigmoid(beta_unscaled[head]);
            const float *key = keys.data() + key_head * kRecurrentHeadSize;
            const float *query = queries.data() + key_head * kRecurrentHeadSize;
            const float *value = values.data() + head * kRecurrentHeadSize;
            float *matrix = state.matrix.data() + head * state_stride;
            std::array<float, kRecurrentHeadSize> delta{};
            for (std::size_t column = 0; column < kRecurrentHeadSize; ++column) {
                float predicted = 0.0F;
                for (std::size_t row = 0; row < kRecurrentHeadSize; ++row) {
                    float &cell = matrix[row * kRecurrentHeadSize + column];
                    cell *= decay;
                    predicted = std::fma(cell, key[row], predicted);
                }
                delta[column] = beta * (value[column] - predicted);
            }
            for (std::size_t row = 0; row < kRecurrentHeadSize; ++row) {
                const float scaled_key = key[row];
                for (std::size_t column = 0; column < kRecurrentHeadSize; ++column) {
                    matrix[row * kRecurrentHeadSize + column] = std::fma(scaled_key, delta[column],
                                                                            matrix[row * kRecurrentHeadSize + column]);
                }
            }
            for (std::size_t column = 0; column < kRecurrentHeadSize; ++column) {
                float sum = 0.0F;
                for (std::size_t row = 0; row < kRecurrentHeadSize; ++row) {
                    sum = std::fma(matrix[row * kRecurrentHeadSize + column], query[row], sum);
                }
                output[head * kRecurrentHeadSize + column] = sum * q_scale;
            }
        }
        auto normalized = rms_norm(output, f32_vector(prefix + "ssm_norm.weight", kRecurrentHeadSize),
                                   kRecurrentHeadSize, epsilon_);
        for (std::size_t index = 0; index < normalized.size(); ++index) {
            normalized[index] *= silu(z[index]);
        }
        state.host_matrix_current = true;
        state.device_matrix_current = false;
        return projection(prefix + "ssm_out.weight", normalized);
    }

    std::vector<float> recurrent_layer(const std::string &prefix, const std::vector<float> &input,
                                       RecurrentState &state) const {
        prepare_host_recurrent_state(state);
        const auto qkv = projection(prefix + "attn_qkv.weight", input);
        const auto z = projection(prefix + "attn_gate.weight", input);
        return recurrent_layer_from_projected(prefix, input, qkv, z, state);
    }

    std::vector<float> recurrent_layer_batch8(const std::string &prefix, const std::vector<float> &input,
                                               RecurrentState &state) const {
        constexpr std::size_t batch_size = 8;
        const auto qkv = projection_batch8(prefix + "attn_qkv.weight", input);
        const auto z = projection_batch8(prefix + "attn_gate.weight", input);
        constexpr std::size_t qkv_width = 2 * kRecurrentKeyHeads * kRecurrentHeadSize +
                                          kRecurrentValueHeads * kRecurrentHeadSize;
        constexpr std::size_t recurrent_width = kRecurrentValueHeads * kRecurrentHeadSize;
        const auto *conv_weights = f32_vector(prefix + "ssm_conv1d.weight", 4 * qkv_width);
        const auto *dt_bias = f32_vector(prefix + "ssm_dt.bias", kRecurrentValueHeads);
        const auto *a = f32_vector(prefix + "ssm_a", kRecurrentValueHeads);

        std::vector<float> all_queries(batch_size * recurrent_width);
        std::vector<float> all_keys(batch_size * recurrent_width);
        std::vector<float> all_values(batch_size * recurrent_width);
        std::vector<float> all_decays(batch_size * kRecurrentValueHeads);
        std::vector<float> all_betas(batch_size * kRecurrentValueHeads);
        std::vector<float> all_z(batch_size * recurrent_width);

        for (std::size_t batch = 0; batch < batch_size; ++batch) {
            const std::vector<float> input_row(input.begin() + batch * kEmbedding,
                                               input.begin() + (batch + 1) * kEmbedding);
            const float *qkv_row = qkv.data() + batch * qkv_width;
            std::copy_n(z.data() + batch * recurrent_width, recurrent_width,
                        all_z.data() + batch * recurrent_width);
            const auto beta_unscaled = f32_projection(prefix + "ssm_beta.weight", input_row);
            const auto alpha = f32_projection(prefix + "ssm_alpha.weight", input_row);
            std::vector<float> convolved(qkv_width);
            for (std::size_t channel = 0; channel < qkv_width; ++channel) {
                float sum = qkv_row[channel] * conv_weights[channel * 4 + 3];
                for (std::size_t tap = 0; tap < 3; ++tap) {
                    sum = std::fma(state.convolution[channel * 3 + tap],
                                   conv_weights[channel * 4 + tap], sum);
                }
                convolved[channel] = silu(sum);
                state.convolution[channel * 3] = state.convolution[channel * 3 + 1];
                state.convolution[channel * 3 + 1] = state.convolution[channel * 3 + 2];
                state.convolution[channel * 3 + 2] = qkv_row[channel];
            }
            float *q_out = all_queries.data() + batch * recurrent_width;
            float *k_out = all_keys.data() + batch * recurrent_width;
            float *v_out = all_values.data() + batch * recurrent_width;
            for (std::size_t head = 0; head < kRecurrentValueHeads; ++head) {
                const std::size_t key_head = head % kRecurrentKeyHeads;
                std::copy_n(convolved.data() + key_head * kRecurrentHeadSize, kRecurrentHeadSize,
                            q_out + head * kRecurrentHeadSize);
                std::copy_n(convolved.data() + (kRecurrentKeyHeads + key_head) * kRecurrentHeadSize,
                            kRecurrentHeadSize, k_out + head * kRecurrentHeadSize);
            }
            std::copy_n(convolved.data() + 2 * kRecurrentKeyHeads * kRecurrentHeadSize,
                        recurrent_width, v_out);
            l2_norm_heads_inplace(q_out, recurrent_width, kRecurrentHeadSize, epsilon_);
            l2_norm_heads_inplace(k_out, recurrent_width, kRecurrentHeadSize, epsilon_);
            for (std::size_t head = 0; head < kRecurrentValueHeads; ++head) {
                all_decays[batch * kRecurrentValueHeads + head] =
                    std::exp(softplus(alpha[head] + dt_bias[head]) * a[head]);
                all_betas[batch * kRecurrentValueHeads + head] = sigmoid(beta_unscaled[head]);
            }
        }

        std::vector<float> all_outputs(batch_size * recurrent_width);
        qwen_native::execute_gated_delta_step_batch8(
            all_queries.data(), all_keys.data(), all_values.data(),
            all_decays.data(), all_betas.data(), state.device_matrix,
            all_outputs.data(), static_cast<int>(batch_size));

        std::vector<float> normalized_batch(batch_size * recurrent_width);
        const auto *norm_weights = f32_vector(prefix + "ssm_norm.weight", kRecurrentHeadSize);
        for (std::size_t batch = 0; batch < batch_size; ++batch) {
            std::vector<float> recurrent_output(all_outputs.begin() + batch * recurrent_width,
                                                 all_outputs.begin() + (batch + 1) * recurrent_width);
            auto normalized = rms_norm(recurrent_output, norm_weights, kRecurrentHeadSize, epsilon_);
            for (std::size_t index = 0; index < normalized.size(); ++index) {
                normalized[index] *= silu(all_z[batch * recurrent_width + index]);
            }
            std::copy(normalized.begin(), normalized.end(),
                      normalized_batch.begin() + batch * recurrent_width);
        }
        state.host_matrix_current = false;
        state.device_matrix_current = true;
        return projection_batch8(prefix + "ssm_out.weight", normalized_batch);
    }

    std::vector<float> full_attention_from_projected(const std::string &prefix,
                                                     const std::vector<float> &q_and_gate,
                                                     std::vector<float> keys,
                                                     const std::vector<float> &values,
                                                     AttentionCache &cache,
                                                     std::size_t position) const {
        std::vector<float> queries(kAttentionHeads * kAttentionHeadSize);
        for (std::size_t head = 0; head < kAttentionHeads; ++head) {
            std::copy_n(q_and_gate.data() + head * 2 * kAttentionHeadSize, kAttentionHeadSize,
                        queries.data() + head * kAttentionHeadSize);
        }
        queries = rms_norm(queries, f32_vector(prefix + "attn_q_norm.weight", kAttentionHeadSize),
                           kAttentionHeadSize, epsilon_);
        keys = rms_norm(keys, f32_vector(prefix + "attn_k_norm.weight", kAttentionHeadSize),
                        kAttentionHeadSize, epsilon_);
        apply_rope(queries, kAttentionHeads, position);
        apply_rope(keys, kAttentionKvHeads, position);
        for (float value : keys) cache.keys.push_back(float_to_fp16(value));
        for (float value : values) cache.values.push_back(float_to_fp16(value));

        const std::size_t tokens = cache.keys.size() / kKvWidth;
        std::vector<float> result(kAttentionHeads * kAttentionHeadSize);
        const float scale = 1.0F / std::sqrt(static_cast<float>(kAttentionHeadSize));
        std::vector<float> scores(tokens);
        for (std::size_t head = 0; head < kAttentionHeads; ++head) {
            const std::size_t kv_head = head / (kAttentionHeads / kAttentionKvHeads);
            const float *query = queries.data() + head * kAttentionHeadSize;
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t token = 0; token < tokens; ++token) {
                const std::uint16_t *key = cache.keys.data() + token * kKvWidth + kv_head * kAttentionHeadSize;
                float dot = 0.0F;
                for (std::size_t dim = 0; dim < kAttentionHeadSize; ++dim) {
                    dot = std::fma(query[dim], fp16_to_float(key[dim]), dot);
                }
                scores[token] = dot * scale;
                maximum = std::max(maximum, scores[token]);
            }
            float denominator = 0.0F;
            for (float &score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            float *output = result.data() + head * kAttentionHeadSize;
            for (std::size_t token = 0; token < tokens; ++token) {
                const float probability = scores[token] / denominator;
                const std::uint16_t *value = cache.values.data() + token * kKvWidth + kv_head * kAttentionHeadSize;
                for (std::size_t dim = 0; dim < kAttentionHeadSize; ++dim) {
                    output[dim] = std::fma(probability, fp16_to_float(value[dim]), output[dim]);
                }
            }
            const float *gate = q_and_gate.data() + head * 2 * kAttentionHeadSize + kAttentionHeadSize;
            for (std::size_t dim = 0; dim < kAttentionHeadSize; ++dim) {
                output[dim] *= sigmoid(gate[dim]);
            }
        }
        return projection(prefix + "attn_output.weight", result);
    }

    std::vector<float> full_attention_layer(const std::string &prefix, const std::vector<float> &input,
                                            AttentionCache &cache) const {
        const auto q_and_gate = projection(prefix + "attn_q.weight", input);
        const auto keys = projection(prefix + "attn_k.weight", input);
        const auto values = projection(prefix + "attn_v.weight", input);
        return full_attention_from_projected(prefix, q_and_gate, keys, values, cache, active_tokens_);
    }

    std::vector<float> full_attention_layer_batch8(const std::string &prefix, const std::vector<float> &input,
                                                   AttentionCache &cache) const {
        constexpr std::size_t batch_size = 8;
        const auto q_and_gate = projection_batch8(prefix + "attn_q.weight", input);
        const auto keys = projection_batch8(prefix + "attn_k.weight", input);
        const auto values = projection_batch8(prefix + "attn_v.weight", input);
        std::vector<float> output(batch_size * kEmbedding);
        constexpr std::size_t q_width = 2 * kAttentionHeads * kAttentionHeadSize;
        for (std::size_t batch = 0; batch < batch_size; ++batch) {
            const std::vector<float> q_row(q_and_gate.begin() + batch * q_width,
                                           q_and_gate.begin() + (batch + 1) * q_width);
            const std::vector<float> k_row(keys.begin() + batch * kKvWidth,
                                           keys.begin() + (batch + 1) * kKvWidth);
            const std::vector<float> v_row(values.begin() + batch * kKvWidth,
                                           values.begin() + (batch + 1) * kKvWidth);
            const auto row = full_attention_from_projected(prefix, q_row, k_row, v_row, cache,
                                                           active_tokens_ + batch);
            std::copy(row.begin(), row.end(), output.begin() + batch * kEmbedding);
        }
        return output;
    }

    qwen_native::GgufFile model_;
    Mapping mapping_;
    std::size_t context_capacity_ = 0;
    std::size_t active_tokens_ = 0;
    std::size_t vocabulary_size_ = 0;
    float epsilon_ = 1.0e-6F;
    float rope_frequency_base_ = 1.0e7F;
    bool allow_dynamic_target_cache_ = true;
    std::vector<RecurrentState> recurrent_;
    std::vector<AttentionCache> attention_;
    AttentionCache mtp_attention_cache_;
    std::vector<float> trunk_hidden_;
    std::vector<float> batch_trunk_hidden_;
    std::array<std::vector<float>, 5> dflash_features_;
    qwen_native::DevicePipeline *pipeline_ = nullptr;
    qwen_native::RecurrentDeviceState *recurrent_device_ = nullptr;
    qwen_native::AttentionDeviceState *attn_device_ = nullptr;
    std::unordered_map<std::string, std::vector<char>> iq3s_converted_;
};

// DFlash2's encoder is intentionally a separate native component: it consumes
// the five target layer-input snapshots and emits the feature that is injected
// into the draft K/V cache.  It shares the target process' resident HIP cache
// but never routes execution through llama.cpp or ggml.
class NativeDflashEncoder {
public:
    explicit NativeDflashEncoder(const std::string &path) : model_(path), mapping_(path) {
        const auto *architecture = model_.find_metadata("general.architecture");
        if (!architecture || !std::holds_alternative<std::string>(*architecture) ||
            std::get<std::string>(*architecture) != "dflash") {
            throw std::runtime_error("native DFlash encoder requires a dflash GGUF model");
        }
        const auto &fc = require_tensor(model_, "fc.weight");
        if (fc.type != 12 || fc.dimensions.size() != 2 || fc.dimensions[0] != 5 * 5120 ||
            fc.dimensions[1] != 5120) {
            throw std::runtime_error("DFlash fc.weight is not the expected Q4_K [25600 x 5120] matrix");
        }
        const auto &norm = require_tensor(model_, "enc.output_norm.weight");
        if (norm.type != 0 || norm.dimensions.size() != 1 || norm.dimensions[0] != 5120) {
            throw std::runtime_error("DFlash encoder output norm is not F32[5120]");
        }
    }

    std::vector<float> encode(const std::array<std::vector<float>, 5> &features) const {
        std::vector<float> fused;
        fused.reserve(5 * 5120);
        for (const auto &feature : features) {
            if (feature.size() != 5120) {
                throw std::runtime_error("target did not capture all five DFlash layer-input features");
            }
            fused.insert(fused.end(), feature.begin(), feature.end());
        }
        const auto &fc = require_tensor(model_, "fc.weight");
        std::vector<float> encoded(5120);
        qwen_native::execute_k_matvec_cached(
            tensor_data(mapping_, model_, fc), fc.type, fused.data(), fused.size(), encoded.data(), encoded.size(), true);
        const auto &norm = require_tensor(model_, "enc.output_norm.weight");
        return rms_norm(encoded, static_cast<const float *>(tensor_data(mapping_, model_, norm)), 5120, 1.0e-6F);
    }

    void preload_all_quantized() const {
        for (const auto &tensor : model_.tensors()) {
            if (tensor.type == 12 || tensor.type == 13 || tensor.type == 14) {
                (void) qwen_native::preload_resident_k_matrix(
                    tensor_data(mapping_, model_, tensor), tensor.type,
                    static_cast<std::size_t>(tensor.dimensions.at(0)),
                    static_cast<std::size_t>(tensor.dimensions.at(1)));
                if (tensor.type == 12) {
                    qwen_native::preload_n23q4_matrix(
                        tensor_data(mapping_, model_, tensor),
                        static_cast<std::size_t>(tensor.dimensions.at(0)),
                        static_cast<std::size_t>(tensor.dimensions.at(1)));
                }
            }
        }
    }

    void inject(const std::array<std::vector<float>, 5> &features, std::size_t position) {
        const auto encoded = encode(features);
        for (int layer = 0; layer < 5; ++layer) {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            auto keys = projection(prefix + "attn_k.weight", encoded);
            const auto values = projection(prefix + "attn_v.weight", encoded);
            keys = rms_norm(keys, f32_vector(prefix + "attn_k_norm.weight", 128), 128, 1.0e-6F);
            apply_rope(keys, 8, position);
            append_injected(layer, keys, values);
        }
    }

    std::array<std::uint32_t, 7> propose(NativeTargetSession &target, std::uint32_t anchor,
                                         std::size_t position) const {
        constexpr std::size_t block = 8;
        constexpr std::size_t width = 5120;
        constexpr std::size_t heads = 32;
        constexpr std::size_t kv_heads = 8;
        constexpr std::size_t head_size = 128;
        constexpr std::size_t kv_width = kv_heads * head_size;
        constexpr std::uint32_t mask = 248070;
        std::vector<float> hidden(block * width);
        for (std::size_t item = 0; item < block; ++item) {
            const auto embedding = target.dflash_token_embedding(item == 0 ? anchor : mask);
            std::copy(embedding.begin(), embedding.end(), hidden.begin() + item * width);
        }
        for (int layer = 0; layer < 5; ++layer) {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            auto attention_input = rms_norm(hidden, f32_vector(prefix + "attn_norm.weight", width), width, 1.0e-6F);
            const auto attention_dynamic = projection_batch8(prefix + "attn_conv_proj.weight", attention_input);
            attention_input = dynamic_conv(attention_input, attention_dynamic,
                                           f32_data(prefix + "attn_conv_base", 2 * 2 * width), 0);
            auto queries = projection_batch8(prefix + "attn_q.weight", attention_input);
            auto keys = projection_batch8(prefix + "attn_k.weight", attention_input);
            const auto values = projection_batch8(prefix + "attn_v.weight", attention_input);
            queries = rms_norm(queries, f32_vector(prefix + "attn_q_norm.weight", head_size), head_size, 1.0e-6F);
            keys = rms_norm(keys, f32_vector(prefix + "attn_k_norm.weight", head_size), head_size, 1.0e-6F);
            for (std::size_t item = 0; item < block; ++item) {
                std::vector<float> q_row(queries.begin() + item * heads * head_size,
                                         queries.begin() + (item + 1) * heads * head_size);
                std::vector<float> k_row(keys.begin() + item * kv_width,
                                         keys.begin() + (item + 1) * kv_width);
                apply_rope(q_row, heads, position + item);
                apply_rope(k_row, kv_heads, position + item);
                std::copy(q_row.begin(), q_row.end(), queries.begin() + item * heads * head_size);
                std::copy(k_row.begin(), k_row.end(), keys.begin() + item * kv_width);
            }
            auto attention = noncausal_attention(layer, queries, keys, values);
            attention = projection_batch8(prefix + "attn_output.weight", attention);
            attention = dynamic_conv(attention, attention_dynamic,
                                     f32_data(prefix + "attn_conv_base", 2 * 2 * width), 1);
            const auto ffn_input = add_vectors(hidden, attention);
            auto ffn_hidden = rms_norm(ffn_input, f32_vector(prefix + "ffn_norm.weight", width), width, 1.0e-6F);
            const auto ffn_dynamic = projection_batch8(prefix + "ffn_conv_proj.weight", ffn_hidden);
            ffn_hidden = dynamic_conv(ffn_hidden, ffn_dynamic,
                                      f32_data(prefix + "ffn_conv_base", 2 * 2 * width), 0);
            auto gate = projection_batch8(prefix + "ffn_gate.weight", ffn_hidden);
            const auto up = projection_batch8(prefix + "ffn_up.weight", ffn_hidden);
            for (std::size_t item = 0; item < gate.size(); ++item) {
                gate[item] = silu(gate[item]) * up[item];
            }
            auto ffn_output = projection_batch8(prefix + "ffn_down.weight", gate);
            ffn_output = dynamic_conv(ffn_output, ffn_dynamic,
                                      f32_data(prefix + "ffn_conv_base", 2 * 2 * width), 1);
            hidden = add_vectors(ffn_input, ffn_output);
        }
        hidden = rms_norm(hidden, f32_vector("output_norm.weight", 5120), 5120, 1.0e-6F);
        const auto logits = target.dflash_output_logits_batch8(hidden);
        const auto selector_hidden = projection_batch8("selector_hidden.weight", hidden);
        return select(anchor, logits, selector_hidden);
    }

private:
    struct InjectedCache {
        std::vector<std::uint16_t> keys;
        std::vector<std::uint16_t> values;
    };

    const qwen_native::TensorInfo &tensor(const std::string &name) const {
        return require_tensor(model_, name);
    }

    const float *f32_vector(const std::string &name, std::size_t values) const {
        const auto &item = tensor(name);
        if (item.type != 0 || item.dimensions.empty() ||
            std::accumulate(item.dimensions.begin(), item.dimensions.end(), std::size_t{1},
                            [](std::size_t left, std::uint64_t right) {
                                return left * static_cast<std::size_t>(right);
                            }) != values) {
            throw std::runtime_error("expected DFlash F32 vector with a different shape: " + name);
        }
        return static_cast<const float *>(tensor_data(mapping_, model_, item));
    }

    const float *f32_data(const std::string &name, std::size_t values) const {
        return f32_vector(name, values);
    }

    std::vector<float> projection(const std::string &name, const std::vector<float> &input) const {
        const auto &item = tensor(name);
        if (item.dimensions.size() != 2 || item.dimensions[0] != input.size() ||
            (item.type != 12 && item.type != 13 && item.type != 14)) {
            throw std::runtime_error("DFlash projection shape/type does not match activation: " + name);
        }
        std::vector<float> output(static_cast<std::size_t>(item.dimensions[1]));
        qwen_native::execute_k_matvec_cached(tensor_data(mapping_, model_, item), item.type,
                                             input.data(), input.size(), output.data(), output.size(), true);
        return output;
    }

    std::vector<float> projection_batch8(const std::string &name, const std::vector<float> &input) const {
        const auto &item = tensor(name);
        if (item.dimensions.size() != 2 || input.size() != 8 * item.dimensions[0] ||
            (item.type != 12 && item.type != 13 && item.type != 14)) {
            throw std::runtime_error("DFlash batch projection shape/type does not match activation: " + name);
        }
        const std::size_t outputs = static_cast<std::size_t>(item.dimensions[1]);
        std::vector<float> output(8 * outputs);
        if (item.type == 12) {
            qwen_native::execute_n23q4_batch8_cached(
                tensor_data(mapping_, model_, item), input.data(),
                static_cast<std::size_t>(item.dimensions[0]), outputs, output.data());
            return output;
        }
        qwen_native::execute_k_matvec_cached_batch8(tensor_data(mapping_, model_, item), item.type,
                                                     input.data(), static_cast<std::size_t>(item.dimensions[0]),
                                                     output.data(), outputs, true);
        return output;
    }

    static void apply_rope(std::vector<float> &values, std::size_t heads, std::size_t position) {
        constexpr std::size_t rotated = 64;
        constexpr std::size_t pairs = rotated / 2;
        const float theta_scale = std::pow(1.0e7F, -2.0F / static_cast<float>(rotated));
        float theta = static_cast<float>(position);
        std::array<float, pairs> cosine{};
        std::array<float, pairs> sine{};
        for (std::size_t pair = 0; pair < pairs; ++pair) {
            cosine[pair] = std::cos(theta);
            sine[pair] = std::sin(theta);
            theta *= theta_scale;
        }
        for (std::size_t head = 0; head < heads; ++head) {
            float *item = values.data() + head * 128;
            for (std::size_t pair = 0; pair < pairs; ++pair) {
                const float left = item[pair];
                const float right = item[pair + pairs];
                item[pair] = left * cosine[pair] - right * sine[pair];
                item[pair + pairs] = left * sine[pair] + right * cosine[pair];
            }
        }
    }

    static std::uint16_t to_fp16(float value) {
        const __half half = __float2half(value);
        std::uint16_t bits = 0;
        std::memcpy(&bits, &half, sizeof(bits));
        return bits;
    }

    void append_injected(int layer, const std::vector<float> &keys, const std::vector<float> &values) {
        auto &cache = injected_.at(layer);
        constexpr std::size_t kv_width = 8 * 128;
        constexpr std::size_t window = 2048;
        if (cache.keys.size() / kv_width == window) {
            cache.keys.erase(cache.keys.begin(), cache.keys.begin() + kv_width);
            cache.values.erase(cache.values.begin(), cache.values.begin() + kv_width);
        }
        for (const float value : keys) cache.keys.push_back(to_fp16(value));
        for (const float value : values) cache.values.push_back(to_fp16(value));
    }

    std::vector<float> dynamic_conv(const std::vector<float> &values, const std::vector<float> &dynamic,
                                    const float *base, int side) const {
        constexpr std::size_t block = 8;
        constexpr std::size_t width = 5120;
        constexpr std::size_t groups = width / 16;
        std::vector<float> output(values.size());
        for (std::size_t item = 0; item < block; ++item) {
            for (std::size_t channel = 0; channel < width; ++channel) {
                const std::size_t group = channel / 16;
                float sum = 0.0F;
                for (std::size_t tap = 0; tap < 2; ++tap) {
                    if (item < tap) continue;
                    const float coefficient = base[channel + width * tap + width * 2 * side]
                        + dynamic[item * (groups * 4) + group + groups * tap + groups * 2 * side];
                    sum = std::fma(coefficient, values[(item - tap) * width + channel], sum);
                }
                output[item * width + channel] = sum;
            }
        }
        return output;
    }

    std::vector<float> noncausal_attention(int layer, const std::vector<float> &queries,
                                            const std::vector<float> &keys,
                                            const std::vector<float> &values) const {
        constexpr std::size_t block = 8;
        constexpr std::size_t heads = 32;
        constexpr std::size_t kv_heads = 8;
        constexpr std::size_t head_size = 128;
        constexpr std::size_t kv_width = kv_heads * head_size;
        const auto &cache = injected_.at(layer);
        const std::size_t cached = cache.keys.size() / kv_width;
        std::vector<float> output(block * heads * head_size);
        std::vector<float> scores(cached + block);
        for (std::size_t item = 0; item < block; ++item) {
            for (std::size_t head = 0; head < heads; ++head) {
                const std::size_t kv_head = head / (heads / kv_heads);
                const float *query = queries.data() + (item * heads + head) * head_size;
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t source = 0; source < cached + block; ++source) {
                    const std::uint16_t *key = source < cached
                        ? cache.keys.data() + source * kv_width + kv_head * head_size
                        : nullptr;
                    float dot = 0.0F;
                    for (std::size_t dim = 0; dim < head_size; ++dim) {
                        const float value = key ? fp16_to_float(key[dim])
                            : keys[(source - cached) * kv_width + kv_head * head_size + dim];
                        dot = std::fma(query[dim], value, dot);
                    }
                    scores[source] = dot * 0.08838834764831843F;
                    maximum = std::max(maximum, scores[source]);
                }
                float denominator = 0.0F;
                for (float &score : scores) {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                float *result = output.data() + (item * heads + head) * head_size;
                for (std::size_t source = 0; source < cached + block; ++source) {
                    const float probability = scores[source] / denominator;
                    const std::uint16_t *value = source < cached
                        ? cache.values.data() + source * kv_width + kv_head * head_size
                        : nullptr;
                    for (std::size_t dim = 0; dim < head_size; ++dim) {
                        const float item_value = value ? fp16_to_float(value[dim])
                            : values[(source - cached) * kv_width + kv_head * head_size + dim];
                        result[dim] = std::fma(probability, item_value, result[dim]);
                    }
                }
            }
        }
        return output;
    }

    std::vector<float> dequantize_selector_row(const std::string &name, std::uint32_t token) const {
        const auto &item = tensor(name);
        if (item.type != 12 || item.dimensions.size() != 2 || item.dimensions[0] != 256 ||
            token >= item.dimensions[1]) {
            throw std::runtime_error("DFlash selector tensor is not Q4_K[256 x vocab]");
        }
        constexpr std::size_t row_bytes = 144;
        return dequantize_q4k_row(static_cast<const char *>(tensor_data(mapping_, model_, item)) + token * row_bytes, 256);
    }

    std::array<std::uint32_t, 7> select(std::uint32_t anchor, const std::vector<float> &logits,
                                        const std::vector<float> &selector_hidden) const {
        constexpr std::size_t block = 8;
        constexpr std::size_t top_k = 16;
        const std::size_t vocabulary = logits.size() / block;
        std::array<std::array<std::uint32_t, top_k>, 8> candidates{};
        std::array<std::array<float, top_k>, 8> unary{};
        std::vector<std::uint32_t> order(vocabulary);
        std::iota(order.begin(), order.end(), 0U);
        for (std::size_t pos = 1; pos < block; ++pos) {
            std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                [&](std::uint32_t left, std::uint32_t right) {
                    return logits[pos * vocabulary + left] > logits[pos * vocabulary + right];
                });
            for (std::size_t item = 0; item < top_k; ++item) {
                candidates[pos][item] = order[item];
                unary[pos][item] = logits[pos * vocabulary + order[item]];
            }
        }
        std::array<std::uint32_t, 7> result{};
        std::array<std::vector<float>, top_k> predecessors{};
        predecessors[0] = dequantize_selector_row("selector_predecessor.weight", anchor);
        std::size_t predecessor = 0;
        for (std::size_t pos = 1; pos < block; ++pos) {
            std::array<std::vector<float>, top_k> successors{};
            for (std::size_t current = 0; current < top_k; ++current) {
                successors[current] = dequantize_selector_row("selector_successor.weight", candidates[pos][current]);
            }
            const float *hidden = selector_hidden.data() + pos * 256;
            std::array<float, top_k> scores{};
            for (std::size_t current = 0; current < top_k; ++current) {
                float score = unary[pos][current];
                const auto &previous = predecessors[predecessor];
                for (std::size_t rank = 0; rank < 256; ++rank) {
                    score = std::fma(successors[current][rank], previous[rank] * hidden[rank], score);
                }
                scores[current] = score;
            }
            predecessor = static_cast<std::size_t>(std::distance(scores.begin(),
                std::max_element(scores.begin(), scores.end())));
            result[pos - 1] = candidates[pos][predecessor];
            if (pos + 1 < block) {
                for (std::size_t item = 0; item < top_k; ++item) {
                    predecessors[item] = dequantize_selector_row("selector_predecessor.weight",
                                                                  candidates[pos][item]);
                }
            }
        }
        return result;
    }

    qwen_native::GgufFile model_;
    Mapping mapping_;
    std::array<InjectedCache, 5> injected_;
};

void mtp_test(const std::string &path, std::size_t context_capacity,
              std::uint32_t first_token, int steps,
              std::size_t resident_weight_budget) {
    NativeTargetSession session(path, context_capacity, resident_weight_budget);
    const qwen_native::GgufFile model(path);
    Tokenizer tokenizer(model);
    std::cout << "native Qwen3.8 MTP head test (blk.64 nextn_predict)\n"
              << "  context capacity: " << context_capacity << " tokens\n"
              << "  steps:            " << steps << "\n";
    std::uint32_t input = first_token;
    for (int i = 0; i < steps; ++i) {
        const auto result = session.step(input);
        const auto mtp_prediction = session.step_mtp(session.trunk_hidden(), result.next_token);
        std::cout << "  step " << (i + 1) << ": input " << input
                  << " -> main " << result.next_token
                  << " (\"" << tokenizer.decode_token(result.next_token) << "\")"
                  << " -> mtp " << mtp_prediction
                  << " (\"" << tokenizer.decode_token(mtp_prediction) << "\")"
                  << " [" << std::fixed << std::setprecision(3) << result.seconds << " s]\n";
        input = result.next_token;
    }
    const auto stats = session.resident_weight_stats();
    std::cout << "  resident weights: " << std::fixed << std::setprecision(2)
              << static_cast<double>(stats.resident_bytes) / (1024.0 * 1024.0 * 1024.0)
              << '/' << static_cast<double>(stats.budget_bytes) / (1024.0 * 1024.0 * 1024.0)
              << " GiB (" << stats.matrix_count << " matrices)\n";
}

void target_generate(const std::string &path, std::size_t context_capacity,
                     std::uint32_t first_token, int generated_tokens,
                     std::size_t resident_weight_budget) {
    if (generated_tokens <= 0) {
        throw std::invalid_argument("generated token count must be positive");
    }
    NativeTargetSession session(path, context_capacity, resident_weight_budget);
    std::uint32_t input = first_token;
    double total_seconds = 0.0;
    std::cout << "native Qwen3.8 target generation\n"
              << "  context capacity: " << session.context_capacity() << " tokens\n"
              << "  cache:            FP16 host reference cache (grows with active tokens)\n"
              << "  runtime:          native TheRock HIP projections; no llama.cpp/ggml runtime\n";
    for (int index = 0; index < generated_tokens; ++index) {
        const auto result = session.step(input);
        total_seconds += result.seconds;
        std::cout << "  step " << (index + 1) << ": input " << input
                  << " -> token " << result.next_token
                  << ", logit " << std::fixed << std::setprecision(5) << result.top_logit
                  << ", " << std::setprecision(3) << result.seconds << " s\n";
        input = result.next_token;
    }
    std::cout << "  active context:   " << session.active_tokens() << '/' << session.context_capacity() << "\n"
              << "  resident weights: " << std::fixed << std::setprecision(2)
              << static_cast<double>(session.resident_weight_stats().resident_bytes) / (1024.0 * 1024.0 * 1024.0)
              << '/' << static_cast<double>(session.resident_weight_stats().budget_bytes) / (1024.0 * 1024.0 * 1024.0)
              << " GiB (" << session.resident_weight_stats().matrix_count << " matrices)\n"
              << "  generation:       " << std::fixed << std::setprecision(3)
              << static_cast<double>(generated_tokens) / total_seconds << " tok/s\n"
              << "  measured time:    " << total_seconds << " s\n";
}

void target_verify_batch8(const std::string &path, std::size_t context_capacity,
                          std::uint32_t first_token, int blocks,
                          std::size_t resident_weight_budget) {
    if (blocks <= 0) {
        throw std::invalid_argument("block count must be positive");
    }
    NativeTargetSession session(path, context_capacity, resident_weight_budget);
    if (context_capacity >= 131072) {
        session.simulate_full_attention_window();
    }
    std::array<std::uint32_t, 8> candidate{};
    candidate[0] = first_token;
    // A verifier deliberately accepts arbitrary candidate tokens.  These mask
    // slots mirror DFlash2's block layout and exercise the real state update;
    // they are not presented as a generation or acceptance-rate measurement.
    candidate.fill(248070);
    candidate[0] = first_token;
    double total_seconds = 0.0;
    std::cout << "native Qwen3.8 target batch-8 verifier\n"
              << "  context capacity: " << context_capacity << " tokens\n"
              << "  runtime:          native TheRock HIP; no llama.cpp/ggml runtime\n";
    for (int block = 0; block < blocks; ++block) {
        const auto result = session.step_batch8_pipelined(candidate);
        total_seconds += result.seconds;
        std::cout << "  block " << block + 1 << ": first prediction " << result.next_tokens[0]
                  << ", " << std::fixed << std::setprecision(3) << result.seconds << " s"
                  << " (recurrent " << result.recurrent_seconds
                  << ", attention " << result.attention_seconds
                  << ", FFN " << result.ffn_seconds
                  << ", head " << result.output_seconds << ")\n";
        candidate = result.next_tokens;
    }
    const double verified_tokens = static_cast<double>(blocks) * candidate.size();
    const double effective_rate = verified_tokens / total_seconds;
    std::cout << "  active context:   " << session.active_tokens() << '/' << session.context_capacity() << "\n"
              << "  resident weights: " << std::fixed << std::setprecision(2)
              << static_cast<double>(session.resident_weight_stats().resident_bytes) / (1024.0 * 1024.0 * 1024.0)
              << '/' << static_cast<double>(session.resident_weight_stats().budget_bytes) / (1024.0 * 1024.0 * 1024.0)
              << " GiB (" << session.resident_weight_stats().matrix_count << " matrices)\n"
              << "  verifier rate:    " << std::fixed << std::setprecision(3)
              << effective_rate << " candidate tok/s\n"
              << "  measured time:    " << total_seconds << " s\n";
}

void dflash_encode(const std::string &target_path, const std::string &dflash_path,
                   std::size_t context_capacity, std::uint32_t token,
                   std::size_t resident_weight_budget) {
    constexpr std::size_t target_cache_budget = 5ULL * 1024ULL * 1024ULL * 1024ULL;
    NativeTargetSession target(target_path, context_capacity,
                               std::min(target_cache_budget, resident_weight_budget));
    NativeDflashEncoder dflash(dflash_path);
    const auto target_result = target.step(token);
    target.set_resident_weight_budget(resident_weight_budget);
    dflash.preload_all_quantized();
    target.freeze_target_weight_cache();
    const auto started = std::chrono::steady_clock::now();
    const auto injected = dflash.encode(target.dflash_features());
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    double squared = 0.0;
    for (const float value : injected) {
        squared += static_cast<double>(value) * value;
    }
    const auto stats = target.resident_weight_stats();
    std::cout << "native DFlash2 feature encoder\n"
              << "  target token:     " << token << " -> " << target_result.next_token << "\n"
              << "  extracted inputs: [6, 20, 34, 48, 62] x 5120\n"
              << "  fused output:     5120 values, RMS " << std::fixed << std::setprecision(6)
              << std::sqrt(squared / injected.size()) << "\n"
              << "  encode time:      " << std::setprecision(3) << elapsed << " s\n"
              << "  resident weights: " << std::setprecision(2)
              << static_cast<double>(stats.resident_bytes) / (1024.0 * 1024.0 * 1024.0)
              << '/' << static_cast<double>(stats.budget_bytes) / (1024.0 * 1024.0 * 1024.0)
              << " GiB (" << stats.matrix_count << " matrices)\n";
}

void dflash_propose(const std::string &target_path, const std::string &dflash_path,
                    std::size_t context_capacity, std::uint32_t token,
                    std::size_t resident_weight_budget) {
    constexpr std::size_t target_cache_budget = 5ULL * 1024ULL * 1024ULL * 1024ULL;
    NativeTargetSession target(target_path, context_capacity,
                               std::min(target_cache_budget, resident_weight_budget));
    NativeDflashEncoder dflash(dflash_path);
    const auto target_result = target.step(token);
    target.set_resident_weight_budget(resident_weight_budget);
    dflash.preload_all_quantized();
    target.freeze_target_weight_cache();
    const auto started = std::chrono::steady_clock::now();
    dflash.inject(target.dflash_features(), target.active_tokens() - 1);
    const auto proposals = dflash.propose(target, target_result.next_token, target.active_tokens());
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const auto stats = target.resident_weight_stats();
    std::cout << "native DFlash2 block proposal\n"
              << "  anchor:           " << target_result.next_token << " (from target input " << token << ")\n"
              << "  selector path:    ";
    for (const auto proposed : proposals) {
        std::cout << proposed << ' ';
    }
    std::cout << "\n"
              << "  block time:       " << std::fixed << std::setprecision(3) << elapsed << " s\n"
              << "  draft rate:       " << std::setprecision(3) << proposals.size() / elapsed << " proposal tok/s\n"
              << "  status:           proposal-only; target acceptance is measured separately\n"
              << "  resident weights: " << std::setprecision(2)
              << static_cast<double>(stats.resident_bytes) / (1024.0 * 1024.0 * 1024.0)
              << '/' << static_cast<double>(stats.budget_bytes) / (1024.0 * 1024.0 * 1024.0)
              << " GiB (" << stats.matrix_count << " matrices)\n";
}

void dflash_verify(const std::string &target_path, const std::string &dflash_path,
                   std::size_t context_capacity, std::uint32_t token,
                   std::size_t resident_weight_budget) {
    constexpr std::size_t target_cache_budget = 5ULL * 1024ULL * 1024ULL * 1024ULL;
    NativeTargetSession target(target_path, context_capacity,
                               std::min(target_cache_budget, resident_weight_budget));
    NativeDflashEncoder dflash(dflash_path);
    const auto initial_started = std::chrono::steady_clock::now();
    const auto anchor_result = target.step(token);
    const double initial_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - initial_started).count();
    target.set_resident_weight_budget(resident_weight_budget);
    dflash.preload_all_quantized();
    target.freeze_target_weight_cache();
    const auto draft_started = std::chrono::steady_clock::now();
    dflash.inject(target.dflash_features(), target.active_tokens() - 1);
    const auto proposals = dflash.propose(target, anchor_result.next_token, target.active_tokens());
    const double draft_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - draft_started).count();
    std::array<std::uint32_t, 8> candidate{};
    candidate[0] = anchor_result.next_token;
    std::copy(proposals.begin(), proposals.end(), candidate.begin() + 1);
    const auto verification = target.step_batch8(candidate);
    std::size_t accepted = 0;
    while (accepted < proposals.size() && verification.next_tokens[accepted] == proposals[accepted]) {
        ++accepted;
    }
    const double total_seconds = initial_seconds + draft_seconds + verification.seconds;
    const auto stats = target.resident_weight_stats();
    std::cout << "native DFlash2 -> target verification block\n"
              << "  anchor:           " << anchor_result.next_token << "\n"
              << "  proposals:        ";
    for (const auto proposed : proposals) std::cout << proposed << ' ';
    std::cout << "\n"
              << "  accepted prefix:  " << accepted << '/' << proposals.size() << "\n"
              << "  timing:           initial " << std::fixed << std::setprecision(3) << initial_seconds
              << " s, draft " << draft_seconds << " s, verifier " << verification.seconds << " s\n"
              << "  accepted rate:    " << std::setprecision(3) << accepted / total_seconds
              << " tok/s (one-block cold-start; not steady-state)\n"
              << "  resident weights: " << std::setprecision(2)
              << static_cast<double>(stats.resident_bytes) / (1024.0 * 1024.0 * 1024.0)
              << '/' << static_cast<double>(stats.budget_bytes) / (1024.0 * 1024.0 * 1024.0)
              << " GiB (" << stats.matrix_count << " matrices)\n";
}

std::vector<std::uint32_t> parse_token_list(const std::string &text) {
    std::vector<std::uint32_t> tokens;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) throw std::invalid_argument("token list contains an empty item");
        tokens.push_back(static_cast<std::uint32_t>(std::stoul(item)));
    }
    if (tokens.empty()) throw std::invalid_argument("token list is empty");
    return tokens;
}

void dflash_verify_prompt(const std::string &target_path, const std::string &dflash_path,
                          std::size_t context_capacity, const std::vector<std::uint32_t> &prompt,
                          std::size_t resident_weight_budget) {
    constexpr std::size_t target_cache_budget = 5ULL * 1024ULL * 1024ULL * 1024ULL;
    NativeTargetSession target(target_path, context_capacity,
                               std::min(target_cache_budget, resident_weight_budget));
    NativeDflashEncoder dflash(dflash_path);
    target.set_resident_weight_budget(resident_weight_budget);
    dflash.preload_all_quantized();
    target.freeze_target_weight_cache();
    const auto prefill_started = std::chrono::steady_clock::now();
    NativeTargetSession::StepResult anchor_result;
    for (const auto token : prompt) {
        anchor_result = target.step(token);
        dflash.inject(target.dflash_features(), target.active_tokens() - 1);
    }
    const double prefill_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - prefill_started).count();
    const auto draft_started = std::chrono::steady_clock::now();
    const auto proposals = dflash.propose(target, anchor_result.next_token, target.active_tokens());
    const double draft_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - draft_started).count();
    std::array<std::uint32_t, 8> candidate{};
    candidate[0] = anchor_result.next_token;
    std::copy(proposals.begin(), proposals.end(), candidate.begin() + 1);
    const auto verification = target.step_batch8(candidate);
    std::size_t accepted = 0;
    while (accepted < proposals.size() && verification.next_tokens[accepted] == proposals[accepted]) ++accepted;
    const double steady_block_seconds = draft_seconds + verification.seconds;
    std::cout << "native DFlash2 prompt verification\n"
              << "  prompt tokens:    " << prompt.size() << " (DFlash injected per target token)\n"
              << "  anchor:           " << anchor_result.next_token << "\n"
              << "  accepted prefix:  " << accepted << '/' << proposals.size() << "\n"
              << "  prefill time:     " << std::fixed << std::setprecision(3) << prefill_seconds << " s\n"
              << "  block timing:     draft " << draft_seconds << " s, verifier " << verification.seconds << " s\n"
              << "  accepted rate:    " << std::setprecision(3) << accepted / steady_block_seconds
              << " tok/s (post-prefill single block)\n";
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


void chat(const std::string &target_path, const std::string &dflash_path,
          std::size_t context_capacity, std::size_t resident_weight_budget) {
    NativeTargetSession target(target_path, context_capacity, resident_weight_budget);
    NativeDflashEncoder dflash(dflash_path);
    dflash.preload_all_quantized();
    target.freeze_target_weight_cache();

    const qwen_native::GgufFile model(target_path);
    Tokenizer tokenizer(model);

    bool show_thinking = false;

    const auto stats = target.resident_weight_stats();
    std::cerr << "qwen38-native chat (speculative decode)\n"
              << "  context:  " << context_capacity << " tokens\n"
              << "  weights:  " << std::fixed << std::setprecision(2)
              << static_cast<double>(stats.resident_bytes) / (1024.0 * 1024.0 * 1024.0)
              << " GiB (" << stats.matrix_count << " matrices)\n"
              << "  commands: /quit /reset /think\n\n";

    auto emit_token = [&](std::uint32_t token, bool &thinking) {
        if (token == 248068) { thinking = true; return; }
        if (token == 248069) { thinking = false; return; }
        if (thinking) {
            if (show_thinking) {
                std::cerr << "\033[2m" << tokenizer.decode_token(token) << "\033[0m" << std::flush;
            }
        } else {
            std::cout << tokenizer.decode_token(token) << std::flush;
        }
    };

    while (true) {
        std::cerr << "\033[1;32mYou:\033[0m ";
        std::string line;
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;
        if (line == "/quit" || line == "/exit") break;
        if (line == "/think") {
            show_thinking = !show_thinking;
            std::cerr << "  [thinking: " << (show_thinking ? "shown" : "hidden") << "]\n\n";
            continue;
        }
        if (line == "/reset") {
            target.~NativeTargetSession();
            new (&target) NativeTargetSession(target_path, context_capacity, resident_weight_budget);
            target.freeze_target_weight_cache();
            std::cerr << "  [context reset]\n\n";
            continue;
        }

        std::vector<std::uint32_t> prompt;
        prompt.push_back(tokenizer.im_start());
        prompt.push_back(846);   // user
        prompt.push_back(198);   // \n
        auto user_text = tokenizer.encode_full(line);
        prompt.insert(prompt.end(), user_text.begin(), user_text.end());
        prompt.push_back(tokenizer.im_end());
        prompt.push_back(198);
        prompt.push_back(tokenizer.im_start());
        prompt.push_back(74455); // assistant
        prompt.push_back(198);

        // Prefill: scalar step + DFlash inject for each prompt token
        const auto prefill_started = std::chrono::steady_clock::now();
        NativeTargetSession::StepResult anchor_result{};
        for (const auto token : prompt) {
            anchor_result = target.step(token);
            dflash.inject(target.dflash_features(), target.active_tokens() - 1);
        }
        const double prefill_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - prefill_started).count();
        std::cerr << "  [prefill " << prompt.size() << " tok, "
                  << std::fixed << std::setprecision(1) << prefill_seconds << "s]\n";

        std::cerr << "\033[1;34mAssistant:\033[0m " << std::flush;
        const auto gen_started = std::chrono::steady_clock::now();
        int generated = 0;
        bool done = false;
        bool thinking = false;

        // Speculative decode loop: DFlash proposes 7, target verifies 8
        std::uint32_t anchor = anchor_result.next_token;

        while (!done && generated < 8192) {
            if (anchor == tokenizer.im_end() || anchor == tokenizer.eos_token()) break;

            // DFlash proposes 7 continuation tokens
            const auto proposals = dflash.propose(target, anchor, target.active_tokens());

            // Build verification batch: [anchor, proposal_0, ..., proposal_6]
            std::array<std::uint32_t, 8> candidate{};
            candidate[0] = anchor;
            std::copy(proposals.begin(), proposals.end(), candidate.begin() + 1);

            // Target verifies all 8 in one pipelined batch
            const auto verification = target.step_batch8_pipelined(candidate);

            // Find accepted prefix (draft agrees with target)
            std::size_t accepted = 0;
            while (accepted < proposals.size() &&
                   verification.next_tokens[accepted] == proposals[accepted]) {
                ++accepted;
            }

            // Emit anchor token
            emit_token(anchor, thinking);
            ++generated;

            // Emit accepted draft tokens
            for (std::size_t i = 0; i < accepted; ++i) {
                if (proposals[i] == tokenizer.im_end() || proposals[i] == tokenizer.eos_token()) {
                    done = true;
                    break;
                }
                emit_token(proposals[i], thinking);
                ++generated;
            }

            if (done) break;

            // Rewind context past rejected positions so attention ignores them
            const std::size_t rejected = 7 - accepted;
            if (rejected > 0) {
                target.rewind(rejected);
            }

            // The bonus token: target's prediction at the rejection point
            anchor = verification.next_tokens[accepted];

            if (target.active_tokens() >= context_capacity - 8) {
                std::cerr << "\n  [context full]";
                done = true;
            }
        }

        const double gen_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - gen_started).count();
        std::cout << '\n';
        std::cerr << "  [" << generated << " tok, "
                  << std::fixed << std::setprecision(2)
                  << static_cast<double>(generated) / gen_seconds << " tok/s, "
                  << target.active_tokens() << "/" << context_capacity << " ctx]\n\n";
    }
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
              << "  " << program << " find-token MODEL.gguf TOKEN\n"
              << "  " << program << " chat TARGET.gguf [CONTEXT] [RESIDENT_GIB]\n"
              << "  " << program << " bench-matvec MODEL.gguf TENSOR [ITERATIONS]\n"
              << "  " << program << " bench-ffn MODEL.gguf LAYER [ITERATIONS]\n"
              << "  " << program << " bench-gdn [ITERATIONS]\n"
              << "  " << program << " target-generate TARGET.gguf CONTEXT TOKEN_ID [TOKENS] [RESIDENT_GIB]\n"
              << "  " << program << " target-verify-batch TARGET.gguf CONTEXT TOKEN_ID [BLOCKS] [RESIDENT_GIB]\n"
              << "  " << program << " dflash-encode TARGET.gguf DFLASH.gguf CONTEXT TOKEN_ID [RESIDENT_GIB]\n"
              << "  " << program << " dflash-propose TARGET.gguf DFLASH.gguf CONTEXT TOKEN_ID [RESIDENT_GIB]\n"
              << "  " << program << " dflash-verify TARGET.gguf DFLASH.gguf CONTEXT TOKEN_ID [RESIDENT_GIB]\n"
              << "  " << program << " dflash-verify-prompt TARGET.gguf DFLASH.gguf CONTEXT TOKENS_CSV [RESIDENT_GIB]\n"
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
        if (argc == 4 && std::string(argv[1]) == "find-token") {
            find_token(argv[2], argv[3]);
            return EXIT_SUCCESS;
        }
        if (argc == 4 && std::string(argv[1]) == "tokenize") {
            const qwen_native::GgufFile model(argv[2]);
            Tokenizer tokenizer(model);
            auto tokens = tokenizer.encode_full(argv[3]);
            std::cout << "tokens (" << tokens.size() << "): ";
            for (auto t : tokens) std::cout << t << ' ';
            std::cout << '\n';
            std::cout << "decode: \"" << tokenizer.decode(tokens) << "\"\n";
            return EXIT_SUCCESS;
        }
        if (argc >= 4 && std::string(argv[1]) == "prompt-test") {
            const std::string model_path = argv[2];
            const std::string user_text = argv[3];
            const int max_tokens = argc >= 5 ? std::stoi(argv[4]) : 20;
            constexpr std::size_t budget = 6ULL * 1024ULL * 1024ULL * 1024ULL + 400ULL * 1024ULL * 1024ULL;
            NativeTargetSession session(model_path, 4096, budget);
            const qwen_native::GgufFile model(model_path);
            Tokenizer tokenizer(model);
            std::vector<std::uint32_t> prompt;
            prompt.push_back(248045); // im_start
            prompt.push_back(846);    // user
            prompt.push_back(198);    // \n
            auto text_tokens = tokenizer.encode_full(user_text);
            prompt.insert(prompt.end(), text_tokens.begin(), text_tokens.end());
            prompt.push_back(248046); // im_end
            prompt.push_back(198);    // \n
            prompt.push_back(248045); // im_start
            prompt.push_back(74455);  // assistant
            prompt.push_back(198);    // \n
            std::cout << "prompt (" << prompt.size() << " tokens): ";
            for (auto t : prompt) std::cout << t << ' ';
            std::cout << '\n';
            NativeTargetSession::StepResult last{};
            for (const auto token : prompt) {
                last = session.step(token);
            }
            std::cout << "generation:\n";
            for (int i = 0; i < max_tokens; ++i) {
                std::cout << "  " << i << ": token " << last.next_token
                          << " logit " << std::fixed << std::setprecision(3) << last.top_logit
                          << " \"" << tokenizer.decode_token(last.next_token) << "\"\n";
                if (last.next_token == 248046) break;
                last = session.step(last.next_token);
            }
            return EXIT_SUCCESS;
        }
        if (argc >= 4 && argc <= 6 && std::string(argv[1]) == "chat") {
            const std::size_t context = argc >= 5 ? static_cast<std::size_t>(std::stoull(argv[4])) : 4096;
            const double gib = argc >= 6 ? std::stod(argv[5]) : 5.5;
            chat(argv[2], argv[3], context,
                 static_cast<std::size_t>(gib * 1024.0 * 1024.0 * 1024.0));
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
        if ((argc == 5 || argc == 6 || argc == 7) && std::string(argv[1]) == "mtp-test") {
            constexpr std::size_t default_resident_weight_budget = 6ULL * 1024ULL * 1024ULL * 1024ULL;
            const double requested_gib = argc == 7 ? std::stod(argv[6]) : 6.0;
            mtp_test(argv[2], static_cast<std::size_t>(std::stoull(argv[3])),
                     static_cast<std::uint32_t>(std::stoul(argv[4])),
                     argc >= 6 ? std::stoi(argv[5]) : 3,
                     argc == 7 ? static_cast<std::size_t>(requested_gib * 1024.0 * 1024.0 * 1024.0)
                               : default_resident_weight_budget);
            return EXIT_SUCCESS;
        }
        if ((argc == 5 || argc == 6 || argc == 7) && std::string(argv[1]) == "target-generate") {
            constexpr std::size_t default_resident_weight_budget = 6ULL * 1024ULL * 1024ULL * 1024ULL;
            const double requested_gib = argc == 7 ? std::stod(argv[6]) : 6.0;
            if (requested_gib < 0.0 || requested_gib > 7.0) {
                throw std::invalid_argument("RESIDENT_GIB must be in [0, 7]");
            }
            target_generate(argv[2], static_cast<std::size_t>(std::stoull(argv[3])),
                            static_cast<std::uint32_t>(std::stoul(argv[4])),
                            argc >= 6 ? std::stoi(argv[5]) : 1,
                            argc == 7 ? static_cast<std::size_t>(requested_gib * 1024.0 * 1024.0 * 1024.0)
                                      : default_resident_weight_budget);
            return EXIT_SUCCESS;
        }
        if ((argc == 5 || argc == 6 || argc == 7) && std::string(argv[1]) == "target-verify-batch") {
            constexpr std::size_t default_resident_weight_budget = 6ULL * 1024ULL * 1024ULL * 1024ULL + 400ULL * 1024ULL * 1024ULL;
            const double requested_gib = argc == 7 ? std::stod(argv[6]) : 6.39;
            if (requested_gib < 0.0 || requested_gib > 7.5) {
                throw std::invalid_argument("RESIDENT_GIB must be in [0, 7.5]");
            }
            target_verify_batch8(argv[2], static_cast<std::size_t>(std::stoull(argv[3])),
                                 static_cast<std::uint32_t>(std::stoul(argv[4])),
                                 argc >= 6 ? std::stoi(argv[5]) : 1,
                                 argc == 7 ? static_cast<std::size_t>(requested_gib * 1024.0 * 1024.0 * 1024.0)
                                           : default_resident_weight_budget);
            return EXIT_SUCCESS;
        }
        if ((argc == 6 || argc == 7) && std::string(argv[1]) == "dflash-encode") {
            const double requested_gib = argc == 7 ? std::stod(argv[6]) : 7.3;
            if (requested_gib < 0.0 || requested_gib > 7.5) {
                throw std::invalid_argument("RESIDENT_GIB must be in [0, 7.5]");
            }
            dflash_encode(argv[2], argv[3], static_cast<std::size_t>(std::stoull(argv[4])),
                          static_cast<std::uint32_t>(std::stoul(argv[5])),
                          static_cast<std::size_t>(requested_gib * 1024.0 * 1024.0 * 1024.0));
            return EXIT_SUCCESS;
        }
        if ((argc == 6 || argc == 7) && std::string(argv[1]) == "dflash-propose") {
            const double requested_gib = argc == 7 ? std::stod(argv[6]) : 7.3;
            if (requested_gib < 0.0 || requested_gib > 7.5) {
                throw std::invalid_argument("RESIDENT_GIB must be in [0, 7.5]");
            }
            dflash_propose(argv[2], argv[3], static_cast<std::size_t>(std::stoull(argv[4])),
                           static_cast<std::uint32_t>(std::stoul(argv[5])),
                           static_cast<std::size_t>(requested_gib * 1024.0 * 1024.0 * 1024.0));
            return EXIT_SUCCESS;
        }
        if ((argc == 6 || argc == 7) && std::string(argv[1]) == "dflash-verify") {
            const double requested_gib = argc == 7 ? std::stod(argv[6]) : 7.3;
            if (requested_gib < 0.0 || requested_gib > 7.5) {
                throw std::invalid_argument("RESIDENT_GIB must be in [0, 7.5]");
            }
            dflash_verify(argv[2], argv[3], static_cast<std::size_t>(std::stoull(argv[4])),
                          static_cast<std::uint32_t>(std::stoul(argv[5])),
                          static_cast<std::size_t>(requested_gib * 1024.0 * 1024.0 * 1024.0));
            return EXIT_SUCCESS;
        }
        if ((argc == 6 || argc == 7) && std::string(argv[1]) == "dflash-verify-prompt") {
            const double requested_gib = argc == 7 ? std::stod(argv[6]) : 7.3;
            if (requested_gib < 0.0 || requested_gib > 7.5) {
                throw std::invalid_argument("RESIDENT_GIB must be in [0, 7.5]");
            }
            dflash_verify_prompt(argv[2], argv[3], static_cast<std::size_t>(std::stoull(argv[4])),
                                 parse_token_list(argv[5]),
                                 static_cast<std::size_t>(requested_gib * 1024.0 * 1024.0 * 1024.0));
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

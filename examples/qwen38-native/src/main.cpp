#include "cpu_matvec.h"
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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <fcntl.h>
#include <map>
#include <numeric>
#include <optional>
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
        if (hipHostRegister(data_, size_, hipHostRegisterReadOnly | hipHostRegisterMapped) != hipSuccess) {
            std::cerr << "  [warning: hipHostRegister failed, PCIe transfers will be slower]\n";
        } else {
            registered_ = true;
        }
    }

    ~Mapping() {
        if (registered_) {
            (void) hipHostUnregister(data_);
        }
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
    bool registered_ = false;
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
    // The current FP16 verifier cache is intentionally bounded.  A true 128K
    // N23KV4 history replaces this limit in the long-context path; until then
    // mode switches reject an oversized host history instead of silently
    // dropping it.
    static constexpr std::size_t kPipelineAttentionCapacity = 4096;

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

    struct SpeculativeCheckpoint {
        std::size_t active_tokens = 0;
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

        // Preload deterministic high-priority matrices before optional entries
        // compete for the remaining cache.  This is a model-load cost, not a
        // decode-timing cost.  Priority: ffn_down (avoids 3-way transfer_stream
        // stall), then recurrent weights that block compute with no prefetch.
        if (resident_weight_budget != 0) {
            if (!std::getenv("QWEN_NATIVE_HYBRID_HEAD")) {
                preload_projection("output.weight");
            }
            for (int layer = 0; layer < 64; ++layer) {
                const std::string prefix = "blk." + std::to_string(layer) + ".";
                preload_projection(prefix + "ffn_down.weight");
            }
            for (int layer = 0; layer < 64; ++layer) {
                if ((layer + 1) % 4 == 0) {
                    const std::string prefix = "blk." + std::to_string(layer) + ".";
                    preload_projection(prefix + "attn_k.weight");
                    preload_projection(prefix + "attn_v.weight");
                    preload_projection(prefix + "attn_output.weight");
                }
            }
            for (int layer = 0; layer < 64; ++layer) {
                const std::string prefix = "blk." + std::to_string(layer) + ".";
                if ((layer + 1) % 4 != 0) {
                    preload_projection(prefix + "attn_qkv.weight");
                    preload_projection(prefix + "attn_gate.weight");
                }
            }
            for (int layer = 0; layer < 64; ++layer) {
                const std::string prefix = "blk." + std::to_string(layer) + ".";
                if ((layer + 1) % 4 != 0) {
                    preload_projection(prefix + "ssm_out.weight");
                }
            }
        }

        constexpr std::size_t max_ffn_weight = (kFfn / 32) * kEmbedding * 34;
        pipeline_ = qwen_native::create_device_pipeline(kEmbedding, kFfn, max_ffn_weight, context_capacity_);

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
            16, static_cast<int>(context_capacity_), rope_frequency_base_);
    }

    [[nodiscard]] std::size_t context_capacity() const { return context_capacity_; }
    [[nodiscard]] static constexpr std::size_t pipeline_attention_capacity() {
        return kPipelineAttentionCapacity;
    }
    [[nodiscard]] std::size_t active_tokens() const { return active_tokens_; }
    [[nodiscard]] std::size_t vocabulary_size() const { return vocabulary_size_; }
    [[nodiscard]] qwen_native::ResidentWeightCacheStats resident_weight_stats() const {
        return qwen_native::resident_weight_cache_stats();
    }
    [[nodiscard]] const std::array<std::vector<float>, 5> &dflash_features() const {
        return dflash_features_;
    }
    [[nodiscard]] std::array<std::vector<float>, 5> batch_dflash_features(
        std::size_t batch_index) const {
        if (batch_index >= 8) {
            throw std::out_of_range("DFlash feature batch index is outside [0, 8)");
        }
        std::array<std::vector<float>, 5> result;
        for (std::size_t feature = 0; feature < result.size(); ++feature) {
            const auto &all_items = batch_dflash_features_.at(feature);
            if (all_items.size() != 8 * kEmbedding) {
                throw std::runtime_error("pipelined target did not capture DFlash features");
            }
            const auto begin = all_items.begin() + batch_index * kEmbedding;
            result[feature].assign(begin, begin + kEmbedding);
        }
        return result;
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
        if (active_tokens_ != 0 || context_capacity_ < 8) {
            throw std::runtime_error("synthetic full-window timing requires a fresh target session");
        }
        active_tokens_ = context_capacity_ - 8;
        qwen_native::attention_advance(attn_device_, static_cast<int>(active_tokens_));
    }
    void rewind(std::size_t count) {
        if (count > active_tokens_) {
            throw std::out_of_range("cannot rewind more target tokens than are active");
        }
        if (pipeline_state_current_) {
            qwen_native::attention_advance(attn_device_, -static_cast<int>(count));
        } else {
            for (auto &cache : attention_) {
                cache.keys.resize((active_tokens_ - count) * kKvWidth);
                cache.values.resize((active_tokens_ - count) * kKvWidth);
            }
        }
        active_tokens_ -= count;
    }
    SpeculativeCheckpoint begin_speculative_checkpoint() {
        // The N23KV4 history remains device-resident at long context. Only
        // recurrent state needs a host mirror: replay overwrites KV entries
        // beyond the restored logical length, so it never reads a rejected
        // suffix. This avoids materializing an 8 GiB FP16 host cache.
        if (context_capacity_ > kPipelineAttentionCapacity && pipeline_state_current_) {
            for (std::size_t layer = 0; layer < recurrent_.size(); ++layer) {
                auto &state = recurrent_[layer];
                prepare_host_recurrent_state(state);
                qwen_native::recurrent_download_convolution_state(
                    recurrent_device_, static_cast<int>(layer), state.convolution.data(), state.convolution.size());
            }
            return {active_tokens_};
        }
        synchronize_host_state();
        for (auto &state : recurrent_) {
            prepare_host_recurrent_state(state);
        }
        return {active_tokens_};
    }

    void commit_speculative_prefix(SpeculativeCheckpoint checkpoint, std::size_t committed_tokens) {
        const std::size_t speculative_tokens = active_tokens_ - checkpoint.active_tokens;
        if (checkpoint.active_tokens > active_tokens_ || committed_tokens == 0 ||
            committed_tokens > speculative_tokens || speculative_tokens > 8) {
            throw std::invalid_argument("invalid speculative prefix commit");
        }
        active_tokens_ = checkpoint.active_tokens + committed_tokens;
        qwen_native::attention_set_history_length(attn_device_, static_cast<int>(active_tokens_));
        for (std::size_t layer = 0; layer < recurrent_.size(); ++layer) {
            auto &state = recurrent_[layer];
            // begin_speculative_checkpoint retained these exact host copies.
            qwen_native::upload_gated_delta_state(state.device_matrix, state.matrix.data());
            qwen_native::recurrent_upload_convolution_state(
                recurrent_device_, static_cast<int>(layer), state.convolution.data(), state.convolution.size());
            qwen_native::recurrent_commit_captured_prefix(
                recurrent_device_, static_cast<int>(layer), state.device_matrix,
                static_cast<int>(committed_tokens));
            state.host_matrix_current = false;
            state.device_matrix_current = true;
        }
        batch_trunk_hidden_.clear();
        pipeline_state_current_ = true;
    }

    void restore_speculative_checkpoint(SpeculativeCheckpoint checkpoint) {
        if (checkpoint.active_tokens > active_tokens_) {
            throw std::invalid_argument("speculative checkpoint is ahead of target state");
        }
        if (context_capacity_ > kPipelineAttentionCapacity) {
            active_tokens_ = checkpoint.active_tokens;
            qwen_native::attention_set_history_length(attn_device_, static_cast<int>(active_tokens_));
            for (auto &state : recurrent_) {
                state.host_matrix_current = true;
                state.device_matrix_current = false;
            }
            batch_trunk_hidden_.clear();
            pipeline_state_current_ = false;
            return;
        }
        for (const auto &cache : attention_) {
            const std::size_t expected = checkpoint.active_tokens * kKvWidth;
            if (cache.keys.size() != expected || cache.values.size() != expected) {
                throw std::runtime_error("speculative checkpoint host cache was modified");
            }
        }
        active_tokens_ = checkpoint.active_tokens;
        for (auto &state : recurrent_) {
            state.host_matrix_current = true;
            state.device_matrix_current = false;
        }
        batch_trunk_hidden_.clear();
        pipeline_state_current_ = false;
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
    void dflash_output_top16_batch8(const std::vector<float> &hidden,
                                    int *out_indices, float *out_scores) const {
        const auto &tensor = require_tensor(model_, "output.weight");
        if (hidden.size() != 8 * kEmbedding) {
            throw std::runtime_error("DFlash top16 hidden has wrong size");
        }
        qwen_native::execute_k_matvec_cached_batch8_top16(
            tensor_data(mapping_, model_, tensor), tensor.type,
            hidden.data(), kEmbedding, out_indices, out_scores,
            static_cast<std::size_t>(tensor.dimensions[1]), false);
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
        synchronize_host_state();
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
            switch (layer) {
            case 5: dflash_features_[0] = hidden; break;
            case 19: dflash_features_[1] = hidden; break;
            case 33: dflash_features_[2] = hidden; break;
            case 47: dflash_features_[3] = hidden; break;
            case 61: dflash_features_[4] = hidden; break;
            default: break;
            }
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
        synchronize_host_state();
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
            switch (layer) {
            case 5: dflash_features_[0] = hidden; break;
            case 19: dflash_features_[1] = hidden; break;
            case 33: dflash_features_[2] = hidden; break;
            case 47: dflash_features_[3] = hidden; break;
            case 61: dflash_features_[4] = hidden; break;
            default: break;
            }
        }
        const auto final_hidden = rms_norm(hidden, f32_vector("output_norm.weight", kEmbedding),
                                           kEmbedding, epsilon_);
        ++active_tokens_;
        return projection("output.weight", final_hidden);
    }

    std::uint32_t step_mtp(const std::vector<float> &trunk_hidden, std::uint32_t predicted_token) {
        synchronize_host_state();
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
        synchronize_host_state();
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

    BatchStepResult step_batch8_pipelined(const std::array<std::uint32_t, 8> &tokens,
                                           std::size_t valid_tokens = 8,
                                           bool capture_dflash = true) {
        synchronize_pipeline_state();
        if (valid_tokens == 0 || valid_tokens > tokens.size()) {
            throw std::invalid_argument("pipelined target batch must contain [1, 8] valid tokens");
        }
        if (active_tokens_ + valid_tokens > context_capacity_) {
            throw std::runtime_error("native target context is full");
        }
        std::array<std::uint32_t, 8> padded_tokens{};
        for (std::size_t index = 0; index < valid_tokens; ++index) {
            if (tokens[index] >= vocabulary_size_) {
                throw std::out_of_range("input token ID is outside the target vocabulary");
            }
            padded_tokens[index] = tokens[index];
        }
        const auto started = std::chrono::steady_clock::now();
        BatchStepResult result;
        const auto target_projection = [](qwen_native::DevicePipeline *pipe,
                                          const void *w, std::uint32_t type,
                                          std::size_t in_size, std::size_t out_size,
                                          bool resident, int slot,
                                          const float *in_buf = nullptr) {
            static const bool hybrid = std::getenv("QWEN_NATIVE_HYBRID") != nullptr;
            if (hybrid) {
                qwen_native::pipeline_project_batch8_hybrid(pipe, w, type, in_size, out_size,
                                                            resident, slot, in_buf);
            } else {
                qwen_native::pipeline_project_batch8(pipe, w, type, in_size, out_size,
                                                     resident, slot, in_buf);
            }
        };
        prepare_batch_recurrent_states();

        auto hidden = embeddings_batch8(padded_tokens);
        constexpr std::size_t hidden_count = 8 * kEmbedding;
        for (auto &features : batch_dflash_features_) {
            features.assign(hidden_count, 0.0F);
        }
        qwen_native::pipeline_upload_hidden(pipeline_, hidden.data(), hidden_count);

        std::size_t recurrent_index = 0;
        std::size_t attention_index = 0;

        {
            constexpr std::size_t first_qkv_width = 2 * kRecurrentKeyHeads * kRecurrentHeadSize +
                                                    kRecurrentValueHeads * kRecurrentHeadSize;
            const auto &first_qkv = tensor("blk.0.attn_qkv.weight");
            qwen_native::pipeline_prefetch_weights(
                pipeline_, tensor_data(mapping_, model_, first_qkv), first_qkv.type,
                kEmbedding, first_qkv_width);
        }

        for (int layer = 0; layer < 64; ++layer) {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            const auto &gate_tensor = tensor(prefix + "ffn_gate.weight");
            const auto &up_tensor = tensor(prefix + "ffn_up.weight");
            const auto &down_tensor = tensor(prefix + "ffn_down.weight");

            const void *gate_data = tensor_data(mapping_, model_, gate_tensor);
            const std::uint32_t gate_type = gate_tensor.type;
            const void *up_data = tensor_data(mapping_, model_, up_tensor);
            const std::uint32_t up_type = up_tensor.type;
            const void *down_data = tensor_data(mapping_, model_, down_tensor);
            const std::uint32_t down_type = down_tensor.type;

            const bool hybrid_soa = std::getenv("QWEN_NATIVE_HYBRID") != nullptr;
            const bool hybrid_ffn = std::getenv("QWEN_NATIVE_HYBRID_FFN") != nullptr;
            // The SoA tensor-parallel path consumes disjoint rows directly and
            // does not stage a complete gate/up matrix. Legacy GPU scheduling
            // retains its overlap prefetch.
            if (!hybrid_soa) {
                if (hybrid_ffn) {
                    qwen_native::pipeline_prefetch_weights(
                        pipeline_, gate_data, gate_type, kEmbedding, kFfn);
                } else {
                    qwen_native::pipeline_prefetch_weights(
                        pipeline_, up_data, up_type, kEmbedding, kFfn);
                }
            }

            const auto attention_started = std::chrono::steady_clock::now();
            if ((layer + 1) % 4 != 0) {
                auto &state = recurrent_.at(recurrent_index);
                constexpr std::size_t qkv_width = 2 * kRecurrentKeyHeads * kRecurrentHeadSize +
                                                  kRecurrentValueHeads * kRecurrentHeadSize;
                constexpr std::size_t recurrent_width = kRecurrentValueHeads * kRecurrentHeadSize;

                float *slot3 = qwen_native::pipeline_slot_ptr(pipeline_, 3);
                float *hidden = qwen_native::pipeline_hidden_ptr(pipeline_);
                float *slot0 = qwen_native::pipeline_slot_ptr(pipeline_, 0);

                qwen_native::pipeline_rms_norm(pipeline_,
                    f32_vector(prefix + "attn_norm.weight", kEmbedding), kEmbedding, epsilon_,
                    slot3, hidden);

                const auto &qkv_tensor = tensor(prefix + "attn_qkv.weight");
                const auto &gate_t = tensor(prefix + "attn_gate.weight");
                const auto &ssm_out_tensor = tensor(prefix + "ssm_out.weight");
                if (!std::getenv("QWEN_NATIVE_GPU_GATE")) {
                    qwen_native::pipeline_project_recurrent_qkv_gate_cpu(
                        pipeline_, tensor_data(mapping_, model_, qkv_tensor), qkv_tensor.type,
                        kEmbedding, qkv_width, tensor_data(mapping_, model_, gate_t), gate_t.type,
                        recurrent_width, true, slot3);
                } else {
                    target_projection(pipeline_,
                        tensor_data(mapping_, model_, qkv_tensor), qkv_tensor.type,
                        kEmbedding, qkv_width, true, 0, slot3);
                    target_projection(pipeline_,
                        tensor_data(mapping_, model_, gate_t), gate_t.type,
                        kEmbedding, recurrent_width, true, 1, slot3);
                }

                qwen_native::pipeline_prefetch_inline(pipeline_,
                    tensor_data(mapping_, model_, ssm_out_tensor), ssm_out_tensor.type,
                    recurrent_width, kEmbedding);

                qwen_native::pipeline_recurrent_batch8(pipeline_, recurrent_device_,
                    static_cast<int>(recurrent_index), state.device_matrix, epsilon_,
                    static_cast<int>(valid_tokens), slot3, slot0);
                state.host_matrix_current = false;
                state.device_matrix_current = true;

                target_projection(pipeline_,
                    tensor_data(mapping_, model_, ssm_out_tensor), ssm_out_tensor.type,
                    recurrent_width, kEmbedding, true, 1, slot0);
                qwen_native::pipeline_residual_add(pipeline_, 1);

                recurrent_index++;
                result.recurrent_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - attention_started).count();
            } else {
                const auto attention_started = std::chrono::steady_clock::now();
                constexpr std::size_t q_width = 2 * kAttentionHeads * kAttentionHeadSize;
                constexpr std::size_t attn_result_width = kAttentionHeads * kAttentionHeadSize;

                float *slot3 = qwen_native::pipeline_slot_ptr(pipeline_, 3);
                float *hidden = qwen_native::pipeline_hidden_ptr(pipeline_);
                qwen_native::pipeline_rms_norm(pipeline_,
                    f32_vector(prefix + "attn_norm.weight", kEmbedding), kEmbedding, epsilon_,
                    slot3, hidden);

                const auto &q_tensor = tensor(prefix + "attn_q.weight");
                const auto &k_tensor = tensor(prefix + "attn_k.weight");
                const auto &v_tensor = tensor(prefix + "attn_v.weight");
                if (hybrid_soa) {
                    qwen_native::pipeline_project_attention_qkv_hybrid(
                        pipeline_,
                        tensor_data(mapping_, model_, q_tensor), q_tensor.type, q_width,
                        tensor_data(mapping_, model_, k_tensor), k_tensor.type, kKvWidth,
                        tensor_data(mapping_, model_, v_tensor), v_tensor.type, kKvWidth,
                        kEmbedding, true, slot3);
                } else {
                    target_projection(pipeline_,
                        tensor_data(mapping_, model_, q_tensor), q_tensor.type,
                        kEmbedding, q_width, true, 0, slot3);
                    target_projection(pipeline_,
                        tensor_data(mapping_, model_, k_tensor), k_tensor.type,
                        kEmbedding, kKvWidth, true, 1, slot3);
                    target_projection(pipeline_,
                        tensor_data(mapping_, model_, v_tensor), v_tensor.type,
                        kEmbedding, kKvWidth, true, 2, slot3);
                }

                const auto &out_tensor = tensor(prefix + "attn_output.weight");
                qwen_native::pipeline_prefetch_inline(pipeline_,
                    tensor_data(mapping_, model_, out_tensor), out_tensor.type,
                    attn_result_width, kEmbedding);

                qwen_native::pipeline_attention_batch8(pipeline_, attn_device_,
                    static_cast<int>(attention_index),
                    f32_vector(prefix + "attn_q_norm.weight", kAttentionHeadSize),
                    f32_vector(prefix + "attn_k_norm.weight", kAttentionHeadSize),
                    epsilon_, static_cast<int>(valid_tokens));

                target_projection(pipeline_,
                    tensor_data(mapping_, model_, out_tensor), out_tensor.type,
                    attn_result_width, kEmbedding, true, 1, qwen_native::pipeline_slot_ptr(pipeline_, 0));
                qwen_native::pipeline_residual_add(pipeline_, 1);

                attention_index++;
                result.attention_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - attention_started).count();
            }


            const auto ffn_started = std::chrono::steady_clock::now();
            const bool keep_gate_up = allow_dynamic_target_cache_;
            if (hybrid_soa) {
                qwen_native::pipeline_fused_ffn_batch8_hybrid_soa(pipeline_,
                    f32_vector(prefix + "post_attention_norm.weight", kEmbedding),
                    kEmbedding, kFfn, epsilon_,
                    gate_data, gate_type, up_data, up_type, down_data, down_type,
                    keep_gate_up);
            } else if (hybrid_ffn) {
                qwen_native::pipeline_fused_ffn_batch8_hybrid_up(pipeline_,
                    f32_vector(prefix + "post_attention_norm.weight", kEmbedding),
                    kEmbedding, kFfn, epsilon_,
                    gate_data, gate_type, up_data, up_type, down_data, down_type,
                    keep_gate_up);
            } else {
                qwen_native::pipeline_fused_ffn_batch8(pipeline_,
                    f32_vector(prefix + "post_attention_norm.weight", kEmbedding),
                    kEmbedding, kFfn, epsilon_,
                    gate_data, gate_type, up_data, up_type, down_data, down_type,
                    keep_gate_up);
            }
            result.ffn_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - ffn_started).count();

            int dflash_feature = -1;
            switch (layer) {
            case 5: dflash_feature = 0; break;
            case 19: dflash_feature = 1; break;
            case 33: dflash_feature = 2; break;
            case 47: dflash_feature = 3; break;
            case 61: dflash_feature = 4; break;
            default: break;
            }
            if (capture_dflash && dflash_feature >= 0) {
                qwen_native::pipeline_download_hidden_async(
                    pipeline_, batch_dflash_features_[dflash_feature].data(), hidden_count);
            }

            if (layer + 1 < 64) {
                const std::string next_prefix = "blk." + std::to_string(layer + 1) + ".";
                if ((layer + 2) % 4 != 0) {
                    constexpr std::size_t next_qkv_width = 2 * kRecurrentKeyHeads * kRecurrentHeadSize +
                                                           kRecurrentValueHeads * kRecurrentHeadSize;
                    const auto &next_qkv = tensor(next_prefix + "attn_qkv.weight");
                    qwen_native::pipeline_prefetch_weights(
                        pipeline_, tensor_data(mapping_, model_, next_qkv), next_qkv.type,
                        kEmbedding, next_qkv_width);
                } else {
                    constexpr std::size_t next_q_width = 2 * kAttentionHeads * kAttentionHeadSize;
                    const auto &next_q = tensor(next_prefix + "attn_q.weight");
                    qwen_native::pipeline_prefetch_weights(
                        pipeline_, tensor_data(mapping_, model_, next_q), next_q.type,
                        kEmbedding, next_q_width);
                }
            }
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
        active_tokens_ += valid_tokens;
        qwen_native::attention_advance(attn_device_, static_cast<int>(valid_tokens));
        pipeline_state_current_ = true;
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return result;
    }

private:
    void synchronize_pipeline_state() {
        if (pipeline_state_current_) return;
        const bool long_context = context_capacity_ > kPipelineAttentionCapacity;
        if (!long_context) {
            for (std::size_t layer = 0; layer < attention_.size(); ++layer) {
                const auto &cache = attention_[layer];
                const std::size_t values = active_tokens_ * kKvWidth;
                if (cache.keys.size() != values || cache.values.size() != values) {
                    throw std::runtime_error("scalar attention cache length does not match target state");
                }
                qwen_native::attention_upload_history(attn_device_, static_cast<int>(layer),
                                                       cache.keys.data(), cache.values.data(), active_tokens_);
            }
        } else if (active_tokens_ != 0 && attention_.front().keys.size() == active_tokens_ * kKvWidth) {
            throw std::runtime_error("long-context pipeline cannot replace an existing scalar FP16 history");
        }
        qwen_native::attention_set_history_length(attn_device_, static_cast<int>(active_tokens_));
        for (std::size_t layer = 0; layer < recurrent_.size(); ++layer) {
            auto &state = recurrent_[layer];
            if (!state.device_matrix_current) {
                qwen_native::upload_gated_delta_state(state.device_matrix, state.matrix.data());
                state.device_matrix_current = true;
            }
            qwen_native::recurrent_upload_convolution_state(
                recurrent_device_, static_cast<int>(layer), state.convolution.data(), state.convolution.size());
        }
        pipeline_state_current_ = true;
    }
public:
    // Fast prefill for large batches (512 tokens/call) with weight reuse and no DFlash
    BatchStepResult step_batch_large(const std::vector<uint32_t> &tokens) {
        if (tokens.empty() || tokens.size() > 1024) throw std::invalid_argument("prefill batch must be in [1,1024]");
        // For large prefill, keep weights resident across the 64 sub-batches of the 512 chunk
        // to cut PCIe 64×. Preload the first 8 layers' QKV to warm the cache.
        if (tokens.size() > 64) {
            for (int layer = 0; layer < 8; ++layer) {
                const std::string prefix = "blk." + std::to_string(layer) + ".";
                if ((layer+1)%4==0) {
                    const auto &q = tensor(prefix+"attn_q.weight"); qwen_native::preload_resident_k_matrix(tensor_data(mapping_,model_,q), q.type, kEmbedding, 2*kAttentionHeads*kAttentionHeadSize);
                } else {
                    const auto &qkv = tensor(prefix+"attn_qkv.weight"); qwen_native::preload_resident_k_matrix(tensor_data(mapping_,model_,qkv), qkv.type, kEmbedding, 2*kRecurrentKeyHeads*kRecurrentHeadSize + kRecurrentValueHeads*kRecurrentHeadSize);
                }
            }
        }
        BatchStepResult result;
        for (size_t offset = 0; offset < tokens.size(); offset += 8) {
            size_t valid = std::min<size_t>(8, tokens.size() - offset);
            std::array<uint32_t,8> batch{}; batch.fill(248070);
            std::copy_n(tokens.begin()+offset, valid, batch.begin());
            auto sub = step_batch8_pipelined(batch, valid, false);
            result.recurrent_seconds += sub.recurrent_seconds;
            result.attention_seconds += sub.attention_seconds;
            result.ffn_seconds += sub.ffn_seconds;
            result.output_seconds += sub.output_seconds;
            result.seconds += sub.seconds;
            if (offset + valid == tokens.size()) {
                result.next_tokens = sub.next_tokens;
                result.top_logits = sub.top_logits;
            }
        }
        return result;
    }
private:
    void synchronize_host_state() {
        if (!pipeline_state_current_) return;
        for (std::size_t layer = 0; layer < attention_.size(); ++layer) {
            auto &cache = attention_[layer];
            const std::size_t values = active_tokens_ * kKvWidth;
            cache.keys.resize(values);
            cache.values.resize(values);
            qwen_native::attention_download_history(attn_device_, static_cast<int>(layer),
                                                     cache.keys.data(), cache.values.data(), active_tokens_);
        }
        for (std::size_t layer = 0; layer < recurrent_.size(); ++layer) {
            auto &state = recurrent_[layer];
            qwen_native::recurrent_download_convolution_state(
                recurrent_device_, static_cast<int>(layer), state.convolution.data(), state.convolution.size());
            state.host_matrix_current = false;
            state.device_matrix_current = true;
        }
        pipeline_state_current_ = false;
    }

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
        qwen_native::execute_k_matvec_cached(tensor_data(mapping_, model_, item), item.type,
                                             input.data(), input.size(), output.data(), output.size(), keep_resident);
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
        if (name == "output.weight" && std::getenv("QWEN_NATIVE_HYBRID_HEAD")) {
            qwen_native::execute_k_matvec_hybrid_cached_batch8(
                tensor_data(mapping_, model_, item), item.type, input.data(),
                static_cast<std::size_t>(item.dimensions[0]), output.data(), outputs, false);
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
            // GGUF conversion tiles V heads as [replica][K head], so the
            // matching repeated Q/K head is selected modulo the K-head count.
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
            const float *gate = q_and_gate.data()
                + head * 2 * kAttentionHeadSize + kAttentionHeadSize;
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
    bool pipeline_state_current_ = false;
    std::vector<RecurrentState> recurrent_;
    std::vector<AttentionCache> attention_;
    AttentionCache mtp_attention_cache_;
    std::vector<float> trunk_hidden_;
    std::vector<float> batch_trunk_hidden_;
    std::array<std::vector<float>, 5> dflash_features_;
    // Each entry is eight contiguous target-layer snapshots, one per input
    // token in the last pipelined verification/replay batch.
    std::array<std::vector<float>, 5> batch_dflash_features_;
    qwen_native::DevicePipeline *pipeline_ = nullptr;
    qwen_native::RecurrentDeviceState *recurrent_device_ = nullptr;
    qwen_native::AttentionDeviceState *attn_device_ = nullptr;
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
        float rms_pre = 0;
        for (float v : encoded) rms_pre += v * v;
        rms_pre = std::sqrt(rms_pre / encoded.size());
        const auto &norm = require_tensor(model_, "enc.output_norm.weight");
        auto result = rms_norm(encoded, static_cast<const float *>(tensor_data(mapping_, model_, norm)), 5120, 1.0e-6F);
        float rms_post = 0;
        for (float v : result) rms_post += v * v;
        rms_post = std::sqrt(rms_post / result.size());
        static int encode_debug_count = 0;
        if (std::getenv("QWEN_NATIVE_TRACE") && encode_debug_count < 1) {
            std::cerr << "  [encode: features rms=[";
            for (int fi = 0; fi < 5; ++fi) {
                float fr = 0;
                for (float v : features[fi]) fr += v * v;
                std::cerr << std::sqrt(fr / features[fi].size());
                if (fi < 4) std::cerr << ", ";
            }
            std::cerr << "] pre=" << rms_pre << " post=" << rms_post << "]\n";
            ++encode_debug_count;
        }
        return result;
    }

    void preload_all_quantized() const {
        // N23 is an optional draft-only experiment. The default preloads the
        // official packed matrices unchanged into the shared exact cache.
        if (std::getenv("QWEN_NATIVE_DRAFT_N23")) {
            for (const auto &tensor : model_.tensors()) {
                if (tensor.type != 12) continue;
                qwen_native::preload_n23q4_matrix(
                    tensor_data(mapping_, model_, tensor),
                    static_cast<std::size_t>(tensor.dimensions.at(0)),
                    static_cast<std::size_t>(tensor.dimensions.at(1)));
            }
            return;
        }
        std::size_t required_bytes = 0;
        for (const auto &item : model_.tensors()) {
            if (item.dimensions.size() != 2 ||
                (item.type != 12 && item.type != 13 && item.type != 14)) continue;
            const std::size_t block_bytes = item.type == 12 ? 144 : item.type == 13 ? 176 : 210;
            required_bytes += static_cast<std::size_t>(item.dimensions[0]) / 256
                            * static_cast<std::size_t>(item.dimensions[1]) * block_bytes;
        }
        const auto existing = qwen_native::resident_weight_cache_stats();
        qwen_native::set_resident_weight_cache_budget(existing.budget_bytes + required_bytes);
        for (const auto &item : model_.tensors()) {
            if (item.dimensions.size() != 2 ||
                (item.type != 12 && item.type != 13 && item.type != 14)) continue;
            if (!qwen_native::preload_resident_k_matrix(
                    tensor_data(mapping_, model_, item), item.type,
                    static_cast<std::size_t>(item.dimensions[0]),
                    static_cast<std::size_t>(item.dimensions[1]))) {
                throw std::runtime_error("VRAM is insufficient for exact DFlash residency");
            }
        }
    }

    [[nodiscard]] std::size_t injected_token_count() const {
        constexpr std::size_t kv_width = 8 * 128;
        const std::size_t count = injected_.front().keys.size() / kv_width;
        for (const auto &cache : injected_) {
            if (cache.keys.size() != cache.values.size() || cache.keys.size() / kv_width != count) {
                throw std::runtime_error("DFlash injected KV layers are out of sync");
            }
        }
        return count;
    }

    void reset() {
        for (auto &cache : injected_) {
            cache.keys.clear();
            cache.values.clear();
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

    void inject_batch(
        const std::array<std::array<std::vector<float>, 5>, 8> &features,
        std::size_t valid_count, std::size_t first_position) {
        if (valid_count == 0 || valid_count > 8) {
            throw std::invalid_argument("DFlash injection batch must contain [1, 8] items");
        }
        constexpr std::size_t feature_width = 5120;
        constexpr std::size_t fused_width = 5 * feature_width;
        std::vector<float> fused(8 * fused_width);
        for (std::size_t batch = 0; batch < 8; ++batch) {
            const std::size_t source_batch = std::min(batch, valid_count - 1);
            for (std::size_t feature = 0; feature < 5; ++feature) {
                if (features[source_batch][feature].size() != feature_width) {
                    throw std::runtime_error("target did not capture a complete DFlash injection batch");
                }
                std::copy(features[source_batch][feature].begin(),
                          features[source_batch][feature].end(),
                          fused.begin() + batch * fused_width + feature * feature_width);
            }
        }
        const auto &fc = tensor("fc.weight");
        std::vector<float> encoded(8 * feature_width);
        qwen_native::execute_k_matvec_cached_batch8(
            tensor_data(mapping_, model_, fc), fc.type, fused.data(), fused_width,
            encoded.data(), feature_width, true);
        encoded = rms_norm(encoded, f32_vector("enc.output_norm.weight", feature_width),
                           feature_width, 1.0e-6F);

        constexpr std::size_t kv_width = 8 * 128;
        for (int layer = 0; layer < 5; ++layer) {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            const auto &k_tensor = tensor(prefix + "attn_k.weight");
            const auto &v_tensor = tensor(prefix + "attn_v.weight");
            std::vector<float> keys(8 * kv_width);
            std::vector<float> values(8 * kv_width);
            qwen_native::execute_k_matvec_cached_batch8(
                tensor_data(mapping_, model_, k_tensor), k_tensor.type,
                encoded.data(), feature_width, keys.data(), kv_width, true);
            qwen_native::execute_k_matvec_cached_batch8(
                tensor_data(mapping_, model_, v_tensor), v_tensor.type,
                encoded.data(), feature_width, values.data(), kv_width, true);
            keys = rms_norm(keys, f32_vector(prefix + "attn_k_norm.weight", 128),
                            128, 1.0e-6F);
            for (std::size_t batch = 0; batch < valid_count; ++batch) {
                float *key_ptr = keys.data() + batch * kv_width;
                const float *val_ptr = values.data() + batch * kv_width;
                apply_rope(key_ptr, 8, first_position + batch);
                append_injected(layer, key_ptr, val_ptr, kv_width);
            }
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
        const auto layers_started = std::chrono::steady_clock::now();
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
                apply_rope(queries.data() + item * heads * head_size, heads, position + item);
                apply_rope(keys.data() + item * kv_width, kv_heads, position + item);
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
        const auto layers_done = std::chrono::steady_clock::now();
        hidden = rms_norm(hidden, f32_vector("output_norm.weight", 5120), 5120, 1.0e-6F);
        const bool use_gpu_topk = !std::getenv("QWEN_NATIVE_GPU_TOPK") || std::string(std::getenv("QWEN_NATIVE_GPU_TOPK")) != "0";
        std::array<std::uint32_t, 7> result{};
        auto logits_done = layers_done;
        std::vector<float> fallback_logits;
        int top_indices[8 * 16];
        float top_scores[8 * 16];
        if (use_gpu_topk) {
            target.dflash_output_top16_batch8(hidden, top_indices, top_scores);
            logits_done = std::chrono::steady_clock::now();
            if (std::getenv("QWEN_NATIVE_TRACE")) {
                std::cerr << "  [dflash top-1 per pos:";
                for (int pos = 0; pos < 8; ++pos) std::cerr << ' ' << top_indices[pos * 16];
                std::cerr << "]\n";
            }
        } else {
            fallback_logits = target.dflash_output_logits_batch8(hidden);
            logits_done = std::chrono::steady_clock::now();
            const std::size_t vocab = fallback_logits.size() / 8;
            if (std::getenv("QWEN_NATIVE_TRACE")) {
                std::cerr << "  [dflash top-1 per pos:";
                for (int pos = 0; pos < 8; ++pos) {
                    const float *begin = fallback_logits.data() + pos * vocab;
                    const auto best = std::max_element(begin, begin + vocab);
                    std::cerr << ' ' << std::distance(begin, best);
                }
                std::cerr << "]\n";
            }
        }
        const auto selector_hidden = projection_batch8("selector_hidden.weight", hidden);
        const auto sel_done = std::chrono::steady_clock::now();
        if (use_gpu_topk) {
            result = select_from_topk(anchor, top_indices, top_scores, selector_hidden);
        } else {
            result = select(anchor, fallback_logits, selector_hidden);
        }
        const auto end_done = std::chrono::steady_clock::now();
        if (std::getenv("QWEN_NATIVE_DFLASH_DEBUG")) {
            std::cerr << "    [dflash breakdown: layers="
                      << std::chrono::duration<double, std::milli>(layers_done - layers_started).count()
                      << "ms, logits="
                      << std::chrono::duration<double, std::milli>(logits_done - layers_done).count()
                      << "ms, sel_hidden="
                      << std::chrono::duration<double, std::milli>(sel_done - logits_done).count()
                      << "ms, select="
                      << std::chrono::duration<double, std::milli>(end_done - sel_done).count()
                      << "ms]\n";
        }
        return result;
    }

    [[nodiscard]] int last_candidate_rank(std::size_t position,
                                          std::uint32_t token) const {
        if (position >= 7) return -1;
        const auto &items = last_candidates_.at(position);
        const auto found = std::find(items.begin(), items.end(), token);
        return found == items.end() ? -1 : static_cast<int>(std::distance(items.begin(), found));
    }

private:
    struct InjectedCache {
        std::vector<std::uint16_t> keys;
        std::vector<std::uint16_t> values;
    };

    const qwen_native::TensorInfo &tensor(const std::string &name) const {
        return require_tensor(model_, name);
    }

    static bool has_n23q4(const void *weights) {
        return qwen_native::has_n23q4_matrix(weights);
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
        if (item.type == 12 && has_n23q4(tensor_data(mapping_, model_, item))) {
            qwen_native::execute_n23q4_batch8_cached(
                tensor_data(mapping_, model_, item), input.data(),
                static_cast<std::size_t>(item.dimensions[0]), outputs, output.data());
            return output;
        }
        qwen_native::execute_k_matvec_cached_batch8(tensor_data(mapping_, model_, item), item.type,
                                                     input.data(), static_cast<std::size_t>(item.dimensions[0]),
                                                     output.data(), outputs, false);
        return output;
    }

    static void apply_rope(float *values, std::size_t heads, std::size_t position) {
        // The DFlash checkpoint uses Qwen3 attention with a full 128-dimension
        // rotary head. Unlike the Qwen3.8 target's partial 64/256 RoPE, no
        // trailing half of a draft head is left unrotated.
        constexpr std::size_t rotated = 128;
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
            float *item = values + head * 128;
            for (std::size_t pair = 0; pair < pairs; ++pair) {
                const float left = item[pair];
                const float right = item[pair + pairs];
                item[pair] = left * cosine[pair] - right * sine[pair];
                item[pair + pairs] = left * sine[pair] + right * cosine[pair];
            }
        }
    }
    static void apply_rope(std::vector<float> &values, std::size_t heads, std::size_t position) {
        apply_rope(values.data(), heads, position);
    }

    static std::uint16_t to_fp16(float value) {
        const __half half = __float2half(value);
        std::uint16_t bits = 0;
        std::memcpy(&bits, &half, sizeof(bits));
        return bits;
    }

    void append_injected(int layer, const float *keys, const float *values, std::size_t count) {
        auto &cache = injected_.at(layer);
        constexpr std::size_t kv_width = 8 * 128;
        constexpr std::size_t window = 2048;
        if (cache.keys.size() / kv_width == window) {
            cache.keys.erase(cache.keys.begin(), cache.keys.begin() + kv_width);
            cache.values.erase(cache.values.begin(), cache.values.begin() + kv_width);
        }
        const std::size_t old_size = cache.keys.size();
        cache.keys.resize(old_size + count);
        cache.values.resize(old_size + count);
        for (std::size_t i = 0; i < count; ++i) {
            cache.keys[old_size + i] = to_fp16(keys[i]);
            cache.values[old_size + i] = to_fp16(values[i]);
        }
    }
    void append_injected(int layer, const std::vector<float> &keys, const std::vector<float> &values) {
        append_injected(layer, keys.data(), values.data(), keys.size());
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
        qwen_native::compute_dflash_noncausal_attention(
            queries.data(), keys.data(), values.data(),
            cache.keys.empty() ? nullptr : cache.keys.data(),
            cache.values.empty() ? nullptr : cache.values.data(),
            cached, output.data());
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
        struct CandidateItem {
            float score;
            std::uint32_t token;
        };
        auto heap_cmp = [](const CandidateItem &a, const CandidateItem &b) {
            return a.score > b.score; // min-heap: smallest at root
        };

        for (std::size_t pos = 1; pos < block; ++pos) {
            const float *row = logits.data() + pos * vocabulary;
            std::array<CandidateItem, top_k> heap{};
            for (std::size_t i = 0; i < top_k; ++i) {
                heap[i] = {row[i], static_cast<std::uint32_t>(i)};
            }
            std::make_heap(heap.begin(), heap.end(), heap_cmp);

            float min_score = heap[0].score;
            for (std::size_t i = top_k; i < vocabulary; ++i) {
                const float val = row[i];
                if (val > min_score) {
                    std::pop_heap(heap.begin(), heap.end(), heap_cmp);
                    heap.back() = {val, static_cast<std::uint32_t>(i)};
                    std::push_heap(heap.begin(), heap.end(), heap_cmp);
                    min_score = heap[0].score;
                }
            }
            std::sort_heap(heap.begin(), heap.end(), heap_cmp);
            for (std::size_t item = 0; item < top_k; ++item) {
                candidates[pos][item] = heap[item].token;
                last_candidates_[pos - 1][item] = heap[item].token;
                unary[pos][item] = heap[item].score;
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
            static const float selector_scale = [] {
                if (const char *env = std::getenv("QWEN_NATIVE_SELECTOR_WEIGHT")) {
                    return std::stof(env);
                }
                return 1.0F;
            }();
            std::array<float, top_k> scores{};
            for (std::size_t current = 0; current < top_k; ++current) {
                float transition = 0.0F;
                const auto &previous = predecessors[predecessor];
                for (std::size_t rank = 0; rank < 256; ++rank) {
                    transition = std::fma(successors[current][rank], previous[rank] * hidden[rank], transition);
                }
                scores[current] = unary[pos][current] + selector_scale * transition;
            }
            if (std::getenv("QWEN_NATIVE_DFLASH_DEBUG") && pos == 1) {
                std::cerr << "    [dflash pos1 debug: ";
                for (std::size_t c = 0; c < 3; ++c) {
                    std::cerr << "c" << c << " tok=" << candidates[pos][c]
                              << " u=" << unary[pos][c] << " s=" << scores[c] << " | ";
                }
                std::cerr << "]\n";
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

    std::array<std::uint32_t, 7> select_from_topk(std::uint32_t anchor,
                                                  const int *top_indices,
                                                  const float *top_scores,
                                                  const std::vector<float> &selector_hidden) const {
        constexpr std::size_t block = 8;
        constexpr std::size_t top_k = 16;
        std::array<std::array<std::uint32_t, top_k>, 8> candidates{};
        std::array<std::array<float, top_k>, 8> unary{};
        for (std::size_t pos = 1; pos < block; ++pos) {
            for (std::size_t k = 0; k < top_k; ++k) {
                const int idx = top_indices[pos * top_k + k];
                const float score = top_scores[pos * top_k + k];
                candidates[pos][k] = static_cast<std::uint32_t>(idx < 0 ? 0 : idx);
                last_candidates_[pos - 1][k] = candidates[pos][k];
                unary[pos][k] = score;
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
            static const float selector_scale = [] {
                if (const char *env = std::getenv("QWEN_NATIVE_SELECTOR_WEIGHT")) {
                    return std::stof(env);
                }
                return 1.0F;
            }();
            std::array<float, top_k> scores{};
            for (std::size_t current = 0; current < top_k; ++current) {
                float transition = 0.0F;
                const auto &previous = predecessors[predecessor];
                for (std::size_t rank = 0; rank < 256; ++rank) {
                    transition = std::fma(successors[current][rank], previous[rank] * hidden[rank], transition);
                }
                scores[current] = unary[pos][current] + selector_scale * transition;
            }
            if (std::getenv("QWEN_NATIVE_DFLASH_DEBUG") && pos == 1) {
                std::cerr << "    [dflash pos1 debug: ";
                for (std::size_t c = 0; c < 3; ++c) {
                    std::cerr << "c" << c << " tok=" << candidates[pos][c]
                              << " u=" << unary[pos][c] << " s=" << scores[c] << " | ";
                }
                std::cerr << "]\n";
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
    mutable std::array<std::array<std::uint32_t, 16>, 7> last_candidates_{};
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

void target_long_window_bench(const std::string &path, std::size_t context_capacity,
                              std::uint32_t token, std::size_t resident_weight_budget) {
    NativeTargetSession session(path, context_capacity, resident_weight_budget);
    session.simulate_full_attention_window();
    std::array<std::uint32_t, 8> candidate{};
    candidate.fill(248070);
    candidate[0] = token;
    std::cout << "native synthetic full-window target timing\n";
    for (int iteration = 0; iteration < 3; ++iteration) {
        const auto result = session.step_batch8_pipelined(candidate);
        std::cout << "  pass " << iteration + 1 << ": "
                  << std::fixed << std::setprecision(3) << result.seconds << " s"
                  << " (recurrent " << result.recurrent_seconds
                  << ", attention " << result.attention_seconds
                  << ", FFN " << result.ffn_seconds
                  << ", head " << result.output_seconds << ")\n";
        if (iteration != 2) session.rewind(8);
    }
    std::cout << "  active context: " << session.active_tokens() << '/' << context_capacity << "\n"
              << "  note: zero-filled KV timing only; not a semantic 128K fixture\n";
}

void target_compare_batch(const std::string &path, std::size_t context_capacity,
                          std::uint32_t first_token, std::size_t resident_weight_budget) {
    std::array<std::uint32_t, 8> candidate{};
    candidate.fill(248070);
    candidate[0] = first_token;
    // The authoritative target-only oracle uses the same declared Q8 cache
    // representation and executes one committed token at a time. This avoids
    // comparing the Q8 long-context engine to the separate FP16 host fixture.
    std::array<std::uint32_t, 8> reference_tokens{};
    {
        NativeTargetSession reference(path, context_capacity, resident_weight_budget);
        for (std::size_t index = 0; index < candidate.size(); ++index) {
            std::array<std::uint32_t, 8> scalar_input{};
            scalar_input.fill(248070);
            scalar_input[0] = candidate[index];
            reference_tokens[index] = reference.step_batch8_pipelined(scalar_input, 1).next_tokens[0];
        }
    }
    std::array<std::uint32_t, 8> pipeline_tokens{};
    {
        NativeTargetSession pipeline(path, context_capacity, resident_weight_budget);
        pipeline_tokens = pipeline.step_batch8_pipelined(candidate).next_tokens;
    }
    bool matches = true;
    std::cout << "native target batch verifier comparison\n  reference:";
    for (const auto token : reference_tokens) std::cout << ' ' << token;
    std::cout << "\n  pipeline: ";
    for (const auto token : pipeline_tokens) std::cout << ' ' << token;
    std::cout << '\n';
    for (std::size_t index = 0; index < reference_tokens.size(); ++index) {
        matches = matches && reference_tokens[index] == pipeline_tokens[index];
    }
    std::cout << "  result:   " << (matches ? "match" : "MISMATCH") << '\n';
    if (!matches) {
        throw std::runtime_error("pipelined target verifier diverges from native reference batch");
    }
}

void target_rollback_test(const std::string &path, std::size_t context_capacity,
                          std::uint32_t first_token, std::size_t resident_weight_budget) {
    std::array<std::uint32_t, 8> candidate{};
    candidate.fill(248070);
    candidate[0] = first_token;
    std::array<std::uint32_t, 8> follow_up{};
    follow_up.fill(248070);
    std::array<std::uint32_t, 8> expected{};
    {
        NativeTargetSession reference(path, context_capacity, resident_weight_budget);
        (void)reference.step_batch8_pipelined(candidate, 1);
        expected = reference.step_batch8_pipelined(follow_up).next_tokens;
    }
    std::array<std::uint32_t, 8> actual{};
    {
        NativeTargetSession transaction(path, context_capacity, resident_weight_budget);
        const auto checkpoint = transaction.begin_speculative_checkpoint();
        const auto speculative = transaction.step_batch8_pipelined(candidate);
        transaction.commit_speculative_prefix(checkpoint, 1);
        if (transaction.active_tokens() != checkpoint.active_tokens + 1) {
            throw std::runtime_error("captured prefix commit advanced the wrong target length");
        }
        actual = transaction.step_batch8_pipelined(follow_up).next_tokens;
        (void)speculative;
    }
    const bool matches = actual == expected;
    std::cout << "native speculative rollback test\n  expected:";
    for (const auto token : expected) std::cout << ' ' << token;
    std::cout << "\n  replayed: ";
    for (const auto token : actual) std::cout << ' ' << token;
    std::cout << "\n  result:   " << (matches ? "match" : "MISMATCH") << '\n';
    if (!matches) {
        throw std::runtime_error("rejected speculative suffix changed subsequent target state");
    }
}

void dflash_commit_test(const std::string &target_path, const std::string &dflash_path,
                        std::size_t context_capacity, std::uint32_t first_token,
                        std::size_t resident_weight_budget) {
    NativeTargetSession target(target_path, context_capacity, resident_weight_budget);
    NativeDflashEncoder dflash(dflash_path);
    std::uint32_t anchor_token = 0;
    if (context_capacity > NativeTargetSession::pipeline_attention_capacity()) {
        std::array<std::uint32_t, 8> prefill{};
        prefill.fill(248070);
        prefill[0] = first_token;
        const auto prefill_result = target.step_batch8_pipelined(prefill, 1);
        anchor_token = prefill_result.next_tokens[0];
        dflash.inject(target.batch_dflash_features(0), 0);
    } else {
        const auto anchor = target.step(first_token);
        anchor_token = anchor.next_token;
        dflash.inject(target.dflash_features(), target.active_tokens() - 1);
    }
    const std::size_t before = target.active_tokens();
    if (dflash.injected_token_count() != before) {
        throw std::runtime_error("DFlash prefill injection did not match target position");
    }

    std::array<std::uint32_t, 8> candidate{};
    candidate.fill(248070);
    candidate[0] = anchor_token;
    const auto checkpoint = target.begin_speculative_checkpoint();
    (void)target.step_batch8_pipelined(candidate);
    target.restore_speculative_checkpoint(checkpoint);
    (void)target.step_batch8_pipelined(candidate, 2);
    for (std::size_t index = 0; index < 2; ++index) {
        dflash.inject(target.batch_dflash_features(index), checkpoint.active_tokens + index);
    }
    const std::size_t expected = std::min<std::size_t>(target.active_tokens(), 2048);
    const bool match = dflash.injected_token_count() == expected && target.active_tokens() == before + 2;
    std::cout << "native DFlash committed-feature test\n"
              << "  target positions: " << before << " -> " << target.active_tokens() << '\n'
              << "  DFlash cache:     " << dflash.injected_token_count() << '/' << expected << '\n'
              << "  result:           " << (match ? "match" : "MISMATCH") << '\n';
    if (!match) {
        throw std::runtime_error("DFlash injected rejected or omitted target-token features");
    }
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
    std::cout << "\n  target says:      ";
    for (std::size_t item = 0; item < 8; ++item) std::cout << verification.next_tokens[item] << ' ';
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
              << "  proposals:        ";
    for (const auto proposed : proposals) std::cout << proposed << ' ';
    std::cout << "\n  target says:      ";
    for (const auto predicted : verification.next_tokens) std::cout << predicted << ' ';
    std::cout << "\n"
              << "  accepted prefix:  " << accepted << '/' << proposals.size() << "\n"
              << "  prefill time:     " << std::fixed << std::setprecision(3) << prefill_seconds << " s\n"
              << "  block timing:     draft " << draft_seconds << " s, verifier " << verification.seconds << " s\n"
              << "  accepted rate:    " << std::setprecision(3) << accepted / steady_block_seconds
              << " tok/s (post-prefill single block)\n";
}

void speculative_equivalence_test(const std::string &target_path,
                                  const std::string &dflash_path,
                                  std::size_t context_capacity,
                                  const std::vector<std::uint32_t> &prompt,
                                  std::size_t generated_count,
                                  std::size_t resident_weight_budget) {
    if (generated_count == 0) {
        throw std::invalid_argument("speculative equivalence count must be nonzero");
    }
    std::vector<std::uint32_t> oracle_tokens;
    oracle_tokens.reserve(generated_count);
    {
        NativeTargetSession oracle(target_path, context_capacity, resident_weight_budget);
        std::uint32_t anchor = 0;
        for (std::size_t begin = 0; begin < prompt.size(); begin += 8) {
            const std::size_t valid = std::min<std::size_t>(8, prompt.size() - begin);
            std::array<std::uint32_t, 8> batch{};
            batch.fill(248070);
            std::copy_n(prompt.begin() + begin, valid, batch.begin());
            anchor = oracle.step_batch8_pipelined(batch, valid).next_tokens[valid - 1];
        }
        while (oracle_tokens.size() < generated_count) {
            oracle_tokens.push_back(anchor);
            std::array<std::uint32_t, 8> input{};
            input.fill(248070);
            input[0] = anchor;
            anchor = oracle.step_batch8_pipelined(input, 1).next_tokens[0];
        }
    }
    // The cache is process-global and keyed by a session's mmap addresses.
    // Release the oracle's entries before opening a second 4.25-GiB Q8 cache.
    qwen_native::clear_resident_weight_cache();

    std::vector<std::uint32_t> speculative_tokens;
    speculative_tokens.reserve(generated_count);
    std::size_t cycles = 0;
    std::size_t accepted_total = 0;
    double speculative_elapsed_seconds = 0.0;
    {
        NativeTargetSession target(target_path, context_capacity, resident_weight_budget);
        NativeDflashEncoder dflash(dflash_path);
        dflash.preload_all_quantized();
        std::uint32_t anchor = 0;
        for (std::size_t begin = 0; begin < prompt.size(); begin += 8) {
            const std::size_t valid = std::min<std::size_t>(8, prompt.size() - begin);
            std::array<std::uint32_t, 8> batch{};
            batch.fill(248070);
            std::copy_n(prompt.begin() + begin, valid, batch.begin());
            const std::size_t position = target.active_tokens();
            const auto result = target.step_batch8_pipelined(batch, valid);
            for (std::size_t index = 0; index < valid; ++index) {
                dflash.inject(target.batch_dflash_features(index), position + index);
            }
            anchor = result.next_tokens[valid - 1];
        }
        const auto speculative_started = std::chrono::steady_clock::now();
        while (speculative_tokens.size() < generated_count) {
            const auto proposals = dflash.propose(target, anchor, target.active_tokens());
            std::array<std::uint32_t, 8> candidate{};
            candidate[0] = anchor;
            std::copy(proposals.begin(), proposals.end(), candidate.begin() + 1);
            const auto checkpoint = target.begin_speculative_checkpoint();
            const auto verification = target.step_batch8_pipelined(candidate);
            std::size_t accepted = 0;
            while (accepted < proposals.size() &&
                   verification.next_tokens[accepted] == proposals[accepted]) {
                ++accepted;
            }
            if (accepted != proposals.size()) {
                target.commit_speculative_prefix(checkpoint, accepted + 1);
            }
            std::array<std::array<std::vector<float>, 5>, 8> injection_features{};
            for (std::size_t index = 0; index <= accepted; ++index) {
                injection_features[index] = target.batch_dflash_features(index);
            }
            dflash.inject_batch(injection_features, accepted + 1, checkpoint.active_tokens);
            speculative_tokens.push_back(anchor);
            for (std::size_t index = 0;
                 index < accepted && speculative_tokens.size() < generated_count; ++index) {
                speculative_tokens.push_back(proposals[index]);
            }
            anchor = verification.next_tokens[accepted];
            accepted_total += accepted;
            ++cycles;
        }
        const double speculative_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - speculative_started).count();
        speculative_elapsed_seconds = speculative_seconds;
    }

    const bool matches = oracle_tokens == speculative_tokens;
    std::cout << "native target-only/speculative greedy equivalence\n"
              << "  prompt tokens:    " << prompt.size() << "\n"
              << "  generated tokens: " << generated_count << "\n"
              << "  cycles:           " << cycles << "\n"
              << "  accepted draft:   " << accepted_total << '/' << cycles * 7 << "\n"
              << "  speculative rate: " << std::fixed << std::setprecision(2)
              << generated_count / speculative_elapsed_seconds << " tok/s ("
              << speculative_elapsed_seconds << " s)\n"
              << "  result:           " << (matches ? "match" : "MISMATCH") << '\n';
    if (!matches) {
        const auto mismatch = std::mismatch(oracle_tokens.begin(), oracle_tokens.end(),
                                            speculative_tokens.begin());
        std::cout << "  first mismatch:   " << std::distance(oracle_tokens.begin(), mismatch.first)
                  << " (target " << *mismatch.first << ", speculative " << *mismatch.second << ")\n";
        throw std::runtime_error("speculative decode changed deterministic target output");
    }
}

std::size_t quant_matrix_bytes(std::uint32_t type, std::size_t input_size,
                               std::size_t output_size) {
    std::size_t block_values = 0;
    std::size_t block_bytes = 0;
    switch (type) {
    case 8: block_values = 32; block_bytes = 34; break;
    case 11: case 21: block_values = 256; block_bytes = 110; break;
    case 12: block_values = 256; block_bytes = 144; break;
    case 13: block_values = 256; block_bytes = 176; break;
    case 14: block_values = 256; block_bytes = 210; break;
    case 20: block_values = 32; block_bytes = 18; break;
    case 23: block_values = 256; block_bytes = 136; break;
    default: throw std::invalid_argument("unsupported exact CPU benchmark format");
    }
    if (input_size % block_values != 0) {
        throw std::invalid_argument("tensor input width is not quant-block aligned");
    }
    return input_size / block_values * block_bytes * output_size;
}

void benchmark_cpu_streamed_batch8(const void *weights, std::uint32_t type,
                                   std::size_t input_size, std::size_t output_size,
                                   int iterations) {
    const std::size_t weight_bytes = quant_matrix_bytes(type, input_size, output_size);
    std::vector<float> input(8 * input_size);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::sin(static_cast<float>(index + 1) * 0.00137F);
    }
    std::vector<float> gpu_output(8 * output_size);
    std::vector<float> cpu_output(8 * output_size);
    qwen_native::execute_k_matvec_cached_batch8(weights, type, input.data(), input_size,
                                                gpu_output.data(), output_size, false);
    const auto gpu_started = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        qwen_native::execute_k_matvec_cached_batch8(weights, type, input.data(), input_size,
                                                    gpu_output.data(), output_size, false);
    }
    const double gpu_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - gpu_started).count() / iterations;
    std::cout << "  streamed GPU (H2D + kernel):\n"
              << "    latency:    " << std::fixed << std::setprecision(3)
              << gpu_seconds * 1000.0 << " ms\n"
              << "    weight BW:  " << std::setprecision(2)
              << static_cast<double>(weight_bytes) / gpu_seconds / 1.0e9 << " GB/s\n";
    try {
        const auto mapped = qwen_native::benchmark_host_mapped_batch8(
            weights, type, input_size, output_size, iterations);
        std::cout << "  mapped-host GPU direct:\n"
                  << "    latency:    " << std::fixed << std::setprecision(3)
                  << mapped.milliseconds << " ms\n"
                  << "    weight BW:  " << std::setprecision(2)
                  << mapped.bandwidth_gbps << " GB/s\n";
    } catch (const std::exception &error) {
        std::cout << "  mapped-host GPU direct: unavailable (" << error.what() << ")\n";
    }

    double best_seconds = std::numeric_limits<double>::infinity();
    int best_threads = 0;
    for (const int threads : {6, 12, 14, 20, 24}) {
        qwen_native::cpu_quant_matvec_batch8_rows(weights, type, input.data(), input_size,
                                                  cpu_output.data(), output_size, threads);
        const auto started = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            qwen_native::cpu_quant_matvec_batch8_rows(weights, type, input.data(), input_size,
                                                      cpu_output.data(), output_size, threads);
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count() / iterations;
        std::cout << "  exact CPU GEMM8, " << threads << " workers:\n"
                  << "    latency:    " << std::fixed << std::setprecision(3)
                  << seconds * 1000.0 << " ms\n"
                  << "    weight BW:  " << std::setprecision(2)
                  << static_cast<double>(weight_bytes) / seconds / 1.0e9 << " GB/s\n";
        if (seconds < best_seconds) {
            best_seconds = seconds;
            best_threads = threads;
        }
    }
    qwen_native::cpu_quant_matvec_batch8_rows(weights, type, input.data(), input_size,
                                              cpu_output.data(), output_size, best_threads);
    float max_error = 0.0F;
    for (std::size_t index = 0; index < cpu_output.size(); ++index) {
        max_error = std::max(max_error, std::abs(cpu_output[index] - gpu_output[index]));
    }
    std::cout << "  best exact CPU: " << best_threads << " workers, "
              << std::fixed << std::setprecision(2)
              << static_cast<double>(weight_bytes) / best_seconds / 1.0e9 << " GB/s\n"
              << "  CPU/GPU max delta: " << std::scientific << max_error << '\n';
    qwen_native::CpuMatvecBenchmarkResult soa;
    const char *soa_name = nullptr;
    if (type == 12) {
        soa = qwen_native::benchmark_cpu_q4k_soa_batch8(
            weights, input.data(), input_size, output_size, iterations,
            best_threads, gpu_output.data());
        soa_name = "Q4_K";
    } else if (type == 13) {
        soa = qwen_native::benchmark_cpu_q5k_soa_batch8(
            weights, input.data(), input_size, output_size, iterations,
            best_threads, gpu_output.data());
        soa_name = "Q5_K";
    } else if (type == 14) {
        soa = qwen_native::benchmark_cpu_q6k_soa_batch8(
            weights, input.data(), input_size, output_size, iterations,
            best_threads, gpu_output.data());
        soa_name = "Q6_K";
    } else if (type == 23) {
        soa = qwen_native::benchmark_cpu_iq4xs_soa_batch8(
            weights, input.data(), input_size, output_size, iterations,
            best_threads, gpu_output.data());
        soa_name = "IQ4_XS";
    }
    if (soa_name) {
        std::cout << "  exact " << soa_name << " SoA GEMM8:\n"
                  << "    latency:    " << std::fixed << std::setprecision(3)
                  << soa.milliseconds << " ms\n"
                  << "    weight BW:  " << std::setprecision(2)
                  << soa.bandwidth_gbps << " GB/s\n"
                  << "    pack once:  " << std::setprecision(1)
                  << soa.packing_milliseconds << " ms\n"
                  << "    max delta:  " << std::scientific
                  << soa.max_absolute_error << '\n';
    }
}

void bench_matvec(const std::string &path, const std::string &tensor_name, int iterations) {
    const qwen_native::GgufFile model(path);
    const auto *tensor = model.find_tensor(tensor_name);
    if (!tensor) {
        throw std::runtime_error("tensor not found: " + tensor_name);
    }
    const bool supported = tensor->type == 8 || tensor->type == 11 || tensor->type == 12 ||
                           tensor->type == 13 || tensor->type == 14 || tensor->type == 20 ||
                           tensor->type == 21 || tensor->type == 23;
    if (!supported || tensor->dimensions.size() != 2) {
        throw std::runtime_error("benchmark requires a supported rank-2 exact quant tensor");
    }
    const Mapping mapping(path);
    const auto input_size = static_cast<std::size_t>(tensor->dimensions[0]);
    const auto output_size = static_cast<std::size_t>(tensor->dimensions[1]);
    std::cout << qwen_native::ggml_type_name(tensor->type) << " real-weight benchmark: " << tensor_name << '\n'
              << "  shape: " << input_size << " x " << output_size << '\n';
    const void *weights = mapping.at(model.data_offset() + tensor->offset);
    if (tensor->type == 12 || tensor->type == 13 || tensor->type == 14) {
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
                  << "    max error:  " << std::scientific << result.max_absolute_error << '\n'
                  << "  eight-token resident verifier:\n"
                  << "    latency:    " << std::fixed << std::setprecision(3) << batch8.milliseconds << " ms\n"
                  << "    weight BW:  " << std::setprecision(2) << batch8.bandwidth_gbps << " GB/s\n"
                  << "    max error:  " << std::scientific << batch8.max_absolute_error << '\n';
    }
    benchmark_cpu_streamed_batch8(weights, tensor->type, input_size, output_size, iterations);
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

struct BenchmarkDecodeStats {
    std::size_t generated = 0;
    std::size_t cycles = 0;
    std::size_t proposed = 0;
    std::size_t accepted = 0;
    double seconds = 0.0;
    double draft_seconds = 0.0;
    double target_seconds = 0.0;
    double injection_seconds = 0.0;
    bool ended_early = false;
    std::vector<std::uint32_t> tokens;
};

std::uint64_t token_hash(const std::vector<std::uint32_t> &tokens) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto token : tokens) {
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= (token >> (8 * byte)) & 0xFFU;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

std::vector<std::uint32_t> make_128k_benchmark_prompt(Tokenizer &tokenizer) {
    constexpr std::size_t prompt_tokens = 131072;
    std::vector<std::uint32_t> prefix{
        tokenizer.im_start(), 8678, 198,
    };
    const auto system_text = tokenizer.encode_full(
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
        "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.");
    prefix.insert(prefix.end(), system_text.begin(), system_text.end());
    prefix.push_back(tokenizer.im_end());
    prefix.push_back(198);
    prefix.push_back(tokenizer.im_start());
    prefix.push_back(846);
    prefix.push_back(198);

    const std::vector<std::uint32_t> suffix{
        tokenizer.im_end(), 198, tokenizer.im_start(), 74455, 198,
    };
    if (prefix.size() + suffix.size() >= prompt_tokens) {
        throw std::runtime_error("128K benchmark template exceeds its token budget");
    }
    const std::size_t filler_needed = prompt_tokens - prefix.size() - suffix.size();
    const auto paragraph_tokens = tokenizer.encode_full(
        "This is deterministic long-context evidence. Preserve every numbered fact, compare distant details, "
        "and finish with a careful technical synthesis rather than a short answer.\n");
    if (paragraph_tokens.empty()) {
        throw std::runtime_error("128K benchmark filler did not tokenize");
    }
    std::vector<std::uint32_t> filler_tokens;
    filler_tokens.reserve(filler_needed);
    while (filler_tokens.size() < filler_needed) {
        const std::size_t count = std::min(paragraph_tokens.size(),
                                           filler_needed - filler_tokens.size());
        filler_tokens.insert(filler_tokens.end(), paragraph_tokens.begin(),
                             paragraph_tokens.begin() + count);
    }
    prefix.insert(prefix.end(), filler_tokens.begin(), filler_tokens.end());
    prefix.insert(prefix.end(), suffix.begin(), suffix.end());
    if (prefix.size() != prompt_tokens) {
        throw std::runtime_error("128K benchmark prompt construction is not exact");
    }
    return prefix;
}

BenchmarkDecodeStats benchmark_decode_tokens(NativeTargetSession &target,
                                              NativeDflashEncoder &dflash,
                                              Tokenizer &tokenizer,
                                              std::uint32_t &anchor,
                                              std::size_t requested_tokens) {
    BenchmarkDecodeStats stats;
    stats.tokens.reserve(requested_tokens);
    const auto started = std::chrono::steady_clock::now();
    while (stats.generated < requested_tokens) {
        if (anchor == tokenizer.im_end() || anchor == tokenizer.eos_token()) {
            stats.ended_early = true;
            break;
        }
        const std::size_t remaining = requested_tokens - stats.generated;
        const std::size_t proposal_count = std::min<std::size_t>(7, remaining - 1);
        std::array<std::uint32_t, 7> proposals{};
        if (proposal_count != 0) {
            const auto draft_started = std::chrono::steady_clock::now();
            proposals = dflash.propose(target, anchor, target.active_tokens());
            stats.draft_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - draft_started).count();
        }
        stats.proposed += proposal_count;

        std::array<std::uint32_t, 8> candidate{};
        candidate.fill(248070);
        candidate[0] = anchor;
        std::copy_n(proposals.begin(), proposal_count, candidate.begin() + 1);
        const auto checkpoint = target.begin_speculative_checkpoint();
        const auto target_started = std::chrono::steady_clock::now();
        const auto verification = target.step_batch8_pipelined(candidate, proposal_count + 1);
        std::size_t accepted = 0;
        while (accepted < proposal_count &&
               verification.next_tokens[accepted] == proposals[accepted]) {
            ++accepted;
        }
        if (accepted != proposal_count) {
            target.commit_speculative_prefix(checkpoint, accepted + 1);
        }
        stats.target_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - target_started).count();
        stats.accepted += accepted;

        const auto injection_started = std::chrono::steady_clock::now();
        std::array<std::array<std::vector<float>, 5>, 8> injection_features{};
        for (std::size_t index = 0; index <= accepted; ++index) {
            injection_features[index] = target.batch_dflash_features(index);
        }
        dflash.inject_batch(injection_features, accepted + 1, checkpoint.active_tokens);
        stats.injection_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - injection_started).count();
        if (dflash.injected_token_count() != std::min<std::size_t>(target.active_tokens(), 2048)) {
            throw std::runtime_error("128K benchmark DFlash cache lost target alignment");
        }

        stats.tokens.push_back(anchor);
        ++stats.generated;
        for (std::size_t index = 0; index < accepted; ++index) {
            stats.tokens.push_back(proposals[index]);
            ++stats.generated;
            if (proposals[index] == tokenizer.im_end() ||
                proposals[index] == tokenizer.eos_token()) {
                stats.ended_early = true;
                break;
            }
        }
        anchor = verification.next_tokens[accepted];
        ++stats.cycles;
        if (stats.ended_early) break;
    }
    stats.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return stats;
}

void benchmark_128k(const std::string &target_path, const std::string &dflash_path,
                    const std::string &output_path, std::size_t resident_weight_budget,
                    int runs) {
    constexpr std::size_t prompt_token_count = 131072;
    constexpr std::size_t warmup_token_count = 8;
    constexpr std::size_t output_token_count = 256;
    constexpr std::size_t context_capacity =
        prompt_token_count + warmup_token_count + output_token_count;
    if (runs < 1) throw std::invalid_argument("128K benchmark RUNS must be positive");

    const qwen_native::GgufFile model(target_path);
    Tokenizer tokenizer(model);
    const auto prompt = make_128k_benchmark_prompt(tokenizer);
    const std::uint64_t prompt_digest = token_hash(prompt);
    const std::filesystem::path records_path(output_path);
    if (!records_path.parent_path().empty()) {
        std::filesystem::create_directories(records_path.parent_path());
    }
    std::ofstream records(records_path, std::ios::trunc);
    if (!records) throw std::runtime_error("cannot create 128K benchmark record file");
    std::ofstream prompt_record(records_path.string() + ".prompt.tokens", std::ios::trunc);
    if (!prompt_record) throw std::runtime_error("cannot create 128K benchmark prompt record");
    for (std::size_t index = 0; index < prompt.size(); ++index) {
        if (index) prompt_record << ' ';
        prompt_record << prompt[index];
    }
    prompt_record << '\n';
    prompt_record.close();

    int device = 0;
    hipDeviceProp_t properties{};
    hip_check(hipGetDevice(&device), "128K benchmark get device");
    hip_check(hipGetDeviceProperties(&properties, device), "128K benchmark device properties");
    double initial_load = 0.0;
    {
        std::ifstream loadavg("/proc/loadavg");
        loadavg >> initial_load;
    }
    std::string cpu_model = "unknown";
    {
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpuinfo, line)) {
            constexpr const char *key = "model name\t: ";
            if (line.rfind(key, 0) == 0) {
                cpu_model = line.substr(std::strlen(key));
                break;
            }
        }
    }
    const auto environment = [](const char *name, const char *fallback) {
        const char *value = std::getenv(name);
        return std::string(value ? value : fallback);
    };
    records << "{\"metadata\":true"
            << ",\"device\":\"" << properties.name << "\""
            << ",\"arch\":\"" << properties.gcnArchName << "\""
            << ",\"cpu\":\"" << cpu_model << "\""
            << ",\"target\":\"" << target_path << "\""
            << ",\"target_bytes\":" << std::filesystem::file_size(target_path)
            << ",\"dflash\":\"" << dflash_path << "\""
            << ",\"dflash_bytes\":" << std::filesystem::file_size(dflash_path)
            << ",\"prompt_tokens\":" << prompt_token_count
            << ",\"prompt_fnv64\":\"0x" << std::hex << prompt_digest << std::dec << "\""
            << ",\"prompt_record\":\"" << records_path.string() << ".prompt.tokens\""
            << ",\"greedy\":true,\"streams\":1,\"draft_block\":8"
            << ",\"warmup_tokens\":" << warmup_token_count
            << ",\"output_tokens\":" << output_token_count
            << ",\"runs\":" << runs
            << ",\"resident_gib\":"
            << static_cast<double>(resident_weight_budget) / (1024.0 * 1024.0 * 1024.0)
            << ",\"QWEN_NATIVE_HYBRID\":\"" << environment("QWEN_NATIVE_HYBRID", "0") << "\""
            << ",\"QWEN_NATIVE_Q8_TILED\":\"" << environment("QWEN_NATIVE_Q8_TILED", "default") << "\""
            << ",\"QWEN_NATIVE_CPU_THREADS\":\"" << environment("QWEN_NATIVE_CPU_THREADS", "default") << "\""
            << ",\"QWEN_NATIVE_SOA_SPLIT\":\"" << environment("QWEN_NATIVE_SOA_SPLIT", "default") << "\""
            << ",\"initial_loadavg\":" << initial_load << "}\n";
    records.flush();
    std::cerr << "native deterministic 128K benchmark\n"
              << "  device: " << properties.name << " / " << properties.gcnArchName << '\n'
              << "  prompt: " << prompt.size() << " actually tokenized tokens, FNV64 0x"
              << std::hex << prompt_digest << std::dec << '\n'
              << "  decode: " << output_token_count
              << " committed greedy tokens, one stream, " << runs << " runs\n"
              << "  warmup: " << warmup_token_count << " generated tokens excluded\n"
              << "  initial loadavg: " << std::fixed << std::setprecision(2) << initial_load << '\n'
              << "  records: " << records_path.string() << "\n";
    if (initial_load > 2.0) {
        std::cerr << "  [warning: system load is above the clean-run threshold]\n";
    }

    std::vector<std::uint32_t> reference_tokens;
    std::vector<double> throughputs;
    for (int run = 1; run <= runs; ++run) {
        qwen_native::clear_resident_weight_cache();
        const auto initialization_started = std::chrono::steady_clock::now();
        NativeTargetSession target(target_path, context_capacity, resident_weight_budget);
        NativeDflashEncoder dflash(dflash_path);
        dflash.preload_all_quantized();
        const double initialization_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - initialization_started).count();

        const auto prefill_started = std::chrono::steady_clock::now();
        std::uint32_t anchor = 0;
        // Fast prefill: first 129k without DFlash, batched as 512 to cut 16k steps -> 252 steps
        constexpr std::size_t dflash_window = 2048;
        constexpr std::size_t prefill_batch = 512;
        const std::size_t fast_end = prompt.size() > dflash_window ? prompt.size() - dflash_window : 0;
        for (std::size_t begin = 0; begin < fast_end; begin += prefill_batch) {
            const std::size_t valid = std::min<std::size_t>(prefill_batch, fast_end - begin);
            std::vector<uint32_t> chunk(prompt.begin() + begin, prompt.begin() + begin + valid);
            const auto result = target.step_batch_large(chunk);
            // For large batch, the last sub-batch has valid = min(8, remaining) and next_tokens[valid-1] is the true anchor
            const std::size_t last_sub_valid = valid % 8 == 0 ? 8 : valid % 8;
            anchor = result.next_tokens[last_sub_valid - 1];
            if ((begin + valid) % 8192 == 0) {
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - prefill_started).count();
                std::cerr << "  [run " << run << " prefill " << begin + valid << '/' << prompt.size() << " (fast 512, no DFlash), " << std::fixed << std::setprecision(1) << elapsed << "s]\n";
            }
        }
        for (std::size_t begin = fast_end; begin < prompt.size(); begin += 8) {
            const std::size_t valid = std::min<std::size_t>(8, prompt.size() - begin);
            std::array<std::uint32_t, 8> batch{};
            batch.fill(248070);
            std::copy_n(prompt.begin() + begin, valid, batch.begin());
            const std::size_t position = target.active_tokens();
            const auto result = target.step_batch8_pipelined(batch, valid, true);
            std::array<std::array<std::vector<float>, 5>, 8> injection_features{};
            for (std::size_t index = 0; index < valid; ++index) {
                injection_features[index] = target.batch_dflash_features(index);
            }
            dflash.inject_batch(injection_features, valid, position);
            anchor = result.next_tokens[valid - 1];
            if ((begin + valid) % 8192 == 0) {
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - prefill_started).count();
                std::cerr << "  [run " << run << " prefill " << begin + valid << '/'
                          << prompt.size() << ", " << std::fixed << std::setprecision(1)
                          << elapsed << "s]\n";
            }
        }
        const double ttft_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - prefill_started).count();
        if (target.active_tokens() != prompt_token_count) {
            throw std::runtime_error("128K benchmark did not physically commit the complete prompt");
        }

        const auto warmup = benchmark_decode_tokens(
            target, dflash, tokenizer, anchor, warmup_token_count);
        if (warmup.generated != warmup_token_count || warmup.ended_early) {
            throw std::runtime_error("128K benchmark warmup ended before eight generated tokens");
        }
        const auto measured = benchmark_decode_tokens(
            target, dflash, tokenizer, anchor, output_token_count);
        if (measured.generated != output_token_count || measured.ended_early) {
            throw std::runtime_error("128K benchmark response ended before 256 generated tokens");
        }
        if (run == 1) {
            reference_tokens = measured.tokens;
        } else if (measured.tokens != reference_tokens) {
            throw std::runtime_error("128K benchmark greedy output changed between runs");
        }

        const double throughput = measured.generated / measured.seconds;
        throughputs.push_back(throughput);
        const double acceptance = measured.proposed == 0 ? 0.0
            : static_cast<double>(measured.accepted) / measured.proposed;
        const auto stats = target.resident_weight_stats();
        records << std::setprecision(9)
                << "{\"run\":" << run
                << ",\"device\":\"" << properties.name << "\""
                << ",\"arch\":\"" << properties.gcnArchName << "\""
                << ",\"target\":\"" << target_path << "\""
                << ",\"dflash\":\"" << dflash_path << "\""
                << ",\"prompt_tokens\":" << prompt_token_count
                << ",\"prompt_fnv64\":\"0x" << std::hex << prompt_digest << std::dec << "\""
                << ",\"context_capacity\":" << context_capacity
                << ",\"warmup_tokens\":" << warmup.generated
                << ",\"output_tokens\":" << measured.generated
                << ",\"output_fnv64\":\"0x" << std::hex << token_hash(measured.tokens)
                << std::dec << "\""
                << ",\"initialization_seconds\":" << initialization_seconds
                << ",\"ttft_seconds\":" << ttft_seconds
                << ",\"decode_seconds\":" << measured.seconds
                << ",\"tokens_per_second\":" << throughput
                << ",\"cycles\":" << measured.cycles
                << ",\"draft_proposed\":" << measured.proposed
                << ",\"draft_accepted\":" << measured.accepted
                << ",\"acceptance\":" << acceptance
                << ",\"draft_seconds\":" << measured.draft_seconds
                << ",\"target_seconds\":" << measured.target_seconds
                << ",\"injection_seconds\":" << measured.injection_seconds
                << ",\"resident_bytes\":" << stats.resident_bytes
                << ",\"resident_matrices\":" << stats.matrix_count
                << ",\"tokens\":[";
        for (std::size_t index = 0; index < measured.tokens.size(); ++index) {
            if (index) records << ',';
            records << measured.tokens[index];
        }
        records << "]}\n";
        records.flush();
        std::cerr << "  run " << run << ": " << std::fixed << std::setprecision(3)
                  << measured.seconds << "s, " << throughput << " tok/s, accepted "
                  << measured.accepted << '/' << measured.proposed << ", TTFT "
                  << ttft_seconds << "s\n";
    }

    const double mean = std::accumulate(throughputs.begin(), throughputs.end(), 0.0)
                      / throughputs.size();
    records << "{\"summary\":true,\"runs\":" << runs
            << ",\"mean_tokens_per_second\":" << std::setprecision(9) << mean
            << ",\"requirement_tokens_per_second\":10.0"
            << ",\"passed\":" << (mean >= 10.0 ? "true" : "false") << "}\n";
    records.flush();
    std::cerr << "  arithmetic mean: " << std::fixed << std::setprecision(3)
              << mean << " generated tok/s — " << (mean >= 10.0 ? "PASS" : "FAIL") << '\n';
    if (mean < 10.0) {
        throw std::runtime_error("128K benchmark mean is below 10.0 generated tokens/s");
    }
}

void chat(const std::string &target_path, const std::string &dflash_path,
          std::size_t context_capacity, std::size_t resident_weight_budget) {
    NativeTargetSession target(target_path, context_capacity, resident_weight_budget);
    NativeDflashEncoder dflash(dflash_path);
    dflash.preload_all_quantized();

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
            dflash.reset();
            std::cerr << "  [context reset]\n\n";
            continue;
        }

        std::vector<std::uint32_t> prompt;
        // Match the checkpoint's default chat template: thinking mode inserts
        // the xhigh reasoning instruction as a system turn even when the user
        // did not provide an explicit system message.
        prompt.push_back(tokenizer.im_start());
        prompt.push_back(8678);  // system
        prompt.push_back(198);   // \n
        const auto system_text = tokenizer.encode_full(
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
            "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.");
        prompt.insert(prompt.end(), system_text.begin(), system_text.end());
        prompt.push_back(tokenizer.im_end());
        prompt.push_back(198);
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

        // Prefill with the same native pipeline used for verification when a
        // genuine long context was requested. This writes every prompt KV into
        // the N23KV4 cache rather than building a truncated scalar cache.
        const auto prefill_started = std::chrono::steady_clock::now();
        std::uint32_t anchor = 0;
        if (context_capacity > NativeTargetSession::pipeline_attention_capacity()) {
            for (std::size_t begin = 0; begin < prompt.size(); begin += 8) {
                const std::size_t valid = std::min<std::size_t>(8, prompt.size() - begin);
                std::array<std::uint32_t, 8> batch{};
                batch.fill(248070);
                std::copy_n(prompt.begin() + begin, valid, batch.begin());
                const std::size_t position = target.active_tokens();
                const auto result = target.step_batch8_pipelined(batch, valid);
                std::array<std::array<std::vector<float>, 5>, 8> injection_features{};
                for (std::size_t index = 0; index < valid; ++index) {
                    injection_features[index] = target.batch_dflash_features(index);
                }
                dflash.inject_batch(injection_features, valid, position);
                anchor = result.next_tokens[valid - 1];
            }
        } else {
            NativeTargetSession::StepResult anchor_result{};
            for (const auto token : prompt) {
                anchor_result = target.step(token);
                dflash.inject(target.dflash_features(), target.active_tokens() - 1);
            }
            anchor = anchor_result.next_token;
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
        int cycle_count = 0;
        std::optional<std::array<std::uint32_t, 7>> pending_stale_block;

        while (!done && generated < 8192) {
            if (anchor == tokenizer.im_end() || anchor == tokenizer.eos_token()) break;

            const auto cycle_start = std::chrono::steady_clock::now();

            // DFlash proposes 7 continuation tokens
            const auto proposals = dflash.propose(target, anchor, target.active_tokens());
            if (pending_stale_block && std::getenv("QWEN_NATIVE_CASCADE_TRACE")) {
                std::size_t matching = 0;
                while (matching + 1 < pending_stale_block->size() &&
                       (*pending_stale_block)[matching + 1] == proposals[matching]) {
                    ++matching;
                }
                std::cerr << "  [cascade stale/fresh continuation: " << matching << "/6]\n";
                pending_stale_block.reset();
            }

            const auto propose_done = std::chrono::steady_clock::now();

            // Build verification batch: [anchor, proposal_0, ..., proposal_6]
            std::array<std::uint32_t, 8> candidate{};
            candidate[0] = anchor;
            std::copy(proposals.begin(), proposals.end(), candidate.begin() + 1);

            // Target verifies all 8 in one pipelined batch.  Keep a host
            // checkpoint so a rejected suffix cannot leak into recurrent or
            // attention state.
            const auto checkpoint = target.begin_speculative_checkpoint();
            auto verification = target.step_batch8_pipelined(candidate);
            const auto step_done = std::chrono::steady_clock::now();

            // Find accepted prefix (draft agrees with target)
            std::size_t accepted = 0;
            while (accepted < proposals.size() &&
                   verification.next_tokens[accepted] == proposals[accepted]) {
                ++accepted;
            }

            if (accepted != proposals.size()) {
                // The verifier's causal prefix activations, KV entries, target
                // features, and logits are already final. Restore only the
                // recurrent checkpoint and replay its captured update factors;
                // rerunning 64 layers would duplicate the dominant weight pass.
                target.commit_speculative_prefix(checkpoint, accepted + 1);
            }
            const auto commit_done = std::chrono::steady_clock::now();

            if (accepted == proposals.size() && std::getenv("QWEN_NATIVE_CASCADE_TRACE")) {
                const auto stale = dflash.propose(
                    target, proposals.back(), checkpoint.active_tokens + 7);
                const bool guessed_bonus = stale[0] == verification.next_tokens[7];
                std::cerr << "  [cascade stale bonus: "
                          << (guessed_bonus ? "match" : "miss") << "]\n";
                if (guessed_bonus) pending_stale_block = stale;
            }

            // DFlash conditions its next block on target features for every
            // committed input token. Encode the committed prefix in one exact
            // batch, and never append the speculative suffix.
            std::array<std::array<std::vector<float>, 5>, 8> injection_features{};
            for (std::size_t index = 0; index <= accepted; ++index) {
                injection_features[index] = target.batch_dflash_features(index);
            }
            dflash.inject_batch(injection_features, accepted + 1, checkpoint.active_tokens);
            if (dflash.injected_token_count() != std::min<std::size_t>(target.active_tokens(), 2048)) {
                throw std::runtime_error("DFlash injected cache did not advance exactly once per committed token");
            }
            const auto inject_done = std::chrono::steady_clock::now();

            if (cycle_count < 5) {
                const double propose_ms = std::chrono::duration<double, std::milli>(propose_done - cycle_start).count();
                const double step_ms = std::chrono::duration<double, std::milli>(step_done - propose_done).count();
                const double commit_ms = std::chrono::duration<double, std::milli>(commit_done - step_done).count();
                const double inject_ms = std::chrono::duration<double, std::milli>(inject_done - commit_done).count();
                const double total_ms = std::chrono::duration<double, std::milli>(inject_done - cycle_start).count();
                const double rate = (accepted + 1) / (total_ms / 1000.0);
                std::cerr << "  [cycle " << cycle_count << ": prop " << std::fixed << std::setprecision(1)
                          << propose_ms << "ms, step " << step_ms << "ms (rec "
                          << verification.recurrent_seconds * 1000.0 << ", attn "
                          << verification.attention_seconds * 1000.0 << ", ffn "
                          << verification.ffn_seconds * 1000.0 << ", head "
                          << verification.output_seconds * 1000.0 << "), commit " << commit_ms
                          << "ms, inj " << inject_ms << "ms, total " << total_ms << "ms ("
                          << rate << " tok/s), acc=" << accepted << "/7";
                if (accepted < proposals.size()) {
                    std::cerr << ", target-rank="
                              << dflash.last_candidate_rank(accepted,
                                                            verification.next_tokens[accepted]);
                }
                std::cerr << '\n';
            }
            ++cycle_count;

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

            // The prefix commit retained exactly [anchor, accepted proposals].
            // The target prediction at the rejection point is the next anchor.
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
              << "  " << program << " tokenize MODEL.gguf TEXT\n"
              << "  " << program << " prompt-test TARGET.gguf TEXT [TOKENS]\n"
              << "  " << program << " chat TARGET.gguf DFLASH.gguf [CONTEXT] [RESIDENT_GIB]\n"
              << "  " << program << " benchmark-128k TARGET.gguf DFLASH.gguf OUTPUT.jsonl [RESIDENT_GIB] [RUNS]\n"
              << "  " << program << " bench-matvec MODEL.gguf TENSOR [ITERATIONS]\n"
              << "  " << program << " bench-ffn MODEL.gguf LAYER [ITERATIONS]\n"
              << "  " << program << " bench-gdn [ITERATIONS]\n"
              << "  " << program << " mtp-test TARGET.gguf CONTEXT TOKEN_ID [TOKENS] [RESIDENT_GIB]\n"
              << "  " << program << " target-generate TARGET.gguf CONTEXT TOKEN_ID [TOKENS] [RESIDENT_GIB]\n"
              << "  " << program << " target-verify-batch TARGET.gguf CONTEXT TOKEN_ID [BLOCKS] [RESIDENT_GIB]\n"
              << "  " << program << " target-long-window-bench TARGET.gguf CONTEXT TOKEN_ID [RESIDENT_GIB]\n"
              << "  " << program << " target-compare-batch TARGET.gguf CONTEXT TOKEN_ID [RESIDENT_GIB]\n"
              << "  " << program << " target-rollback-test TARGET.gguf CONTEXT TOKEN_ID [RESIDENT_GIB]\n"
              << "  " << program << " dflash-commit-test TARGET.gguf DFLASH.gguf CONTEXT TOKEN_ID [RESIDENT_GIB]\n"
              << "  " << program << " dflash-encode TARGET.gguf DFLASH.gguf CONTEXT TOKEN_ID [RESIDENT_GIB]\n"
              << "  " << program << " dflash-propose TARGET.gguf DFLASH.gguf CONTEXT TOKEN_ID [RESIDENT_GIB]\n"
              << "  " << program << " dflash-verify TARGET.gguf DFLASH.gguf CONTEXT TOKEN_ID [RESIDENT_GIB]\n"
              << "  " << program << " dflash-verify-prompt TARGET.gguf DFLASH.gguf CONTEXT TOKENS_CSV [RESIDENT_GIB]\n"
              << "  " << program << " speculative-equivalence TARGET.gguf DFLASH.gguf CONTEXT TOKENS_CSV COUNT [RESIDENT_GIB]\n"
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
        if (argc >= 5 && argc <= 7 && std::string(argv[1]) == "benchmark-128k") {
            const double gib = argc >= 6 ? std::stod(argv[5]) : 1.0;
            if (gib < 0.0 || gib > 2.0) {
                throw std::invalid_argument("benchmark RESIDENT_GIB must be in [0, 2]");
            }
            const int runs = argc >= 7 ? std::stoi(argv[6]) : 3;
            benchmark_128k(argv[2], argv[3], argv[4],
                           static_cast<std::size_t>(gib * 1024.0 * 1024.0 * 1024.0),
                           runs);
            return EXIT_SUCCESS;
        }
        if (argc >= 4 && argc <= 6 && std::string(argv[1]) == "chat") {
            const std::size_t context = argc >= 5 ? static_cast<std::size_t>(std::stoull(argv[4])) : 4096;
            const double gib = argc >= 6 ? std::stod(argv[5]) : 6.0;
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
        if ((argc == 5 || argc == 6) && std::string(argv[1]) == "target-long-window-bench") {
            const char *kv_env = std::getenv("QWEN_NATIVE_KV");
            const double default_gib = (kv_env && std::strcmp(kv_env, "q4") == 0) ? 4.0 : 1.0;
            const double requested_gib = argc == 6 ? std::stod(argv[5]) : default_gib;
            if (requested_gib < 0.0 || requested_gib > 7.0) {
                throw std::invalid_argument("RESIDENT_GIB must be in [0, 7]");
            }
            target_long_window_bench(
                argv[2], static_cast<std::size_t>(std::stoull(argv[3])),
                static_cast<std::uint32_t>(std::stoul(argv[4])),
                static_cast<std::size_t>(requested_gib * 1024.0 * 1024.0 * 1024.0));
            return EXIT_SUCCESS;
        }
        if ((argc == 5 || argc == 6) && std::string(argv[1]) == "target-compare-batch") {
            const double requested_gib = argc == 6 ? std::stod(argv[5]) : 0.0;
            if (requested_gib < 0.0 || requested_gib > 7.0) {
                throw std::invalid_argument("RESIDENT_GIB must be in [0, 7]");
            }
            target_compare_batch(argv[2], static_cast<std::size_t>(std::stoull(argv[3])),
                                 static_cast<std::uint32_t>(std::stoul(argv[4])),
                                 static_cast<std::size_t>(requested_gib * 1024.0 * 1024.0 * 1024.0));
            return EXIT_SUCCESS;
        }
        if ((argc == 5 || argc == 6) && std::string(argv[1]) == "target-rollback-test") {
            const double requested_gib = argc == 6 ? std::stod(argv[5]) : 0.0;
            if (requested_gib < 0.0 || requested_gib > 7.0) {
                throw std::invalid_argument("RESIDENT_GIB must be in [0, 7]");
            }
            target_rollback_test(argv[2], static_cast<std::size_t>(std::stoull(argv[3])),
                                 static_cast<std::uint32_t>(std::stoul(argv[4])),
                                 static_cast<std::size_t>(requested_gib * 1024.0 * 1024.0 * 1024.0));
            return EXIT_SUCCESS;
        }
        if ((argc == 6 || argc == 7) && std::string(argv[1]) == "dflash-commit-test") {
            const double requested_gib = argc == 7 ? std::stod(argv[6]) : 0.0;
            if (requested_gib < 0.0 || requested_gib > 7.0) {
                throw std::invalid_argument("RESIDENT_GIB must be in [0, 7]");
            }
            dflash_commit_test(argv[2], argv[3], static_cast<std::size_t>(std::stoull(argv[4])),
                                static_cast<std::uint32_t>(std::stoul(argv[5])),
                                static_cast<std::size_t>(requested_gib * 1024.0 * 1024.0 * 1024.0));
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
        if ((argc == 7 || argc == 8) && std::string(argv[1]) == "speculative-equivalence") {
            const double requested_gib = argc == 8 ? std::stod(argv[7]) : 1.0;
            if (requested_gib < 0.0 || requested_gib > 7.0) {
                throw std::invalid_argument("RESIDENT_GIB must be in [0, 7]");
            }
            speculative_equivalence_test(
                argv[2], argv[3], static_cast<std::size_t>(std::stoull(argv[4])),
                parse_token_list(argv[5]), static_cast<std::size_t>(std::stoull(argv[6])),
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

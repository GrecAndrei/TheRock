#include "gguf.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace qwen_native {
namespace {

constexpr std::uint32_t kGgufMagic = 0x46554747U;
constexpr std::uint64_t kDefaultAlignment = 32;
constexpr std::uint64_t kMaxStringLength = 1ULL << 30;
constexpr std::uint64_t kMaxArrayLength = 1ULL << 32;

enum class ValueType : std::uint32_t {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
};

template <typename T>
T read_scalar(std::ifstream &stream) {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    stream.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!stream) {
        throw std::runtime_error("unexpected end of GGUF file");
    }
    return value;
}

std::string read_string(std::ifstream &stream) {
    const auto length = read_scalar<std::uint64_t>(stream);
    if (length > kMaxStringLength || length > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("invalid GGUF string length");
    }
    std::string value(static_cast<std::size_t>(length), '\0');
    stream.read(value.data(), static_cast<std::streamsize>(length));
    if (!stream) {
        throw std::runtime_error("unexpected end of GGUF string");
    }
    return value;
}

template <typename T>
void skip_scalars(std::ifstream &stream, std::uint64_t count) {
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) / sizeof(T)) {
        throw std::runtime_error("GGUF array is too large");
    }
    stream.seekg(static_cast<std::streamoff>(count * sizeof(T)), std::ios::cur);
    if (!stream) {
        throw std::runtime_error("unexpected end of GGUF array");
    }
}

MetadataValue read_value(std::ifstream &stream, ValueType type);

MetadataValue read_array(std::ifstream &stream) {
    const auto element_type = static_cast<ValueType>(read_scalar<std::uint32_t>(stream));
    const auto count = read_scalar<std::uint64_t>(stream);
    if (count > kMaxArrayLength) {
        throw std::runtime_error("GGUF array exceeds safety limit");
    }

    if (element_type == ValueType::String) {
        std::vector<std::string> values;
        values.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            values.push_back(read_string(stream));
        }
        return values;
    }

    switch (element_type) {
    case ValueType::Uint8: {
        std::vector<std::uint64_t> values(count);
        for (auto &value : values) value = read_scalar<std::uint8_t>(stream);
        return values;
    }
    case ValueType::Uint16: {
        std::vector<std::uint64_t> values(count);
        for (auto &value : values) value = read_scalar<std::uint16_t>(stream);
        return values;
    }
    case ValueType::Uint32: {
        std::vector<std::uint64_t> values(count);
        for (auto &value : values) value = read_scalar<std::uint32_t>(stream);
        return values;
    }
    case ValueType::Uint64: {
        std::vector<std::uint64_t> values(count);
        for (auto &value : values) value = read_scalar<std::uint64_t>(stream);
        return values;
    }
    case ValueType::Bool: {
        std::vector<std::uint64_t> values(count);
        for (auto &value : values) value = read_scalar<std::uint8_t>(stream) != 0;
        return values;
    }
    case ValueType::Int8: {
        std::vector<std::int64_t> values(count);
        for (auto &value : values) value = read_scalar<std::int8_t>(stream);
        return values;
    }
    case ValueType::Int16: {
        std::vector<std::int64_t> values(count);
        for (auto &value : values) value = read_scalar<std::int16_t>(stream);
        return values;
    }
    case ValueType::Int32: {
        std::vector<std::int64_t> values(count);
        for (auto &value : values) value = read_scalar<std::int32_t>(stream);
        return values;
    }
    case ValueType::Int64: {
        std::vector<std::int64_t> values(count);
        for (auto &value : values) value = read_scalar<std::int64_t>(stream);
        return values;
    }
    case ValueType::Float32: {
        std::vector<double> values(count);
        for (auto &value : values) value = read_scalar<float>(stream);
        return values;
    }
    case ValueType::Float64: {
        std::vector<double> values(count);
        for (auto &value : values) value = read_scalar<double>(stream);
        return values;
    }
    case ValueType::String:
    case ValueType::Array:
        throw std::runtime_error("nested or invalid GGUF array type");
    }
}

MetadataValue read_value(std::ifstream &stream, ValueType type) {
    switch (type) {
    case ValueType::Uint8:   return static_cast<std::uint64_t>(read_scalar<std::uint8_t>(stream));
    case ValueType::Int8:    return static_cast<std::int64_t>(read_scalar<std::int8_t>(stream));
    case ValueType::Uint16:  return static_cast<std::uint64_t>(read_scalar<std::uint16_t>(stream));
    case ValueType::Int16:   return static_cast<std::int64_t>(read_scalar<std::int16_t>(stream));
    case ValueType::Uint32:  return static_cast<std::uint64_t>(read_scalar<std::uint32_t>(stream));
    case ValueType::Int32:   return static_cast<std::int64_t>(read_scalar<std::int32_t>(stream));
    case ValueType::Float32: return static_cast<double>(read_scalar<float>(stream));
    case ValueType::Bool:    return read_scalar<std::uint8_t>(stream) != 0;
    case ValueType::String:  return read_string(stream);
    case ValueType::Array:   return read_array(stream);
    case ValueType::Uint64:  return read_scalar<std::uint64_t>(stream);
    case ValueType::Int64:   return read_scalar<std::int64_t>(stream);
    case ValueType::Float64: return read_scalar<double>(stream);
    }
    throw std::runtime_error("unknown GGUF metadata value type");
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::runtime_error("GGUF alignment is not a power of two");
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace

GgufFile::GgufFile(const std::filesystem::path &path) : path_(path) {
    file_size_ = std::filesystem::file_size(path_);
    std::ifstream stream(path_, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + path_.string());
    }

    if (read_scalar<std::uint32_t>(stream) != kGgufMagic) {
        throw std::runtime_error("not a GGUF file: " + path_.string());
    }
    version_ = read_scalar<std::uint32_t>(stream);
    if (version_ != 3) {
        throw std::runtime_error("unsupported GGUF version " + std::to_string(version_));
    }

    const auto tensor_count = read_scalar<std::uint64_t>(stream);
    const auto metadata_count = read_scalar<std::uint64_t>(stream);
    if (tensor_count > 1'000'000 || metadata_count > 1'000'000) {
        throw std::runtime_error("implausible GGUF directory size");
    }

    metadata_.reserve(static_cast<std::size_t>(metadata_count));
    for (std::uint64_t i = 0; i < metadata_count; ++i) {
        auto key = read_string(stream);
        const auto type = static_cast<ValueType>(read_scalar<std::uint32_t>(stream));
        metadata_.insert_or_assign(std::move(key), read_value(stream, type));
    }

    tensors_.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo tensor;
        tensor.name = read_string(stream);
        const auto dimensions = read_scalar<std::uint32_t>(stream);
        if (dimensions == 0 || dimensions > 8) {
            throw std::runtime_error("invalid tensor rank for " + tensor.name);
        }
        tensor.dimensions.resize(dimensions);
        for (auto &dimension : tensor.dimensions) {
            dimension = read_scalar<std::uint64_t>(stream);
        }
        tensor.type = read_scalar<std::uint32_t>(stream);
        tensor.offset = read_scalar<std::uint64_t>(stream);
        tensors_.push_back(std::move(tensor));
    }

    std::uint64_t alignment = kDefaultAlignment;
    if (const auto *value = find_metadata("general.alignment")) {
        if (const auto *number = std::get_if<std::uint64_t>(value)) {
            alignment = *number;
        }
    }
    const auto directory_end = static_cast<std::uint64_t>(stream.tellg());
    data_offset_ = align_up(directory_end, alignment);
    if (data_offset_ > file_size_) {
        throw std::runtime_error("GGUF tensor data begins past end of file");
    }
}

const MetadataValue *GgufFile::find_metadata(const std::string &key) const {
    const auto it = metadata_.find(key);
    return it == metadata_.end() ? nullptr : &it->second;
}

const TensorInfo *GgufFile::find_tensor(const std::string &name) const {
    const auto it = std::find_if(tensors_.begin(), tensors_.end(), [&](const TensorInfo &tensor) {
        return tensor.name == name;
    });
    return it == tensors_.end() ? nullptr : &*it;
}

std::string metadata_to_string(const MetadataValue &value) {
    return std::visit([](const auto &item) -> std::string {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, std::string>) {
            return item;
        } else if constexpr (std::is_same_v<T, bool>) {
            return item ? "true" : "false";
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            return "[" + std::to_string(item.size()) + " strings]";
        } else if constexpr (std::is_same_v<T, std::vector<std::uint64_t>> ||
                             std::is_same_v<T, std::vector<std::int64_t>> ||
                             std::is_same_v<T, std::vector<double>>) {
            std::ostringstream output;
            output << '[';
            for (std::size_t index = 0; index < item.size(); ++index) {
                output << (index == 0 ? "" : ",") << item[index];
            }
            output << ']';
            return output.str();
        } else {
            std::ostringstream output;
            output << item;
            return output.str();
        }
    }, value);
}

const char *ggml_type_name(std::uint32_t type) {
    constexpr std::array<const char *, 40> names = {
        "F32", "F16", "Q4_0", "Q4_1", "Q4_2", "Q4_3", "Q5_0", "Q5_1",
        "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K",
        "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S",
        "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16", "Q4_0_4_4",
        "Q4_0_4_8", "Q4_0_8_8", "TQ1_0", "TQ2_0", "MXFP4", "NVFP4", "UNKNOWN_38", "UNKNOWN_39"
    };
    return type < names.size() ? names[type] : "UNKNOWN";
}

} // namespace qwen_native

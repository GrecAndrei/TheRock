#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace qwen_native {

using MetadataValue = std::variant<std::string, std::uint64_t, std::int64_t,
                                   double, bool, std::vector<std::string>,
                                   std::vector<std::uint64_t>,
                                   std::vector<std::int64_t>,
                                   std::vector<double>>;

struct TensorInfo {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t type = 0;
    std::uint64_t offset = 0;
};

class GgufFile {
public:
    explicit GgufFile(const std::filesystem::path &path);

    [[nodiscard]] std::uint32_t version() const { return version_; }
    [[nodiscard]] std::uint64_t data_offset() const { return data_offset_; }
    [[nodiscard]] std::uint64_t file_size() const { return file_size_; }
    [[nodiscard]] const std::filesystem::path &path() const { return path_; }
    [[nodiscard]] const std::unordered_map<std::string, MetadataValue> &metadata() const { return metadata_; }
    [[nodiscard]] const std::vector<TensorInfo> &tensors() const { return tensors_; }

    [[nodiscard]] const MetadataValue *find_metadata(const std::string &key) const;
    [[nodiscard]] const TensorInfo *find_tensor(const std::string &name) const;

private:
    std::filesystem::path path_;
    std::uint32_t version_ = 0;
    std::uint64_t data_offset_ = 0;
    std::uint64_t file_size_ = 0;
    std::unordered_map<std::string, MetadataValue> metadata_;
    std::vector<TensorInfo> tensors_;
};

std::string metadata_to_string(const MetadataValue &value);
const char *ggml_type_name(std::uint32_t type);

} // namespace qwen_native

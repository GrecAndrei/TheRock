#include "gguf.h"
#include "kernels.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fcntl.h>
#include <map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>
#include <string>

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

void bench_q4(const std::string &path, const std::string &tensor_name, int iterations) {
    const qwen_native::GgufFile model(path);
    const auto *tensor = model.find_tensor(tensor_name);
    if (!tensor) {
        throw std::runtime_error("tensor not found: " + tensor_name);
    }
    if (tensor->type != 12 || tensor->dimensions.size() != 2) {
        throw std::runtime_error("benchmark requires a rank-2 Q4_K tensor");
    }
    const Mapping mapping(path);
    const auto input_size = static_cast<std::size_t>(tensor->dimensions[0]);
    const auto output_size = static_cast<std::size_t>(tensor->dimensions[1]);
    std::cout << "Q4_K real-weight benchmark: " << tensor_name << '\n'
              << "  shape: " << input_size << " x " << output_size << '\n';
    const auto result = qwen_native::benchmark_q4k(
        mapping.at(model.data_offset() + tensor->offset), input_size, output_size, iterations);
    std::cout << "  Q4_K native:\n"
              << "    latency:    " << std::fixed << std::setprecision(3) << result.milliseconds << " ms\n"
              << "    bandwidth:  " << std::setprecision(2) << result.bandwidth_gbps << " GB/s\n"
              << "    max error:  " << std::scientific << result.max_absolute_error << '\n';
    const auto n23 = qwen_native::benchmark_n23q4_from_q4k(
        mapping.at(model.data_offset() + tensor->offset), input_size, output_size, iterations);
    std::cout << "  N23Q4 wave32 + DP4A:\n"
              << "    latency:    " << std::fixed << std::setprecision(3) << n23.milliseconds << " ms\n"
              << "    bandwidth:  " << std::setprecision(2) << n23.bandwidth_gbps << " GB/s\n"
              << "    max delta:  " << std::scientific << n23.max_absolute_error << '\n'
              << "    rel. RMSE:  " << n23.relative_rmse << '\n'
              << "    kernel err: " << n23.kernel_max_absolute_error << '\n';
    const auto batch8 = qwen_native::benchmark_q4k_batch8(
        mapping.at(model.data_offset() + tensor->offset), input_size, output_size, iterations);
    std::cout << "  Q4_K eight-token verifier:\n"
              << "    latency:    " << std::fixed << std::setprecision(3) << batch8.milliseconds << " ms\n"
              << "    weight BW:  " << std::setprecision(2) << batch8.bandwidth_gbps << " GB/s\n"
              << "    max error:  " << std::scientific << batch8.max_absolute_error << '\n';
    const auto n23_batch8 = qwen_native::benchmark_n23q4_batch8_from_q4k(
        mapping.at(model.data_offset() + tensor->offset), input_size, output_size, iterations);
    std::cout << "  N23Q4 eight-token verifier:\n"
              << "    latency:    " << std::fixed << std::setprecision(3) << n23_batch8.milliseconds << " ms\n"
              << "    weight BW:  " << std::setprecision(2) << n23_batch8.bandwidth_gbps << " GB/s\n"
              << "    max delta:  " << std::scientific << n23_batch8.max_absolute_error << '\n'
              << "    rel. RMSE:  " << n23_batch8.relative_rmse << '\n';
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

void usage(const char *program) {
    std::cout << "Usage:\n"
              << "  " << program << " device\n"
              << "  " << program << " inspect MODEL.gguf [--tensors]\n"
              << "  " << program << " bench-q4 MODEL.gguf TENSOR [ITERATIONS]\n"
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
        if ((argc == 4 || argc == 5) && std::string(argv[1]) == "bench-q4") {
            const int iterations = argc == 5 ? std::stoi(argv[4]) : 100;
            bench_q4(argv[2], argv[3], iterations);
            return EXIT_SUCCESS;
        }
        if (argc == 4 && std::string(argv[1]) == "plan-128k") {
            plan_128k(argv[2], argv[3]);
            return EXIT_SUCCESS;
        }
        usage(argv[0]);
        return EXIT_FAILURE;
    } catch (const std::exception &error) {
        std::cerr << "qwen38-native: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

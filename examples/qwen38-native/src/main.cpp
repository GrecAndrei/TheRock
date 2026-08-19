#include "gguf.h"
#include "kernels.h"

#include <hip/hip_runtime.h>

#include <algorithm>
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
}

void usage(const char *program) {
    std::cout << "Usage:\n"
              << "  " << program << " device\n"
              << "  " << program << " inspect MODEL.gguf [--tensors]\n"
              << "  " << program << " bench-q4 MODEL.gguf TENSOR [ITERATIONS]\n";
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
        usage(argv[0]);
        return EXIT_FAILURE;
    } catch (const std::exception &error) {
        std::cerr << "qwen38-native: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

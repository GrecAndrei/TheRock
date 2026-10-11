#include "attention_tile.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void require_bits(float expected, float actual) {
    require(std::memcmp(&expected, &actual, sizeof(float)) == 0,
            "packed LDS coefficient/FMA differs from FP32 tile");
}

void coefficient_test() {
    qwen_native::AttentionSharedTile<16, false> reference{};
    qwen_native::AttentionSharedTile<16, true> packed{};
    constexpr std::array<float, 7> scales = {
        0.0F, -0.0F, 0x1p-24F, -0.125F, 1.0F, 65504.0F, 0.03125F};
    for (int item = 0; item < 16 * 256; ++item) {
        const auto key = static_cast<std::int8_t>(item % 256 - 128);
        const auto value = static_cast<std::int8_t>((item * 37) % 256 - 128);
        const float key_scale = scales[(item / 32) % scales.size()];
        const float value_scale = scales[(item / 32 + 3) % scales.size()];
        reference.stage(item, key, value, key_scale, value_scale);
        packed.stage(item, key, value, key_scale, value_scale);
    }
    for (int item = 0; item < 16 * 256; ++item) {
        require_bits(reference.key(item), packed.key(item));
        require_bits(reference.value(item), packed.value(item));
    }
    for (int token = 0; token < 16; ++token) {
        for (int lane = 0; lane < 32; ++lane) {
            float expected = 0.0F;
            float actual = 0.0F;
            for (int group = 0; group < 8; ++group) {
                const int index = token * 256 + group * 32 + lane;
                const float query = (group * 32 + lane - 117) / 256.0F;
                expected = std::fma(query, reference.key(index), expected);
                actual = std::fma(query, packed.key(index), actual);
            }
            require_bits(expected, actual);
        }
    }
}

void cooperative_map_test() {
    for (int valid_tokens : {1, 7, 15, 16}) {
        std::array<int, 16 * 256> code_writers{};
        std::array<int, 16 * 8> scale_writers{};
        for (int thread = 0; thread < 512; ++thread) {
            for (int item = thread; item < valid_tokens * 256; item += 512) {
                ++code_writers[item];
                if (item % 32 == 0) ++scale_writers[item / 32];
            }
        }
        for (int item = 0; item < 16 * 256; ++item) {
            require(code_writers[item] == (item < valid_tokens * 256 ? 1 : 0),
                    "missing/duplicate code writer or access beyond tile tail");
        }
        for (int group = 0; group < 16 * 8; ++group) {
            require(scale_writers[group] == (group < valid_tokens * 8 ? 1 : 0),
                    "missing/duplicate group-scale writer");
        }
    }
    std::array<int, 8 * 24> queries{};
    for (int kv_head = 0; kv_head < 4; ++kv_head) {
        for (int query_group = 0; query_group < 3; ++query_group) {
            for (int warp = 0; warp < 16; ++warp) {
                const int local_query = query_group * 16 + warp;
                const int batch = local_query / 6;
                const int head = kv_head * 6 + local_query % 6;
                ++queries[batch * 24 + head];
            }
        }
    }
    for (int count : queries) require(count == 1, "GQA query missing or duplicated");
}

void causal_partition_test() {
    for (int history : {0, 1, 7, 16, 513, 131072}) {
        const int context = history + 8;
        for (int partitions : {1, 3, 48, 256}) {
            const int chunk = (context + partitions - 1) / partitions;
            for (int batch = 0; batch < 8; ++batch) {
                const int causal_end = history + batch + 1;
                int covered = 0;
                for (int partition = 0; partition < partitions; ++partition) {
                    const int begin = partition * chunk;
                    const int end = std::min({context, begin + chunk, causal_end});
                    if (end <= begin) continue;
                    require(begin == covered, "causal partition gap/overlap");
                    require(end <= causal_end, "future history exposed");
                    covered = end;
                }
                require(covered == causal_end, "complete physical history not covered");
            }
        }
    }
}

} // namespace

int main() {
    try {
        coefficient_test();
        cooperative_map_test();
        causal_partition_test();
        std::cout << "packed attention tile: coefficient/FMA bits, tail ownership, "
                     "GQA mapping and causal coverage passed (CPU only)\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

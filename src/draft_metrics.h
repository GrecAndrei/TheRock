#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>

namespace qwen_native {

struct DraftAcceptanceMetrics {
    std::size_t draft_cycles = 0;
    std::array<std::size_t, 8> prefix_histogram{};
    std::array<std::size_t, 7> conditional_attempts{};
    std::array<std::size_t, 7> conditional_accepted{};
    std::array<std::size_t, 7> conditional_top16_hits{};
    // Bucket 0 means absent; buckets 1..16 encode zero-based rank plus one.
    std::array<std::size_t, 17> first_reject_rank{};

    void observe(std::size_t proposed, std::size_t accepted,
                 const std::array<int, 7> &target_ranks) {
        if (proposed > 7 || accepted > proposed) {
            throw std::invalid_argument("invalid speculative prefix metrics");
        }
        if (proposed == 0) return;
        const auto labelled = accepted < proposed ? accepted + 1 : accepted;
        for (std::size_t pos = 0; pos < labelled; ++pos) {
            if (target_ranks[pos] < -1 || target_ranks[pos] > 15) {
                throw std::invalid_argument("invalid draft candidate rank");
            }
        }
        ++draft_cycles;
        ++prefix_histogram[accepted];
        // Past the first rejection, verifier logits see a wrong draft prefix;
        // they are not labels from the true greedy continuation.
        for (std::size_t pos = 0; pos < labelled; ++pos) {
            ++conditional_attempts[pos];
            conditional_accepted[pos] += pos < accepted;
            conditional_top16_hits[pos] += target_ranks[pos] >= 0;
        }
        if (accepted < proposed) {
            ++first_reject_rank[static_cast<std::size_t>(target_ranks[accepted] + 1)];
        }
    }
};

} // namespace qwen_native

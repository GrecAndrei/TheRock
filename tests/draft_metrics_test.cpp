#include "draft_metrics.h"

#include <array>
#include <iostream>
#include <numeric>
#include <stdexcept>

int main() {
    try {
        qwen_native::DraftAcceptanceMetrics metrics;
        for (std::size_t accepted = 0; accepted <= 7; ++accepted) {
            std::array<int, 7> ranks{};
            ranks.fill(99);
            const auto labelled = accepted < 7 ? accepted + 1 : accepted;
            for (std::size_t pos = 0; pos < labelled; ++pos) ranks[pos] = 0;
            if (accepted < 7) ranks[accepted] = accepted % 2 ? 15 : -1;
            metrics.observe(7, accepted, ranks);
        }
        if (metrics.draft_cycles != 8 ||
            std::accumulate(metrics.prefix_histogram.begin(), metrics.prefix_histogram.end(),
                            std::size_t{}) != 8 ||
            metrics.first_reject_rank[0] != 4 || metrics.first_reject_rank[16] != 3) {
            throw std::runtime_error("prefix or rejection histogram mismatch");
        }
        for (std::size_t pos = 0; pos < 7; ++pos) {
            const auto hits = 8 - pos - (pos % 2 ? 0 : 1);
            if (metrics.conditional_attempts[pos] != 8 - pos ||
                metrics.conditional_accepted[pos] != 7 - pos ||
                metrics.conditional_top16_hits[pos] != hits) {
                throw std::runtime_error("conditional label count mismatch");
            }
        }
        const auto cycles = metrics.draft_cycles;
        std::array<int, 7> invalid{};
        invalid.fill(99);
        metrics.observe(0, 0, invalid);
        if (metrics.draft_cycles != cycles) throw std::runtime_error("anchor-only cycle counted");
        for (const auto counts : {std::array<std::size_t, 2>{8, 0}, {2, 3}, {0, 1}, {1, 0}}) {
            bool rejected = false;
            try {
                metrics.observe(counts[0], counts[1], invalid);
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            if (!rejected || metrics.draft_cycles != cycles) {
                throw std::runtime_error("invalid metrics mutated counters");
            }
        }
        std::array<int, 7> tail{};
        tail.fill(99);
        tail[0] = 2;
        metrics.observe(1, 1, tail);
        if (metrics.draft_cycles != cycles + 1 || metrics.prefix_histogram[1] != 2) {
            throw std::runtime_error("partial proposal tail not counted");
        }
        std::cout << "draft prefix/conditional-rank fixtures passed; CPU-only, not inference timing\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

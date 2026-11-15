// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rxinfer/sampler.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>

namespace rxinfer {
namespace {

float fp16_bits_to_f32_scalar(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            uint32_t m = mant;
            uint32_t e = 127 - 14;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                --e;
            }
            bits = sign | (e << 23) | ((m & 0x03FFu) << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

const std::array<float, 65536>& fp16_to_f32_lut() {
    static const std::array<float, 65536> lut = []() {
        std::array<float, 65536> arr{};
        for (uint32_t i = 0; i < 65536; ++i) {
            arr[i] = fp16_bits_to_f32_scalar(static_cast<uint16_t>(i));
        }
        return arr;
    }();
    return lut;
}

uint64_t choose_seed(uint64_t req_seed) {
    if (req_seed != 0) return req_seed;
    std::random_device rd;
    const uint64_t r0 = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
    const uint64_t clk = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return r0 ^ clk ^ 0x9E3779B97F4A7C15ULL;
}

} // namespace

Sampler::Sampler(SamplerConfig cfg) : cfg_(cfg), rng_(choose_seed(cfg.seed)) {
    (void)fp16_to_f32_lut();
}

void Sampler::set_config(const SamplerConfig& cfg) {
    const bool reseed_needed = (cfg.seed != cfg_.seed);
    cfg_ = cfg;
    if (reseed_needed) {
        reseed(cfg_.seed);
    }
}

void Sampler::reseed(uint64_t seed) {
    cfg_.seed = seed;
    rng_.seed(choose_seed(seed));
}

void Sampler::clear_history() {
    history_.clear();
}

void Sampler::record_token(int32_t tok) {
    if (tok < 0) return;
    history_.push_back(tok);
    const size_t cap = static_cast<size_t>(std::max<int32_t>(cfg_.repeat_last_n, 256));
    while (history_.size() > cap) {
        history_.pop_front();
    }
}

void Sampler::record_tokens(const std::vector<int32_t>& toks) {
    for (int32_t t : toks) record_token(t);
}

int32_t Sampler::sample_from_f16_bits(const uint16_t* logits_f16, int32_t vocab_size) {
    if (vocab_size <= 0 || !logits_f16) return 0;
    logits_buf_.resize(static_cast<size_t>(vocab_size));
    const auto& lut = fp16_to_f32_lut();
    for (int32_t i = 0; i < vocab_size; ++i) {
        logits_buf_[i] = lut[logits_f16[i]];
    }
    return sample_buf(vocab_size);
}

int32_t Sampler::sample_from_f32(const float* logits_f32, int32_t vocab_size) {
    if (vocab_size <= 0 || !logits_f32) return 0;
    logits_buf_.assign(logits_f32, logits_f32 + vocab_size);
    return sample_buf(vocab_size);
}

int32_t Sampler::sample_buf(int32_t vocab_size) {
    // 1. Apply repetition & presence penalties on recent window.
    const bool do_rep = (cfg_.repeat_penalty > 0.0f && cfg_.repeat_penalty != 1.0f);
    const bool do_pres = (cfg_.presence_penalty != 0.0f);
    if ((do_rep || do_pres) && cfg_.repeat_last_n > 0 && !history_.empty()) {
        if (seen_stamp_.size() < static_cast<size_t>(vocab_size)) {
            seen_stamp_.assign(static_cast<size_t>(vocab_size), 0u);
            cur_stamp_ = 1;
        } else {
            ++cur_stamp_;
            if (cur_stamp_ == 0) {
                std::fill(seen_stamp_.begin(), seen_stamp_.end(), 0u);
                cur_stamp_ = 1;
            }
        }
        const size_t win = std::min(history_.size(), static_cast<size_t>(cfg_.repeat_last_n));
        const size_t start = history_.size() - win;
        for (size_t idx = start; idx < history_.size(); ++idx) {
            const int32_t tok = history_[idx];
            if (tok < 0 || tok >= vocab_size) continue;
            if (seen_stamp_[tok] == cur_stamp_) continue;
            seen_stamp_[tok] = cur_stamp_;
            float& l = logits_buf_[tok];
            if (!std::isfinite(l)) continue;
            if (do_rep) {
                l = (l > 0.0f) ? (l / cfg_.repeat_penalty) : (l * cfg_.repeat_penalty);
            }
            if (do_pres) {
                l -= cfg_.presence_penalty;
            }
        }
    }

    // 2. Greedy argmax when temperature <= 0 or top_k == 1.
    if (cfg_.temperature <= 0.0f || cfg_.top_k == 1) {
        float best_v = -std::numeric_limits<float>::infinity();
        int32_t best_i = 0;
        for (int32_t i = 0; i < vocab_size; ++i) {
            const float v = logits_buf_[i];
            if (v > best_v) {
                best_v = v;
                best_i = i;
            }
        }
        return best_i;
    }

    // 3. Find max finite logit.
    float max_logit = -std::numeric_limits<float>::infinity();
    int32_t max_id = 0;
    for (int32_t i = 0; i < vocab_size; ++i) {
        const float v = logits_buf_[i];
        if (std::isfinite(v) && v > max_logit) {
            max_logit = v;
            max_id = i;
        }
    }
    if (!std::isfinite(max_logit)) return 0;

    const float temp = cfg_.temperature;
    const float inv_temp = 1.0f / temp;
    candidates_.clear();

    // 4. Filter / Top-K selection in O(V) (zero-allocation reuse of candidates_).
    if (cfg_.top_k > 0 && cfg_.top_k < vocab_size) {
        const size_t k = static_cast<size_t>(cfg_.top_k);
        auto cmp_min = [](const Candidate& a, const Candidate& b) {
            if (a.logit != b.logit) return a.logit > b.logit;
            return a.id < b.id;
        };
        candidates_.reserve(k);
        for (int32_t i = 0; i < vocab_size; ++i) {
            const float v = logits_buf_[i];
            if (!std::isfinite(v)) continue;
            if (candidates_.size() < k) {
                candidates_.push_back(Candidate{v, i, 0.0f});
                if (candidates_.size() == k) {
                    std::make_heap(candidates_.begin(), candidates_.end(), cmp_min);
                }
            } else if (v > candidates_.front().logit ||
                       (v == candidates_.front().logit && i < candidates_.front().id)) {
                std::pop_heap(candidates_.begin(), candidates_.end(), cmp_min);
                candidates_.back() = Candidate{v, i, 0.0f};
                std::push_heap(candidates_.begin(), candidates_.end(), cmp_min);
            }
        }
    } else {
        // Cutoff tokens with negligible softmax weight (< ~4e-11 per token relative to peak)
        float logit_cutoff = max_logit - 24.0f * std::max(0.05f, temp);
        if (cfg_.min_p > 0.0f && cfg_.min_p < 1.0f) {
            logit_cutoff = std::max(logit_cutoff, max_logit + temp * std::log(cfg_.min_p));
        }
        for (int32_t i = 0; i < vocab_size; ++i) {
            const float v = logits_buf_[i];
            if (std::isfinite(v) && v >= logit_cutoff) {
                candidates_.push_back(Candidate{v, i, 0.0f});
            }
        }
        if (candidates_.empty()) {
            candidates_.push_back(Candidate{max_logit, max_id, 1.0f});
        }
    }

    if (candidates_.empty()) return 0;

    // 5. Compute unnormalized probabilities exp((logit - max_logit) / temp).
    double sum_p = 0.0;
    for (auto& c : candidates_) {
        c.prob = std::exp((c.logit - max_logit) * inv_temp);
        sum_p += c.prob;
    }

    // 6. Min-P filtering (note peak prob is 1.0 before normalization).
    if (cfg_.min_p > 0.0f && cfg_.min_p < 1.0f && candidates_.size() > 1) {
        const float min_thresh = cfg_.min_p;
        size_t keep = 0;
        for (size_t i = 0; i < candidates_.size(); ++i) {
            if (candidates_[i].prob >= min_thresh || candidates_[i].id == max_id) {
                candidates_[keep++] = candidates_[i];
            }
        }
        if (keep > 0) candidates_.resize(keep);
    }

    // 7. Top-P (nucleus) filtering.
    if (cfg_.top_p > 0.0f && cfg_.top_p < 1.0f && candidates_.size() > 1) {
        std::sort(candidates_.begin(), candidates_.end(), [](const Candidate& a, const Candidate& b) {
            if (a.logit != b.logit) return a.logit > b.logit;
            return a.id < b.id;
        });
        double total = 0.0;
        for (const auto& c : candidates_) total += c.prob;
        const double target = static_cast<double>(cfg_.top_p) * total;
        double cum = 0.0;
        size_t keep = 0;
        for (size_t i = 0; i < candidates_.size(); ++i) {
            cum += candidates_[i].prob;
            keep = i + 1;
            if (cum >= target) break;
        }
        candidates_.resize(std::max<size_t>(1, keep));
        sum_p = cum;
    } else if (cfg_.min_p > 0.0f) {
        sum_p = 0.0;
        for (const auto& c : candidates_) sum_p += c.prob;
    }

    if (candidates_.size() == 1 || sum_p <= 0.0) {
        return candidates_[0].id;
    }

    std::uniform_real_distribution<double> dist(0.0, sum_p);
    double r = dist(rng_);
    for (const auto& c : candidates_) {
        r -= c.prob;
        if (r <= 0.0) return c.id;
    }
    return candidates_.back().id;
}

} // namespace rxinfer

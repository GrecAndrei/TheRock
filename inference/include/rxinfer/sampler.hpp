// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <deque>
#include <random>
#include <vector>

namespace rxinfer {

struct SamplerConfig {
    float temperature = 0.0f;      // <= 0.0f means greedy (GPU argmax fast path when no penalties)
    int32_t top_k = 40;            // <= 0 disables top-k
    float top_p = 0.95f;           // >= 1.0f disables top-p
    float min_p = 0.0f;            // <= 0.0f disables min-p
    float repeat_penalty = 1.0f;   // 1.0f disables repetition penalty
    float presence_penalty = 0.0f; // 0.0f disables presence penalty
    int32_t repeat_last_n = 64;    // token window for penalties (<= 0 disables)
    uint64_t seed = 0;             // 0 = random seed, >0 = deterministic seed

    bool uses_host_logits() const {
        return temperature > 0.0f || repeat_penalty != 1.0f || presence_penalty != 0.0f;
    }
};

class Sampler {
public:
    explicit Sampler(SamplerConfig cfg = {});

    void set_config(const SamplerConfig& cfg);
    const SamplerConfig& config() const { return cfg_; }

    void clear_history();
    void record_token(int32_t tok);
    void record_tokens(const std::vector<int32_t>& toks);
    void reseed(uint64_t seed);

    int32_t sample_from_f16_bits(const uint16_t* logits_f16, int32_t vocab_size);
    int32_t sample_from_f32(const float* logits_f32, int32_t vocab_size);

private:
    int32_t sample_buf(int32_t vocab_size);

    SamplerConfig cfg_{};
    std::mt19937_64 rng_;
    std::deque<int32_t> history_;

    std::vector<float> logits_buf_;
    std::vector<uint32_t> seen_stamp_;
    uint32_t cur_stamp_ = 1;

    struct Candidate {
        float logit;
        int32_t id;
        float prob;
    };
    std::vector<Candidate> candidates_;
};

} // namespace rxinfer

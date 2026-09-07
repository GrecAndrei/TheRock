#pragma once

#include <cstddef>
#include <cstdint>

namespace qwen_native {

struct Q4KBenchmarkResult {
    float milliseconds = 0.0F;
    float bandwidth_gbps = 0.0F;
    float max_absolute_error = 0.0F;
    float relative_rmse = 0.0F;
    float kernel_max_absolute_error = 0.0F;
};

struct FfnBenchmarkResult {
    float milliseconds = 0.0F;
    float weight_bandwidth_gbps = 0.0F;
    float verifier_tokens_per_second = 0.0F;
    float max_absolute_error = 0.0F;
};

struct ResidentWeightCacheStats {
    std::size_t budget_bytes = 0;
    std::size_t resident_bytes = 0;
    std::size_t matrix_count = 0;
};

Q4KBenchmarkResult benchmark_q4k(const void *weights, std::size_t input_size,
                                 std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_q4k_batch8(const void *weights, std::size_t input_size,
                                        std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_q5k(const void *weights, std::size_t input_size,
                                 std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_q5k_batch8(const void *weights, std::size_t input_size,
                                        std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_q6k(const void *weights, std::size_t input_size,
                                 std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_q6k_batch8(const void *weights, std::size_t input_size,
                                        std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_host_mapped_batch8(const void *weights, std::uint32_t type,
                                                std::size_t input_size,
                                                std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_n23q4_from_q4k(const void *weights, std::size_t input_size,
                                             std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_n23q4_batch8_from_q4k(const void *weights, std::size_t input_size,
                                                    std::size_t output_size, int iterations);
Q4KBenchmarkResult benchmark_n23kv4(std::size_t context_size, int iterations);
Q4KBenchmarkResult benchmark_n23kv4_batch8(std::size_t context_size, int iterations);
Q4KBenchmarkResult benchmark_n23kv4_flash_batch8(std::size_t context_size,
                                                  int partitions, int iterations);
Q4KBenchmarkResult benchmark_gated_delta_step(int iterations);
FfnBenchmarkResult benchmark_ffn_batch8(const void *gate_weights, std::uint32_t gate_type,
                                         const void *up_weights, std::uint32_t up_type,
                                         const void *down_weights, std::uint32_t down_type,
                                         std::size_t input_size, std::size_t hidden_size,
                                         std::size_t output_size, int iterations);
FfnBenchmarkResult benchmark_ffn_batch8_n23q4(const void *gate_weights,
                                               const void *up_weights,
                                               const void *down_weights,
                                               std::uint32_t down_type,
                                               std::size_t input_size,
                                               std::size_t hidden_size,
                                               std::size_t output_size, int iterations);
void execute_k_matvec(const void *weights, std::uint32_t weight_type,
                      const float *input, std::size_t input_size,
                      float *output, std::size_t output_size);
void set_resident_weight_cache_budget(std::size_t bytes);
void clear_resident_weight_cache();
ResidentWeightCacheStats resident_weight_cache_stats();
bool preload_resident_k_matrix(const void *weights, std::uint32_t weight_type,
                               std::size_t input_size, std::size_t output_size);
void execute_k_matvec_cached(const void *weights, std::uint32_t weight_type,
                             const float *input, std::size_t input_size,
                             float *output, std::size_t output_size,
                             bool keep_resident);
void execute_k_matvec_batch8(const void *weights, std::uint32_t weight_type,
                             const float *input, std::size_t input_size,
                             float *output, std::size_t output_size);
// Shares the scalar executor's resident-weight cache.  The native verifier
// therefore streams a cold matrix once per 8-token block, not once per token.
void execute_k_matvec_cached_batch8(const void *weights, std::uint32_t weight_type,
                                    const float *input, std::size_t input_size,
                                    float *output, std::size_t output_size,
                                    bool keep_resident);
void execute_k_matvec_hybrid_cached_batch8(const void *weights, std::uint32_t weight_type,
                                           const float *input, std::size_t input_size,
                                           float *output, std::size_t output_size,
                                           bool keep_resident);
// GPU top-16: matvec + device-side reduction, downloads only 16*8 indices/scores.
void execute_k_matvec_cached_batch8_top16(const void *weights, std::uint32_t weight_type,
                                           const float *input, std::size_t input_size,
                                           int *out_indices, float *out_scores,
                                           std::size_t output_size, bool keep_resident);

// Persistent FP16 gated-delta state for the 48 x 128 x 128 recurrent target
// heads.  The scalar reference keeps an FP32 mirror; the batched verifier uses
// this device state to avoid running hundreds of millions of recurrence FLOPs
// on the CPU for every proposed block.
void *create_gated_delta_state();
void destroy_gated_delta_state(void *state);
void upload_gated_delta_state(void *state, const float *host_state);
void download_gated_delta_state(const void *state, float *host_state);
void execute_gated_delta_step(const float *queries, const float *keys,
                              const float *values, const float *decays,
                              const float *betas, void *state, float *output);
void execute_gated_delta_step_batch8(const float *queries, const float *keys,
                                     const float *values, const float *decays,
                                     const float *betas, void *state, float *output,
                                     int batch_count);
// N23Q4 is the draft-only, verification-safe RDNA2 layout: DFlash proposals
// may be approximate because the target batch verifier accepts or rejects
// every token.  Its Q8 activations unlock the gfx1032 DP4A path.
void preload_n23q4_matrix(const void *weights, std::size_t input_size, std::size_t output_size);
bool has_n23q4_matrix(const void *weights);
void execute_n23q4_batch8_cached(const void *weights, const float *input,
                                  std::size_t input_size, std::size_t output_size,
                                  float *output);

// Device-resident pipelined executor for the batch-8 target verifier.
// Keeps hidden states on-device and overlaps PCIe weight transfer with
// kernel execution using two streams and double-buffered weight staging.
struct DevicePipeline;
DevicePipeline *create_device_pipeline(std::size_t embedding_size, std::size_t ffn_size,
                                       std::size_t max_weight_bytes, std::size_t max_context = 4096);
void destroy_device_pipeline(DevicePipeline *pipeline);

// Upload/download the batch-8 hidden state (8 × embedding_size floats).
void pipeline_upload_hidden(DevicePipeline *pipeline, const float *host, std::size_t count);
void pipeline_download_hidden(DevicePipeline *pipeline, float *host, std::size_t count);
void pipeline_download_hidden_async(DevicePipeline *pipeline, float *host, std::size_t count);

// RMS norm on pipeline hidden buffer or out-of-place from source to destination.
void pipeline_rms_norm(DevicePipeline *pipeline, const float *norm_weights,
                       std::size_t width, float epsilon,
                       float *destination = nullptr, const float *source = nullptr);

// Residual add: hidden += source (both device-resident).
void pipeline_residual_add(DevicePipeline *pipeline, int source_slot);

// Buffer pointer accessors
float *pipeline_slot_ptr(DevicePipeline *pipeline, int slot);
float *pipeline_hidden_ptr(DevicePipeline *pipeline);

// Run a batch-8 projection from input_buffer (defaulting to hidden) into an output slot.
// Uses the resident weight cache when available; otherwise streams via
// the double-buffer on the transfer stream.
void pipeline_project_batch8(DevicePipeline *pipeline, const void *weights,
                             std::uint32_t type, std::size_t input_size,
                             std::size_t output_size, bool keep_resident, int output_slot,
                             const float *input_buffer = nullptr);
// Exact row split between CPU AVX2 and GPU. Intended for streamed target
// projections; resident matrices stay on the ordinary GPU path.
void pipeline_project_batch8_hybrid(DevicePipeline *pipeline, const void *weights,
                                    std::uint32_t type, std::size_t input_size,
                                    std::size_t output_size, bool keep_resident,
                                    int output_slot,
                                    const float *input_buffer = nullptr);
void pipeline_project_batch_large(DevicePipeline *pipeline, const void *weights,
                                   std::uint32_t type, std::size_t input_size,
                                   std::size_t output_size, std::size_t batch_size,
                                   bool keep_resident, int output_slot,
                                   const float *input_buffer = nullptr);
std::size_t hybrid_cpu_row_count(std::uint32_t type, std::size_t output_size);
std::size_t hybrid_cpu_row_count(std::uint32_t type, std::size_t output_size, std::size_t max_context);
void pipeline_project_attention_qkv_hybrid(
    DevicePipeline *pipeline,
    const void *q_weights, std::uint32_t q_type, std::size_t q_size,
    const void *k_weights, std::uint32_t k_type, std::size_t k_size,
    const void *v_weights, std::uint32_t v_type, std::size_t v_size,
    std::size_t input_size, bool keep_resident,
    const float *input_buffer = nullptr);
void pipeline_project_recurrent_qkv_gate_cpu(DevicePipeline *pipeline,
                                             const void *qkv_weights, std::uint32_t qkv_type,
                                             std::size_t input_size, std::size_t qkv_size,
                                             const void *gate_weights, std::uint32_t gate_type,
                                             std::size_t gate_size, bool keep_resident,
                                             const float *input_buffer = nullptr);

// Run SiLU-gated multiply: out_slot = silu(gate_slot) * up_slot.
void pipeline_silu_multiply(DevicePipeline *pipeline, int gate_slot, int up_slot,
                            std::size_t count);

// Copy an output slot back into the hidden buffer (for FFN down output → hidden).
void pipeline_slot_to_hidden(DevicePipeline *pipeline, int slot, std::size_t count);

// Save hidden state to a secondary slot (for residual connections).
void pipeline_save_hidden(DevicePipeline *pipeline, int slot, std::size_t count);

// Download a pipeline slot to host memory.
void pipeline_download_slot(DevicePipeline *pipeline, int slot, float *host, std::size_t count);
void pipeline_download_slots3(DevicePipeline *pipeline,
                              int slot0, float *host0, std::size_t count0,
                              int slot1, float *host1, std::size_t count1,
                              int slot2, float *host2, std::size_t count2);

// Synchronize both streams (call before reading results or at end of block).
void pipeline_sync(DevicePipeline *pipeline);

// Begin async prefetch of a weight matrix (returns immediately).
void pipeline_prefetch_weights(DevicePipeline *pipeline, const void *weights,
                               std::uint32_t type, std::size_t input_size,
                               std::size_t output_size);
void pipeline_prefetch_hybrid_weights(DevicePipeline *pipeline, const void *weights,
                                      std::uint32_t type, std::size_t input_size,
                                      std::size_t output_size);

// Prefetch a small matrix on the dedicated inline buffer (no conflict with main bufs).
void pipeline_prefetch_inline(DevicePipeline *pipeline, const void *weights,
                              std::uint32_t type, std::size_t input_size,
                              std::size_t output_size);

// Run a full fused FFN layer batch-8 on the pipeline (norm → gate → up → silu → down → add).
// Hidden state is updated in-place. norm_weights are the post-attention norm.
void pipeline_fused_ffn_batch8(DevicePipeline *pipeline, const float *norm_weights,
                               std::size_t embedding_size, std::size_t ffn_size, float epsilon,
                               const void *gate_weights, std::uint32_t gate_type,
                               const void *up_weights, std::uint32_t up_type,
                               const void *down_weights, std::uint32_t down_type,
                               bool keep_gate_up_resident);
void pipeline_fused_ffn_batch8_hybrid_up(DevicePipeline *pipeline, const float *norm_weights,
                                         std::size_t embedding_size, std::size_t ffn_size,
                                         float epsilon,
                                         const void *gate_weights, std::uint32_t gate_type,
                                         const void *up_weights, std::uint32_t up_type,
                                         const void *down_weights, std::uint32_t down_type,
                                         bool keep_gate_resident);
void pipeline_fused_ffn_batch8_hybrid_soa(DevicePipeline *pipeline, const float *norm_weights,
                                          std::size_t embedding_size, std::size_t ffn_size,
                                          float epsilon,
                                          const void *gate_weights, std::uint32_t gate_type,
                                          const void *up_weights, std::uint32_t up_type,
                                          const void *down_weights, std::uint32_t down_type,
                                          bool keep_gate_up_resident);

// GPU-native GQA attention for the batch-8 target verifier.
// FP16 KV cache on device, eliminates CPU attention round-trips.
struct AttentionDeviceState;
AttentionDeviceState *create_attention_device_state(int num_layers, int max_context,
                                                     float rope_freq_base);
void destroy_attention_device_state(AttentionDeviceState *state);
void attention_reset(AttentionDeviceState *state);
void attention_advance(AttentionDeviceState *state, int tokens);
void pipeline_attention_batch8(DevicePipeline *pipeline, AttentionDeviceState *attn,
                               int layer_index,
                               const float *q_norm_weights, const float *k_norm_weights,
                               float epsilon, int batch_count = 8);
// Synchronize the scalar reference cache (token-major K then V) with the
// device verifier cache.  A batch verifier must never begin from an empty or
// stale attention history after scalar prompt prefill.
void attention_upload_history(AttentionDeviceState *state, int layer_index,
                              const std::uint16_t *keys, const std::uint16_t *values,
                              std::size_t token_count);
void attention_download_history(const AttentionDeviceState *state, int layer_index,
                                std::uint16_t *keys, std::uint16_t *values,
                                std::size_t token_count);
void attention_set_history_length(AttentionDeviceState *state, int token_count);

// GPU-native recurrent: all conv1d/decay/gdn/norm stay on device.
struct RecurrentDeviceState;
RecurrentDeviceState *create_recurrent_device_state(int num_layers, int qkv_width,
                                                    int recurrent_width, int embedding_size,
                                                    int value_heads, int head_size);
void destroy_recurrent_device_state(RecurrentDeviceState *state);
void recurrent_upload_weights(RecurrentDeviceState *state, int layer,
                              const float *conv_weights, const float *beta_weights,
                              const float *alpha_weights, const float *dt_bias,
                              const float *a_weights, const float *norm_weights);
void recurrent_reset_conv_states(RecurrentDeviceState *state);
void pipeline_recurrent_batch8(DevicePipeline *pipeline, RecurrentDeviceState *recurrent,
                               int layer_index, void *gdn_state, float epsilon,
                               int batch_count = 8,
                               const float *normalized_hidden = nullptr,
                               float *output_buffer = nullptr);
// Restore the checkpoint matrix/convolution history on the caller side, then
// replay only the inexpensive recurrent updates captured by the last batch.
// No quantized target projection is rerun.
void recurrent_commit_captured_prefix(RecurrentDeviceState *state, int layer_index,
                                      void *gdn_state, int committed_tokens);
// Keep scalar prompt prefill and batch verification on the same recurrent
// convolution history.  The gated-delta matrix is synchronized separately by
// the target session when it changes execution modes.
void recurrent_upload_convolution_state(RecurrentDeviceState *state, int layer_index,
                                        const float *history, std::size_t value_count);
void recurrent_download_convolution_state(const RecurrentDeviceState *state, int layer_index,
                                          float *history, std::size_t value_count);

} // namespace qwen_native

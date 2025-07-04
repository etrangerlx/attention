#pragma once

#include "api/attention_interface.hpp"
#include "api/tensor.hpp"
#include <memory>
#include <vector>
#include <string>
#include <cmath>

namespace attention_hpc {

// Forward declarations
template<typename T>
class Tensor;

// FlashAttention algorithm parameters
struct FlashAttentionConfig {
    // Block sizes for tiling strategy
    size_t block_size_q = 64;  // Query block size
    size_t block_size_k = 64;  // Key block size
    size_t block_size_v = 64;  // Value block size
    
    // Memory optimization settings
    bool use_causal_mask = false;
    bool apply_dropout = false;
    float dropout_probability = 0.0f;
    
    // Numerical stability settings
    bool use_safe_softmax = true;
    float softmax_scale = 0.0f; // 0 means auto-compute as 1/sqrt(head_dim)
    
    // Performance tuning
    bool prefetch_next_block = true;
    bool use_async_copy = false;
    size_t num_parallel_blocks = 1;
};

// Online softmax state for memory-efficient computation
struct OnlineSoftmaxState {
    float max_val;
    float sum_exp;
    
    OnlineSoftmaxState() : max_val(-std::numeric_limits<float>::infinity()), sum_exp(0.0f) {}
    
    void reset() {
        max_val = -std::numeric_limits<float>::infinity();
        sum_exp = 0.0f;
    }
    
    void update(float new_max, float new_sum) {
        if (new_max > max_val) {
            sum_exp = sum_exp * std::exp(max_val - new_max) + new_sum;
            max_val = new_max;
        } else {
            sum_exp += new_sum * std::exp(new_max - max_val);
        }
    }
    
    float get_normalizer() const {
        return sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
    }
};

// Block-wise attention computation state
struct AttentionBlockState {
    OnlineSoftmaxState softmax_state;
    std::vector<float> output_buffer;
    std::vector<float> attention_weights_buffer;
    size_t current_block_idx;
    
    AttentionBlockState(size_t output_size, size_t attention_size) 
        : output_buffer(output_size, 0.0f), 
          attention_weights_buffer(attention_size, 0.0f),
          current_block_idx(0) {}
    
    void reset() {
        softmax_state.reset();
        std::fill(output_buffer.begin(), output_buffer.end(), 0.0f);
        std::fill(attention_weights_buffer.begin(), attention_weights_buffer.end(), 0.0f);
        current_block_idx = 0;
    }
};

// FlashAttention implementation class
class FlashAttention : public AttentionInterface {
public:
    FlashAttention();
    virtual ~FlashAttention() = default;
    
    // Configuration methods (inherited from AttentionInterface)
    void configure(const AttentionParams& params, 
                   const PerformanceConfig& perf_config = PerformanceConfig{}) override;
    
    // FlashAttention specific configuration
    void configure_flash_attention(const FlashAttentionConfig& flash_config);
    
    // Core computation methods (inherited from AttentionInterface)
    void forward(const TensorShape& query_shape, const void* query_data,
                 const TensorShape& key_shape, const void* key_data,
                 const TensorShape& value_shape, const void* value_data,
                 const TensorShape& output_shape, void* output_data,
                 const void* attention_mask = nullptr) override;
    
    void backward(const TensorShape& grad_output_shape, const void* grad_output_data,
                  const TensorShape& query_shape, const void* query_data,
                  const TensorShape& key_shape, const void* key_data,
                  const TensorShape& value_shape, const void* value_data,
                  const TensorShape& grad_query_shape, void* grad_query_data,
                  const TensorShape& grad_key_shape, void* grad_key_data,
                  const TensorShape& grad_value_shape, void* grad_value_data,
                  const void* attention_mask = nullptr) override;
    
    // Memory management (inherited from AttentionInterface)
    size_t get_workspace_size(const AttentionParams& params) const override;
    void set_workspace(void* workspace_ptr, size_t workspace_size) override;
    
    // Query methods (inherited from AttentionInterface)
    bool supports_data_type(DataType data_type) const override;
    bool supports_memory_layout(MemoryLayout layout) const override;
    std::string get_backend_name() const override;
    std::string get_version() const override;
    
    // Performance and profiling (inherited from AttentionInterface)
    void enable_profiling(bool enable) override;
    std::vector<std::pair<std::string, double>> get_profiling_results() const override;
    void reset_profiling() override;
    
    // Validation methods (inherited from AttentionInterface)
    AttentionError validate_inputs(const TensorShape& query_shape,
                                   const TensorShape& key_shape,
                                   const TensorShape& value_shape,
                                   const TensorShape& output_shape,
                                   const AttentionParams& params) const override;

protected:
    // Core FlashAttention algorithm components
    
    // Block-wise attention computation
    virtual void compute_attention_block(
        const float* query_block, const float* key_block, const float* value_block,
        float* output_block, AttentionBlockState& block_state,
        size_t q_block_size, size_t kv_block_size, size_t head_dim,
        bool is_causal, const float* attention_mask_block = nullptr);
    
    // Online softmax computation
    virtual void compute_online_softmax(
        const float* attention_scores, float* attention_weights,
        OnlineSoftmaxState& softmax_state,
        size_t num_keys, float scale_factor);
    
    // Multi-head attention computation
    virtual void compute_multihead_attention(
        const TensorShape& query_shape, const float* query_data,
        const TensorShape& key_shape, const float* key_data,
        const TensorShape& value_shape, const float* value_data,
        const TensorShape& output_shape, float* output_data,
        const float* attention_mask = nullptr);
    
    // Block scheduling and tiling
    virtual void compute_optimal_block_sizes(
        size_t seq_len_q, size_t seq_len_kv, size_t head_dim,
        size_t available_memory, FlashAttentionConfig& config);
    
    virtual std::vector<std::pair<size_t, size_t>> schedule_attention_blocks(
        size_t seq_len_q, size_t seq_len_kv, 
        size_t block_size_q, size_t block_size_k);
    
    // Gradient computation for backward pass
    virtual void compute_query_gradients(
        const float* grad_output, const float* key_data, const float* value_data,
        const float* attention_weights, float* grad_query,
        size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim);
    
    virtual void compute_key_value_gradients(
        const float* grad_output, const float* query_data,
        const float* attention_weights, float* grad_key, float* grad_value,
        size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim);
    
    // Utility methods
    virtual void apply_causal_mask(float* attention_scores, 
                                   size_t seq_len_q, size_t seq_len_k,
                                   size_t q_offset, size_t k_offset);
    
    virtual void apply_attention_mask(float* attention_scores,
                                      const float* mask_data,
                                      size_t seq_len_q, size_t seq_len_k,
                                      size_t q_offset, size_t k_offset);
    
    virtual float compute_attention_scale(size_t head_dim) const;
    
    // Memory layout utilities
    virtual void transpose_for_attention(const float* input, float* output,
                                         size_t batch_size, size_t seq_len,
                                         size_t num_heads, size_t head_dim);
    
    virtual void transpose_from_attention(const float* input, float* output,
                                          size_t batch_size, size_t seq_len,
                                          size_t num_heads, size_t head_dim);
    
    // Data type conversion utilities
    virtual void convert_to_float(const void* input, float* output,
                                  size_t num_elements, DataType input_type);
    
    virtual void convert_from_float(const float* input, void* output,
                                    size_t num_elements, DataType output_type);

private:
    // Configuration state
    FlashAttentionConfig flash_config_;
    bool is_configured_;
    
    // Workspace management
    void* workspace_ptr_;
    size_t workspace_size_;
    size_t workspace_offset_;
    
    // Profiling state
    bool profiling_enabled_;
    std::vector<std::pair<std::string, double>> profiling_results_;
    
    // Temporary buffers for computation
    std::unique_ptr<float[]> query_buffer_;
    std::unique_ptr<float[]> key_buffer_;
    std::unique_ptr<float[]> value_buffer_;
    std::unique_ptr<float[]> output_buffer_;
    std::unique_ptr<float[]> attention_scores_buffer_;
    
    // Buffer sizes
    size_t query_buffer_size_;
    size_t key_buffer_size_;
    size_t value_buffer_size_;
    size_t output_buffer_size_;
    size_t attention_scores_buffer_size_;
    
    // Helper methods for memory management
    void allocate_buffers(const AttentionParams& params);
    void deallocate_buffers();
    void* allocate_workspace(size_t size, size_t alignment = 64);
    
    // Validation helpers
    bool validate_tensor_shapes(const TensorShape& query_shape,
                                const TensorShape& key_shape,
                                const TensorShape& value_shape) const;
    
    bool validate_attention_dimensions(const AttentionParams& params) const;
    
    // Performance optimization helpers
    void optimize_block_sizes_for_cache(FlashAttentionConfig& config,
                                        size_t cache_size) const;
    
    void estimate_memory_requirements(const AttentionParams& params,
                                      const FlashAttentionConfig& config,
                                      size_t& workspace_size,
                                      size_t& peak_memory) const;
    
    // Profiling helpers
    void profile_operation(const std::string& operation_name,
                           std::function<void()> operation);
    
    double get_current_time() const;
};

// Template specializations for different data types
template<typename T>
class TypedFlashAttention : public FlashAttention {
public:
    TypedFlashAttention() = default;
    virtual ~TypedFlashAttention() = default;
    
    // Type-specific implementations
    void forward_typed(const Tensor<T>& query, const Tensor<T>& key, 
                       const Tensor<T>& value, Tensor<T>& output,
                       const Tensor<T>* attention_mask = nullptr);
    
    void backward_typed(const Tensor<T>& grad_output,
                        const Tensor<T>& query, const Tensor<T>& key, const Tensor<T>& value,
                        Tensor<T>& grad_query, Tensor<T>& grad_key, Tensor<T>& grad_value,
                        const Tensor<T>* attention_mask = nullptr);

protected:
    // Type-specific computation kernels
    virtual void compute_attention_block_typed(
        const T* query_block, const T* key_block, const T* value_block,
        T* output_block, AttentionBlockState& block_state,
        size_t q_block_size, size_t kv_block_size, size_t head_dim,
        bool is_causal, const T* attention_mask_block = nullptr);
    
    virtual void compute_online_softmax_typed(
        const T* attention_scores, T* attention_weights,
        OnlineSoftmaxState& softmax_state,
        size_t num_keys, T scale_factor);
};

// Type aliases for common instantiations
using FlashAttentionFloat = TypedFlashAttention<float>;
using FlashAttentionHalf = TypedFlashAttention<half>;
using FlashAttentionBFloat16 = TypedFlashAttention<bfloat16>;

// Utility functions for FlashAttention
namespace flash_attention_utils {
    // Memory estimation utilities
    size_t estimate_flash_attention_memory(const AttentionParams& params,
                                           const FlashAttentionConfig& config);
    
    // Optimal configuration detection
    FlashAttentionConfig get_optimal_config(const AttentionParams& params,
                                            size_t available_memory);
    
    // Block size calculation utilities
    std::tuple<size_t, size_t, size_t> calculate_optimal_block_sizes(
        size_t seq_len, size_t head_dim, size_t available_memory);
    
    // Causal mask utilities
    bool needs_causal_mask(const AttentionParams& params);
    
    // Performance benchmarking
    double benchmark_flash_attention_config(const AttentionParams& params,
                                            const FlashAttentionConfig& config,
                                            int num_iterations = 10);
}

} // namespace attention_hpc

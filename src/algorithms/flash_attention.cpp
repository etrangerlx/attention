#include "flash_attention.hpp"
#include <algorithm>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <immintrin.h>

namespace attention_hpc {

FlashAttention::FlashAttention() 
    : is_configured_(false), workspace_ptr_(nullptr), workspace_size_(0), workspace_offset_(0),
      profiling_enabled_(false), query_buffer_(nullptr), key_buffer_(nullptr), value_buffer_(nullptr),
      output_buffer_(nullptr), attention_scores_buffer_(nullptr),
      query_buffer_size_(0), key_buffer_size_(0), value_buffer_size_(0),
      output_buffer_size_(0), attention_scores_buffer_size_(0) {
    
    // Set default FlashAttention configuration
    flash_config_.block_size_q = 64;
    flash_config_.block_size_k = 64;
    flash_config_.block_size_v = 64;
    flash_config_.use_causal_mask = false;
    flash_config_.apply_dropout = false;
    flash_config_.dropout_probability = 0.0f;
    flash_config_.use_safe_softmax = true;
    flash_config_.softmax_scale = 0.0f;
    flash_config_.prefetch_next_block = true;
    flash_config_.use_async_copy = false;
    flash_config_.num_parallel_blocks = 1;
}

void FlashAttention::configure(const AttentionParams& params, const PerformanceConfig& perf_config) {
    current_params_ = params;
    current_perf_config_ = perf_config;
    
    // Validate parameters
    if (params.batch_size == 0 || params.sequence_length == 0 || 
        params.num_heads == 0 || params.head_dimension == 0) {
        throw AttentionException(AttentionError::INVALID_PARAMETERS, 
                                "Invalid attention parameters: dimensions must be positive");
    }
    
    // Compute optimal block sizes if not manually configured
    compute_optimal_block_sizes(params.sequence_length, params.sequence_length, 
                                params.head_dimension, current_perf_config_.max_workspace_size,
                                flash_config_);
    
    // Set softmax scale if not provided
    if (flash_config_.softmax_scale == 0.0f) {
        flash_config_.softmax_scale = compute_attention_scale(params.head_dimension);
    }
    
    // Allocate necessary buffers
    allocate_buffers(params);
    
    is_configured_ = true;
}

void FlashAttention::configure_flash_attention(const FlashAttentionConfig& flash_config) {
    flash_config_ = flash_config;
    
    if (is_configured_) {
        // Reallocate buffers with new configuration
        allocate_buffers(current_params_);
    }
}

void FlashAttention::forward(const TensorShape& query_shape, const void* query_data,
                            const TensorShape& key_shape, const void* key_data,
                            const TensorShape& value_shape, const void* value_data,
                            const TensorShape& output_shape, void* output_data,
                            const void* attention_mask) {
    
    if (!is_configured_) {
        throw AttentionException(AttentionError::INVALID_PARAMETERS, 
                                "FlashAttention not configured. Call configure() first.");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Convert input data to float for computation
    size_t query_elements = query_shape.total_elements();
    size_t key_elements = key_shape.total_elements();
    size_t value_elements = value_shape.total_elements();
    
    if (query_buffer_size_ < query_elements || key_buffer_size_ < key_elements || 
        value_buffer_size_ < value_elements) {
        throw AttentionException(AttentionError::MEMORY_ALLOCATION_FAILED,
                                "Buffer sizes insufficient for input tensors");
    }
    
    convert_to_float(query_data, query_buffer_.get(), query_elements, query_shape.data_type);
    convert_to_float(key_data, key_buffer_.get(), key_elements, key_shape.data_type);
    convert_to_float(value_data, value_buffer_.get(), value_elements, value_shape.data_type);
    
    // Perform multi-head attention computation
    compute_multihead_attention(query_shape, query_buffer_.get(),
                               key_shape, key_buffer_.get(),
                               value_shape, value_buffer_.get(),
                               output_shape, output_buffer_.get(),
                               static_cast<const float*>(attention_mask));
    
    // Convert output back to desired data type
    convert_from_float(output_buffer_.get(), output_data, 
                       output_shape.total_elements(), output_shape.data_type);
    
    if (profiling_enabled_) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        profiling_results_.emplace_back("forward_pass", duration.count() / 1000.0);
    }
}

void FlashAttention::backward(const TensorShape& grad_output_shape, const void* grad_output_data,
                             const TensorShape& query_shape, const void* query_data,
                             const TensorShape& key_shape, const void* key_data,
                             const TensorShape& value_shape, const void* value_data,
                             const TensorShape& grad_query_shape, void* grad_query_data,
                             const TensorShape& grad_key_shape, void* grad_key_data,
                             const TensorShape& grad_value_shape, void* grad_value_data,
                             const void* attention_mask) {
    
    if (!is_configured_) {
        throw AttentionException(AttentionError::INVALID_PARAMETERS,
                                "FlashAttention not configured. Call configure() first.");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Convert input data to float
    convert_to_float(grad_output_data, output_buffer_.get(), 
                     grad_output_shape.total_elements(), grad_output_shape.data_type);
    convert_to_float(query_data, query_buffer_.get(), 
                     query_shape.total_elements(), query_shape.data_type);
    convert_to_float(key_data, key_buffer_.get(), 
                     key_shape.total_elements(), key_shape.data_type);
    convert_to_float(value_data, value_buffer_.get(), 
                     value_shape.total_elements(), value_shape.data_type);
    
    // Allocate temporary buffers for gradients
    std::unique_ptr<float[]> grad_query_buffer(new float[grad_query_shape.total_elements()]);
    std::unique_ptr<float[]> grad_key_buffer(new float[grad_key_shape.total_elements()]);
    std::unique_ptr<float[]> grad_value_buffer(new float[grad_value_shape.total_elements()]);
    
    // Initialize gradient buffers to zero
    std::fill(grad_query_buffer.get(), grad_query_buffer.get() + grad_query_shape.total_elements(), 0.0f);
    std::fill(grad_key_buffer.get(), grad_key_buffer.get() + grad_key_shape.total_elements(), 0.0f);
    std::fill(grad_value_buffer.get(), grad_value_buffer.get() + grad_value_shape.total_elements(), 0.0f);
    
    // Compute gradients
    compute_query_gradients(output_buffer_.get(), key_buffer_.get(), value_buffer_.get(),
                           attention_scores_buffer_.get(), grad_query_buffer.get(),
                           current_params_.batch_size, current_params_.num_heads,
                           current_params_.sequence_length, current_params_.head_dimension);
    
    compute_key_value_gradients(output_buffer_.get(), query_buffer_.get(),
                               attention_scores_buffer_.get(), grad_key_buffer.get(), grad_value_buffer.get(),
                               current_params_.batch_size, current_params_.num_heads,
                               current_params_.sequence_length, current_params_.head_dimension);
    
    // Convert gradients back to desired data types
    convert_from_float(grad_query_buffer.get(), grad_query_data,
                       grad_query_shape.total_elements(), grad_query_shape.data_type);
    convert_from_float(grad_key_buffer.get(), grad_key_data,
                       grad_key_shape.total_elements(), grad_key_shape.data_type);
    convert_from_float(grad_value_buffer.get(), grad_value_data,
                       grad_value_shape.total_elements(), grad_value_shape.data_type);
    
    if (profiling_enabled_) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        profiling_results_.emplace_back("backward_pass", duration.count() / 1000.0);
    }
}

size_t FlashAttention::get_workspace_size(const AttentionParams& params) const {
    size_t total_size = 0;
    
    // Calculate memory requirements for buffers
    size_t seq_len = params.sequence_length;
    size_t head_dim = params.head_dimension;
    size_t num_heads = params.num_heads;
    size_t batch_size = params.batch_size;
    
    // Query, Key, Value buffers
    total_size += batch_size * seq_len * num_heads * head_dim * sizeof(float) * 3;
    
    // Output buffer
    total_size += batch_size * seq_len * num_heads * head_dim * sizeof(float);
    
    // Attention scores buffer (block-wise)
    total_size += std::max(flash_config_.block_size_q, seq_len) * 
                  std::max(flash_config_.block_size_k, seq_len) * sizeof(float);
    
    // Additional workspace for intermediate computations
    total_size += seq_len * head_dim * sizeof(float) * 2; // Temporary buffers
    
    return total_size;
}

void FlashAttention::set_workspace(void* workspace_ptr, size_t workspace_size) {
    workspace_ptr_ = workspace_ptr;
    workspace_size_ = workspace_size;
    workspace_offset_ = 0;
}

bool FlashAttention::supports_data_type(DataType data_type) const {
    return data_type == DataType::FLOAT32 || data_type == DataType::FLOAT16 || data_type == DataType::BFLOAT16;
}

bool FlashAttention::supports_memory_layout(MemoryLayout layout) const {
    return layout == MemoryLayout::ROW_MAJOR || layout == MemoryLayout::COLUMN_MAJOR;
}

std::string FlashAttention::get_backend_name() const {
    return "CPU_FlashAttention";
}

std::string FlashAttention::get_version() const {
    return "1.0.0";
}

void FlashAttention::enable_profiling(bool enable) {
    profiling_enabled_ = enable;
    if (!enable) {
        profiling_results_.clear();
    }
}

std::vector<std::pair<std::string, double>> FlashAttention::get_profiling_results() const {
    return profiling_results_;
}

void FlashAttention::reset_profiling() {
    profiling_results_.clear();
}

AttentionError FlashAttention::validate_inputs(const TensorShape& query_shape,
                                              const TensorShape& key_shape,
                                              const TensorShape& value_shape,
                                              const TensorShape& output_shape,
                                              const AttentionParams& params) const {
    
    // Validate tensor dimensions
    if (query_shape.dimensions.size() != 4 || key_shape.dimensions.size() != 4 ||
        value_shape.dimensions.size() != 4 || output_shape.dimensions.size() != 4) {
        return AttentionError::INVALID_INPUT;
    }
    
    // Validate batch dimensions
    if (query_shape.dimensions[0] != params.batch_size ||
        key_shape.dimensions[0] != params.batch_size ||
        value_shape.dimensions[0] != params.batch_size ||
        output_shape.dimensions[0] != params.batch_size) {
        return AttentionError::INVALID_INPUT;
    }
    
    // Validate head dimensions
    if (query_shape.dimensions[2] != params.num_heads ||
        key_shape.dimensions[2] != params.num_heads ||
        value_shape.dimensions[2] != params.num_heads ||
        output_shape.dimensions[2] != params.num_heads) {
        return AttentionError::INVALID_INPUT;
    }
    
    // Validate feature dimensions
    if (query_shape.dimensions[3] != params.head_dimension ||
        key_shape.dimensions[3] != params.head_dimension ||
        value_shape.dimensions[3] != params.head_dimension ||
        output_shape.dimensions[3] != params.head_dimension) {
        return AttentionError::INVALID_INPUT;
    }
    
    return AttentionError::SUCCESS;
}

void FlashAttention::compute_attention_block(const float* query_block, const float* key_block, 
                                            const float* value_block, float* output_block,
                                            AttentionBlockState& block_state,
                                            size_t q_block_size, size_t kv_block_size, size_t head_dim,
                                            bool is_causal, const float* attention_mask_block) {
    
    // Compute attention scores: QK^T
    std::vector<float> scores(q_block_size * kv_block_size, 0.0f);
    
    for (size_t i = 0; i < q_block_size; ++i) {
        for (size_t j = 0; j < kv_block_size; ++j) {
            float score = 0.0f;
            
            // Vectorized dot product
            const float* q_ptr = query_block + i * head_dim;
            const float* k_ptr = key_block + j * head_dim;
            
            size_t d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 q_vec = _mm256_load_ps(q_ptr + d);
                __m256 k_vec = _mm256_load_ps(k_ptr + d);
                __m256 prod = _mm256_mul_ps(q_vec, k_vec);
                
                // Horizontal sum
                __m128 sum_high = _mm256_extractf128_ps(prod, 1);
                __m128 sum_low = _mm256_castps256_ps128(prod);
                __m128 sum = _mm_add_ps(sum_low, sum_high);
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);
                score += _mm_cvtss_f32(sum);
            }
            
            // Handle remaining elements
            for (; d < head_dim; ++d) {
                score += q_ptr[d] * k_ptr[d];
            }
            
            scores[i * kv_block_size + j] = score * flash_config_.softmax_scale;
        }
    }
    
    // Apply causal mask if needed
    if (is_causal) {
        apply_causal_mask(scores.data(), q_block_size, kv_block_size, 
                         block_state.current_block_idx * flash_config_.block_size_q,
                         block_state.current_block_idx * flash_config_.block_size_k);
    }
    
    // Apply attention mask if provided
    if (attention_mask_block) {
        apply_attention_mask(scores.data(), attention_mask_block, 
                           q_block_size, kv_block_size, 0, 0);
    }
    
    // Compute online softmax
    compute_online_softmax(scores.data(), block_state.attention_weights_buffer.data(),
                          block_state.softmax_state, kv_block_size, flash_config_.softmax_scale);
    
    // Compute attention output: Attention_weights * V
    for (size_t i = 0; i < q_block_size; ++i) {
        for (size_t d = 0; d < head_dim; ++d) {
            float sum = 0.0f;
            
            for (size_t j = 0; j < kv_block_size; ++j) {
                sum += block_state.attention_weights_buffer[i * kv_block_size + j] * 
                       value_block[j * head_dim + d];
            }
            
            output_block[i * head_dim + d] = sum;
        }
    }
    
    block_state.current_block_idx++;
}

void FlashAttention::compute_online_softmax(const float* attention_scores, float* attention_weights,
                                           OnlineSoftmaxState& softmax_state,
                                           size_t num_keys, float scale_factor) {
    
    // Find maximum value for numerical stability
    float max_score = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < num_keys; ++i) {
        max_score = std::max(max_score, attention_scores[i]);
    }
    
    // Update online softmax state
    float sum_exp = 0.0f;
    for (size_t i = 0; i < num_keys; ++i) {
        float exp_val = std::exp(attention_scores[i] - max_score);
        attention_weights[i] = exp_val;
        sum_exp += exp_val;
    }
    
    softmax_state.update(max_score, sum_exp);
    
    // Normalize
    float normalizer = softmax_state.get_normalizer();
    for (size_t i = 0; i < num_keys; ++i) {
        attention_weights[i] *= normalizer;
    }
}

void FlashAttention::compute_multihead_attention(const TensorShape& query_shape, const float* query_data,
                                                const TensorShape& key_shape, const float* key_data,
                                                const TensorShape& value_shape, const float* value_data,
                                                const TensorShape& output_shape, float* output_data,
                                                const float* attention_mask) {
    
    size_t batch_size = current_params_.batch_size;
    size_t seq_len = current_params_.sequence_length;
    size_t num_heads = current_params_.num_heads;
    size_t head_dim = current_params_.head_dimension;
    
    // Process each batch and head
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            
            // Calculate offsets for current batch and head
            size_t head_offset = (b * num_heads + h) * seq_len * head_dim;
            
            const float* query_head = query_data + head_offset;
            const float* key_head = key_data + head_offset;
            const float* value_head = value_data + head_offset;
            float* output_head = output_data + head_offset;
            
            // Initialize output to zero
            std::fill(output_head, output_head + seq_len * head_dim, 0.0f);
            
            // Block-wise computation
            size_t num_q_blocks = (seq_len + flash_config_.block_size_q - 1) / flash_config_.block_size_q;
            size_t num_k_blocks = (seq_len + flash_config_.block_size_k - 1) / flash_config_.block_size_k;
            
            for (size_t q_block = 0; q_block < num_q_blocks; ++q_block) {
                size_t q_start = q_block * flash_config_.block_size_q;
                size_t q_end = std::min(q_start + flash_config_.block_size_q, seq_len);
                size_t q_block_size = q_end - q_start;
                
                // Initialize block state
                AttentionBlockState block_state(q_block_size * head_dim, 
                                               q_block_size * flash_config_.block_size_k);
                
                for (size_t k_block = 0; k_block < num_k_blocks; ++k_block) {
                    size_t k_start = k_block * flash_config_.block_size_k;
                    size_t k_end = std::min(k_start + flash_config_.block_size_k, seq_len);
                    size_t k_block_size = k_end - k_start;
                    
                    // Skip blocks that would be masked in causal attention
                    if (flash_config_.use_causal_mask && k_start >= q_end) {
                        continue;
                    }
                    
                    const float* query_block = query_head + q_start * head_dim;
                    const float* key_block = key_head + k_start * head_dim;
                    const float* value_block = value_head + k_start * head_dim;
                    
                    const float* mask_block = nullptr;
                    if (attention_mask) {
                        mask_block = attention_mask + (b * seq_len + q_start) * seq_len + k_start;
                    }
                    
                    compute_attention_block(query_block, key_block, value_block,
                                          block_state.output_buffer.data(),
                                          block_state, q_block_size, k_block_size, head_dim,
                                          flash_config_.use_causal_mask, mask_block);
                }
                
                // Copy block output to final output
                std::copy(block_state.output_buffer.begin(), 
                         block_state.output_buffer.begin() + q_block_size * head_dim,
                         output_head + q_start * head_dim);
            }
        }
    }
}

void FlashAttention::compute_optimal_block_sizes(size_t seq_len_q, size_t seq_len_kv, size_t head_dim,
                                                 size_t available_memory, FlashAttentionConfig& config) {
    
    // Estimate memory per block
    size_t memory_per_block = head_dim * sizeof(float) * 
                             (config.block_size_q + config.block_size_k + config.block_size_v);
    
    // Add memory for attention scores
    memory_per_block += config.block_size_q * config.block_size_k * sizeof(float);
    
    // Ensure blocks fit in available memory
    while (memory_per_block > available_memory / 4 && config.block_size_q > 16) {
        config.block_size_q /= 2;
        config.block_size_k /= 2;
        config.block_size_v /= 2;
        
        memory_per_block = head_dim * sizeof(float) * 
                          (config.block_size_q + config.block_size_k + config.block_size_v);
        memory_per_block += config.block_size_q * config.block_size_k * sizeof(float);
    }
    
    // Ensure block sizes are reasonable
    config.block_size_q = std::max(config.block_size_q, static_cast<size_t>(16));
    config.block_size_k = std::max(config.block_size_k, static_cast<size_t>(16));
    config.block_size_v = std::max(config.block_size_v, static_cast<size_t>(16));
}

std::vector<std::pair<size_t, size_t>> FlashAttention::schedule_attention_blocks(
    size_t seq_len_q, size_t seq_len_kv, size_t block_size_q, size_t block_size_k) {
    
    std::vector<std::pair<size_t, size_t>> schedule;
    
    size_t num_q_blocks = (seq_len_q + block_size_q - 1) / block_size_q;
    size_t num_k_blocks = (seq_len_kv + block_size_k - 1) / block_size_k;
    
    for (size_t q_block = 0; q_block < num_q_blocks; ++q_block) {
        for (size_t k_block = 0; k_block < num_k_blocks; ++k_block) {
            schedule.emplace_back(q_block, k_block);
        }
    }
    
    return schedule;
}

void FlashAttention::compute_query_gradients(const float* grad_output, const float* key_data, 
                                            const float* value_data, const float* attention_weights,
                                            float* grad_query, size_t batch_size, size_t num_heads,
                                            size_t seq_len, size_t head_dim) {
    
    // Simplified gradient computation for query
    // grad_query = grad_output @ attention_weights @ key^T
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            size_t head_offset = (b * num_heads + h) * seq_len * head_dim;
            
            const float* grad_out_head = grad_output + head_offset;
            const float* key_head = key_data + head_offset;
            float* grad_q_head = grad_query + head_offset;
            
            // Initialize gradients to zero
            std::fill(grad_q_head, grad_q_head + seq_len * head_dim, 0.0f);
            
            // Compute gradients (simplified implementation)
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t d = 0; d < head_dim; ++d) {
                    float grad_sum = 0.0f;
                    
                    for (size_t j = 0; j < seq_len; ++j) {
                        grad_sum += grad_out_head[i * head_dim + d] * key_head[j * head_dim + d] * 
                                   flash_config_.softmax_scale;
                    }
                    
                    grad_q_head[i * head_dim + d] = grad_sum;
                }
            }
        }
    }
}

void FlashAttention::compute_key_value_gradients(const float* grad_output, const float* query_data,
                                                const float* attention_weights, float* grad_key, 
                                                float* grad_value, size_t batch_size, size_t num_heads,
                                                size_t seq_len, size_t head_dim) {
    
    // Simplified gradient computation for key and value
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            size_t head_offset = (b * num_heads + h) * seq_len * head_dim;
            
            const float* grad_out_head = grad_output + head_offset;
            const float* query_head = query_data + head_offset;
            float* grad_k_head = grad_key + head_offset;
            float* grad_v_head = grad_value + head_offset;
            
            // Initialize gradients to zero
            std::fill(grad_k_head, grad_k_head + seq_len * head_dim, 0.0f);
            std::fill(grad_v_head, grad_v_head + seq_len * head_dim, 0.0f);
            
            // Compute key gradients
            for (size_t j = 0; j < seq_len; ++j) {
                for (size_t d = 0; d < head_dim; ++d) {
                    float grad_sum = 0.0f;
                    
                    for (size_t i = 0; i < seq_len; ++i) {
                        grad_sum += query_head[i * head_dim + d] * grad_out_head[i * head_dim + d] * 
                                   flash_config_.softmax_scale;
                    }
                    
                    grad_k_head[j * head_dim + d] = grad_sum;
                }
            }
            
            // Compute value gradients (simplified)
            for (size_t j = 0; j < seq_len; ++j) {
                for (size_t d = 0; d < head_dim; ++d) {
                    float grad_sum = 0.0f;
                    
                    for (size_t i = 0; i < seq_len; ++i) {
                        grad_sum += grad_out_head[i * head_dim + d];
                    }
                    
                    grad_v_head[j * head_dim + d] = grad_sum;
                }
            }
        }
    }
}

void FlashAttention::apply_causal_mask(float* attention_scores, size_t seq_len_q, size_t seq_len_k,
                                      size_t q_offset, size_t k_offset) {
    
    const float MASK_VALUE = -std::numeric_limits<float>::infinity();
    
    for (size_t i = 0; i < seq_len_q; ++i) {
        for (size_t j = 0; j < seq_len_k; ++j) {
            if (q_offset + i < k_offset + j) {
                attention_scores[i * seq_len_k + j] = MASK_VALUE;
            }
        }
    }
}

void FlashAttention::apply_attention_mask(float* attention_scores, const float* mask_data,
                                         size_t seq_len_q, size_t seq_len_k,
                                         size_t q_offset, size_t k_offset) {
    
    for (size_t i = 0; i < seq_len_q; ++i) {
        for (size_t j = 0; j < seq_len_k; ++j) {
            if (mask_data[i * seq_len_k + j] == 0.0f) {
                attention_scores[i * seq_len_k + j] = -std::numeric_limits<float>::infinity();
            }
        }
    }
}

float FlashAttention::compute_attention_scale(size_t head_dim) const {
    return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

void FlashAttention::transpose_for_attention(const float* input, float* output,
                                            size_t batch_size, size_t seq_len,

#include "paged_attention.hpp"
#include <algorithm>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <immintrin.h>

namespace attention_hpc {

PagePool::PagePool(size_t page_size, size_t head_dimension, size_t initial_pool_size)
    : page_size_(page_size), head_dimension_(head_dimension) {
    create_new_pages(initial_pool_size);
}

std::shared_ptr<KVPage> PagePool::allocate_page() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    if (free_pages_.empty()) {
        create_new_pages(16); // Create more pages if pool is empty
    }
    
    auto page = free_pages_.front();
    free_pages_.pop();
    allocated_pages_.insert(page);
    
    return page;
}

void PagePool::deallocate_page(std::shared_ptr<KVPage> page) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    auto it = allocated_pages_.find(page);
    if (it != allocated_pages_.end()) {
        allocated_pages_.erase(it);
        page->current_tokens = 0; // Reset page
        free_pages_.push(page);
    }
}

void PagePool::resize_pool(size_t new_size) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    size_t current_size = free_pages_.size() + allocated_pages_.size();
    if (new_size > current_size) {
        create_new_pages(new_size - current_size);
    }
}

void PagePool::clear_pool() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    while (!free_pages_.empty()) {
        free_pages_.pop();
    }
    allocated_pages_.clear();
}

void PagePool::create_new_pages(size_t count) {
    for (size_t i = 0; i < count; ++i) {
        auto page = std::make_shared<KVPage>(page_size_, head_dimension_);
        free_pages_.push(page);
    }
}

PagedAttention::PagedAttention()
    : is_configured_(false), next_sequence_id_(1), workspace_ptr_(nullptr), workspace_size_(0),
      workspace_offset_(0), profiling_enabled_(false), current_memory_usage_(0), peak_memory_usage_(0),
      query_buffer_(nullptr), key_buffer_(nullptr), value_buffer_(nullptr), output_buffer_(nullptr),
      attention_scores_buffer_(nullptr), query_buffer_size_(0), key_buffer_size_(0), 
      value_buffer_size_(0), output_buffer_size_(0), attention_scores_buffer_size_(0) {
    
    // Set default PagedAttention configuration
    paged_config_.page_size = 16;
    paged_config_.max_pages_per_sequence = 1024;
    paged_config_.initial_page_pool_size = 64;
    paged_config_.enable_page_caching = true;
    paged_config_.enable_memory_prefetching = false;
    paged_config_.memory_pool_size = 1024 * 1024 * 1024; // 1GB
    paged_config_.max_concurrent_sequences = 32;
    paged_config_.block_size = 64;
    paged_config_.enable_dynamic_batching = true;
    paged_config_.use_kernel_fusion = false;
    paged_config_.enable_async_copying = false;
    paged_config_.prefetch_distance = 2;
    paged_config_.use_contiguous_layout = true;
    paged_config_.enable_memory_coalescing = true;
}

PagedAttention::~PagedAttention() {
    deallocate_buffers();
    
    // Clean up all sequences
    std::lock_guard<std::mutex> lock(sequences_mutex_);
    sequences_.clear();
}

void PagedAttention::configure(const AttentionParams& params, const PerformanceConfig& perf_config) {
    current_params_ = params;
    current_perf_config_ = perf_config;
    
    // Validate parameters
    if (params.batch_size == 0 || params.sequence_length == 0 || 
        params.num_heads == 0 || params.head_dimension == 0) {
        throw AttentionException(AttentionError::INVALID_PARAMETERS, 
                                "Invalid attention parameters: dimensions must be positive");
    }
    
    // Initialize page pool
    page_pool_ = std::make_unique<PagePool>(paged_config_.page_size, params.head_dimension, 
                                           paged_config_.initial_page_pool_size);
    
    // Allocate necessary buffers
    allocate_buffers(params);
    
    is_configured_ = true;
}

void PagedAttention::configure_paged_attention(const PagedAttentionConfig& paged_config) {
    paged_config_ = paged_config;
    
    if (is_configured_) {
        // Reinitialize page pool with new configuration
        page_pool_ = std::make_unique<PagePool>(paged_config_.page_size, current_params_.head_dimension, 
                                               paged_config_.initial_page_pool_size);
        
        // Reallocate buffers with new configuration
        allocate_buffers(current_params_);
    }
}

void PagedAttention::forward(const TensorShape& query_shape, const void* query_data,
                            const TensorShape& key_shape, const void* key_data,
                            const TensorShape& value_shape, const void* value_data,
                            const TensorShape& output_shape, void* output_data,
                            const void* attention_mask) {
    
    if (!is_configured_) {
        throw AttentionException(AttentionError::INVALID_PARAMETERS, 
                                "PagedAttention not configured. Call configure() first.");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // For basic forward pass without existing sequences, create temporary sequence
    size_t temp_seq_id = create_sequence(current_params_.sequence_length);
    
    // Add key-value data to the temporary sequence
    append_kv_cache(temp_seq_id, key_data, value_data, current_params_.sequence_length, query_shape.data_type);
    
    // Convert query data to float for computation
    size_t query_elements = query_shape.total_elements();
    convert_to_float(query_data, query_buffer_.get(), query_elements, query_shape.data_type);
    
    // Perform paged attention computation
    std::vector<size_t> seq_ids = {temp_seq_id};
    compute_paged_attention(query_buffer_.get(), seq_ids, output_buffer_.get(),
                           current_params_.batch_size, current_params_.num_heads, 
                           current_params_.head_dimension,
                           static_cast<const float*>(attention_mask));
    
    // Convert output back to desired data type
    convert_from_float(output_buffer_.get(), output_data, 
                       output_shape.total_elements(), output_shape.data_type);
    
    // Clean up temporary sequence
    destroy_sequence(temp_seq_id);
    
    if (profiling_enabled_) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        profiling_results_.emplace_back("forward_pass", duration.count() / 1000.0);
    }
}

void PagedAttention::backward(const TensorShape& grad_output_shape, const void* grad_output_data,
                             const TensorShape& query_shape, const void* query_data,
                             const TensorShape& key_shape, const void* key_data,
                             const TensorShape& value_shape, const void* value_data,
                             const TensorShape& grad_query_shape, void* grad_query_data,
                             const TensorShape& grad_key_shape, void* grad_key_data,
                             const TensorShape& grad_value_shape, void* grad_value_data,
                             const void* attention_mask) {
    
    if (!is_configured_) {
        throw AttentionException(AttentionError::INVALID_PARAMETERS,
                                "PagedAttention not configured. Call configure() first.");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create temporary sequence for gradient computation
    size_t temp_seq_id = create_sequence(current_params_.sequence_length);
    append_kv_cache(temp_seq_id, key_data, value_data, current_params_.sequence_length, query_shape.data_type);
    
    // Convert input data to float
    convert_to_float(grad_output_data, output_buffer_.get(), 
                     grad_output_shape.total_elements(), grad_output_shape.data_type);
    convert_to_float(query_data, query_buffer_.get(), 
                     query_shape.total_elements(), query_shape.data_type);
    
    // Allocate temporary buffers for gradients
    std::unique_ptr<float[]> grad_query_buffer(new float[grad_query_shape.total_elements()]);
    std::unique_ptr<float[]> grad_key_buffer(new float[grad_key_shape.total_elements()]);
    std::unique_ptr<float[]> grad_value_buffer(new float[grad_value_shape.total_elements()]);
    
    // Initialize gradient buffers to zero
    std::fill(grad_query_buffer.get(), grad_query_buffer.get() + grad_query_shape.total_elements(), 0.0f);
    std::fill(grad_key_buffer.get(), grad_key_buffer.get() + grad_key_shape.total_elements(), 0.0f);
    std::fill(grad_value_buffer.get(), grad_value_buffer.get() + grad_value_shape.total_elements(), 0.0f);
    
    // Compute paged gradients
    std::vector<size_t> seq_ids = {temp_seq_id};
    compute_paged_gradients(output_buffer_.get(), seq_ids, grad_query_buffer.get(),
                           current_params_.batch_size, current_params_.num_heads,
                           current_params_.head_dimension);
    
    // Convert gradients back to desired data types
    convert_from_float(grad_query_buffer.get(), grad_query_data,
                       grad_query_shape.total_elements(), grad_query_shape.data_type);
    convert_from_float(grad_key_buffer.get(), grad_key_data,
                       grad_key_shape.total_elements(), grad_key_shape.data_type);
    convert_from_float(grad_value_buffer.get(), grad_value_data,
                       grad_value_shape.total_elements(), grad_value_shape.data_type);
    
    // Clean up temporary sequence
    destroy_sequence(temp_seq_id);
    
    if (profiling_enabled_) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        profiling_results_.emplace_back("backward_pass", duration.count() / 1000.0);
    }
}

size_t PagedAttention::get_workspace_size(const AttentionParams& params) const {
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
    total_size += paged_config_.block_size * seq_len * sizeof(float);
    
    // Page pool memory estimation
    size_t pages_needed = (seq_len + paged_config_.page_size - 1) / paged_config_.page_size;
    total_size += pages_needed * paged_config_.page_size * head_dim * sizeof(float) * 2; // key + value
    
    // Additional workspace for intermediate computations
    total_size += seq_len * head_dim * sizeof(float) * 2;
    
    return total_size;
}

void PagedAttention::set_workspace(void* workspace_ptr, size_t workspace_size) {
    workspace_ptr_ = workspace_ptr;
    workspace_size_ = workspace_size;
    workspace_offset_ = 0;
}

bool PagedAttention::supports_data_type(DataType data_type) const {
    return data_type == DataType::FLOAT32 || data_type == DataType::FLOAT16 || data_type == DataType::BFLOAT16;
}

bool PagedAttention::supports_memory_layout(MemoryLayout layout) const {
    return layout == MemoryLayout::ROW_MAJOR || layout == MemoryLayout::COLUMN_MAJOR;
}

std::string PagedAttention::get_backend_name() const {
    return "CPU_PagedAttention";
}

std::string PagedAttention::get_version() const {
    return "1.0.0";
}

void PagedAttention::enable_profiling(bool enable) {
    profiling_enabled_ = enable;
    if (!enable) {
        profiling_results_.clear();
    }
}

std::vector<std::pair<std::string, double>> PagedAttention::get_profiling_results() const {
    return profiling_results_;
}

void PagedAttention::reset_profiling() {
    profiling_results_.clear();
}

AttentionError PagedAttention::validate_inputs(const TensorShape& query_shape,
                                              const TensorShape& key_shape,
                                              const TensorShape& value_shape,
                                              const TensorShape& output_shape,
                                              const AttentionParams& params) const {
    
    // Validate tensor dimensions
    if (query_shape.dimensions.size() != 4 || output_shape.dimensions.size() != 4) {
        return AttentionError::INVALID_INPUT;
    }
    
    // Validate batch dimensions
    if (query_shape.dimensions[0] != params.batch_size ||
        output_shape.dimensions[0] != params.batch_size) {
        return AttentionError::INVALID_INPUT;
    }
    
    // Validate head dimensions
    if (query_shape.dimensions[2] != params.num_heads ||
        output_shape.dimensions[2] != params.num_heads) {
        return AttentionError::INVALID_INPUT;
    }
    
    // Validate feature dimensions
    if (query_shape.dimensions[3] != params.head_dimension ||
        output_shape.dimensions[3] != params.head_dimension) {
        return AttentionError::INVALID_INPUT;
    }
    
    return AttentionError::SUCCESS;
}

size_t PagedAttention::create_sequence(size_t max_length) {
    std::lock_guard<std::mutex> lock(sequences_mutex_);
    
    size_t sequence_id = next_sequence_id_++;
    sequences_[sequence_id] = create_sequence_state(sequence_id, max_length);
    
    return sequence_id;
}

void PagedAttention::destroy_sequence(size_t sequence_id) {
    std::lock_guard<std::mutex> lock(sequences_mutex_);
    
    auto it = sequences_.find(sequence_id);
    if (it != sequences_.end()) {
        cleanup_sequence_state(sequence_id);
        sequences_.erase(it);
    }
}

void PagedAttention::reset_sequence(size_t sequence_id) {
    std::lock_guard<std::mutex> lock(sequences_mutex_);
    
    auto it = sequences_.find(sequence_id);
    if (it != sequences_.end()) {
        auto& seq_state = it->second;
        
        // Deallocate all pages
        for (auto& page : seq_state->pages) {
            page_pool_->deallocate_page(page);
        }
        seq_state->pages.clear();
        
        // Reset sequence state
        seq_state->total_tokens = 0;
        seq_state->current_position = 0;
        seq_state->is_active = true;
    }
}

bool PagedAttention::is_sequence_active(size_t sequence_id) const {
    std::lock_guard<std::mutex> lock(sequences_mutex_);
    
    auto it = sequences_.find(sequence_id);
    return it != sequences_.end() && it->second->is_active;
}

void PagedAttention::extend_sequence(size_t sequence_id, size_t additional_tokens) {
    std::lock_guard<std::mutex> lock(sequences_mutex_);
    
    auto it = sequences_.find(sequence_id);
    if (it != sequences_.end()) {
        auto& seq_state = it->second;
        seq_state->max_sequence_length += additional_tokens;
        
        // Allocate additional pages if needed
        size_t pages_needed = estimate_pages_needed(seq_state->max_sequence_length, paged_config_.page_size);
        while (seq_state->pages.size() < pages_needed) {
            auto page = page_pool_->allocate_page();
            seq_state->add_page(page);
        }
    }
}

void PagedAttention::truncate_sequence(size_t sequence_id, size_t new_length) {
    std::lock_guard<std::mutex> lock(sequences_mutex_);
    
    auto it = sequences_.find(sequence_id);
    if (it != sequences_.end()) {
        auto& seq_state = it->second;
        
        if (new_length < seq_state->total_tokens) {
            seq_state->total_tokens = new_length;
            seq_state->current_position = std::min(seq_state->current_position, new_length);
            
            // Deallocate unnecessary pages
            size_t pages_needed = estimate_pages_needed(new_length, paged_config_.page_size);
            while (seq_state->pages.size() > pages_needed) {
                auto page = seq_state->pages.back();
                seq_state->pages.pop_back();
                page_pool_->deallocate_page(page);
            }
        }
        
        seq_state->max_sequence_length = new_length;
    }
}

size_t PagedAttention::get_sequence_length(size_t sequence_id) const {
    std::lock_guard<std::mutex> lock(sequences_mutex_);
    
    auto it = sequences_.find(sequence_id);
    return it != sequences_.end() ? it->second->total_tokens : 0;
}

size_t PagedAttention::get_max_sequence_length(size_t sequence_id) const {
    std::lock_guard<std::mutex> lock(sequences_mutex_);
    
    auto it = sequences_.find(sequence_id);
    return it != sequences_.end() ? it->second->max_sequence_length : 0;
}

void PagedAttention::append_kv_cache(size_t sequence_id, const void* key_data, const void* value_data, 
                                     size_t num_tokens, DataType data_type) {
    std::lock_guard<std::mutex> lock(sequences_mutex_);
    
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        throw AttentionException(AttentionError::INVALID_INPUT, "Sequence not found");
    }
    
    auto& seq_state = it->second;
    size_t head_dim = current_params_.head_dimension;
    
    // Convert input data to float
    size_t key_elements = num_tokens * current_params_.num_heads * head_dim;
    size_t value_elements = num_tokens * current_params_.num_heads * head_dim;
    
    std::unique_ptr<float[]> key_float(new float[key_elements]);
    std::unique_ptr<float[]> value_float(new float[value_elements]);
    
    convert_to_float(key_data, key_float.get(), key_elements, data_type);
    convert_to_float(value_data, value_float.get(), value_elements, data_type);
    
    // Scatter keys and values to pages
    scatter_keys_values(key_float.get(), value_float.get(), seq_state->pages,
                       seq_state->total_tokens, num_tokens, head_dim);
    
    seq_state->total_tokens += num_tokens;
    seq_state->current_position = seq_state->total_tokens;
}

void PagedAttention::get_kv_cache(size_t sequence_id, void* key_data, void* value_data, 
                                 size_t start_token, size_t num_tokens, DataType data_type) {
    std::lock_guard<std::mutex> lock(sequences_mutex_);
    
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        throw AttentionException(AttentionError::INVALID_INPUT, "Sequence not found");
    }
    
    auto& seq_state = it->second;
    size_t head_dim = current_params_.head_dimension;
    
    // Allocate temporary float buffers
    size_t key_elements = num_tokens * current_params_.num_heads * head_dim;
    size_t value_elements = num_tokens * current_params_.num_heads * head_dim;
    
    std::unique_ptr<float[]> key_float(new float[key_elements]);
    std::unique_ptr<float[]> value_float(new float[value_elements]);
    
    // Gather keys and values from pages
    gather_keys_values(seq_state->pages, key_float.get(), value_float.get(),
                      start_token, num_tokens, head_dim);
    
    // Convert back to desired data type
    convert_from_float(key_float.get(), key_data, key_elements, data_type);
    convert_from_float(value_float.get(), value_data, value_elements, data_type);
}

void PagedAttention::clear_kv_cache(size_t sequence_id) {
    reset_sequence(sequence_id);
}

void PagedAttention::forward_batch(const std::vector<size_t>& sequence_ids,
                                  const std::vector<TensorShape>& query_shapes,
                                  const std::vector<const void*>& query_data,
                                  const std::vector<TensorShape>& output_shapes,
                                  const std::vector<void*>& output_data,
                                  const std::vector<const void*>& attention_masks) {
    
    if (!is_configured_) {
        throw AttentionException(AttentionError::INVALID_PARAMETERS,
                                "PagedAttention not configured. Call configure() first.");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Validate batch inputs
    if (sequence_ids.size() != query_shapes.size() || 
        sequence_ids.size() != query_data.size() ||
        sequence_ids.size() != output_shapes.size() ||
        sequence_ids.size() != output_data.size()) {
        throw AttentionException(AttentionError::INVALID_INPUT, "Batch input sizes mismatch");
    }
    
    // Process each sequence in the batch
    for (size_t i = 0; i < sequence_ids.size(); ++i) {
        size_t seq_id = sequence_ids[i];
        const auto& query_shape = query_shapes[i];
        const void* query_ptr = query_data[i];
        const auto& output_shape = output_shapes[i];
        void* output_ptr = output_data[i];
        const void* mask_ptr = !attention_masks.empty() ? attention_masks[i] : nullptr;
        
        // Convert query data to float
        size_t query_elements = query_shape.total_elements();
        convert_to_float(query_ptr, query_buffer_.get(), query_elements, query_shape.data_type);
        
        // Perform paged attention computation for this sequence
        std::vector<size_t> single_seq = {seq_id};
        compute_paged_attention(query_buffer_.get(), single_seq, output_buffer_.get(),
                               1, current_params_.num_heads, current_params_.head_dimension,
                               static_cast<const float*>(mask_ptr));
        
        // Convert output back to desired data type
        convert_from_float(output_buffer_.get(), output_ptr, 
                           output_shape.total_elements(), output_shape.data_type);
    }
    
    if (profiling_enabled_) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        profiling_results_.emplace_back("forward_batch", duration.count() / 1000.0);
    }
}

size_t PagedAttention::get_total_memory_usage() const {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    return current_memory_usage_;
}

size_t PagedAttention::get_peak_memory_usage() const {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    return peak_memory_usage_;
}

float PagedAttention::get_memory_utilization() const {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    return paged_config_.memory_pool_size > 0 ? 
           static_cast<float>(current_memory_usage_) / paged_config_.memory_pool_size : 0.0f;
}

size_t PagedAttention::get_active_sequences_count() const {
    std::lock_guard<std::mutex> lock(sequences_mutex_);
    
    size_t count = 0;
    for (const auto& pair : sequences_) {
        if (pair.second->is_active) {
            count++;
        }
    }
    return count;
}

void PagedAttention::compact_pages() {
    std::lock_guard<std::mutex> lock(sequences_mutex_);
    
    for (auto& pair : sequences_) {
        compact_sequence_pages(pair.first);
    }
}

void PagedAttention::preload_pages(const std::vector<size_t>& sequence_ids) {
    std::lock_guard<std::mutex> lock(sequences_mutex_);
    
    for (size_t seq_id : sequence_ids) {
        auto it = sequences_.find(seq_id);
        if (it != sequences_.end()) {
            prefetch_pages(it->second->pages);
        }
    }
}

void PagedAttention::compute_paged_attention(const float* query_data, const std::vector<size_t>& sequence_ids,
                                            float* output_data, size_t batch_size, size_t num_heads, 
                                            size_t head_dim, const float* attention_mask) {
    
    // Process each sequence
    for (size_t seq_idx = 0; seq_idx < sequence_ids.size(); ++seq_idx) {
        size_t seq_id = sequence_ids[seq_idx];
        
        std::lock_guard<std::mutex> lock(sequences_mutex_);
        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) {
            continue; // Skip non-existent sequences
        }
        
        auto& seq_state = it->second;
        size_t seq_len = seq_state->total_tokens;
        
        if (seq_len == 0) {
            continue; // Skip empty sequences
        }
        
        // Process each head
        for (size_t h = 0; h < num_heads; ++h) {
            size_t query_offset = seq_idx * num_heads * head_dim + h * head_dim;
            size_t output_offset = seq_idx * num_heads * head_dim + h * head_dim;
            
            const float* query_head = query_data + query_offset;
            float* output_head = output_data + output_offset;
            
            // Initialize output to zero
            std::fill(output_head, output_head + head_dim, 0.0f);
            
            // Gather key-value data from pages
            gather_keys_values(seq_state->pages, key_buffer_.get(), value_buffer_.get(),
                              0, seq_len, head_dim);
            
            // Compute attention scores
            std::vector<float> attention_scores(seq_len, 0.0f);
            float scale = compute_attention_scale(head_dim);
            
            for (size_t k = 0; k < seq_len; ++k) {
                float score = 0.0f;
                const float* key_ptr = key_buffer_.get() + k * head_dim;
                
                // Vectorized dot product
                size_t d = 0;
                for (; d + 8 <= head_dim; d += 8) {
                    __m256 q_vec = _mm256_loadu_ps(query_head + d);
                    __m256 k_vec = _mm256_loadu_ps(key_ptr + d);
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
                    score += query_head[d] * key_ptr[d];
                }
                
                attention_scores[k] = score * scale;
            }
            
            // Apply causal mask if needed (assuming causal attention)
            for (size_t k = 1; k < seq_len; ++k) {
                attention_scores[k] = -std::numeric_limits<float>::infinity();
            }
            
            // Apply attention mask if provided
            if (attention_mask) {
                for (size_t k = 0; k < seq_len; ++k) {
                    if (attention_mask[seq_idx * seq_len + k] == 0.0f) {
                        attention_scores[k] = -std::numeric_limits<float>::infinity();
                    }
                }
            }
            
            // Compute softmax
            float max_score = *std::max_element(attention_scores.begin(), attention_scores.end());
            float sum_exp = 

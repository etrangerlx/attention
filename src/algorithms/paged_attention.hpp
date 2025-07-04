#pragma once

#include "api/attention_interface.hpp"
#include "api/tensor.hpp"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <mutex>

namespace attention_hpc {

// Forward declarations
template<typename T>
class Tensor;

// Page structure for key-value cache storage
struct KVPage {
    std::unique_ptr<float[]> key_data;
    std::unique_ptr<float[]> value_data;
    size_t page_size;
    size_t head_dimension;
    size_t current_tokens;  // Number of tokens currently stored in this page
    bool is_allocated;
    
    KVPage(size_t page_sz, size_t head_dim) 
        : page_size(page_sz), head_dimension(head_dim), current_tokens(0), is_allocated(false) {
        allocate();
    }
    
    void allocate() {
        if (!is_allocated) {
            key_data = std::make_unique<float[]>(page_size * head_dimension);
            value_data = std::make_unique<float[]>(page_size * head_dimension);
            is_allocated = true;
        }
    }
    
    void deallocate() {
        key_data.reset();
        value_data.reset();
        is_allocated = false;
        current_tokens = 0;
    }
    
    bool is_full() const {
        return current_tokens >= page_size;
    }
    
    bool is_empty() const {
        return current_tokens == 0;
    }
    
    size_t available_slots() const {
        return page_size - current_tokens;
    }
};

// Page pool for efficient memory management
class PagePool {
public:
    PagePool(size_t page_size, size_t head_dimension, size_t initial_pool_size = 64);
    ~PagePool() = default;
    
    std::shared_ptr<KVPage> allocate_page();
    void deallocate_page(std::shared_ptr<KVPage> page);
    
    size_t get_pool_size() const { return free_pages_.size() + allocated_pages_.size(); }
    size_t get_free_pages() const { return free_pages_.size(); }
    size_t get_allocated_pages() const { return allocated_pages_.size(); }
    
    void resize_pool(size_t new_size);
    void clear_pool();

private:
    size_t page_size_;
    size_t head_dimension_;
    std::queue<std::shared_ptr<KVPage>> free_pages_;
    std::unordered_set<std::shared_ptr<KVPage>> allocated_pages_;
    mutable std::mutex pool_mutex_;
    
    void create_new_pages(size_t count);
};

// Sequence state for tracking paged attention sequences
struct SequenceState {
    size_t sequence_id;
    std::vector<std::shared_ptr<KVPage>> pages;
    size_t total_tokens;
    size_t max_sequence_length;
    size_t current_position;
    bool is_active;
    
    SequenceState(size_t seq_id, size_t max_len) 
        : sequence_id(seq_id), total_tokens(0), max_sequence_length(max_len), 
          current_position(0), is_active(true) {}
    
    void add_page(std::shared_ptr<KVPage> page) {
        pages.push_back(page);
    }
    
    void remove_last_page() {
        if (!pages.empty()) {
            pages.pop_back();
        }
    }
    
    size_t get_num_pages() const {
        return pages.size();
    }
    
    bool needs_new_page(size_t page_size) const {
        if (pages.empty()) return true;
        return pages.back()->is_full();
    }
    
    std::shared_ptr<KVPage> get_current_page() {
        if (pages.empty()) return nullptr;
        return pages.back();
    }
};

// PagedAttention configuration parameters
struct PagedAttentionConfig {
    // Page management settings
    size_t page_size = 16;  // Number of tokens per page
    size_t max_pages_per_sequence = 1024;
    size_t initial_page_pool_size = 64;
    
    // Memory management settings
    bool enable_page_caching = true;
    bool enable_memory_prefetching = false;
    size_t memory_pool_size = 1024 * 1024 * 1024; // 1GB
    
    // Sequence processing settings
    size_t max_concurrent_sequences = 32;
    size_t block_size = 64;  // Block size for attention computation
    bool enable_dynamic_batching = true;
    
    // Performance optimization settings
    bool use_kernel_fusion = false;
    bool enable_async_copying = false;
    size_t prefetch_distance = 2;  // Number of pages to prefetch ahead
    
    // Memory layout optimization
    bool use_contiguous_layout = true;
    bool enable_memory_coalescing = true;
};

// Block mapping for efficient attention computation
struct BlockMapping {
    std::vector<size_t> sequence_ids;
    std::vector<size_t> page_indices;
    std::vector<size_t> token_positions;
    size_t total_blocks;
    
    BlockMapping() : total_blocks(0) {}
    
    void add_block(size_t seq_id, size_t page_idx, size_t token_pos) {
        sequence_ids.push_back(seq_id);
        page_indices.push_back(page_idx);
        token_positions.push_back(token_pos);
        total_blocks++;
    }
    
    void clear() {
        sequence_ids.clear();
        page_indices.clear();
        token_positions.clear();
        total_blocks = 0;
    }
};

// PagedAttention implementation class
class PagedAttention : public AttentionInterface {
public:
    PagedAttention();
    virtual ~PagedAttention();
    
    // Configuration methods (inherited from AttentionInterface)
    void configure(const AttentionParams& params, 
                   const PerformanceConfig& perf_config = PerformanceConfig{}) override;
    
    // PagedAttention specific configuration
    void configure_paged_attention(const PagedAttentionConfig& paged_config);
    
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
    
    // PagedAttention specific methods
    
    // Sequence management
    size_t create_sequence(size_t max_length);
    void destroy_sequence(size_t sequence_id);
    void reset_sequence(size_t sequence_id);
    bool is_sequence_active(size_t sequence_id) const;
    
    // Dynamic sequence length support
    void extend_sequence(size_t sequence_id, size_t additional_tokens);
    void truncate_sequence(size_t sequence_id, size_t new_length);
    size_t get_sequence_length(size_t sequence_id) const;
    size_t get_max_sequence_length(size_t sequence_id) const;
    
    // Key-value cache management
    void append_kv_cache(size_t sequence_id, const void* key_data, const void* value_data, 
                        size_t num_tokens, DataType data_type);
    void get_kv_cache(size_t sequence_id, void* key_data, void* value_data, 
                     size_t start_token, size_t num_tokens, DataType data_type);
    void clear_kv_cache(size_t sequence_id);
    
    // Batch processing for multiple sequences
    void forward_batch(const std::vector<size_t>& sequence_ids,
                      const std::vector<TensorShape>& query_shapes,
                      const std::vector<const void*>& query_data,
                      const std::vector<TensorShape>& output_shapes,
                      const std::vector<void*>& output_data,
                      const std::vector<const void*>& attention_masks = {});
    
    // Memory statistics and monitoring
    size_t get_total_memory_usage() const;
    size_t get_peak_memory_usage() const;
    float get_memory_utilization() const;
    size_t get_active_sequences_count() const;
    
    // Page management utilities
    void compact_pages();  // Defragment and optimize page layout
    void preload_pages(const std::vector<size_t>& sequence_ids);  // Preload pages for efficiency

protected:
    // Core PagedAttention algorithm components
    
    // Paged attention computation
    virtual void compute_paged_attention(
        const float* query_data, const std::vector<size_t>& sequence_ids,
        float* output_data, size_t batch_size, size_t num_heads, size_t head_dim,
        const float* attention_mask = nullptr);
    
    // Block-wise attention for long sequences
    virtual void compute_blocked_attention(
        const float* query_block, const BlockMapping& block_mapping,
        float* output_block, size_t block_size, size_t num_heads, size_t head_dim);
    
    // Page-aware memory access
    virtual void gather_keys_values(
        const std::vector<std::shared_ptr<KVPage>>& pages,
        float* key_buffer, float* value_buffer,
        size_t start_token, size_t num_tokens, size_t head_dim);
    
    virtual void scatter_keys_values(
        const float* key_data, const float* value_data,
        const std::vector<std::shared_ptr<KVPage>>& pages,
        size_t start_token, size_t num_tokens, size_t head_dim);
    
    // Dynamic sequence length handling
    virtual void resize_sequence_storage(size_t sequence_id, size_t new_max_length);
    virtual void migrate_sequence_data(const std::vector<std::shared_ptr<KVPage>>& old_pages,
                                      std::vector<std::shared_ptr<KVPage>>& new_pages,
                                      size_t tokens_to_copy, size_t head_dim);
    
    // Block scheduling and mapping
    virtual BlockMapping create_block_mapping(const std::vector<size_t>& sequence_ids,
                                             size_t block_size);
    
    virtual std::vector<std::vector<size_t>> schedule_attention_blocks(
        const std::vector<size_t>& sequence_ids, size_t max_blocks_per_batch);
    
    // Memory prefetching and optimization
    virtual void prefetch_pages(const std::vector<std::shared_ptr<KVPage>>& pages);
    virtual void optimize_memory_layout(const std::vector<size_t>& sequence_ids);
    
    // Gradient computation for paged attention
    virtual void compute_paged_gradients(
        const float* grad_output, const std::vector<size_t>& sequence_ids,
        float* grad_query, size_t batch_size, size_t num_heads, size_t head_dim);
    
    // Utility methods
    virtual float compute_attention_scale(size_t head_dim) const;
    virtual void apply_sequence_mask(float* attention_scores, size_t sequence_id,
                                   size_t query_length, size_t key_length);
    
    // Data type conversion utilities
    virtual void convert_to_float(const void* input, float* output,
                                  size_t num_elements, DataType input_type);
    virtual void convert_from_float(const float* input, void* output,
                                    size_t num_elements, DataType output_type);

private:
    // Configuration state
    PagedAttentionConfig paged_config_;
    bool is_configured_;
    
    // Page management
    std::unique_ptr<PagePool> page_pool_;
    std::unordered_map<size_t, std::unique_ptr<SequenceState>> sequences_;
    mutable std::mutex sequences_mutex_;
    size_t next_sequence_id_;
    
    // Workspace management
    void* workspace_ptr_;
    size_t workspace_size_;
    size_t workspace_offset_;
    
    // Profiling state
    bool profiling_enabled_;
    std::vector<std::pair<std::string, double>> profiling_results_;
    
    // Memory usage tracking
    size_t current_memory_usage_;
    size_t peak_memory_usage_;
    mutable std::mutex memory_mutex_;
    
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
    
    // Sequence management helpers
    std::unique_ptr<SequenceState> create_sequence_state(size_t sequence_id, size_t max_length);
    void cleanup_sequence_state(size_t sequence_id);
    bool validate_sequence_id(size_t sequence_id) const;
    
    // Page allocation helpers
    std::shared_ptr<KVPage> allocate_page_for_sequence(size_t sequence_id);
    void deallocate_pages_for_sequence(size_t sequence_id);
    size_t estimate_pages_needed(size_t sequence_length, size_t page_size) const;
    
    // Memory tracking helpers
    void update_memory_usage(int64_t delta);
    void track_peak_memory();
    
    // Validation helpers
    bool validate_tensor_shapes_for_paged(const TensorShape& query_shape,
                                          const TensorShape& output_shape) const;
    bool validate_sequence_batch(const std::vector<size_t>& sequence_ids) const;
    
    // Performance optimization helpers
    void optimize_page_layout();
    void compact_sequence_pages(size_t sequence_id);
    
    // Profiling helpers
    void profile_operation(const std::string& operation_name,
                           std::function<void()> operation);
    double get_current_time() const;
};

// Template specializations for different data types
template<typename T>
class TypedPagedAttention : public PagedAttention {
public:
    TypedPagedAttention() = default;
    virtual ~TypedPagedAttention() = default;
    
    // Type-specific implementations
    void forward_typed(const Tensor<T>& query, size_t sequence_id, Tensor<T>& output,
                       const Tensor<T>* attention_mask = nullptr);
    
    void append_kv_cache_typed(size_t sequence_id, const Tensor<T>& key, const Tensor<T>& value);
    
    void forward_batch_typed(const std::vector<size_t>& sequence_ids,
                            const std::vector<Tensor<T>>& queries,
                            std::vector<Tensor<T>>& outputs,
                            const std::vector<Tensor<T>*>& attention_masks = {});

protected:
    // Type-specific computation kernels
    virtual void compute_paged_attention_typed(
        const T* query_data, const std::vector<size_t>& sequence_ids,
        T* output_data, size_t batch_size, size_t num_heads, size_t head_dim,
        const T* attention_mask = nullptr);
    
    virtual void gather_keys_values_typed(
        const std::vector<std::shared_ptr<KVPage>>& pages,
        T* key_buffer, T* value_buffer,
        size_t start_token, size_t num_tokens, size_t head_dim);
};

// Type aliases for common instantiations
using PagedAttentionFloat = TypedPagedAttention<float>;
using PagedAttentionHalf = TypedPagedAttention<half>;
using PagedAttentionBFloat16 = TypedPagedAttention<bfloat16>;

// Utility functions for PagedAttention
namespace paged_attention_utils {
    // Memory estimation utilities
    size_t estimate_paged_attention_memory(const AttentionParams& params,
                                          const PagedAttentionConfig& config);
    
    // Optimal configuration detection
    PagedAttentionConfig get_optimal_paged_config(const AttentionParams& params,
                                                 size_t available_memory,
                                                 size_t max_sequence_length);
    
    // Page size optimization
    size_t calculate_optimal_page_size(size_t typical_sequence_length,
                                      size_t head_dimension,
                                      size_t available_memory);
    
    // Sequence batching utilities
    std::vector<std::vector<size_t>> group_sequences_for_batching(
        const std::vector<size_t>& sequence_ids,
        const std::vector<size_t>& sequence_lengths,
        size_t max_batch_size);
    
    // Memory defragmentation
    void defragment_page_pool(PagePool& pool);
    
    // Performance benchmarking
    double benchmark_paged_attention_config(const AttentionParams& params,
                                           const PagedAttentionConfig& config,
                                           const std::vector<size_t>& sequence_lengths,
                                           int num_iterations = 10);
    
    // Statistics and monitoring
    struct PagedAttentionStats {
        size_t total_sequences;
        size_t active_sequences;
        size_t total_pages;
        size_t free_pages;
        size_t memory_usage;
        size_t peak_memory_usage;
        float average_sequence_length;
        float page_utilization;
    };
    
    PagedAttentionStats get_paged_attention_stats(const PagedAttention& attention);
}

} // namespace attention_hpc

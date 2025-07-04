#pragma once

#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>

namespace attention_hpc {

// Forward declarations
template<typename T>
class Tensor;

// Data types supported by the attention implementations
enum class DataType {
    FLOAT32,
    FLOAT16,
    BFLOAT16
};

// Memory layout options
enum class MemoryLayout {
    ROW_MAJOR,
    COLUMN_MAJOR
};

// Performance configuration parameters
struct PerformanceConfig {
    // Block sizes for tiling
    size_t block_size_m = 64;
    size_t block_size_n = 64;
    size_t block_size_k = 64;
    
    // Number of threads/workers
    size_t num_threads = 0; // 0 means auto-detect
    
    // Memory optimization flags
    bool use_memory_pool = true;
    bool enable_memory_reuse = true;
    
    // Computation optimization flags
    bool use_fast_math = false;
    bool enable_mixed_precision = false;
    
    // Backend-specific parameters
    size_t shared_memory_size = 0; // 0 means auto-determine
    size_t max_workspace_size = 1024 * 1024 * 1024; // 1GB default
    
    // Profiling and debugging
    bool enable_profiling = false;
    bool enable_debug_checks = true;
};

// Attention computation parameters
struct AttentionParams {
    size_t batch_size;
    size_t sequence_length;
    size_t num_heads;
    size_t head_dimension;
    size_t key_value_heads = 0; // 0 means same as num_heads (for GQA/MQA)
    
    // Attention-specific parameters
    float scale_factor = 0.0f; // 0 means auto-compute as 1/sqrt(head_dimension)
    bool causal_mask = false;
    bool apply_rotary_embedding = false;
    
    // Memory management
    size_t max_sequence_length = 0; // For paged attention
    size_t page_size = 16; // For paged attention
};

// Error codes for attention operations
enum class AttentionError {
    SUCCESS = 0,
    INVALID_INPUT,
    INVALID_PARAMETERS,
    MEMORY_ALLOCATION_FAILED,
    COMPUTATION_FAILED,
    BACKEND_NOT_AVAILABLE,
    UNSUPPORTED_OPERATION,
    INTERNAL_ERROR
};

// Custom exception class for attention operations
class AttentionException : public std::runtime_error {
public:
    explicit AttentionException(AttentionError error_code, const std::string& message)
        : std::runtime_error(message), error_code_(error_code) {}
    
    AttentionError error_code() const noexcept { return error_code_; }
    
private:
    AttentionError error_code_;
};

// Tensor shape information
struct TensorShape {
    std::vector<size_t> dimensions;
    DataType data_type;
    MemoryLayout layout;
    
    TensorShape() = default;
    TensorShape(std::vector<size_t> dims, DataType dtype = DataType::FLOAT32, 
                MemoryLayout mem_layout = MemoryLayout::ROW_MAJOR)
        : dimensions(std::move(dims)), data_type(dtype), layout(mem_layout) {}
    
    size_t total_elements() const {
        size_t total = 1;
        for (size_t dim : dimensions) {
            total *= dim;
        }
        return total;
    }
    
    size_t element_size() const {
        switch (data_type) {
            case DataType::FLOAT32: return 4;
            case DataType::FLOAT16: return 2;
            case DataType::BFLOAT16: return 2;
            default: return 4;
        }
    }
    
    size_t total_bytes() const {
        return total_elements() * element_size();
    }
};

// Abstract base interface for attention implementations
class AttentionInterface {
public:
    virtual ~AttentionInterface() = default;
    
    // Configuration methods
    virtual void configure(const AttentionParams& params, 
                          const PerformanceConfig& perf_config = PerformanceConfig{}) = 0;
    
    // Core computation methods
    virtual void forward(const TensorShape& query_shape, const void* query_data,
                        const TensorShape& key_shape, const void* key_data,
                        const TensorShape& value_shape, const void* value_data,
                        const TensorShape& output_shape, void* output_data,
                        const void* attention_mask = nullptr) = 0;
    
    virtual void backward(const TensorShape& grad_output_shape, const void* grad_output_data,
                         const TensorShape& query_shape, const void* query_data,
                         const TensorShape& key_shape, const void* key_data,
                         const TensorShape& value_shape, const void* value_data,
                         const TensorShape& grad_query_shape, void* grad_query_data,
                         const TensorShape& grad_key_shape, void* grad_key_data,
                         const TensorShape& grad_value_shape, void* grad_value_data,
                         const void* attention_mask = nullptr) = 0;
    
    // Memory management
    virtual size_t get_workspace_size(const AttentionParams& params) const = 0;
    virtual void set_workspace(void* workspace_ptr, size_t workspace_size) = 0;
    
    // Query methods
    virtual bool supports_data_type(DataType data_type) const = 0;
    virtual bool supports_memory_layout(MemoryLayout layout) const = 0;
    virtual std::string get_backend_name() const = 0;
    virtual std::string get_version() const = 0;
    
    // Performance and profiling
    virtual void enable_profiling(bool enable) = 0;
    virtual std::vector<std::pair<std::string, double>> get_profiling_results() const = 0;
    virtual void reset_profiling() = 0;
    
    // Validation methods
    virtual AttentionError validate_inputs(const TensorShape& query_shape,
                                          const TensorShape& key_shape,
                                          const TensorShape& value_shape,
                                          const TensorShape& output_shape,
                                          const AttentionParams& params) const = 0;
    
    // Convenience methods with error handling
    void safe_forward(const TensorShape& query_shape, const void* query_data,
                     const TensorShape& key_shape, const void* key_data,
                     const TensorShape& value_shape, const void* value_data,
                     const TensorShape& output_shape, void* output_data,
                     const void* attention_mask = nullptr) {
        AttentionError error = validate_inputs(query_shape, key_shape, value_shape, output_shape, current_params_);
        if (error != AttentionError::SUCCESS) {
            throw AttentionException(error, "Input validation failed in forward pass");
        }
        forward(query_shape, query_data, key_shape, key_data, value_shape, value_data, 
                output_shape, output_data, attention_mask);
    }
    
    void safe_backward(const TensorShape& grad_output_shape, const void* grad_output_data,
                      const TensorShape& query_shape, const void* query_data,
                      const TensorShape& key_shape, const void* key_data,
                      const TensorShape& value_shape, const void* value_data,
                      const TensorShape& grad_query_shape, void* grad_query_data,
                      const TensorShape& grad_key_shape, void* grad_key_data,
                      const TensorShape& grad_value_shape, void* grad_value_data,
                      const void* attention_mask = nullptr) {
        AttentionError error = validate_inputs(query_shape, key_shape, value_shape, grad_output_shape, current_params_);
        if (error != AttentionError::SUCCESS) {
            throw AttentionException(error, "Input validation failed in backward pass");
        }
        backward(grad_output_shape, grad_output_data, query_shape, query_data, 
                key_shape, key_data, value_shape, value_data,
                grad_query_shape, grad_query_data, grad_key_shape, grad_key_data, 
                grad_value_shape, grad_value_data, attention_mask);
    }

protected:
    AttentionParams current_params_;
    PerformanceConfig current_perf_config_;
    void* workspace_ptr_ = nullptr;
    size_t workspace_size_ = 0;
    bool profiling_enabled_ = false;
};

// Helper functions for error handling
inline std::string attention_error_to_string(AttentionError error) {
    switch (error) {
        case AttentionError::SUCCESS:
            return "Success";
        case AttentionError::INVALID_INPUT:
            return "Invalid input tensor";
        case AttentionError::INVALID_PARAMETERS:
            return "Invalid attention parameters";
        case AttentionError::MEMORY_ALLOCATION_FAILED:
            return "Memory allocation failed";
        case AttentionError::COMPUTATION_FAILED:
            return "Computation failed";
        case AttentionError::BACKEND_NOT_AVAILABLE:
            return "Backend not available";
        case AttentionError::UNSUPPORTED_OPERATION:
            return "Unsupported operation";
        case AttentionError::INTERNAL_ERROR:
            return "Internal error";
        default:
            return "Unknown error";
    }
}

} // namespace attention_hpc

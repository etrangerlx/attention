#pragma once

#include "algorithms/flash_attention.hpp"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <memory>
#include <vector>
#include <unordered_map>

namespace attention_hpc {

// CUDA-specific error handling
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw AttentionException(AttentionError::COMPUTATION_FAILED, \
                                "CUDA error: " + std::string(cudaGetErrorString(error))); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw AttentionException(AttentionError::COMPUTATION_FAILED, \
                                "cuBLAS error: " + std::to_string(status)); \
    } \
} while(0)

#define CUDNN_CHECK(call) do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        throw AttentionException(AttentionError::COMPUTATION_FAILED, \
                                "cuDNN error: " + std::string(cudnnGetErrorString(status))); \
    } \
} while(0)

// Forward declarations for CUDA kernels
extern "C" {
    void launch_flash_attention_kernel(
        const float* query, const float* key, const float* value,
        float* output, float* attention_scores,
        int batch_size, int num_heads, int seq_len, int head_dim,
        float scale, bool causal_mask,
        int block_size_q, int block_size_k,
        cudaStream_t stream);
    
    void launch_flash_attention_backward_kernel(
        const float* grad_output, const float* query, const float* key, const float* value,
        float* grad_query, float* grad_key, float* grad_value,
        const float* attention_weights,
        int batch_size, int num_heads, int seq_len, int head_dim,
        float scale, bool causal_mask,
        cudaStream_t stream);
    
    void launch_online_softmax_kernel(
        const float* attention_scores, float* attention_weights,
        float* max_vals, float* sum_exp,
        int batch_size, int num_heads, int seq_len,
        cudaStream_t stream);
    
    void launch_transpose_kernel(
        const float* input, float* output,
        int batch_size, int seq_len, int num_heads, int head_dim,
        bool forward, cudaStream_t stream);
}

// CUDA memory management utility
class CudaMemoryManager {
public:
    CudaMemoryManager(int device_id = 0);
    ~CudaMemoryManager();
    
    void* allocate(size_t size, size_t alignment = 256);
    void deallocate(void* ptr);
    void copy_host_to_device(void* dst, const void* src, size_t size, cudaStream_t stream = 0);
    void copy_device_to_host(void* dst, const void* src, size_t size, cudaStream_t stream = 0);
    void copy_device_to_device(void* dst, const void* src, size_t size, cudaStream_t stream = 0);
    void set_zero(void* ptr, size_t size, cudaStream_t stream = 0);
    
    size_t get_free_memory() const;
    size_t get_total_memory() const;
    size_t get_allocated_memory() const;
    
    void synchronize();
    void set_device(int device_id);
    int get_device() const;

private:
    int device_id_;
    size_t allocated_memory_;
    std::unordered_map<void*, size_t> allocations_;
    mutable std::mutex allocation_mutex_;
};

// CUDA stream management
class CudaStreamManager {
public:
    CudaStreamManager();
    ~CudaStreamManager();
    
    cudaStream_t get_stream(const std::string& stream_name = "default");
    cudaStream_t create_stream(const std::string& stream_name, int priority = 0);
    void destroy_stream(const std::string& stream_name);
    void synchronize_stream(const std::string& stream_name);
    void synchronize_all();
    
    cudaEvent_t create_event(const std::string& event_name);
    void record_event(const std::string& event_name, const std::string& stream_name = "default");
    void wait_event(const std::string& event_name, const std::string& stream_name = "default");
    float get_elapsed_time(const std::string& start_event, const std::string& end_event);
    
    void set_stream_priority(const std::string& stream_name, int priority);

private:
    std::unordered_map<std::string, cudaStream_t> streams_;
    std::unordered_map<std::string, cudaEvent_t> events_;
    mutable std::mutex streams_mutex_;
    mutable std::mutex events_mutex_;
};

// Multi-GPU management
class MultiGpuManager {
public:
    MultiGpuManager();
    ~MultiGpuManager() = default;
    
    int get_device_count() const;
    std::vector<int> get_available_devices() const;
    void set_device(int device_id);
    int get_current_device() const;
    
    void enable_peer_access();
    void disable_peer_access();
    bool can_access_peer(int device_from, int device_to) const;
    
    void distribute_computation(
        const std::vector<int>& devices,
        const std::function<void(int)>& compute_func);
    
    void synchronize_devices(const std::vector<int>& devices);
    
    struct DeviceInfo {
        int device_id;
        std::string name;
        size_t total_memory;
        size_t free_memory;
        int compute_capability_major;
        int compute_capability_minor;
        int multiprocessor_count;
        int max_threads_per_block;
        int max_shared_memory_per_block;
        bool supports_tensor_cores;
    };
    
    std::vector<DeviceInfo> get_device_info() const;

private:
    int device_count_;
    std::vector<int> available_devices_;
    std::vector<std::vector<bool>> peer_access_matrix_;
};

// CUDA-specific FlashAttention configuration
struct CudaFlashAttentionConfig : public FlashAttentionConfig {
    // CUDA-specific parameters
    int device_id = 0;
    bool use_tensor_cores = true;
    bool use_fast_math = true;
    size_t shared_memory_size = 48 * 1024; // 48KB default
    
    // Multi-GPU settings
    std::vector<int> device_ids;
    bool enable_multi_gpu = false;
    bool enable_peer_to_peer = true;
    
    // Stream and synchronization settings
    int num_streams = 4;
    bool enable_async_execution = true;
    bool enable_memory_pool = true;
    
    // Kernel launch parameters
    dim3 block_dim = dim3(256, 1, 1);
    dim3 grid_dim = dim3(1, 1, 1);
    size_t dynamic_shared_mem = 0;
    
    // Performance tuning
    bool use_optimized_kernels = true;
    bool enable_kernel_fusion = true;
    int occupancy_target = 100; // Percentage of theoretical occupancy
};

// CUDA FlashAttention implementation
class CudaFlashAttention : public FlashAttention {
public:
    CudaFlashAttention(int device_id = 0);
    virtual ~CudaFlashAttention();
    
    // Override configuration methods
    void configure(const AttentionParams& params, 
                   const PerformanceConfig& perf_config = PerformanceConfig{}) override;
    
    // CUDA-specific configuration
    void configure_cuda(const CudaFlashAttentionConfig& cuda_config);
    
    // Override core computation methods
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
    
    // Override memory management
    size_t get_workspace_size(const AttentionParams& params) const override;
    void set_workspace(void* workspace_ptr, size_t workspace_size) override;
    
    // Override query methods
    bool supports_data_type(DataType data_type) const override;
    std::string get_backend_name() const override;
    std::string get_version() const override;
    
    // CUDA-specific methods
    
    // Multi-GPU support
    void enable_multi_gpu(const std::vector<int>& device_ids);
    void disable_multi_gpu();
    bool is_multi_gpu_enabled() const;
    void set_primary_device(int device_id);
    int get_primary_device() const;
    
    // Stream management
    void set_compute_stream(cudaStream_t stream);
    cudaStream_t get_compute_stream() const;
    void create_additional_streams(int num_streams);
    void synchronize_streams();
    
    // Event-based synchronization
    void record_event(const std::string& event_name);
    void wait_for_event(const std::string& event_name);
    float get_execution_time(const std::string& start_event, const std::string& end_event);
    
    // Memory management
    void* allocate_device_memory(size_t size);
    void deallocate_device_memory(void* ptr);
    void copy_to_device(void* dst, const void* src, size_t size, cudaStream_t stream = 0);
    void copy_from_device(void* dst, const void* src, size_t size, cudaStream_t stream = 0);
    
    // Performance optimization
    void optimize_kernel_launch_config(const AttentionParams& params);
    void enable_tensor_core_usage(bool enable = true);
    void set_math_mode(cudnnMathType_t math_type);
    
    // Profiling and debugging
    void enable_cuda_profiling(bool enable = true);
    void synchronize_device();
    size_t get_gpu_memory_usage() const;
    
    // Batch processing with multiple sequences
    void forward_multi_sequence(
        const std::vector<TensorShape>& query_shapes,
        const std::vector<const void*>& query_data,
        const std::vector<TensorShape>& key_shapes,
        const std::vector<const void*>& key_data,
        const std::vector<TensorShape>& value_shapes,
        const std::vector<const void*>& value_data,
        const std::vector<TensorShape>& output_shapes,
        const std::vector<void*>& output_data,
        const std::vector<const void*>& attention_masks = {});

protected:
    // CUDA-specific implementations of base class methods
    void compute_multihead_attention(
        const TensorShape& query_shape, const float* query_data,
        const TensorShape& key_shape, const float* key_data,
        const TensorShape& value_shape, const float* value_data,
        const TensorShape& output_shape, float* output_data,
        const float* attention_mask = nullptr) override;
    
    void compute_attention_block(
        const float* query_block, const float* key_block, const float* value_block,
        float* output_block, AttentionBlockState& block_state,
        size_t q_block_size, size_t kv_block_size, size_t head_dim,
        bool is_causal, const float* attention_mask_block = nullptr) override;
    
    void compute_online_softmax(
        const float* attention_scores, float* attention_weights,
        OnlineSoftmaxState& softmax_state,
        size_t num_keys, float scale_factor) override;
    
    // CUDA-specific helper methods
    void launch_flash_attention_cuda_kernel(
        const float* query, const float* key, const float* value,
        float* output, float* workspace,
        size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
        bool causal_mask, const float* attention_mask = nullptr);
    
    void launch_attention_backward_cuda_kernel(
        const float* grad_output, const float* query, const float* key, const float* value,
        float* grad_query, float* grad_key, float* grad_value,
        float* workspace, size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim);
    
    void optimize_cuda_memory_access(const AttentionParams& params);
    void configure_cuda_streams();
    void setup_cublas_handle();
    void setup_cudnn_handle();
    
    // Multi-GPU specific methods
    void distribute_attention_computation(
        const std::vector<int>& devices,
        const float* query, const float* key, const float* value,
        float* output, size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim);
    
    void gather_multi_gpu_results(
        const std::vector<void*>& device_outputs,
        void* final_output, size_t output_size);
    
    // Memory layout optimization
    void transpose_for_cuda_attention(const float* input, float* output,
                                     size_t batch_size, size_t seq_len,
                                     size_t num_heads, size_t head_dim,
                                     cudaStream_t stream = 0);
    
    void transpose_from_cuda_attention(const float* input, float* output,
                                      size_t batch_size, size_t seq_len,
                                      size_t num_heads, size_t head_dim,
                                      cudaStream_t stream = 0);

private:
    // Configuration state
    CudaFlashAttentionConfig cuda_config_;
    
    // Device management
    std::unique_ptr<CudaMemoryManager> memory_manager_;
    std::unique_ptr<CudaStreamManager> stream_manager_;
    std::unique_ptr<MultiGpuManager> multi_gpu_manager_;
    
    // CUDA handles
    cublasHandle_t cublas_handle_;
    cudnnHandle_t cudnn_handle_;
    
    // Device memory buffers
    void* d_query_buffer_;
    void* d_key_buffer_;
    void* d_value_buffer_;
    void* d_output_buffer_;
    void* d_attention_scores_buffer_;
    void* d_workspace_;
    
    // Buffer sizes
    size_t d_query_buffer_size_;
    size_t d_key_buffer_size_;
    size_t d_value_buffer_size_;
    size_t d_output_buffer_size_;
    size_t d_attention_scores_buffer_size_;
    size_t d_workspace_size_;
    
    // Multi-GPU state
    std::vector<int> active_devices_;
    std::vector<void*> device_buffers_;
    std::vector<cudaStream_t> device_streams_;
    bool multi_gpu_enabled_;
    int primary_device_;
    
    // Performance state
    bool tensor_cores_enabled_;
    cudnnMathType_t math_type_;
    
    // Helper methods
    void allocate_cuda_buffers(const AttentionParams& params);
    void deallocate_cuda_buffers();
    void setup_cuda_environment();
    void cleanup_cuda_environment();
    
    void validate_cuda_configuration() const;
    void optimize_cuda_kernel_parameters(const AttentionParams& params);
    
    dim3 calculate_grid_dim(size_t total_elements, dim3 block_dim) const;
    size_t calculate_shared_memory_usage(const AttentionParams& params) const;
    
    void handle_cuda_error(cudaError_t error, const std::string& operation) const;
    void handle_cublas_error(cublasStatus_t status, const std::string& operation) const;
    void handle_cudnn_error(cudnnStatus_t status, const std::string& operation) const;
    
    // Data type conversion utilities for CUDA
    void convert_to_cuda_float(const void* input, float* output,
                              size_t num_elements, DataType input_type,
                              cudaStream_t stream = 0);
    
    void convert_from_cuda_float(const float* input, void* output,
                                size_t num_elements, DataType output_type,
                                cudaStream_t stream = 0);
};

// Factory functions for CUDA FlashAttention
std::unique_ptr<CudaFlashAttention> create_cuda_flash_attention(int device_id = 0);
std::unique_ptr<CudaFlashAttention> create_multi_gpu_flash_attention(
    const std::vector<int>& device_ids);

// CUDA FlashAttention utilities
namespace cuda_flash_attention_utils {
    // Device capability checking
    bool is_cuda_available();
    bool supports_tensor_cores(int device_id = 0);
    bool supports_unified_memory(int device_id = 0);
    
    // Performance optimization
    CudaFlashAttentionConfig get_optimal_cuda_config(
        const AttentionParams& params, int device_id = 0);
    
    std::tuple<dim3, dim3, size_t> calculate_optimal_launch_config(
        const AttentionParams& params, int device_id = 0);
    
    // Memory usage estimation
    size_t estimate_cuda_memory_usage(
        const AttentionParams& params,
        const CudaFlashAttentionConfig& config);
    
    // Multi-GPU load balancing
    std::vector<AttentionParams> distribute_workload(
        const AttentionParams& params,
        const std::vector<int>& device_ids);
    
    // Performance benchmarking
    double benchmark_cuda_flash_attention(
        const AttentionParams& params,
        const CudaFlashAttentionConfig& config,
        int num_iterations = 10);
}

} // namespace attention_hpc

#endif // HAVE_CUDA

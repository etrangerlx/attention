#pragma once

#include "algorithms/flash_attention.hpp"

#ifdef HAVE_OPENCL
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>

namespace attention_hpc {

// OpenCL-specific error handling
#define CL_CHECK(call) do { \
    cl_int error = call; \
    if (error != CL_SUCCESS) { \
        throw AttentionException(AttentionError::COMPUTATION_FAILED, \
                                "OpenCL error: " + std::to_string(error)); \
    } \
} while(0)

// OpenCL device information structure
struct OpenCLDeviceInfo {
    cl_device_id device_id;
    cl_device_type device_type;
    std::string device_name;
    std::string vendor;
    std::string version;
    std::string driver_version;
    size_t global_memory_size;
    size_t local_memory_size;
    size_t max_work_group_size;
    std::vector<size_t> max_work_item_sizes;
    cl_uint max_compute_units;
    cl_uint max_clock_frequency;
    bool supports_double_precision;
    bool supports_half_precision;
    std::vector<std::string> extensions;
};

// OpenCL platform information structure
struct OpenCLPlatformInfo {
    cl_platform_id platform_id;
    std::string platform_name;
    std::string vendor;
    std::string version;
    std::string profile;
    std::vector<std::string> extensions;
    std::vector<OpenCLDeviceInfo> devices;
};

// OpenCL context management
class OpenCLContext {
public:
    OpenCLContext();
    ~OpenCLContext();
    
    // Platform and device detection
    std::vector<OpenCLPlatformInfo> get_available_platforms();
    bool initialize_context(cl_platform_id platform_id, cl_device_id device_id);
    bool initialize_context_from_type(cl_device_type device_type);
    void release_context();
    
    // Context accessors
    cl_context get_context() const { return context_; }
    cl_device_id get_device() const { return device_; }
    cl_platform_id get_platform() const { return platform_; }
    
    // Device information
    OpenCLDeviceInfo get_device_info() const;
    OpenCLPlatformInfo get_platform_info() const;
    
    // Memory management
    cl_mem create_buffer(size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE);
    void release_buffer(cl_mem buffer);
    void* map_buffer(cl_mem buffer, size_t size, cl_map_flags flags);
    void unmap_buffer(cl_mem buffer, void* mapped_ptr);
    
    // Utility methods
    bool is_initialized() const { return context_ != nullptr; }
    std::string get_device_extensions() const;
    bool supports_extension(const std::string& extension) const;
    
private:
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    bool initialized_;
    mutable std::mutex context_mutex_;
    
    void query_device_info();
    void query_platform_info();
};

// OpenCL command queue management
class OpenCLCommandQueue {
public:
    OpenCLCommandQueue(cl_context context, cl_device_id device);
    ~OpenCLCommandQueue();
    
    // Queue management
    bool create_queue(cl_command_queue_properties properties = 0);
    void release_queue();
    cl_command_queue get_queue() const { return queue_; }
    
    // Command execution
    void enqueue_kernel(cl_kernel kernel, size_t global_work_size, 
                       size_t local_work_size = 0, cl_event* event = nullptr);
    void enqueue_kernel_nd(cl_kernel kernel, cl_uint work_dim,
                          const size_t* global_work_size,
                          const size_t* local_work_size = nullptr,
                          cl_event* event = nullptr);
    
    // Memory operations
    void enqueue_read_buffer(cl_mem buffer, void* host_ptr, size_t size,
                            size_t offset = 0, cl_event* event = nullptr);
    void enqueue_write_buffer(cl_mem buffer, const void* host_ptr, size_t size,
                             size_t offset = 0, cl_event* event = nullptr);
    void enqueue_copy_buffer(cl_mem src_buffer, cl_mem dst_buffer, size_t size,
                            size_t src_offset = 0, size_t dst_offset = 0,
                            cl_event* event = nullptr);
    void enqueue_fill_buffer(cl_mem buffer, const void* pattern, size_t pattern_size,
                            size_t offset, size_t size, cl_event* event = nullptr);
    
    // Synchronization
    void finish();
    void flush();
    void wait_for_events(const std::vector<cl_event>& events);
    
    // Profiling
    void enable_profiling(bool enable = true);
    bool is_profiling_enabled() const { return profiling_enabled_; }
    
private:
    cl_context context_;
    cl_device_id device_;
    cl_command_queue queue_;
    bool profiling_enabled_;
    bool initialized_;
    mutable std::mutex queue_mutex_;
};

// OpenCL kernel program management
class OpenCLKernelManager {
public:
    OpenCLKernelManager(cl_context context, cl_device_id device);
    ~OpenCLKernelManager();
    
    // Program compilation
    bool compile_program_from_source(const std::string& source, 
                                   const std::string& program_name,
                                   const std::string& build_options = "");
    bool compile_program_from_file(const std::string& filename,
                                 const std::string& program_name,
                                 const std::string& build_options = "");
    bool load_program_from_binary(const std::vector<unsigned char>& binary,
                                const std::string& program_name);
    
    // Kernel creation and management
    cl_kernel create_kernel(const std::string& program_name, 
                           const std::string& kernel_name);
    void release_kernel(cl_kernel kernel);
    void release_program(const std::string& program_name);
    
    // Kernel argument setting
    void set_kernel_arg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value);
    template<typename T>
    void set_kernel_arg(cl_kernel kernel, cl_uint arg_index, const T& arg_value) {
        set_kernel_arg(kernel, arg_index, sizeof(T), &arg_value);
    }
    
    // Program caching
    bool save_program_binary(const std::string& program_name, const std::string& filename);
    std::vector<unsigned char> get_program_binary(const std::string& program_name);
    
    // Build information
    std::string get_build_log(const std::string& program_name) const;
    std::string get_build_options(const std::string& program_name) const;
    cl_build_status get_build_status(const std::string& program_name) const;
    
    // Utility methods
    bool has_program(const std::string& program_name) const;
    std::vector<std::string> get_program_names() const;
    void clear_all_programs();
    
private:
    cl_context context_;
    cl_device_id device_;
    std::unordered_map<std::string, cl_program> programs_;
    std::unordered_map<cl_kernel, std::string> kernel_names_;
    mutable std::mutex programs_mutex_;
    
    std::string load_source_from_file(const std::string& filename);
    bool build_program(cl_program program, const std::string& build_options);
};

// Cross-platform device detection and selection
class OpenCLDeviceSelector {
public:
    OpenCLDeviceSelector() = default;
    ~OpenCLDeviceSelector() = default;
    
    // Platform and device enumeration
    static std::vector<OpenCLPlatformInfo> enumerate_platforms();
    static std::vector<OpenCLDeviceInfo> enumerate_devices(cl_device_type device_type = CL_DEVICE_TYPE_ALL);
    static std::vector<OpenCLDeviceInfo> enumerate_devices_for_platform(cl_platform_id platform);
    
    // Device selection strategies
    static OpenCLDeviceInfo select_best_device();
    static OpenCLDeviceInfo select_best_device_by_type(cl_device_type device_type);
    static OpenCLDeviceInfo select_device_by_name(const std::string& device_name);
    static OpenCLDeviceInfo select_device_by_vendor(const std::string& vendor_name);
    
    // Device capability checking
    static bool supports_flash_attention(const OpenCLDeviceInfo& device_info);
    static bool supports_data_type(const OpenCLDeviceInfo& device_info, DataType data_type);
    static size_t estimate_memory_usage(const OpenCLDeviceInfo& device_info, 
                                       const AttentionParams& params);
    
    // Performance estimation
    static int score_device_for_attention(const OpenCLDeviceInfo& device_info,
                                        const AttentionParams& params);
    static std::vector<OpenCLDeviceInfo> rank_devices_for_attention(
        const std::vector<OpenCLDeviceInfo>& devices,
        const AttentionParams& params);
    
    // Compatibility checking
    static bool check_opencl_version_compatibility(const OpenCLDeviceInfo& device_info,
                                                  int major_version, int minor_version);
    static bool check_extension_support(const OpenCLDeviceInfo& device_info,
                                       const std::vector<std::string>& required_extensions);
    
private:
    static OpenCLDeviceInfo create_device_info(cl_device_id device_id);
    static OpenCLPlatformInfo create_platform_info(cl_platform_id platform_id);
    static std::string get_device_info_string(cl_device_id device_id, cl_device_info param_name);
    static std::string get_platform_info_string(cl_platform_id platform_id, cl_platform_info param_name);
};

// OpenCL-specific FlashAttention configuration
struct OpenCLFlashAttentionConfig : public FlashAttentionConfig {
    // Device selection
    cl_device_type preferred_device_type = CL_DEVICE_TYPE_GPU;
    std::string preferred_device_name;
    std::string preferred_vendor;
    
    // Kernel compilation options
    std::string build_options = "-cl-fast-relaxed-math";
    bool enable_kernel_caching = true;
    std::string kernel_cache_directory = "./opencl_cache";
    
    // Work group configuration
    size_t work_group_size = 256;
    std::vector<size_t> local_work_size;
    bool auto_tune_work_size = true;
    
    // Memory optimization
    bool use_local_memory = true;
    bool enable_memory_coalescing = true;
    size_t preferred_vector_width = 4;
    
    // Performance tuning
    bool enable_async_execution = true;
    bool enable_event_profiling = false;
    int max_concurrent_kernels = 4;
    
    // Platform-specific optimizations
    bool enable_intel_optimizations = false;
    bool enable_nvidia_optimizations = false;
    bool enable_amd_optimizations = false;
};

// OpenCL FlashAttention implementation
class OpenCLFlashAttention : public FlashAttention {
public:
    OpenCLFlashAttention();
    virtual ~OpenCLFlashAttention();
    
    // Override configuration methods
    void configure(const AttentionParams& params, 
                   const PerformanceConfig& perf_config = PerformanceConfig{}) override;
    
    // OpenCL-specific configuration
    void configure_opencl(const OpenCLFlashAttentionConfig& opencl_config);
    
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
    
    // OpenCL-specific methods
    
    // Device management
    bool initialize_device(cl_device_type device_type = CL_DEVICE_TYPE_GPU);
    bool initialize_device(const std::string& device_name, const std::string& vendor = "");
    void release_device();
    OpenCLDeviceInfo get_device_info() const;
    
    // Kernel management
    void precompile_kernels();
    void reload_kernels();
    void clear_kernel_cache();
    
    // Memory management
    cl_mem allocate_device_buffer(size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE);
    void deallocate_device_buffer(cl_mem buffer);
    void copy_to_device(cl_mem buffer, const void* host_data, size_t size, size_t offset = 0);
    void copy_from_device(void* host_data, cl_mem buffer, size_t size, size_t offset = 0);
    
    // Performance optimization
    void auto_tune_work_group_size(const AttentionParams& params);
    void optimize_memory_access_pattern(const AttentionParams& params);
    void enable_vendor_specific_optimizations();
    
    // Profiling and debugging
    void enable_opencl_profiling(bool enable = true);
    std::vector<std::pair<std::string, double>> get_kernel_profiling_results() const;
    void print_device_capabilities() const;
    
    // Event-based synchronization
    void wait_for_completion();
    void create_event_marker(const std::string& event_name);
    double get_elapsed_time(const std::string& start_event, const std::string& end_event);
    
    // Multi-device support
    void enable_multi_device(const std::vector<OpenCLDeviceInfo>& devices);
    void disable_multi_device();
    bool is_multi_device_enabled() const;
    
protected:
    // OpenCL-specific implementations of base class methods
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
    
    // OpenCL-specific helper methods
    void launch_flash_attention_kernel(
        cl_mem query_buffer, cl_mem key_buffer, cl_mem value_buffer,
        cl_mem output_buffer, cl_mem workspace_buffer,
        size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
        bool causal_mask, cl_mem attention_mask_buffer = nullptr);
    
    void launch_attention_backward_kernel(
        cl_mem grad_output_buffer, cl_mem query_buffer, cl_mem key_buffer, cl_mem value_buffer,
        cl_mem grad_query_buffer, cl_mem grad_key_buffer, cl_mem grad_value_buffer,
        cl_mem workspace_buffer, size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim);
    
    void setup_opencl_kernels();
    void optimize_kernel_parameters(const AttentionParams& params);
    
    // Memory layout optimization
    void transpose_for_opencl_attention(cl_mem input_buffer, cl_mem output_buffer,
                                       size_t batch_size, size_t seq_len,
                                       size_t num_heads, size_t head_dim);
    
    void transpose_from_opencl_attention(cl_mem input_buffer, cl_mem output_buffer,
                                        size_t batch_size, size_t seq_len,
                                        size_t num_heads, size_t head_dim);
    
    // Work group size optimization
    std::vector<size_t> calculate_optimal_work_group_size(
        size_t global_work_size, const OpenCLDeviceInfo& device_info);
    size_t calculate_local_memory_usage(const AttentionParams& params);
    
    // Vendor-specific optimizations
    void apply_intel_optimizations();
    void apply_nvidia_optimizations();
    void apply_amd_optimizations();

private:
    // Configuration state
    OpenCLFlashAttentionConfig opencl_config_;
    
    // OpenCL management objects
    std::unique_ptr<OpenCLContext> context_;
    std::unique_ptr<OpenCLCommandQueue> command_queue_;
    std::unique_ptr<OpenCLKernelManager> kernel_manager_;
    
    // Device information
    OpenCLDeviceInfo device_info_;
    bool device_initialized_;
    
    // Kernels
    cl_kernel flash_attention_kernel_;
    cl_kernel attention_backward_kernel_;
    cl_kernel softmax_kernel_;
    cl_kernel transpose_kernel_;
    cl_kernel reduce_kernel_;
    
    // Device memory buffers
    cl_mem d_query_buffer_;
    cl_mem d_key_buffer_;
    cl_mem d_value_buffer_;
    cl_mem d_output_buffer_;
    cl_mem d_attention_scores_buffer_;
    cl_mem d_workspace_buffer_;
    
    // Buffer sizes
    size_t d_query_buffer_size_;
    size_t d_key_buffer_size_;
    size_t d_value_buffer_size_;
    size_t d_output_buffer_size_;
    size_t d_attention_scores_buffer_size_;
    size_t d_workspace_buffer_size_;
    
    // Performance and profiling
    std::vector<cl_event> profiling_events_;
    std::unordered_map<std::string, cl_event> event_markers_;
    bool profiling_enabled_;
    
    // Multi-device state
    std::vector<OpenCLDeviceInfo> active_devices_;
    std::vector<std::unique_ptr<OpenCLContext>> device_contexts_;
    std::vector<std::unique_ptr<OpenCLCommandQueue>> device_queues_;
    bool multi_device_enabled_;
    
    // Helper methods
    void allocate_opencl_buffers(const AttentionParams& params);
    void deallocate_opencl_buffers();
    void setup_opencl_environment();
    void cleanup_opencl_environment();
    
    void validate_opencl_configuration() const;
    void load_kernel_sources();
    void compile_kernels_with_optimization();
    
    std::string get_kernel_source(const std::string& kernel_name);
    std::string generate_build_options(const AttentionParams& params);
    
    void handle_opencl_error(cl_int error, const std::string& operation) const;
    
    // Data type conversion utilities for OpenCL
    void convert_to_opencl_float(const void* input, cl_mem output_buffer,
                                size_t num_elements, DataType input_type);
    
    void convert_from_opencl_float(cl_mem input_buffer, void* output,
                                  size_t num_elements, DataType output_type);
    
    // Kernel source storage
    static const char* flash_attention_kernel_source_;
    static const char* attention_backward_kernel_source_;
    static const char* softmax_kernel_source_;
    static const char* transpose_kernel_source_;
    static const char* reduce_kernel_source_;
};

// Factory functions for OpenCL FlashAttention
std::unique_ptr<OpenCLFlashAttention> create_opencl_flash_attention();
std::unique_ptr<OpenCLFlashAttention> create_opencl_flash_attention_with_device(
    const std::string& device_name, const std::string& vendor = "");

// OpenCL FlashAttention utilities
namespace opencl_flash_attention_utils {
    // OpenCL availability checking
    bool is_opencl_available();
    std::vector<std::string> get_available_platforms();
    std::vector<std::string> get_available_devices();
    
    // Configuration optimization
    OpenCLFlashAttentionConfig get_optimal_opencl_config(
        const AttentionParams& params, const OpenCLDeviceInfo& device_info);
    
    std::vector<size_t> calculate_optimal_work_group_size(
        const AttentionParams& params, const OpenCLDeviceInfo& device_info);
    
    // Memory usage estimation
    size_t estimate_opencl_memory_usage(
        const AttentionParams& params,
        const OpenCLFlashAttentionConfig& config);
    
    // Performance benchmarking
    double benchmark_opencl_flash_attention(
        const AttentionParams& params,
        const OpenCLFlashAttentionConfig& config,
        int num_iterations = 10);
    
    // Device capability analysis
    struct OpenCLCapabilityReport {
        bool supports_flash_attention;
        bool supports_fp16;
        bool supports_fp64;
        size_t max_work_group_size;
        size_t max_local_memory;
        size_t max_global_memory;
        std::vector<std::string> supported_extensions;
        int performance_score;
    };
    
    OpenCLCapabilityReport analyze_device_capabilities(const OpenCLDeviceInfo& device_info);
    
    // Platform-specific utilities
    bool is_intel_platform(const OpenCLDeviceInfo& device_info);
    bool is_nvidia_platform(const OpenCLDeviceInfo& device_info);
    bool is_amd_platform(const OpenCLDeviceInfo& device_info);
    
    std::string get_optimal_build_options(const OpenCLDeviceInfo& device_info);
    std::vector<std::string> get_recommended_extensions(const OpenCLDeviceInfo& device_info);
}

} // namespace attention_hpc

#endif // HAVE_OPENCL

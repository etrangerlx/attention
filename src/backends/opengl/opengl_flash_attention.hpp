#pragma once

#include "algorithms/flash_attention.hpp"

#ifdef HAVE_OPENGL
#include <GL/gl.h>
#include <GL/glext.h>
#ifdef _WIN32
#include <GL/wglext.h>
#elif defined(__linux__)
#include <GL/glx.h>
#elif defined(__APPLE__)
#include <OpenGL/OpenGL.h>
#endif
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>

namespace attention_hpc {

// OpenGL-specific error handling
#define GL_CHECK(call) do { \
    call; \
    GLenum error = glGetError(); \
    if (error != GL_NO_ERROR) { \
        throw AttentionException(AttentionError::COMPUTATION_FAILED, \
                                "OpenGL error: " + std::to_string(error)); \
    } \
} while(0)

// OpenGL context management
class OpenGLContext {
public:
    OpenGLContext();
    ~OpenGLContext();
    
    // Context creation and management
    bool create_context(int width = 1, int height = 1);
    bool make_current();
    void release_context();
    bool is_initialized() const { return initialized_; }
    
    // OpenGL version and capability queries
    std::string get_version() const;
    std::string get_vendor() const;
    std::string get_renderer() const;
    std::string get_extensions() const;
    bool supports_compute_shaders() const;
    bool supports_shader_storage_buffer() const;
    
    // Compute shader limits
    int get_max_compute_work_group_count(int dimension) const;
    int get_max_compute_work_group_size(int dimension) const;
    int get_max_compute_work_group_invocations() const;
    int get_max_compute_shared_memory_size() const;
    
private:
    bool initialized_;
    
#ifdef _WIN32
    HWND hwnd_;
    HDC hdc_;
    HGLRC context_;
#elif defined(__linux__)
    Display* display_;
    GLXContext context_;
    GLXPbuffer pbuffer_;
#elif defined(__APPLE__)
    CGLContextObj context_;
#endif
    
    void cleanup_platform_specific();
    bool create_platform_context();
};

// Compute shader program management
class OpenGLComputeShader {
public:
    OpenGLComputeShader() = default;
    ~OpenGLComputeShader();
    
    // Shader compilation and linking
    bool compile_from_source(const std::string& source);
    bool compile_from_file(const std::string& filename);
    bool link_program();
    void release_program();
    
    // Program usage
    void use_program();
    GLuint get_program_id() const { return program_id_; }
    bool is_valid() const { return program_id_ != 0; }
    
    // Uniform setting
    void set_uniform(const std::string& name, int value);
    void set_uniform(const std::string& name, float value);
    void set_uniform(const std::string& name, const std::vector<int>& values);
    void set_uniform(const std::string& name, const std::vector<float>& values);
    
    // Dispatch compute shader
    void dispatch(GLuint num_groups_x, GLuint num_groups_y = 1, GLuint num_groups_z = 1);
    void dispatch_indirect(GLintptr indirect);
    
    // Shader storage buffer binding
    void bind_storage_buffer(GLuint binding_point, GLuint buffer_id);
    
    // Memory barriers and synchronization
    void memory_barrier(GLbitfield barriers = GL_ALL_BARRIER_BITS);
    void finish();
    
    // Program introspection
    std::vector<std::string> get_active_uniforms() const;
    std::vector<std::string> get_active_storage_blocks() const;
    int get_uniform_location(const std::string& name) const;
    int get_storage_block_index(const std::string& name) const;
    
private:
    GLuint program_id_ = 0;
    GLuint shader_id_ = 0;
    std::unordered_map<std::string, GLint> uniform_locations_;
    std::unordered_map<std::string, GLuint> storage_block_indices_;
    
    std::string load_source_from_file(const std::string& filename);
    bool check_compile_errors(GLuint shader, const std::string& type);
    bool check_link_errors();
    void cache_uniform_locations();
    void cache_storage_block_indices();
};

// Shader Storage Buffer Object (SSBO) management
class OpenGLStorageBuffer {
public:
    OpenGLStorageBuffer() = default;
    ~OpenGLStorageBuffer();
    
    // Buffer creation and management
    bool create_buffer(size_t size, GLenum usage = GL_DYNAMIC_DRAW);
    void release_buffer();
    bool is_valid() const { return buffer_id_ != 0; }
    GLuint get_buffer_id() const { return buffer_id_; }
    size_t get_size() const { return size_; }
    
    // Data operations
    void upload_data(const void* data, size_t size, size_t offset = 0);
    void download_data(void* data, size_t size, size_t offset = 0);
    void clear_data(int value = 0);
    void copy_from_buffer(const OpenGLStorageBuffer& src, size_t src_offset = 0, 
                         size_t dst_offset = 0, size_t size = 0);
    
    // Buffer binding
    void bind_to_index(GLuint binding_point);
    void bind_range(GLuint binding_point, size_t offset, size_t size);
    
    // Memory mapping
    void* map_buffer(GLenum access = GL_READ_WRITE);
    void unmap_buffer();
    
    // Buffer state queries
    GLenum get_usage() const { return usage_; }
    bool is_mapped() const;
    
private:
    GLuint buffer_id_ = 0;
    size_t size_ = 0;
    GLenum usage_ = GL_DYNAMIC_DRAW;
    void* mapped_ptr_ = nullptr;
};

// OpenGL buffer pool for efficient memory management
class OpenGLBufferPool {
public:
    OpenGLBufferPool(size_t initial_pool_size = 16);
    ~OpenGLBufferPool();
    
    // Buffer allocation and deallocation
    std::shared_ptr<OpenGLStorageBuffer> allocate_buffer(size_t size, GLenum usage = GL_DYNAMIC_DRAW);
    void deallocate_buffer(std::shared_ptr<OpenGLStorageBuffer> buffer);
    
    // Pool management
    void resize_pool(size_t new_size);
    void clear_pool();
    size_t get_pool_size() const;
    size_t get_free_buffers() const;
    size_t get_allocated_buffers() const;
    
    // Memory statistics
    size_t get_total_memory_usage() const;
    size_t get_peak_memory_usage() const;
    
private:
    std::vector<std::shared_ptr<OpenGLStorageBuffer>> free_buffers_;
    std::vector<std::shared_ptr<OpenGLStorageBuffer>> allocated_buffers_;
    size_t total_memory_usage_;
    size_t peak_memory_usage_;
    mutable std::mutex pool_mutex_;
    
    void create_new_buffers(size_t count, size_t size);
};

// OpenGL-specific FlashAttention configuration
struct OpenGLFlashAttentionConfig : public FlashAttentionConfig {
    // Compute shader work group configuration
    GLuint work_group_size_x = 16;
    GLuint work_group_size_y = 16;
    GLuint work_group_size_z = 1;
    
    // Memory management settings
    bool use_buffer_pool = true;
    size_t initial_buffer_pool_size = 32;
    GLenum buffer_usage = GL_DYNAMIC_DRAW;
    
    // Synchronization settings
    bool enable_memory_barriers = true;
    GLbitfield memory_barrier_flags = GL_SHADER_STORAGE_BARRIER_BIT;
    bool enable_finish_after_dispatch = false;
    
    // Shader compilation settings
    std::string shader_version = "#version 430";
    std::vector<std::string> shader_defines;
    bool enable_shader_caching = true;
    std::string shader_cache_directory = "./opengl_cache";
    
    // Performance optimization
    bool enable_async_operations = false;
    bool optimize_memory_layout = true;
    size_t preferred_block_size = 64;
    
    // Debug and profiling
    bool enable_gpu_profiling = false;
    bool validate_shader_compilation = true;
};

// OpenGL FlashAttention implementation
class OpenGLFlashAttention : public FlashAttention {
public:
    OpenGLFlashAttention();
    virtual ~OpenGLFlashAttention();
    
    // Override configuration methods
    void configure(const AttentionParams& params, 
                   const PerformanceConfig& perf_config = PerformanceConfig{}) override;
    
    // OpenGL-specific configuration
    void configure_opengl(const OpenGLFlashAttentionConfig& opengl_config);
    
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
    
    // OpenGL-specific methods
    
    // Context management
    bool initialize_opengl_context();
    void release_opengl_context();
    bool is_opengl_initialized() const;
    OpenGLContext* get_opengl_context() const;
    
    // Shader management
    bool compile_compute_shaders();
    void reload_shaders();
    void clear_shader_cache();
    std::vector<std::string> get_available_shaders() const;
    
    // Buffer management
    std::shared_ptr<OpenGLStorageBuffer> create_storage_buffer(size_t size, GLenum usage = GL_DYNAMIC_DRAW);
    void upload_tensor_data(std::shared_ptr<OpenGLStorageBuffer> buffer, const void* data, 
                           size_t size, DataType data_type);
    void download_tensor_data(std::shared_ptr<OpenGLStorageBuffer> buffer, void* data, 
                             size_t size, DataType data_type);
    
    // Compute shader execution
    void dispatch_attention_kernel(const AttentionParams& params,
                                  std::shared_ptr<OpenGLStorageBuffer> query_buffer,
                                  std::shared_ptr<OpenGLStorageBuffer> key_buffer,
                                  std::shared_ptr<OpenGLStorageBuffer> value_buffer,
                                  std::shared_ptr<OpenGLStorageBuffer> output_buffer,
                                  std::shared_ptr<OpenGLStorageBuffer> workspace_buffer = nullptr);
    
    void dispatch_backward_kernel(const AttentionParams& params,
                                 std::shared_ptr<OpenGLStorageBuffer> grad_output_buffer,
                                 std::shared_ptr<OpenGLStorageBuffer> query_buffer,
                                 std::shared_ptr<OpenGLStorageBuffer> key_buffer,
                                 std::shared_ptr<OpenGLStorageBuffer> value_buffer,
                                 std::shared_ptr<OpenGLStorageBuffer> grad_query_buffer,
                                 std::shared_ptr<OpenGLStorageBuffer> grad_key_buffer,
                                 std::shared_ptr<OpenGLStorageBuffer> grad_value_buffer);
    
    // Performance optimization
    void optimize_work_group_size(const AttentionParams& params);
    void optimize_memory_layout();
    void benchmark_shader_performance(const AttentionParams& params, int num_iterations = 10);
    
    // Synchronization and debugging
    void synchronize_gpu();
    void insert_memory_barrier(GLbitfield barriers = GL_ALL_BARRIER_BITS);
    void validate_buffers() const;
    void print_opengl_info() const;
    
    // Profiling support
    void enable_opengl_profiling(bool enable = true);
    std::vector<std::pair<std::string, double>> get_gpu_profiling_results() const;
    void reset_gpu_profiling();

protected:
    // OpenGL-specific implementations of base class methods
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
    
    // OpenGL-specific helper methods
    void setup_compute_shaders();
    void setup_storage_buffers(const AttentionParams& params);
    void cleanup_opengl_resources();
    
    std::string load_shader_source(const std::string& shader_name);
    std::string generate_shader_defines(const AttentionParams& params);
    std::string preprocess_shader_source(const std::string& source, const AttentionParams& params);
    
    void calculate_dispatch_dimensions(const AttentionParams& params, 
                                     GLuint& num_groups_x, GLuint& num_groups_y, GLuint& num_groups_z);
    
    // Data type conversion utilities
    void convert_to_opengl_format(const void* input, std::shared_ptr<OpenGLStorageBuffer> output_buffer,
                                 size_t num_elements, DataType input_type);
    void convert_from_opengl_format(std::shared_ptr<OpenGLStorageBuffer> input_buffer, void* output,
                                   size_t num_elements, DataType output_type);

private:
    // Configuration state
    OpenGLFlashAttentionConfig opengl_config_;
    
    // OpenGL management objects
    std::unique_ptr<OpenGLContext> context_;
    std::unique_ptr<OpenGLBufferPool> buffer_pool_;
    
    // Compute shaders
    std::unique_ptr<OpenGLComputeShader> flash_attention_shader_;
    std::unique_ptr<OpenGLComputeShader> attention_backward_shader_;
    std::unique_ptr<OpenGLComputeShader> softmax_shader_;
    std::unique_ptr<OpenGLComputeShader> transpose_shader_;
    std::unique_ptr<OpenGLComputeShader> reduce_shader_;
    
    // Storage buffers
    std::shared_ptr<OpenGLStorageBuffer> query_buffer_;
    std::shared_ptr<OpenGLStorageBuffer> key_buffer_;
    std::shared_ptr<OpenGLStorageBuffer> value_buffer_;
    std::shared_ptr<OpenGLStorageBuffer> output_buffer_;
    std::shared_ptr<OpenGLStorageBuffer> attention_scores_buffer_;
    std::shared_ptr<OpenGLStorageBuffer> workspace_buffer_;
    
    // Buffer sizes
    size_t query_buffer_size_;
    size_t key_buffer_size_;
    size_t value_buffer_size_;
    size_t output_buffer_size_;
    size_t attention_scores_buffer_size_;
    size_t workspace_buffer_size_;
    
    // Performance and profiling
    std::vector<std::pair<std::string, double>> profiling_results_;
    bool gpu_profiling_enabled_;
    
    // State management
    bool opengl_initialized_;
    bool shaders_compiled_;
    bool buffers_allocated_;
    
    // Helper methods
    void allocate_opengl_buffers(const AttentionParams& params);
    void deallocate_opengl_buffers();
    void validate_opengl_configuration() const;
    
    void handle_opengl_error(const std::string& operation) const;
    
    // Shader source storage
    static const char* flash_attention_compute_shader_source_;
    static const char* attention_backward_compute_shader_source_;
    static const char* softmax_compute_shader_source_;
    static const char* transpose_compute_shader_source_;
    static const char* reduce_compute_shader_source_;
};

// Factory functions for OpenGL FlashAttention
std::unique_ptr<OpenGLFlashAttention> create_opengl_flash_attention();

// OpenGL FlashAttention utilities
namespace opengl_flash_attention_utils {
    // OpenGL availability checking
    bool is_opengl_available();
    bool supports_compute_shaders();
    std::string get_opengl_version();
    std::string get_opengl_vendor();
    std::string get_opengl_renderer();
    
    // Configuration optimization
    OpenGLFlashAttentionConfig get_optimal_opengl_config(const AttentionParams& params);
    
    std::tuple<GLuint, GLuint, GLuint> calculate_optimal_work_group_size(
        const AttentionParams& params);
    
    // Memory usage estimation
    size_t estimate_opengl_memory_usage(
        const AttentionParams& params,
        const OpenGLFlashAttentionConfig& config);
    
    // Performance benchmarking
    double benchmark_opengl_flash_attention(
        const AttentionParams& params,
        const OpenGLFlashAttentionConfig& config,
        int num_iterations = 10);
    
    // Capability analysis
    struct OpenGLCapabilityReport {
        bool supports_compute_shaders;
        bool supports_shader_storage_buffers;
        int max_work_group_size[3];
        int max_work_group_invocations;
        int max_shared_memory_size;
        int max_storage_buffer_bindings;
        std::string version;
        std::string vendor;
        std::string renderer;
        std::vector<std::string> extensions;
    };
    
    OpenGLCapabilityReport analyze_opengl_capabilities();
    
    // Shader utilities
    std::string load_compute_shader_source(const std::string& filename);
    bool validate_compute_shader_source(const std::string& source);
    std::vector<std::string> extract_shader_dependencies(const std::string& source);
}

} // namespace attention_hpc

#endif // HAVE_OPENGL

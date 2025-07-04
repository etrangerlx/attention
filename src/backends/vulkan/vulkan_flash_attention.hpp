#pragma once

#include "algorithms/flash_attention.hpp"

#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <functional>

namespace attention_hpc {

// Vulkan-specific error handling
#define VK_CHECK(call) do { \
    VkResult result = call; \
    if (result != VK_SUCCESS) { \
        throw AttentionException(AttentionError::COMPUTATION_FAILED, \
                                "Vulkan error: " + std::to_string(result)); \
    } \
} while(0)

// Vulkan instance management
class VulkanInstance {
public:
    VulkanInstance();
    ~VulkanInstance();
    
    // Instance creation and management
    bool create_instance(const std::vector<const char*>& required_extensions = {},
                        const std::vector<const char*>& validation_layers = {},
                        bool enable_validation = false);
    void destroy_instance();
    VkInstance get_instance() const { return instance_; }
    bool is_valid() const { return instance_ != VK_NULL_HANDLE; }
    
    // Extension and layer support
    std::vector<VkExtensionProperties> get_available_extensions() const;
    std::vector<VkLayerProperties> get_available_layers() const;
    bool supports_extension(const std::string& extension_name) const;
    bool supports_layer(const std::string& layer_name) const;
    
    // Debug and validation
    void setup_debug_messenger();
    void cleanup_debug_messenger();
    
private:
    VkInstance instance_;
    VkDebugUtilsMessengerEXT debug_messenger_;
    bool validation_enabled_;
    
    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
        VkDebugUtilsMessageTypeFlagsEXT message_type,
        const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
        void* user_data);
};

// Vulkan physical device information
struct VulkanDeviceInfo {
    VkPhysicalDevice physical_device;
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceFeatures features;
    VkPhysicalDeviceMemoryProperties memory_properties;
    std::vector<VkQueueFamilyProperties> queue_families;
    std::vector<VkExtensionProperties> extensions;
    
    // Compute-specific properties
    VkPhysicalDeviceComputeShaderPropertiesEXT compute_properties;
    bool supports_compute_shaders;
    bool supports_storage_buffers;
    bool supports_push_constants;
    
    // Memory and performance info
    size_t max_memory_allocation_size;
    size_t buffer_alignment;
    uint32_t max_work_group_count[3];
    uint32_t max_work_group_size[3];
    uint32_t max_work_group_invocations;
    
    // Queue family indices
    uint32_t compute_queue_family_index;
    uint32_t transfer_queue_family_index;
    bool has_dedicated_compute_queue;
    bool has_dedicated_transfer_queue;
};

// Vulkan device management
class VulkanDevice {
public:
    VulkanDevice(VkInstance instance);
    ~VulkanDevice();
    
    // Device selection and creation
    std::vector<VulkanDeviceInfo> enumerate_physical_devices();
    VulkanDeviceInfo select_best_device(const std::vector<VulkanDeviceInfo>& devices);
    bool create_logical_device(const VulkanDeviceInfo& device_info,
                              const std::vector<const char*>& required_extensions = {});
    void destroy_device();
    
    // Device accessors
    VkDevice get_device() const { return device_; }
    VkPhysicalDevice get_physical_device() const { return physical_device_; }
    const VulkanDeviceInfo& get_device_info() const { return device_info_; }
    bool is_valid() const { return device_ != VK_NULL_HANDLE; }
    
    // Queue management
    VkQueue get_compute_queue() const { return compute_queue_; }
    VkQueue get_transfer_queue() const { return transfer_queue_; }
    uint32_t get_compute_queue_family_index() const { return device_info_.compute_queue_family_index; }
    uint32_t get_transfer_queue_family_index() const { return device_info_.transfer_queue_family_index; }
    
    // Memory management
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
    VkDeviceMemory allocate_memory(VkMemoryRequirements requirements, VkMemoryPropertyFlags properties);
    void free_memory(VkDeviceMemory memory);
    
private:
    VkInstance instance_;
    VkPhysicalDevice physical_device_;
    VkDevice device_;
    VulkanDeviceInfo device_info_;
    
    VkQueue compute_queue_;
    VkQueue transfer_queue_;
    
    void query_device_properties(VkPhysicalDevice physical_device, VulkanDeviceInfo& device_info);
    bool check_device_extension_support(VkPhysicalDevice physical_device, 
                                       const std::vector<const char*>& required_extensions);
    int score_device(const VulkanDeviceInfo& device_info);
};

// Vulkan buffer management
class VulkanBuffer {
public:
    VulkanBuffer(VkDevice device, VkPhysicalDevice physical_device);
    ~VulkanBuffer();
    
    // Buffer creation and management
    bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
    void destroy_buffer();
    bool is_valid() const { return buffer_ != VK_NULL_HANDLE; }
    
    // Data operations
    void upload_data(const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    void download_data(void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    void copy_from_buffer(const VulkanBuffer& src, VkCommandBuffer command_buffer,
                         VkDeviceSize src_offset = 0, VkDeviceSize dst_offset = 0, VkDeviceSize size = 0);
    
    // Memory mapping
    void* map_memory(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
    void unmap_memory();
    
    // Buffer properties
    VkBuffer get_buffer() const { return buffer_; }
    VkDeviceMemory get_memory() const { return memory_; }
    VkDeviceSize get_size() const { return size_; }
    VkBufferUsageFlags get_usage() const { return usage_; }
    VkMemoryPropertyFlags get_memory_properties() const { return memory_properties_; }
    
private:
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    VkBuffer buffer_;
    VkDeviceMemory memory_;
    VkDeviceSize size_;
    VkBufferUsageFlags usage_;
    VkMemoryPropertyFlags memory_properties_;
    void* mapped_ptr_;
    
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
};

// Vulkan descriptor set management
class VulkanDescriptorSet {
public:
    VulkanDescriptorSet(VkDevice device);
    ~VulkanDescriptorSet();
    
    // Descriptor set layout and pool creation
    bool create_descriptor_set_layout(const std::vector<VkDescriptorSetLayoutBinding>& bindings);
    bool create_descriptor_pool(const std::vector<VkDescriptorPoolSize>& pool_sizes, uint32_t max_sets);
    bool allocate_descriptor_sets(uint32_t count = 1);
    void cleanup();
    
    // Descriptor set updates
    void update_buffer_descriptor(uint32_t binding, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range);
    void update_storage_buffer_descriptor(uint32_t binding, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range);
    void flush_updates();
    
    // Accessors
    VkDescriptorSetLayout get_layout() const { return descriptor_set_layout_; }
    VkDescriptorPool get_pool() const { return descriptor_pool_; }
    VkDescriptorSet get_descriptor_set(uint32_t index = 0) const { 
        return index < descriptor_sets_.size() ? descriptor_sets_[index] : VK_NULL_HANDLE; 
    }
    
private:
    VkDevice device_;
    VkDescriptorSetLayout descriptor_set_layout_;
    VkDescriptorPool descriptor_pool_;
    std::vector<VkDescriptorSet> descriptor_sets_;
    std::vector<VkWriteDescriptorSet> pending_writes_;
};

// Vulkan compute pipeline management
class VulkanComputePipeline {
public:
    VulkanComputePipeline(VkDevice device);
    ~VulkanComputePipeline();
    
    // Pipeline creation
    bool create_pipeline(const std::string& shader_source, 
                        const std::string& entry_point = "main",
                        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE,
                        const std::vector<VkPushConstantRange>& push_constants = {});
    bool create_pipeline_from_spirv(const std::vector<uint32_t>& spirv_code,
                                   const std::string& entry_point = "main",
                                   VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE,
                                   const std::vector<VkPushConstantRange>& push_constants = {});
    void destroy_pipeline();
    
    // Pipeline usage
    void bind_pipeline(VkCommandBuffer command_buffer);
    void bind_descriptor_sets(VkCommandBuffer command_buffer, 
                             const std::vector<VkDescriptorSet>& descriptor_sets);
    void push_constants(VkCommandBuffer command_buffer, VkShaderStageFlags stage_flags,
                       uint32_t offset, uint32_t size, const void* values);
    void dispatch(VkCommandBuffer command_buffer, uint32_t group_count_x, 
                 uint32_t group_count_y = 1, uint32_t group_count_z = 1);
    
    // Pipeline properties
    VkPipeline get_pipeline() const { return compute_pipeline_; }
    VkPipelineLayout get_layout() const { return pipeline_layout_; }
    bool is_valid() const { return compute_pipeline_ != VK_NULL_HANDLE; }
    
private:
    VkDevice device_;
    VkShaderModule shader_module_;
    VkPipelineLayout pipeline_layout_;
    VkPipeline compute_pipeline_;
    
    bool compile_shader(const std::string& source, std::vector<uint32_t>& spirv_code);
    VkShaderModule create_shader_module(const std::vector<uint32_t>& spirv_code);
};

// Vulkan command buffer management
class VulkanCommandBuffer {
public:
    VulkanCommandBuffer(VkDevice device, uint32_t queue_family_index);
    ~VulkanCommandBuffer();
    
    // Command pool and buffer management
    bool create_command_pool(VkCommandPoolCreateFlags flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    bool allocate_command_buffers(uint32_t count = 1, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    void free_command_buffers();
    void reset_command_pool();
    
    // Command buffer recording
    void begin_recording(uint32_t buffer_index = 0, VkCommandBufferUsageFlags flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    void end_recording(uint32_t buffer_index = 0);
    
    // Command submission
    void submit_to_queue(VkQueue queue, uint32_t buffer_index = 0,
                        const std::vector<VkSemaphore>& wait_semaphores = {},
                        const std::vector<VkPipelineStageFlags>& wait_stages = {},
                        const std::vector<VkSemaphore>& signal_semaphores = {},
                        VkFence fence = VK_NULL_HANDLE);
    
    // Synchronization barriers
    void pipeline_barrier(uint32_t buffer_index,
                         VkPipelineStageFlags src_stage_mask,
                         VkPipelineStageFlags dst_stage_mask,
                         const std::vector<VkMemoryBarrier>& memory_barriers = {},
                         const std::vector<VkBufferMemoryBarrier>& buffer_barriers = {},
                         const std::vector<VkImageMemoryBarrier>& image_barriers = {});
    
    // Accessors
    VkCommandPool get_command_pool() const { return command_pool_; }
    VkCommandBuffer get_command_buffer(uint32_t index = 0) const {
        return index < command_buffers_.size() ? command_buffers_[index] : VK_NULL_HANDLE;
    }
    uint32_t get_buffer_count() const { return static_cast<uint32_t>(command_buffers_.size()); }
    
private:
    VkDevice device_;
    uint32_t queue_family_index_;
    VkCommandPool command_pool_;
    std::vector<VkCommandBuffer> command_buffers_;
};

// Vulkan synchronization objects
class VulkanSynchronization {
public:
    VulkanSynchronization(VkDevice device);
    ~VulkanSynchronization();
    
    // Fence management
    VkFence create_fence(VkFenceCreateFlags flags = 0);
    void destroy_fence(VkFence fence);
    void reset_fence(VkFence fence);
    bool wait_for_fence(VkFence fence, uint64_t timeout = UINT64_MAX);
    bool wait_for_fences(const std::vector<VkFence>& fences, bool wait_all = true, uint64_t timeout = UINT64_MAX);
    
    // Semaphore management
    VkSemaphore create_semaphore();
    void destroy_semaphore(VkSemaphore semaphore);
    
    // Event management
    VkEvent create_event();
    void destroy_event(VkEvent event);
    void set_event(VkEvent event);
    void reset_event(VkEvent event);
    VkResult get_event_status(VkEvent event);
    
    // Cleanup all synchronization objects
    void cleanup_all();
    
private:
    VkDevice device_;
    std::vector<VkFence> fences_;
    std::vector<VkSemaphore> semaphores_;
    std::vector<VkEvent> events_;
    std::mutex sync_mutex_;
};

// Vulkan-specific FlashAttention configuration
struct VulkanFlashAttentionConfig : public FlashAttentionConfig {
    // Instance and device selection
    std::vector<const char*> required_instance_extensions;
    std::vector<const char*> required_device_extensions;
    std::vector<const char*> validation_layers;
    bool enable_validation_layers = false;
    
    // Device selection criteria
    VkPhysicalDeviceType preferred_device_type = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    bool require_compute_queue = true;
    bool prefer_dedicated_compute_queue = true;
    
    // Pipeline configuration
    std::string compute_shader_entry_point = "main";
    std::vector<VkPushConstantRange> push_constant_ranges;
    VkPipelineCreateFlags pipeline_flags = 0;
    
    // Command buffer and synchronization
    uint32_t command_buffer_count = 2;
    VkCommandPoolCreateFlags command_pool_flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    bool use_dedicated_compute_queue = false;
    
    // Memory management
    VkMemoryPropertyFlags buffer_memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkBufferUsageFlags buffer_usage_flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bool use_staging_buffers = true;
    
    // Work group configuration
    uint32_t work_group_size_x = 16;
    uint32_t work_group_size_y = 16;
    uint32_t work_group_size_z = 1;
    
    // Performance optimization
    bool enable_pipeline_cache = true;
    std::string pipeline_cache_file = "vulkan_pipeline.cache";
    bool enable_memory_barriers = true;
    bool optimize_memory_layout = true;
    
    // Debug and profiling
    bool enable_debug_markers = false;
    bool enable_gpu_profiling = false;
};

// Vulkan FlashAttention implementation
class VulkanFlashAttention : public FlashAttention {
public:
    VulkanFlashAttention();
    virtual ~VulkanFlashAttention();
    
    // Override configuration methods
    void configure(const AttentionParams& params, 
                   const PerformanceConfig& perf_config = PerformanceConfig{}) override;
    
    // Vulkan-specific configuration
    void configure_vulkan(const VulkanFlashAttentionConfig& vulkan_config);
    
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
    
    // Vulkan-specific methods
    
    // Instance and device management
    bool initialize_vulkan();
    void cleanup_vulkan();
    bool is_vulkan_initialized() const;
    VulkanInstance* get_vulkan_instance() const;
    VulkanDevice* get_vulkan_device() const;
    
    // Pipeline management
    bool create_compute_pipelines();
    void destroy_compute_pipelines();
    bool recompile_shaders();
    
    // Buffer management
    std::shared_ptr<VulkanBuffer> create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
    void upload_tensor_data(std::shared_ptr<VulkanBuffer> buffer, const void* data, size_t size, DataType data_type);
    void download_tensor_data(std::shared_ptr<VulkanBuffer> buffer, void* data, size_t size, DataType data_type);
    
    // Compute dispatch
    void dispatch_attention_compute(const AttentionParams& params,
                                   std::shared_ptr<VulkanBuffer> query_buffer,
                                   std::shared_ptr<VulkanBuffer> key_buffer,
                                   std::shared_ptr<VulkanBuffer> value_buffer,
                                   std::shared_ptr<VulkanBuffer> output_buffer,
                                   std::shared_ptr<VulkanBuffer> workspace_buffer = nullptr);
    
    void dispatch_backward_compute(const AttentionParams& params,
                                  std::shared_ptr<VulkanBuffer> grad_output_buffer,
                                  std::shared_ptr<VulkanBuffer> query_buffer,
                                  std::shared_ptr<VulkanBuffer> key_buffer,
                                  std::shared_ptr<VulkanBuffer> value_buffer,
                                  std::shared_ptr<VulkanBuffer> grad_query_buffer,
                                  std::shared_ptr<VulkanBuffer> grad_key_buffer,
                                  std::shared_ptr<VulkanBuffer> grad_value_buffer);
    
    // Synchronization
    void submit_commands_and_wait();
    void insert_memory_barrier();
    VkFence create_fence();
    VkSemaphore create_semaphore();
    
    // Performance optimization
    void optimize_work_group_size(const AttentionParams& params);
    void optimize_memory_layout();
    void benchmark_pipeline_performance(const AttentionParams& params, int num_iterations = 10);
    
    // Debug and profiling
    void enable_vulkan_profiling(bool enable = true);
    std::vector<std::pair<std::string, double>> get_vulkan_profiling_results() const;
    void print_vulkan_info() const;

protected:
    // Vulkan-specific implementations of base class methods
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
    
    // Vulkan-specific helper methods
    void setup_vulkan_resources();
    void cleanup_vulkan_resources();
    void create_descriptor_sets();
    void update_descriptor_sets();
    
    std::string load_compute_shader_source(const std::string& shader_name);
    std::vector<uint32_t> compile_shader_to_spirv(const std::string& source);
    void validate_vulkan_configuration() const;
    
    void calculate_dispatch_dimensions(const AttentionParams& params,
                                     uint32_t& group_count_x, uint32_t& group_count_y, uint32_t& group_count_z);
    
    // Push constants structure
    struct PushConstants {
        uint32_t batch_size;
        uint32_t num_heads;
        uint32_t seq_len;
        uint32_t head_dim;
        float scale;
        uint32_t causal_mask;
        uint32_t block_size_q;
        uint32_t block_size_k;
    };
    
    // Data type conversion utilities
    void convert_to_vulkan_format(const void* input, std::shared_ptr<VulkanBuffer> output_buffer,
                                 size_t num_elements, DataType input_type);
    void convert_from_vulkan_format(std::shared_ptr<VulkanBuffer> input_buffer, void* output,
                                   size_t num_elements, DataType output_type);

private:
    // Configuration state
    VulkanFlashAttentionConfig vulkan_config_;
    
    // Vulkan management objects
    std::unique_ptr<VulkanInstance> instance_;
    std::unique_ptr<VulkanDevice> device_;
    std::unique_ptr<VulkanCommandBuffer> command_buffer_;
    std::unique_ptr<VulkanSynchronization> synchronization_;
    
    // Compute pipelines
    std::unique_ptr<VulkanComputePipeline> flash_attention_pipeline_;
    std::unique_ptr<VulkanComputePipeline> attention_backward_pipeline_;
    std::unique_ptr<VulkanComputePipeline> softmax_pipeline_;
    std::unique_ptr<VulkanComputePipeline> transpose_pipeline_;
    
    // Descriptor sets
    std::unique_ptr<VulkanDescriptorSet> compute_descriptor_set_;
    std::unique_ptr<VulkanDescriptorSet> backward_descriptor_set_;
    
    // Storage buffers
    std::shared_ptr<VulkanBuffer> query_buffer_;
    std::shared_ptr<VulkanBuffer> key_buffer_;
    std::shared_ptr<VulkanBuffer> value_buffer_;
    std::shared_ptr<VulkanBuffer> output_buffer_;
    std::shared_ptr<VulkanBuffer> attention_scores_buffer_;
    std::shared_ptr<VulkanBuffer> workspace_buffer_;
    
    // Staging buffers for host-device data transfer
    std::shared_ptr<VulkanBuffer> staging_buffer_upload_;
    std::shared_ptr<VulkanBuffer> staging_buffer_download_;
    
    // Buffer sizes
    size_t query_buffer_size_;
    size_t key_buffer_size_;
    size_t value_buffer_size_;
    size_t output_buffer_size_;
    size_t attention_scores_buffer_size_;
    size_t workspace_buffer_size_;
    
    // Synchronization objects
    VkFence compute_fence_;
    VkSemaphore compute_semaphore_;
    
    // Performance and profiling
    std::vector<std::pair<std::string, double>> profiling_results_;
    bool vulkan_profiling_enabled_;
    
    // State management
    bool vulkan_initialized_;
    bool pipelines_created_;
    bool buffers_allocated_;
    bool descriptor_sets_created_;
    
    // Helper methods
    void allocate_vulkan_buffers(const AttentionParams& params);
    void deallocate_vulkan_buffers();
    void create_staging_buffers();
    void setup_synchronization_objects();
    
    void handle_vulkan_error(VkResult result, const std::string& operation) const;
    
    // Shader source storage
    static const char* flash_attention_compute_shader_source_;
    static const char* attention_backward_compute_shader_source_;
    static const char* softmax_compute_shader_source_;
    static const char* transpose_compute_shader_source_;
};

// Factory functions for Vulkan FlashAttention
std::unique_ptr<VulkanFlashAttention> create_vulkan_flash_attention();

// Vulkan FlashAttention utilities
namespace vulkan_flash_attention_utils {
    // Vulkan availability checking
    bool is_vulkan_available();
    std::vector<std::string> get_available_instance_extensions();
    std::vector<std::string> get_available_validation_layers();
    
    // Device enumeration and selection
    std::vector<VulkanDeviceInfo> enumerate_vulkan_devices();
    VulkanDeviceInfo select_best_vulkan_device(const std::vector<VulkanDeviceInfo>& devices);
    
    // Configuration optimization
    VulkanFlashAttentionConfig get_optimal_vulkan_config(const AttentionParams& params);
    
    std::tuple<uint32_t, uint32_t, uint32_t> calculate_optimal_work_group_size(
        const AttentionParams& params, const VulkanDeviceInfo& device_info);
    
    // Memory usage estimation
    size_t estimate_vulkan_memory_usage(
        const AttentionParams& params,
        const VulkanFlashAttentionConfig& config);
    
    // Performance benchmarking
    double benchmark_vulkan_flash_attention(
        const AttentionParams& params,
        const VulkanFlashAttentionConfig& config,
        int num_iterations = 10);
    
    // Capability analysis
    struct VulkanCapabilityReport {
        bool supports_compute_shaders;
        bool supports_storage_buffers;
        bool supports_push_constants;
        uint32_t max_work_group_size[3];
        uint32_t max_work_group_invocations;
        size_t max_storage_buffer_range;
        size_t max_push_constants_size;
        size_t max_memory_allocation_size;
        std::string device_name;
        std::string driver_version;
        VkPhysicalDeviceType device_type;
        std::vector<std::string> supported_extensions;
    };
    
    VulkanCapabilityReport analyze_vulkan_capabilities(const VulkanDeviceInfo& device_info);
    
    // Shader utilities
    std::vector<uint32_t> compile_glsl_to_spirv(const std::string& glsl_source, const std::string& entry_point = "main");
    bool validate_spirv_shader(const std::vector<uint32_t>& spirv_code);
    std::string disassemble_spirv(const std::vector<uint32_t>& spirv_code);
    
    // Memory management utilities
    VkMemoryPropertyFlags get_optimal_memory_properties(const VulkanDeviceInfo& device_info, VkBufferUsageFlags usage);
    size_t align_buffer_size(size_t size, const VulkanDeviceInfo& device_info);
}

} // namespace attention_hpc

#endif // HAVE_VULKAN

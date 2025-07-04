#include "vulkan_flash_attention.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <set>

namespace attention_hpc {

// Vulkan error string conversion
const char* get_vulkan_error_string(VkResult result) {
    switch (result) {
        case VK_SUCCESS: return "Success";
        case VK_NOT_READY: return "Not ready";
        case VK_TIMEOUT: return "Timeout";
        case VK_EVENT_SET: return "Event set";
        case VK_EVENT_RESET: return "Event reset";
        case VK_INCOMPLETE: return "Incomplete";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "Out of host memory";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "Out of device memory";
        case VK_ERROR_INITIALIZATION_FAILED: return "Initialization failed";
        case VK_ERROR_DEVICE_LOST: return "Device lost";
        case VK_ERROR_MEMORY_MAP_FAILED: return "Memory map failed";
        case VK_ERROR_LAYER_NOT_PRESENT: return "Layer not present";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "Extension not present";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "Feature not present";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "Incompatible driver";
        case VK_ERROR_TOO_MANY_OBJECTS: return "Too many objects";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "Format not supported";
        case VK_ERROR_FRAGMENTED_POOL: return "Fragmented pool";
        case VK_ERROR_UNKNOWN: return "Unknown error";
        default: return "Unhandled VkResult";
    }
}

// VulkanInstance implementation
VulkanInstance::VulkanInstance() 
    : instance_(VK_NULL_HANDLE), debug_messenger_(VK_NULL_HANDLE), validation_enabled_(false) {
}

VulkanInstance::~VulkanInstance() {
    destroy_instance();
}

bool VulkanInstance::create_instance(const std::vector<const char*>& required_extensions,
                                    const std::vector<const char*>& validation_layers,
                                    bool enable_validation) {
    validation_enabled_ = enable_validation;
    
    // Application info
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "AttentionHPC";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "AttentionHPC";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;
    
    // Instance create info
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    
    // Extensions
    std::vector<const char*> extensions = required_extensions;
    if (validation_enabled_) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    
    // Validation layers
    if (validation_enabled_ && !validation_layers.empty()) {
        create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
        create_info.ppEnabledLayerNames = validation_layers.data();
    } else {
        create_info.enabledLayerCount = 0;
    }
    
    VkResult result = vkCreateInstance(&create_info, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan instance: " << get_vulkan_error_string(result) << std::endl;
        return false;
    }
    
    if (validation_enabled_) {
        setup_debug_messenger();
    }
    
    return true;
}

void VulkanInstance::destroy_instance() {
    if (validation_enabled_ && debug_messenger_ != VK_NULL_HANDLE) {
        cleanup_debug_messenger();
    }
    
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

std::vector<VkExtensionProperties> VulkanInstance::get_available_extensions() const {
    uint32_t extension_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
    
    std::vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data());
    
    return extensions;
}

std::vector<VkLayerProperties> VulkanInstance::get_available_layers() const {
    uint32_t layer_count = 0;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    
    std::vector<VkLayerProperties> layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, layers.data());
    
    return layers;
}

bool VulkanInstance::supports_extension(const std::string& extension_name) const {
    auto extensions = get_available_extensions();
    for (const auto& ext : extensions) {
        if (extension_name == ext.extensionName) {
            return true;
        }
    }
    return false;
}

bool VulkanInstance::supports_layer(const std::string& layer_name) const {
    auto layers = get_available_layers();
    for (const auto& layer : layers) {
        if (layer_name == layer.layerName) {
            return true;
        }
    }
    return false;
}

void VulkanInstance::setup_debug_messenger() {
    if (!validation_enabled_ || instance_ == VK_NULL_HANDLE) return;
    
    VkDebugUtilsMessengerCreateInfoEXT create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = debug_callback;
    create_info.pUserData = nullptr;
    
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance_, &create_info, nullptr, &debug_messenger_);
    }
}

void VulkanInstance::cleanup_debug_messenger() {
    if (debug_messenger_ != VK_NULL_HANDLE) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr) {
            func(instance_, debug_messenger_, nullptr);
        }
        debug_messenger_ = VK_NULL_HANDLE;
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanInstance::debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data) {
    
    std::cerr << "Validation layer: " << callback_data->pMessage << std::endl;
    return VK_FALSE;
}

// VulkanDevice implementation
VulkanDevice::VulkanDevice(VkInstance instance)
    : instance_(instance), physical_device_(VK_NULL_HANDLE), device_(VK_NULL_HANDLE),
      compute_queue_(VK_NULL_HANDLE), transfer_queue_(VK_NULL_HANDLE) {
}

VulkanDevice::~VulkanDevice() {
    destroy_device();
}

std::vector<VulkanDeviceInfo> VulkanDevice::enumerate_physical_devices() {
    std::vector<VulkanDeviceInfo> device_infos;
    
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    
    if (device_count == 0) {
        return device_infos;
    }
    
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
    
    for (VkPhysicalDevice device : devices) {
        VulkanDeviceInfo device_info;
        device_info.physical_device = device;
        query_device_properties(device, device_info);
        device_infos.push_back(device_info);
    }
    
    return device_infos;
}

VulkanDeviceInfo VulkanDevice::select_best_device(const std::vector<VulkanDeviceInfo>& devices) {
    VulkanDeviceInfo best_device = {};
    int best_score = -1;
    
    for (const auto& device : devices) {
        int score = score_device(device);
        if (score > best_score) {
            best_score = score;
            best_device = device;
        }
    }
    
    return best_device;
}

bool VulkanDevice::create_logical_device(const VulkanDeviceInfo& device_info,
                                        const std::vector<const char*>& required_extensions) {
    device_info_ = device_info;
    physical_device_ = device_info.physical_device;
    
    // Queue create infos
    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    std::set<uint32_t> unique_queue_families = {
        device_info.compute_queue_family_index,
        device_info.transfer_queue_family_index
    };
    
    float queue_priority = 1.0f;
    for (uint32_t queue_family : unique_queue_families) {
        VkDeviceQueueCreateInfo queue_create_info = {};
        queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.queueFamilyIndex = queue_family;
        queue_create_info.queueCount = 1;
        queue_create_info.pQueuePriorities = &queue_priority;
        queue_create_infos.push_back(queue_create_info);
    }
    
    // Device features
    VkPhysicalDeviceFeatures device_features = {};
    
    // Device create info
    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
    create_info.pQueueCreateInfos = queue_create_infos.data();
    create_info.pEnabledFeatures = &device_features;
    create_info.enabledExtensionCount = static_cast<uint32_t>(required_extensions.size());
    create_info.ppEnabledExtensionNames = required_extensions.data();
    
    VkResult result = vkCreateDevice(physical_device_, &create_info, nullptr, &device_);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create logical device: " << get_vulkan_error_string(result) << std::endl;
        return false;
    }
    
    // Get queue handles
    vkGetDeviceQueue(device_, device_info.compute_queue_family_index, 0, &compute_queue_);
    vkGetDeviceQueue(device_, device_info.transfer_queue_family_index, 0, &transfer_queue_);
    
    return true;
}

void VulkanDevice::destroy_device() {
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    
    physical_device_ = VK_NULL_HANDLE;
    compute_queue_ = VK_NULL_HANDLE;
    transfer_queue_ = VK_NULL_HANDLE;
}

uint32_t VulkanDevice::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties);
    
    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && 
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    throw std::runtime_error("Failed to find suitable memory type");
}

VkDeviceMemory VulkanDevice::allocate_memory(VkMemoryRequirements requirements, VkMemoryPropertyFlags properties) {
    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(requirements.memoryTypeBits, properties);
    
    VkDeviceMemory memory;
    VkResult result = vkAllocateMemory(device_, &alloc_info, nullptr, &memory);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate device memory: " + std::string(get_vulkan_error_string(result)));
    }
    
    return memory;
}

void VulkanDevice::free_memory(VkDeviceMemory memory) {
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(device_, memory, nullptr);
    }
}

void VulkanDevice::query_device_properties(VkPhysicalDevice physical_device, VulkanDeviceInfo& device_info) {
    // Basic properties
    vkGetPhysicalDeviceProperties(physical_device, &device_info.properties);
    vkGetPhysicalDeviceFeatures(physical_device, &device_info.features);
    vkGetPhysicalDeviceMemoryProperties(physical_device, &device_info.memory_properties);
    
    // Queue families
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
    device_info.queue_families.resize(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, device_info.queue_families.data());
    
    // Find compute and transfer queue families
    device_info.compute_queue_family_index = UINT32_MAX;
    device_info.transfer_queue_family_index = UINT32_MAX;
    device_info.has_dedicated_compute_queue = false;
    device_info.has_dedicated_transfer_queue = false;
    
    for (uint32_t i = 0; i < queue_family_count; i++) {
        const auto& queue_family = device_info.queue_families[i];
        
        // Look for compute queue
        if (queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            if (device_info.compute_queue_family_index == UINT32_MAX) {
                device_info.compute_queue_family_index = i;
            }
            
            // Prefer dedicated compute queue
            if (!(queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                device_info.compute_queue_family_index = i;
                device_info.has_dedicated_compute_queue = true;
            }
        }
        
        // Look for transfer queue
        if (queue_family.queueFlags & VK_QUEUE_TRANSFER_BIT) {
            if (device_info.transfer_queue_family_index == UINT32_MAX) {
                device_info.transfer_queue_family_index = i;
            }
            
            // Prefer dedicated transfer queue
            if (!(queue_family.queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
                device_info.transfer_queue_family_index = i;
                device_info.has_dedicated_transfer_queue = true;
            }
        }
    }
    
    // Fallback to any queue if no dedicated queues
    if (device_info.transfer_queue_family_index == UINT32_MAX) {
        device_info.transfer_queue_family_index = device_info.compute_queue_family_index;
    }
    
    // Extensions
    uint32_t extension_count;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
    device_info.extensions.resize(extension_count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, device_info.extensions.data());
    
    // Compute capabilities
    device_info.supports_compute_shaders = (device_info.compute_queue_family_index != UINT32_MAX);
    device_info.supports_storage_buffers = true; // Basic Vulkan support
    device_info.supports_push_constants = true; // Basic Vulkan support
    
    // Memory limits
    device_info.max_memory_allocation_size = device_info.properties.limits.maxMemoryAllocationCount;
    device_info.buffer_alignment = device_info.properties.limits.minStorageBufferOffsetAlignment;
    
    // Work group limits
    device_info.max_work_group_count[0] = device_info.properties.limits.maxComputeWorkGroupCount[0];
    device_info.max_work_group_count[1] = device_info.properties.limits.maxComputeWorkGroupCount[1];
    device_info.max_work_group_count[2] = device_info.properties.limits.maxComputeWorkGroupCount[2];
    
    device_info.max_work_group_size[0] = device_info.properties.limits.maxComputeWorkGroupSize[0];
    device_info.max_work_group_size[1] = device_info.properties.limits.maxComputeWorkGroupSize[1];
    device_info.max_work_group_size[2] = device_info.properties.limits.maxComputeWorkGroupSize[2];
    
    device_info.max_work_group_invocations = device_info.properties.limits.maxComputeWorkGroupInvocations;
}

bool VulkanDevice::check_device_extension_support(VkPhysicalDevice physical_device,
                                                 const std::vector<const char*>& required_extensions) {
    uint32_t extension_count;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
    
    std::vector<VkExtensionProperties> available_extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, available_extensions.data());
    
    std::set<std::string> required_extension_set(required_extensions.begin(), required_extensions.end());
    
    for (const auto& extension : available_extensions) {
        required_extension_set.erase(extension.extensionName);
    }
    
    return required_extension_set.empty();
}

int VulkanDevice::score_device(const VulkanDeviceInfo& device_info) {
    int score = 0;
    
    // Device type scoring
    if (device_info.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 1000;
    } else if (device_info.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        score += 500;
    }
    
    // Compute support
    if (device_info.supports_compute_shaders) {
        score += 200;
        if (device_info.has_dedicated_compute_queue) {
            score += 100;
        }
    }
    
    // Memory size
    size_t total_memory = 0;
    for (uint32_t i = 0; i < device_info.memory_properties.memoryHeapCount; i++) {
        if (device_info.memory_properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            total_memory += device_info.memory_properties.memoryHeaps[i].size;
        }
    }
    score += static_cast<int>(total_memory / (1024 * 1024)); // MB
    
    return score;
}

// VulkanBuffer implementation
VulkanBuffer::VulkanBuffer(VkDevice device, VkPhysicalDevice physical_device)
    : device_(device), physical_device_(physical_device), buffer_(VK_NULL_HANDLE),
      memory_(VK_NULL_HANDLE), size_(0), usage_(0), memory_properties_(0), mapped_ptr_(nullptr) {
}

VulkanBuffer::~VulkanBuffer() {
    destroy_buffer();
}

bool VulkanBuffer::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
    size_ = size;
    usage_ = usage;
    memory_properties_ = properties;
    
    // Create buffer
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VkResult result = vkCreateBuffer(device_, &buffer_info, nullptr, &buffer_);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create buffer: " << get_vulkan_error_string(result) << std::endl;
        return false;
    }
    
    // Get memory requirements
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device_, buffer_, &mem_requirements);
    
    // Allocate memory
    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_requirements.memoryTypeBits, properties);
    
    result = vkAllocateMemory(device_, &alloc_info, nullptr, &memory_);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to allocate buffer memory: " << get_vulkan_error_string(result) << std::endl;
        vkDestroyBuffer(device_, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        return false;
    }
    
    // Bind memory
    vkBindBufferMemory(device_, buffer_, memory_, 0);
    
    return true;
}

void VulkanBuffer::destroy_buffer() {
    if (mapped_ptr_) {
        unmap_memory();
    }
    
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
    }
    
    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, memory_, nullptr);
        memory_ = VK_NULL_HANDLE;
    }
    
    size_ = 0;
}

void VulkanBuffer::upload_data(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    void* mapped = map_memory(size, offset);
    memcpy(mapped, data, size);
    unmap_memory();
}

void VulkanBuffer::download_data(void* data, VkDeviceSize size, VkDeviceSize offset) {
    void* mapped = map_memory(size, offset);
    memcpy(data, mapped, size);
    unmap_memory();
}

void VulkanBuffer::copy_from_buffer(const VulkanBuffer& src, VkCommandBuffer command_buffer,
                                   VkDeviceSize src_offset, VkDeviceSize dst_offset, VkDeviceSize size) {
    VkBufferCopy copy_region = {};
    copy_region.srcOffset = src_offset;
    copy_region.dstOffset = dst_offset;
    copy_region.size = (size == 0) ? src.size_ : size;
    
    vkCmdCopyBuffer(command_buffer, src.buffer_, buffer_, 1, &copy_region);
}

void* VulkanBuffer::map_memory(VkDeviceSize size, VkDeviceSize offset) {
    if (mapped_ptr_) {
        return mapped_ptr_;
    }
    
    VkDeviceSize map_size = (size == VK_WHOLE_SIZE) ? size_ : size;
    VkResult result = vkMapMemory(device_, memory_, offset, map_size, 0, &mapped_ptr_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to map buffer memory: " + std::string(get_vulkan_error_string(result)));
    }
    
    return mapped_ptr_;
}

void VulkanBuffer::unmap_memory() {
    if (mapped_ptr_) {
        vkUnmapMemory(device_, memory_);
        mapped_ptr_ = nullptr;
    }
}

uint32_t VulkanBuffer::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties);
    
    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && 
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    throw std::runtime_error("Failed to find suitable memory type");
}

// VulkanDescriptorSet implementation
VulkanDescriptorSet::VulkanDescriptorSet(VkDevice device)
    : device_(device), descriptor_set_layout_(VK_NULL_HANDLE), descriptor_pool_(VK_NULL_HANDLE) {
}

VulkanDescriptorSet::~VulkanDescriptorSet() {
    cleanup();
}

bool VulkanDescriptorSet::create_descriptor_set_layout(const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();
    
    VkResult result = vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_set_layout_);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor set layout: " << get_vulkan_error_string(result) << std::endl;
        return false;
    }
    
    return true;
}

bool VulkanDescriptorSet::create_descriptor_pool(const std::vector<VkDescriptorPoolSize>& pool_sizes, uint32_t max_sets) {
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = max_sets;
    
    VkResult result = vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool: " << get_vulkan_error_string(result) << std::endl;
        return false;
    }
    
    return true;
}

bool VulkanDescriptorSet::allocate_descriptor_sets(uint32_t count) {
    if (descriptor_set_layout_ == VK_NULL_HANDLE || descriptor_pool_ == VK_NULL_HANDLE) {
        return false;
    }
    
    std::vector<VkDescriptorSetLayout> layouts(count, descriptor_set_layout_);
    
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = count;
    alloc_info.pSetLayouts = layouts.data();
    
    descriptor_sets_.resize(count);
    VkResult result = vkAllocateDescriptorSets(device_, &alloc_info, descriptor_sets_.data());
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor sets: " << get_vulkan_error_string(result) << std::endl;
        return false;
    }
    
    return true;
}

void VulkanDescriptorSet::cleanup() {
    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
    }
    
    if (descriptor_set_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
        descriptor_set_layout_ = VK_NULL_HANDLE;
    }
    
    descriptor_sets_.clear();
    pending_writes_.clear();
}

void VulkanDescriptorSet::update_buffer_descriptor(uint32_t binding, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range) {
    VkDescriptorBufferInfo buffer_info = {};
    buffer_info.buffer = buffer;
    buffer_info.offset

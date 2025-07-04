#include "backend_factory.hpp"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <thread>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cudnn.h>
#endif

#ifdef HAVE_OPENCL
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#endif

#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

#ifdef HAVE_OPENGL
#include <GL/gl.h>
#include <GL/glext.h>
#endif

namespace attention_hpc {

// Forward declarations for backend implementations
class CPUFlashAttention;
class CPUPagedAttention;

#ifdef HAVE_CUDA
class CUDAFlashAttention;
class CUDAPagedAttention;
#endif

#ifdef HAVE_OPENCL
class OpenCLFlashAttention;
class OpenCLPagedAttention;
#endif

#ifdef HAVE_OPENGL
class OpenGLFlashAttention;
class OpenGLPagedAttention;
#endif

#ifdef HAVE_VULKAN
class VulkanFlashAttention;
class VulkanPagedAttention;
#endif

BackendFactory::BackendFactory() 
    : default_backend_(BackendType::CPU), debug_mode_(false), initialized_(false) {
    initializeBackends();
}

BackendFactory& BackendFactory::getInstance() {
    static BackendFactory instance;
    return instance;
}

void BackendFactory::initializeBackends() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) return;
    
    // Initialize backend names
    backend_names_[BackendType::CPU] = "CPU";
    backend_names_[BackendType::CUDA] = "CUDA";
    backend_names_[BackendType::OPENCL] = "OpenCL";
    backend_names_[BackendType::OPENGL] = "OpenGL";
    backend_names_[BackendType::VULKAN] = "Vulkan";
    
    // Set default priority order
    backend_priority_ = {
        BackendType::CUDA,
        BackendType::VULKAN,
        BackendType::OPENCL,
        BackendType::OPENGL,
        BackendType::CPU
    };
    
    // Detect backend capabilities
    detectBackendCapabilities();
    
    // Register backend creators
    registerBackend(BackendType::CPU, "CPU", []() -> std::unique_ptr<AttentionInterface> {
        throw std::runtime_error("CPU backend not implemented yet");
    });
    
#ifdef HAVE_CUDA
    if (backend_capabilities_[BackendType::CUDA].available) {
        registerBackend(BackendType::CUDA, "CUDA", []() -> std::unique_ptr<AttentionInterface> {
            throw std::runtime_error("CUDA backend not implemented yet");
        });
    }
#endif

#ifdef HAVE_OPENCL
    if (backend_capabilities_[BackendType::OPENCL].available) {
        registerBackend(BackendType::OPENCL, "OpenCL", []() -> std::unique_ptr<AttentionInterface> {
            throw std::runtime_error("OpenCL backend not implemented yet");
        });
    }
#endif

#ifdef HAVE_OPENGL
    if (backend_capabilities_[BackendType::OPENGL].available) {
        registerBackend(BackendType::OPENGL, "OpenGL", []() -> std::unique_ptr<AttentionInterface> {
            throw std::runtime_error("OpenGL backend not implemented yet");
        });
    }
#endif

#ifdef HAVE_VULKAN
    if (backend_capabilities_[BackendType::VULKAN].available) {
        registerBackend(BackendType::VULKAN, "Vulkan", []() -> std::unique_ptr<AttentionInterface> {
            throw std::runtime_error("Vulkan backend not implemented yet");
        });
    }
#endif
    
    initialized_ = true;
}

void BackendFactory::detectBackendCapabilities() {
    // CPU backend is always available
    backend_capabilities_[BackendType::CPU] = BackendCapability{
        .available = true,
        .version = "1.0.0",
        .device_name = "CPU",
        .memory_size = 0, // Will be detected later
        .supported_data_types = {DataType::FLOAT32, DataType::FLOAT16, DataType::BFLOAT16},
        .supported_layouts = {MemoryLayout::ROW_MAJOR, MemoryLayout::COLUMN_MAJOR},
        .max_threads = std::thread::hardware_concurrency(),
        .max_shared_memory = 0,
        .supports_mixed_precision = true,
        .supports_tensor_cores = false
    };
    
    // Check CUDA availability
    backend_capabilities_[BackendType::CUDA] = BackendCapability{};
    backend_capabilities_[BackendType::CUDA].available = checkCudaAvailability();
    
    // Check OpenCL availability
    backend_capabilities_[BackendType::OPENCL] = BackendCapability{};
    backend_capabilities_[BackendType::OPENCL].available = checkOpenCLAvailability();
    
    // Check OpenGL availability
    backend_capabilities_[BackendType::OPENGL] = BackendCapability{};
    backend_capabilities_[BackendType::OPENGL].available = checkOpenGLAvailability();
    
    // Check Vulkan availability
    backend_capabilities_[BackendType::VULKAN] = BackendCapability{};
    backend_capabilities_[BackendType::VULKAN].available = checkVulkanAvailability();
}

bool BackendFactory::checkCudaAvailability() {
#ifdef HAVE_CUDA
    try {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        
        if (error != cudaSuccess || device_count == 0) {
            return false;
        }
        
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, 0);
        if (error != cudaSuccess) {
            return false;
        }
        
        // Fill CUDA capability information
        auto& cuda_cap = backend_capabilities_[BackendType::CUDA];
        cuda_cap.available = true;
        cuda_cap.version = std::to_string(prop.major) + "." + std::to_string(prop.minor);
        cuda_cap.device_name = prop.name;
        cuda_cap.memory_size = prop.totalGlobalMem;
        cuda_cap.compute_capability_major = prop.major;
        cuda_cap.compute_capability_minor = prop.minor;
        cuda_cap.supported_data_types = {DataType::FLOAT32, DataType::FLOAT16};
        cuda_cap.supported_layouts = {MemoryLayout::ROW_MAJOR, MemoryLayout::COLUMN_MAJOR};
        cuda_cap.max_threads = prop.maxThreadsPerBlock;
        cuda_cap.max_shared_memory = prop.sharedMemPerBlock;
        cuda_cap.supports_mixed_precision = prop.major >= 7; // Tensor cores available from Volta
        cuda_cap.supports_tensor_cores = prop.major >= 7;
        
        if (prop.major >= 8) {
            cuda_cap.supported_data_types.push_back(DataType::BFLOAT16);
        }
        
        return true;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

bool BackendFactory::checkOpenCLAvailability() {
#ifdef HAVE_OPENCL
    try {
        cl_uint num_platforms = 0;
        cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
        
        if (err != CL_SUCCESS || num_platforms == 0) {
            return false;
        }
        
        std::vector<cl_platform_id> platforms(num_platforms);
        err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        if (err != CL_SUCCESS) {
            return false;
        }
        
        // Check for at least one device
        for (cl_platform_id platform : platforms) {
            cl_uint num_devices = 0;
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
            if (err == CL_SUCCESS && num_devices > 0) {
                std::vector<cl_device_id> devices(num_devices);
                err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);
                if (err == CL_SUCCESS) {
                    // Get device information for the first device
                    cl_device_id device = devices[0];
                    
                    char device_name[256];
                    size_t max_work_group_size;
                    cl_ulong global_mem_size;
                    cl_ulong local_mem_size;
                    
                    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
                    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, nullptr);
                    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, nullptr);
                    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, nullptr);
                    
                    // Fill OpenCL capability information
                    auto& opencl_cap = backend_capabilities_[BackendType::OPENCL];
                    opencl_cap.available = true;
                    opencl_cap.version = "2.0"; // Default assumption
                    opencl_cap.device_name = device_name;
                    opencl_cap.memory_size = global_mem_size;
                    opencl_cap.supported_data_types = {DataType::FLOAT32};
                    opencl_cap.supported_layouts = {MemoryLayout::ROW_MAJOR, MemoryLayout::COLUMN_MAJOR};
                    opencl_cap.max_threads = max_work_group_size;
                    opencl_cap.max_shared_memory = local_mem_size;
                    opencl_cap.supports_mixed_precision = false;
                    opencl_cap.supports_tensor_cores = false;
                    
                    return true;
                }
            }
        }
        
        return false;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

bool BackendFactory::checkOpenGLAvailability() {
#ifdef HAVE_OPENGL
    try {
        // This is a simplified check - in practice, we'd need an OpenGL context
        // For now, we'll assume OpenGL is available if compiled with support
        auto& opengl_cap = backend_capabilities_[BackendType::OPENGL];
        opengl_cap.available = true;
        opengl_cap.version = "4.3"; // Minimum version for compute shaders
        opengl_cap.device_name = "OpenGL Device";
        opengl_cap.memory_size = 0; // Cannot determine without context
        opengl_cap.supported_data_types = {DataType::FLOAT32};
        opengl_cap.supported_layouts = {MemoryLayout::ROW_MAJOR};
        opengl_cap.max_threads = 1024; // Typical workgroup size
        opengl_cap.max_shared_memory = 32768; // Typical shared memory size
        opengl_cap.supports_mixed_precision = false;
        opengl_cap.supports_tensor_cores = false;
        
        return true;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

bool BackendFactory::checkVulkanAvailability() {
#ifdef HAVE_VULKAN
    try {
        VkApplicationInfo app_info = {};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "AttentionHPC";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "AttentionHPC";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_0;
        
        VkInstanceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;
        
        VkInstance instance;
        VkResult result = vkCreateInstance(&create_info, nullptr, &instance);
        
        if (result != VK_SUCCESS) {
            return false;
        }
        
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
        
        if (device_count == 0) {
            vkDestroyInstance(instance, nullptr);
            return false;
        }
        
        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
        
        VkPhysicalDeviceProperties device_properties;
        vkGetPhysicalDeviceProperties(devices[0], &device_properties);
        
        VkPhysicalDeviceMemoryProperties memory_properties;
        vkGetPhysicalDeviceMemoryProperties(devices[0], &memory_properties);
        
        // Calculate total memory
        size_t total_memory = 0;
        for (uint32_t i = 0; i < memory_properties.memoryHeapCount; ++i) {
            if (memory_properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                total_memory += memory_properties.memoryHeaps[i].size;
            }
        }
        
        // Fill Vulkan capability information
        auto& vulkan_cap = backend_capabilities_[BackendType::VULKAN];
        vulkan_cap.available = true;
        vulkan_cap.version = std::to_string(VK_VERSION_MAJOR(device_properties.apiVersion)) + "." +
                            std::to_string(VK_VERSION_MINOR(device_properties.apiVersion));
        vulkan_cap.device_name = device_properties.deviceName;
        vulkan_cap.memory_size = total_memory;
        vulkan_cap.supported_data_types = {DataType::FLOAT32};
        vulkan_cap.supported_layouts = {MemoryLayout::ROW_MAJOR, MemoryLayout::COLUMN_MAJOR};
        vulkan_cap.max_threads = device_properties.limits.maxComputeWorkGroupSize[0];
        vulkan_cap.max_shared_memory = device_properties.limits.maxComputeSharedMemorySize;
        vulkan_cap.supports_mixed_precision = false;
        vulkan_cap.supports_tensor_cores = false;
        
        vkDestroyInstance(instance, nullptr);
        return true;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

std::unique_ptr<AttentionInterface> BackendFactory::createBackend(BackendType type) {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    
    if (!factory.initialized_) {
        factory.initializeBackends();
    }
    
    if (!isBackendAvailable(type)) {
        throw BackendException(type, "Backend not available");
    }
    
    auto it = factory.backend_creators_.find(type);
    if (it == factory.backend_creators_.end()) {
        throw BackendException(type, "Backend creator not found");
    }
    
    try {
        return it->second();
    } catch (const std::exception& e) {
        throw BackendException(type, "Failed to create backend: " + std::string(e.what()));
    }
}

std::unique_ptr<AttentionInterface> BackendFactory::createFlashAttention(BackendType type) {
    return createBackend(type);
}

std::unique_ptr<AttentionInterface> BackendFactory::createPagedAttention(BackendType type) {
    return createBackend(type);
}

bool BackendFactory::isBackendAvailable(BackendType type) {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    
    if (!factory.initialized_) {
        factory.initializeBackends();
    }
    
    auto it = factory.backend_capabilities_.find(type);
    return it != factory.backend_capabilities_.end() && it->second.available;
}

std::vector<BackendType> BackendFactory::getAvailableBackends() {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    
    if (!factory.initialized_) {
        factory.initializeBackends();
    }
    
    std::vector<BackendType> available;
    for (const auto& pair : factory.backend_capabilities_) {
        if (pair.second.available) {
            available.push_back(pair.first);
        }
    }
    
    return available;
}

BackendCapability BackendFactory::getBackendCapability(BackendType type) {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    
    if (!factory.initialized_) {
        factory.initializeBackends();
    }
    
    auto it = factory.backend_capabilities_.find(type);
    if (it != factory.backend_capabilities_.end()) {
        return it->second;
    }
    
    return BackendCapability{}; // Return empty capability if not found
}

BackendType BackendFactory::selectBestBackend(const AttentionParams& params) {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    
    if (!factory.initialized_) {
        factory.initializeBackends();
    }
    
    BackendType best_backend = BackendType::CPU;
    int best_score = -1;
    
    for (BackendType type : factory.backend_priority_) {
        if (isBackendAvailable(type) && isBackendSuitable(type, params)) {
            int score = scoreBackend(type, params);
            if (score > best_score) {
                best_backend = type;
                best_score = score;
            }
        }
    }
    
    return best_backend;
}

BackendType BackendFactory::selectBestBackend(const std::vector<BackendType>& preferred_backends) {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    
    if (!factory.initialized_) {
        factory.initializeBackends();
    }
    
    for (BackendType type : preferred_backends) {
        if (isBackendAvailable(type)) {
            return type;
        }
    }
    
    // Fall back to default priority order
    return selectBestBackend();
}

void BackendFactory::setBackendPriority(const std::vector<BackendType>& priority_order) {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    factory.backend_priority_ = priority_order;
}

std::vector<BackendType> BackendFactory::getBackendPriority() {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    return factory.backend_priority_;
}

void BackendFactory::registerBackend(BackendType type, const std::string& name, BackendCreator creator) {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    
    factory.backend_creators_[type] = creator;
    factory.backend_names_[type] = name;
}

void BackendFactory::unregisterBackend(BackendType type) {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    
    factory.backend_creators_.erase(type);
    factory.backend_names_.erase(type);
}

std::string BackendFactory::backendTypeToString(BackendType type) {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    
    auto it = factory.backend_names_.find(type);
    if (it != factory.backend_names_.end()) {
        return it->second;
    }
    
    switch (type) {
        case BackendType::CPU: return "CPU";
        case BackendType::CUDA: return "CUDA";
        case BackendType::OPENCL: return "OpenCL";
        case BackendType::OPENGL: return "OpenGL";
        case BackendType::VULKAN: return "Vulkan";
        case BackendType::AUTO: return "AUTO";
        default: return "Unknown";
    }
}

BackendType BackendFactory::stringToBackendType(const std::string& type_str) {
    std::string lower_str = type_str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);
    
    if (lower_str == "cpu") return BackendType::CPU;
    if (lower_str == "cuda") return BackendType::CUDA;
    if (lower_str == "opencl") return BackendType::OPENCL;
    if (lower_str == "opengl") return BackendType::OPENGL;
    if (lower_str == "vulkan") return BackendType::VULKAN;
    if (lower_str == "auto") return BackendType::AUTO;
    
    throw std::invalid_argument("Unknown backend type: " + type_str);
}

std::vector<std::string> BackendFactory::getSupportedBackendNames() {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    
    std::vector<std::string> names;
    for (const auto& pair : factory.backend_names_) {
        names.push_back(pair.second);
    }
    
    return names;
}

void BackendFactory::printAvailableBackends() {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    
    if (!factory.initialized_) {
        factory.initializeBackends();
    }
    
    std::cout << "Available backends:\n";
    for (const auto& pair : factory.backend_capabilities_) {
        const auto& cap = pair.second;
        if (cap.available) {
            std::cout << "  " << backendTypeToString(pair.first) 
                     << " - " << cap.device_name 
                     << " (version: " << cap.version << ")\n";
        }
    }
}

std::string BackendFactory::getSystemInfo() {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    
    if (!factory.initialized_) {
        factory.initializeBackends();
    }
    
    std::ostringstream oss;
    oss << "System Information:\n";
    oss << "CPU cores: " << std::thread::hardware_concurrency() << "\n";
    
    for (const auto& pair : factory.backend_capabilities_) {
        const auto& cap = pair.second;
        if (cap.available) {
            oss << backendTypeToString(pair.first) << ":\n";
            oss << "  Device: " << cap.device_name << "\n";
            oss << "  Version: " << cap.version << "\n";
            oss << "  Memory: " << (cap.memory_size / (1024 * 1024)) << " MB\n";
            oss << "  Max threads: " << cap.max_threads << "\n";
        }
    }
    
    return oss.str();
}

bool BackendFactory::validateBackendCompatibility(BackendType type, const AttentionParams& params) {
    if (!isBackendAvailable(type)) {
        return false;
    }
    
    BackendCapability cap = getBackendCapability(type);
    
    // Check memory requirements (simplified)
    size_t required_memory = params.batch_size * params.sequence_length * 
                            params.num_heads * params.head_dimension * sizeof(float);
    
    if (cap.memory_size > 0 && required_memory > cap.memory_size) {
        return false;
    }
    
    return true;
}

void BackendFactory::enableDebugMode(bool enable) {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    factory.debug_mode_ = enable;
}

bool BackendFactory::isDebugModeEnabled() {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    return factory.debug_mode_;
}

void BackendFactory::setDefaultBackend(BackendType type) {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    factory.default_backend_ = type;
}

BackendType BackendFactory::getDefaultBackend() {
    auto& factory = getInstance();
    std::lock_guard<std::mutex> lock(factory.mutex_);
    return factory.default_backend_;
}

int BackendFactory::scoreBackend(BackendType type, const AttentionParams& params) {
    BackendCapability cap = getBackendCapability(type);
    
    if (!cap.available) {
        return -1;
    }
    
    int score = 0;
    
    // Base scores by backend type
    switch (type) {
        case BackendType::CUDA:
            score = 1000;
            if (cap.supports_tensor_cores) score += 200;
            break;
        case BackendType::VULKAN:
            score = 800;
            break;
        case BackendType::OPENCL:
            score = 600;
            break;
        case BackendType::OPENGL:
            score = 400;
            break;
        case BackendType::CPU:
            score = 200;
            break;
        default:
            score = 0;
            break;
    }
    
    // Adjust score based on memory requirements
    size_t required_memory = params.batch_size * params.sequence_length * 
                            params.num_heads * params.head_dimension * sizeof(float);
    
    if (cap.memory_size > 0) {
        float memory_ratio = static_cast<float>(required_memory) / cap.memory_size;
        if (memory_ratio > 0.8f) {
            score -= 100; // Penalize if using too much memory
        }
    }
    
    // Bonus for mixed precision support
    if (cap.supports_mixed_precision) {
        score += 50;
    }
    
    return score;
}

bool BackendFactory::isBackendSuitable(BackendType type, const AttentionParams& params) {
    BackendCapability cap = getBackendCapability(type);
    
    if (!cap.available) {
        return false;
    }
    
    // Check if backend supports required sequence length
    if (params.sequence_length > 8192 && type == BackendType::CPU) {
        return false; // CPU might be too slow for very long sequences
    }
    
    // Check memory requirements
    size_t required_memory = params.batch_size * params.sequence_length * 
                            params.num_heads * params.head_dimension * sizeof(float) * 4; // Factor for intermediate results
    
    if (cap.memory_size > 0 && required_memory > cap.memory_size) {
        return false;
    }
    
    return true;
}

} // namespace attention_hpc

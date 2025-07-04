#pragma once

#include "attention_interface.hpp"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <functional>

namespace attention_hpc {

// Enumeration of all supported HPC backend types
enum class BackendType {
    CPU,           // CPU implementation using optimized BLAS
    CUDA,          // NVIDIA CUDA implementation
    OPENCL,        // OpenCL implementation for cross-platform GPU computing
    OPENGL,        // OpenGL compute shader implementation
    VULKAN,        // Vulkan compute pipeline implementation
    AUTO           // Automatically select the best available backend
};

// Backend capability information
struct BackendCapability {
    bool available = false;
    std::string version;
    std::string device_name;
    size_t memory_size = 0;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    std::vector<DataType> supported_data_types;
    std::vector<MemoryLayout> supported_layouts;
    size_t max_threads = 0;
    size_t max_shared_memory = 0;
    bool supports_mixed_precision = false;
    bool supports_tensor_cores = false;
};

// Backend factory class using singleton pattern
class BackendFactory {
public:
    // Get the singleton instance
    static BackendFactory& getInstance();
    
    // Delete copy constructor and assignment operator
    BackendFactory(const BackendFactory&) = delete;
    BackendFactory& operator=(const BackendFactory&) = delete;
    
    // Factory methods for creating backend instances
    static std::unique_ptr<AttentionInterface> createBackend(BackendType type);
    static std::unique_ptr<AttentionInterface> createFlashAttention(BackendType type);
    static std::unique_ptr<AttentionInterface> createPagedAttention(BackendType type);
    
    // Backend availability detection
    static bool isBackendAvailable(BackendType type);
    static std::vector<BackendType> getAvailableBackends();
    static BackendCapability getBackendCapability(BackendType type);
    
    // Automatic backend selection
    static BackendType selectBestBackend(const AttentionParams& params = AttentionParams{});
    static BackendType selectBestBackend(const std::vector<BackendType>& preferred_backends);
    
    // Backend priority and preference management
    static void setBackendPriority(const std::vector<BackendType>& priority_order);
    static std::vector<BackendType> getBackendPriority();
    
    // Registration system for new backends
    using BackendCreator = std::function<std::unique_ptr<AttentionInterface>()>;
    static void registerBackend(BackendType type, const std::string& name, 
                               BackendCreator creator);
    static void unregisterBackend(BackendType type);
    
    // Utility methods
    static std::string backendTypeToString(BackendType type);
    static BackendType stringToBackendType(const std::string& type_str);
    static std::vector<std::string> getSupportedBackendNames();
    
    // Debug and information methods
    static void printAvailableBackends();
    static std::string getSystemInfo();
    static bool validateBackendCompatibility(BackendType type, const AttentionParams& params);
    
    // Configuration methods
    static void enableDebugMode(bool enable = true);
    static bool isDebugModeEnabled();
    static void setDefaultBackend(BackendType type);
    static BackendType getDefaultBackend();

private:
    // Private constructor for singleton
    BackendFactory();
    ~BackendFactory() = default;
    
    // Internal methods
    void initializeBackends();
    void detectBackendCapabilities();
    bool checkCudaAvailability();
    bool checkOpenCLAvailability();
    bool checkOpenGLAvailability();
    bool checkVulkanAvailability();
    
    // Member variables
    mutable std::mutex mutex_;
    std::unordered_map<BackendType, BackendCapability> backend_capabilities_;
    std::unordered_map<BackendType, BackendCreator> backend_creators_;
    std::unordered_map<BackendType, std::string> backend_names_;
    std::vector<BackendType> backend_priority_;
    BackendType default_backend_;
    bool debug_mode_;
    bool initialized_;
    
    // Static utility methods
    static int scoreBackend(BackendType type, const AttentionParams& params);
    static bool isBackendSuitable(BackendType type, const AttentionParams& params);
};

// Convenience macros for backend detection
#ifdef HAVE_CUDA
#define CUDA_BACKEND_AVAILABLE true
#else
#define CUDA_BACKEND_AVAILABLE false
#endif

#ifdef HAVE_OPENCL
#define OPENCL_BACKEND_AVAILABLE true
#else
#define OPENCL_BACKEND_AVAILABLE false
#endif

#ifdef HAVE_OPENGL
#define OPENGL_BACKEND_AVAILABLE true
#else
#define OPENGL_BACKEND_AVAILABLE false
#endif

#ifdef HAVE_VULKAN
#define VULKAN_BACKEND_AVAILABLE true
#else
#define VULKAN_BACKEND_AVAILABLE false
#endif

// Helper functions for backend management
namespace backend_utils {
    // Performance benchmarking
    double benchmarkBackend(BackendType type, const AttentionParams& params, 
                           int num_iterations = 10);
    std::vector<std::pair<BackendType, double>> benchmarkAllBackends(
        const AttentionParams& params, int num_iterations = 10);
    
    // Memory usage estimation
    size_t estimateMemoryUsage(BackendType type, const AttentionParams& params);
    
    // Backend recommendation
    BackendType recommendBackend(const AttentionParams& params, 
                                size_t available_memory = 0);
    
    // Compatibility checking
    bool checkDataTypeSupport(BackendType type, DataType data_type);
    bool checkMemoryLayoutSupport(BackendType type, MemoryLayout layout);
    
    // Version information
    std::string getBackendVersion(BackendType type);
    std::string getDriverVersion(BackendType type);
}

// Exception class for backend-related errors
class BackendException : public AttentionException {
public:
    explicit BackendException(BackendType backend_type, const std::string& message)
        : AttentionException(AttentionError::BACKEND_NOT_AVAILABLE, 
                            "Backend " + BackendFactory::backendTypeToString(backend_type) + 
                            ": " + message), backend_type_(backend_type) {}
    
    BackendType backend_type() const noexcept { return backend_type_; }
    
private:
    BackendType backend_type_;
};

} // namespace attention_hpc

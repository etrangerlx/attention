# AttentionHPC API Reference

## Table of Contents

1. [Overview](#overview)
2. [Core Interfaces](#core-interfaces)
   - [AttentionInterface](#attentioninterface)
   - [BackendFactory](#backendfactory)
   - [Tensor](#tensor)
3. [Data Types and Enumerations](#data-types-and-enumerations)
4. [Configuration Structures](#configuration-structures)
5. [Error Handling](#error-handling)
6. [Memory Management](#memory-management)
7. [Utilities](#utilities)
8. [Code Examples](#code-examples)
9. [Best Practices](#best-practices)

## Overview

AttentionHPC provides a comprehensive C++ API for high-performance attention mechanisms. The library is designed around a modular architecture with clear separation between algorithm interfaces and backend implementations.

### Key Design Principles

- **Backend Agnostic**: Uniform interface across CUDA, OpenCL, OpenGL, and Vulkan backends
- **Type Safety**: Template-based design with compile-time type checking
- **Memory Efficient**: Optimized memory management with pooling and reuse
- **Exception Safe**: RAII-based resource management with comprehensive error handling

## Core Interfaces

### AttentionInterface

The abstract base class for all attention implementations.

```cpp
class AttentionInterface {
public:
    virtual ~AttentionInterface() = default;
    
    // Configuration
    virtual void configure(const AttentionParams& params, 
                          const PerformanceConfig& perf_config = PerformanceConfig{}) = 0;
    
    // Core computation
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
    
    // Capabilities
    virtual bool supports_data_type(DataType data_type) const = 0;
    virtual bool supports_memory_layout(MemoryLayout layout) const = 0;
    virtual std::string get_backend_name() const = 0;
    virtual std::string get_version() const = 0;
    
    // Profiling
    virtual void enable_profiling(bool enable) = 0;
    virtual std::vector<std::pair<std::string, double>> get_profiling_results() const = 0;
    virtual void reset_profiling() = 0;
    
    // Validation
    virtual AttentionError validate_inputs(const TensorShape& query_shape,
                                          const TensorShape& key_shape,
                                          const TensorShape& value_shape,
                                          const TensorShape& output_shape,
                                          const AttentionParams& params) const = 0;
};
```

#### Methods

##### configure()

Configures the attention implementation with specified parameters.

**Parameters:**
- `params` (const AttentionParams&): Attention computation parameters
- `perf_config` (const PerformanceConfig&): Performance optimization configuration (optional)

**Throws:**
- `AttentionException`: If configuration parameters are invalid or incompatible with backend

**Example:**
```cpp
AttentionParams params;
params.batch_size = 2;
params.sequence_length = 512;
params.num_heads = 8;
params.head_dimension = 64;
params.causal_mask = true;

PerformanceConfig perf_config;
perf_config.block_size_m = 64;
perf_config.use_fast_math = true;

attention->configure(params, perf_config);
```

##### forward()

Executes the forward pass of attention computation.

**Parameters:**
- `query_shape` (const TensorShape&): Shape of query tensor
- `query_data` (const void*): Pointer to query data
- `key_shape` (const TensorShape&): Shape of key tensor
- `key_data` (const void*): Pointer to key data
- `value_shape` (const TensorShape&): Shape of value tensor
- `value_data` (const void*): Pointer to value data
- `output_shape` (const TensorShape&): Shape of output tensor
- `output_data` (void*): Pointer to output buffer
- `attention_mask` (const void*, optional): Attention mask data

**Throws:**
- `AttentionException`: If computation fails or inputs are invalid

**Example:**
```cpp
TensorShape shape({2, 512, 8, 64}, DataType::FLOAT32);
std::vector<float> query(shape.total_elements());
std::vector<float> key(shape.total_elements());
std::vector<float> value(shape.total_elements());
std::vector<float> output(shape.total_elements());

// Fill input data...

attention->forward(shape, query.data(),
                  shape, key.data(),
                  shape, value.data(),
                  shape, output.data());
```

##### backward()

Executes the backward pass for gradient computation.

**Parameters:**
- `grad_output_shape` (const TensorShape&): Shape of gradient output tensor
- `grad_output_data` (const void*): Pointer to gradient output data
- `query_shape` (const TensorShape&): Shape of query tensor
- `query_data` (const void*): Pointer to query data
- `key_shape` (const TensorShape&): Shape of key tensor
- `key_data` (const void*): Pointer to key data
- `value_shape` (const TensorShape&): Shape of value tensor
- `value_data` (const void*): Pointer to value data
- `grad_query_shape` (const TensorShape&): Shape of query gradient tensor
- `grad_query_data` (void*): Pointer to query gradient buffer
- `grad_key_shape` (const TensorShape&): Shape of key gradient tensor
- `grad_key_data` (void*): Pointer to key gradient buffer
- `grad_value_shape` (const TensorShape&): Shape of value gradient tensor
- `grad_value_data` (void*): Pointer to value gradient buffer
- `attention_mask` (const void*, optional): Attention mask data

**Throws:**
- `AttentionException`: If computation fails or inputs are invalid

##### get_workspace_size()

Returns the required workspace memory size for given parameters.

**Parameters:**
- `params` (const AttentionParams&): Attention parameters

**Returns:**
- `size_t`: Required workspace size in bytes

**Example:**
```cpp
size_t workspace_size = attention->get_workspace_size(params);
std::vector<uint8_t> workspace(workspace_size);
attention->set_workspace(workspace.data(), workspace_size);
```

##### set_workspace()

Sets the workspace memory for attention computation.

**Parameters:**
- `workspace_ptr` (void*): Pointer to workspace memory
- `workspace_size` (size_t): Size of workspace in bytes

##### supports_data_type()

Checks if the backend supports a specific data type.

**Parameters:**
- `data_type` (DataType): Data type to check

**Returns:**
- `bool`: True if supported, false otherwise

##### supports_memory_layout()

Checks if the backend supports a specific memory layout.

**Parameters:**
- `layout` (MemoryLayout): Memory layout to check

**Returns:**
- `bool`: True if supported, false otherwise

##### get_backend_name()

Returns the name of the backend implementation.

**Returns:**
- `std::string`: Backend name (e.g., "CUDA", "OpenCL", "CPU")

##### get_version()

Returns the version string of the implementation.

**Returns:**
- `std::string`: Version string

##### enable_profiling()

Enables or disables performance profiling.

**Parameters:**
- `enable` (bool): True to enable, false to disable

##### get_profiling_results()

Returns profiling results as key-value pairs.

**Returns:**
- `std::vector<std::pair<std::string, double>>`: Profiling data with operation names and times

**Example:**
```cpp
attention->enable_profiling(true);
// ... perform computations ...
auto results = attention->get_profiling_results();
for (const auto& [op_name, time_ms] : results) {
    std::cout << op_name << ": " << time_ms << " ms" << std::endl;
}
```

##### reset_profiling()

Resets profiling counters and statistics.

##### validate_inputs()

Validates input tensor shapes and parameters.

**Parameters:**
- `query_shape` (const TensorShape&): Query tensor shape
- `key_shape` (const TensorShape&): Key tensor shape
- `value_shape` (const TensorShape&): Value tensor shape
- `output_shape` (const TensorShape&): Output tensor shape
- `params` (const AttentionParams&): Attention parameters

**Returns:**
- `AttentionError`: Error code indicating validation result

### BackendFactory

Factory class for creating attention backend instances.

```cpp
class BackendFactory {
public:
    // Factory methods
    static std::unique_ptr<AttentionInterface> createBackend(BackendType type);
    static std::unique_ptr<AttentionInterface> createFlashAttention(BackendType type);
    static std::unique_ptr<AttentionInterface> createPagedAttention(BackendType type);
    
    // Backend detection
    static bool isBackendAvailable(BackendType type);
    static std::vector<BackendType> getAvailableBackends();
    static BackendCapability getBackendCapability(BackendType type);
    
    // Backend selection
    static BackendType selectBestBackend(const AttentionParams& params = AttentionParams{});
    static BackendType selectBestBackend(const std::vector<BackendType>& preferred_backends);
    
    // Configuration
    static void setBackendPriority(const std::vector<BackendType>& priority_order);
    static std::vector<BackendType> getBackendPriority();
    
    // Utilities
    static std::string backendTypeToString(BackendType type);
    static BackendType stringToBackendType(const std::string& type_str);
    static void printAvailableBackends();
};
```

#### Methods

##### createBackend()

Creates a generic attention backend instance.

**Parameters:**
- `type` (BackendType): Backend type to create

**Returns:**
- `std::unique_ptr<AttentionInterface>`: Pointer to created backend instance

**Throws:**
- `AttentionException`: If backend is not available or creation fails

**Example:**
```cpp
try {
    auto attention = BackendFactory::createBackend(BackendType::CUDA);
    // Use attention instance...
} catch (const AttentionException& e) {
    std::cerr << "Failed to create CUDA backend: " << e.what() << std::endl;
}
```

##### createFlashAttention()

Creates a FlashAttention-specific backend instance.

**Parameters:**
- `type` (BackendType): Backend type for FlashAttention

**Returns:**
- `std::unique_ptr<AttentionInterface>`: FlashAttention instance

##### createPagedAttention()

Creates a PagedAttention-specific backend instance.

**Parameters:**
- `type` (BackendType): Backend type for PagedAttention

**Returns:**
- `std::unique_ptr<AttentionInterface>`: PagedAttention instance

##### isBackendAvailable()

Checks if a specific backend is available on the current system.

**Parameters:**
- `type` (BackendType): Backend type to check

**Returns:**
- `bool`: True if available, false otherwise

**Example:**
```cpp
if (BackendFactory::isBackendAvailable(BackendType::CUDA)) {
    auto cuda_attention = BackendFactory::createFlashAttention(BackendType::CUDA);
} else {
    std::cout << "CUDA not available, falling back to CPU" << std::endl;
    auto cpu_attention = BackendFactory::createFlashAttention(BackendType::CPU);
}
```

##### getAvailableBackends()

Returns a list of all available backends on the system.

**Returns:**
- `std::vector<BackendType>`: List of available backend types

##### getBackendCapability()

Returns detailed capability information for a backend.

**Parameters:**
- `type` (BackendType): Backend type to query

**Returns:**
- `BackendCapability`: Structure containing backend capabilities

**Example:**
```cpp
auto capability = BackendFactory::getBackendCapability(BackendType::CUDA);
std::cout << "Device: " << capability.device_name << std::endl;
std::cout << "Memory: " << capability.memory_size / (1024*1024) << " MB" << std::endl;
std::cout << "Compute Capability: " << capability.compute_capability_major 
          << "." << capability.compute_capability_minor << std::endl;
```

##### selectBestBackend()

Automatically selects the best available backend for given parameters.

**Parameters:**
- `params` (const AttentionParams&): Attention parameters for optimization (optional)

**Returns:**
- `BackendType`: Best backend type for the given parameters

**Example:**
```cpp
AttentionParams params;
params.batch_size = 4;
params.sequence_length = 2048;
params.num_heads = 16;
params.head_dimension = 64;

BackendType best_backend = BackendFactory::selectBestBackend(params);
auto attention = BackendFactory::createFlashAttention(best_backend);
```

### Tensor

Template class for multi-dimensional tensor operations.

```cpp
template<typename T>
class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<size_t>& dimensions);
    Tensor(const std::vector<size_t>& dimensions, const T& initial_value);
    Tensor(const TensorShape& shape);
    
    // Data access
    T* data();
    const T* data() const;
    T& operator[](size_t index);
    const T& operator[](size_t index) const;
    T& at(const std::vector<size_t>& indices);
    const T& at(const std::vector<size_t>& indices) const;
    
    // Shape and size
    const std::vector<size_t>& dimensions() const;
    size_t total_elements() const;
    size_t size_bytes() const;
    DataType data_type() const;
    MemoryLayout memory_layout() const;
    
    // Memory management
    void allocate();
    void deallocate();
    void resize(const std::vector<size_t>& new_dimensions);
    void copy_from(const Tensor<T>& other);
    void copy_to_device();
    void copy_from_device();
    
    // Utilities
    void fill(const T& value);
    void zero();
    bool is_contiguous() const;
    Tensor<T> reshape(const std::vector<size_t>& new_dimensions) const;
    Tensor<T> transpose(const std::vector<size_t>& axes) const;
};
```

#### Methods

##### Constructor

Creates a tensor with specified dimensions.

**Example:**
```cpp
// Create a 4D tensor for batch=2, seq_len=512, heads=8, head_dim=64
Tensor<float> tensor({2, 512, 8, 64});
tensor.allocate();
tensor.zero();
```

##### data()

Returns a pointer to the underlying data.

**Returns:**
- `T*`: Pointer to tensor data

##### at()

Accesses tensor element at specified multi-dimensional indices.

**Parameters:**
- `indices` (const std::vector<size_t>&): Multi-dimensional indices

**Returns:**
- `T&`: Reference to tensor element

**Example:**
```cpp
Tensor<float> tensor({2, 3, 4});
tensor.allocate();
tensor.at({1, 2, 3}) = 42.0f;
float value = tensor.at({1, 2, 3});
```

##### resize()

Resizes the tensor to new dimensions.

**Parameters:**
- `new_dimensions` (const std::vector<size_t>&): New tensor dimensions

##### copy_from()

Copies data from another tensor.

**Parameters:**
- `other` (const Tensor<T>&): Source tensor

## Data Types and Enumerations

### BackendType

```cpp
enum class BackendType {
    CPU,           // CPU implementation
    CUDA,          // NVIDIA CUDA implementation
    OPENCL,        // OpenCL implementation
    OPENGL,        // OpenGL compute shader implementation
    VULKAN,        // Vulkan compute pipeline implementation
    AUTO           // Automatically select best backend
};
```

### DataType

```cpp
enum class DataType {
    FLOAT32,       // 32-bit floating point
    FLOAT16,       // 16-bit floating point
    BFLOAT16       // 16-bit brain floating point
};
```

### MemoryLayout

```cpp
enum class MemoryLayout {
    ROW_MAJOR,     // Row-major memory layout (C-style)
    COLUMN_MAJOR   // Column-major memory layout (Fortran-style)
};
```

### AttentionError

```cpp
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
```

## Configuration Structures

### AttentionParams

Configuration parameters for attention computation.

```cpp
struct AttentionParams {
    size_t batch_size;              // Batch size
    size_t sequence_length;         // Sequence length
    size_t num_heads;              // Number of attention heads
    size_t head_dimension;         // Dimension per head
    size_t key_value_heads = 0;    // Number of key-value heads (for GQA/MQA)
    
    float scale_factor = 0.0f;     // Attention scale factor (0 = auto-compute)
    bool causal_mask = false;      // Apply causal masking
    bool apply_rotary_embedding = false; // Apply rotary position embedding
    
    size_t max_sequence_length = 0; // Maximum sequence length (for paged attention)
    size_t page_size = 16;         // Page size for paged attention
};
```

### PerformanceConfig

Performance optimization configuration.

```cpp
struct PerformanceConfig {
    // Block sizes for tiling
    size_t block_size_m = 64;
    size_t block_size_n = 64;
    size_t block_size_k = 64;
    
    // Threading
    size_t num_threads = 0;        // 0 = auto-detect
    
    // Memory optimization
    bool use_memory_pool = true;
    bool enable_memory_reuse = true;
    
    // Computation optimization
    bool use_fast_math = false;
    bool enable_mixed_precision = false;
    
    // Backend-specific
    size_t shared_memory_size = 0; // 0 = auto-determine
    size_t max_workspace_size = 1024 * 1024 * 1024; // 1GB default
    
    // Debug and profiling
    bool enable_profiling = false;
    bool enable_debug_checks = true;
};
```

### TensorShape

Tensor shape and metadata information.

```cpp
struct TensorShape {
    std::vector<size_t> dimensions; // Tensor dimensions
    DataType data_type;            // Element data type
    MemoryLayout layout;           // Memory layout
    
    // Constructor
    TensorShape(std::vector<size_t> dims, 
                DataType dtype = DataType::FLOAT32, 
                MemoryLayout mem_layout = MemoryLayout::ROW_MAJOR);
    
    // Utility methods
    size_t total_elements() const;
    size_t element_size() const;
    size_t total_bytes() const;
};
```

### BackendCapability

Backend capability information.

```cpp
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
```

## Error Handling

### AttentionException

Custom exception class for attention-related errors.

```cpp
class AttentionException : public std::runtime_error {
public:
    explicit AttentionException(AttentionError error_code, const std::string& message);
    AttentionError error_code() const noexcept;
};
```

#### Usage Example

```cpp
try {
    auto attention = BackendFactory::createFlashAttention(BackendType::CUDA);
    attention->configure(params);
    attention->forward(/* ... */);
} catch (const AttentionException& e) {
    switch (e.error_code()) {
        case AttentionError::BACKEND_NOT_AVAILABLE:
            std::cerr << "Backend not available: " << e.what() << std::endl;
            // Fall back to CPU backend
            break;
        case AttentionError::INVALID_INPUT:
            std::cerr << "Invalid input: " << e.what() << std::endl;
            // Check and fix input parameters
            break;
        case AttentionError::MEMORY_ALLOCATION_FAILED:
            std::cerr << "Memory allocation failed: " << e.what() << std::endl;
            // Reduce batch size or sequence length
            break;
        default:
            std::cerr << "Attention error: " << e.what() << std::endl;
            break;
    }
}
```

### Error Checking Functions

```cpp
// Convert error code to string
std::string attention_error_to_string(AttentionError error);

// Validate tensor shapes
AttentionError validate_tensor_compatibility(const TensorShape& shape1, 
                                           const TensorShape& shape2);

// Check parameter validity
AttentionError validate_attention_params(const AttentionParams& params);
```

## Memory Management

### MemoryPool

Memory pool for efficient memory allocation and reuse.

```cpp
class MemoryPool {
public:
    // Constructor
    explicit MemoryPool(size_t initial_size = 0);
    
    // Allocation
    void* allocate(size_t size, size_t alignment = 32);
    void deallocate(void* ptr);
    
    // Pool management
    void reserve(size_t size);
    void clear();
    size_t total_allocated() const;
    size_t available_memory() const;
    
    // Statistics
    size_t allocation_count() const;
    size_t deallocation_count() const;
    size_t peak_usage() const;
};
```

#### Usage Example

```cpp
// Create memory pool with 1GB initial size
MemoryPool pool(1024 * 1024 * 1024);

// Allocate tensors using the pool
auto* query_data = pool.allocate(query_size);
auto* key_data = pool.allocate(key_size);
auto* value_data = pool.allocate(value_size);

// Use tensors...

// Deallocate when done
pool.deallocate(query_data);
pool.deallocate(key_data);
pool.deallocate(value_data);
```

## Utilities

### Timer

High-precision timer for performance measurement.

```cpp
class Timer {
public:
    void start();
    void stop();
    void reset();
    double elapsed_ms() const;
    double elapsed_s() const;
    
    // Statistics
    void add_measurement();
    double average_ms() const;
    double min_ms() const;
    double max_ms() const;
    size_t measurement_count() const;
};
```

### Logger

Logging system for debugging and monitoring.

```cpp
class Logger {
public:
    enum Level { DEBUG, INFO, WARNING, ERROR };
    
    static void log(Level level, const std::string& message);
    static void set_level(Level min_level);
    static void set_output_file(const std::string& filename);
    
    // Convenience macros
    #define LOG_DEBUG(msg) Logger::log(Logger::DEBUG, msg)
    #define LOG_INFO(msg) Logger::log(Logger::INFO, msg)
    #define LOG_WARNING(msg) Logger::log(Logger::WARNING, msg)
    #define LOG_ERROR(msg) Logger::log(Logger::ERROR, msg)
};
```

## Code Examples

### Basic FlashAttention Usage

```cpp
#include "api/attention_interface.hpp"
#include "api/backend_factory.hpp"

int main() {
    try {
        // Create FlashAttention with best available backend
        auto attention = BackendFactory::createFlashAttention(BackendType::AUTO);
        
        // Configure parameters
        AttentionParams params;
        params.batch_size = 2;
        params.sequence_length = 1024;
        params.num_heads = 12;
        params.head_dimension = 64;
        params.causal_mask = true;
        
        PerformanceConfig perf_config;
        perf_config.enable_mixed_precision = true;
        perf_config.use_fast_math = true;
        
        attention->configure(params, perf_config);
        
        // Prepare tensors
        TensorShape tensor_shape({params.batch_size, params.sequence_length, 
                                 params.num_heads, params.head_dimension}, 
                                DataType::FLOAT32);
        
        Tensor<float> query(tensor_shape);
        Tensor<float> key(tensor_shape);
        Tensor<float> value(tensor_shape);
        Tensor<float> output(tensor_shape);
        
        query.allocate();
        key.allocate();
        value.allocate();
        output.allocate();
        
        // Initialize with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < query.total_elements(); ++i) {
            query.data()[i] = dist(gen);
            key.data()[i] = dist(gen);
            value.data()[i] = dist(gen);
        }
        
        // Execute forward pass
        attention->forward(tensor_shape, query.data(),
                          tensor_shape, key.data(),
                          tensor_shape, value.data(),
                          tensor_shape, output.data());
        
        std::cout << "Forward pass completed successfully!" << std::endl;
        
    } catch (const AttentionException& e) {
        std::cerr << "Attention error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

### Performance Benchmarking

```cpp
#include "utils/timer.hpp"

void benchmark_attention(AttentionInterface* attention, 
                        const AttentionParams& params) {
    // Setup tensors...
    TensorShape shape({params.batch_size, params.sequence_length,
                      params.num_heads, params.head_dimension});
    
    Tensor<float> query(shape), key(shape), value(shape), output(shape);
    query.allocate(); key.allocate(); value.allocate(); output.allocate();
    
    // Fill with random data...
    
    // Warmup runs
    for (int i = 0; i < 5; ++i) {
        attention->forward(shape, query.data(), shape, key.data(),
                          shape, value.data(), shape, output.data());
    }
    
    // Benchmark runs
    Timer timer;
    const int num_runs = 100;
    
    timer.start();
    for (int i = 0; i < num_runs; ++i) {
        attention->forward(shape, query.data(), shape, key.data(),
                          shape, value.data(), shape, output.data());
    }
    timer.stop();
    
    double avg_time = timer.elapsed_ms() / num_runs;
    double gflops = calculate_gflops(params, avg_time);
    
    std::cout << "Average time: " << avg_time << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
}
```

### Multi-Backend Comparison

```cpp
void compare_backends(const AttentionParams& params) {
    std::vector<BackendType> backends = BackendFactory::getAvailableBackends();
    
    for (BackendType backend : backends) {
        try {
            auto attention = BackendFactory::createFlashAttention(backend);
            attention->configure(params);
            
            std::cout << "Testing " << BackendFactory::backendTypeToString(backend) 
                     << " backend..." << std::endl;
            
            benchmark_attention(attention.get(), params);
            
        } catch (const AttentionException& e) {
            std::cout << "Backend " << BackendFactory::backendTypeToString(backend)
                     << " failed: " << e.what() << std::endl;
        }
    }
}
```

### Error Handling Best Practices

```cpp
std::unique_ptr<AttentionInterface> create_robust_attention(
    const AttentionParams& params) {
    
    // Try backends in order of preference
    std::vector<BackendType> preference = {
        BackendType::CUDA,
        BackendType::OPENCL,
        BackendType::VULKAN,
        BackendType::OPENGL,
        BackendType::CPU
    };
    
    for (BackendType backend : preference) {
        if (BackendFactory::isBackendAvailable(backend)) {
            try {
                auto attention = BackendFactory::createFlashAttention(backend);
                attention->configure(params);
                
                std::cout << "Successfully created " 
                         << BackendFactory::backendTypeToString(backend)
                         << " backend" << std::endl;
                return attention;
                
            } catch (const AttentionException& e) {
                std::cout << "Failed to create " 
                         << BackendFactory::backendTypeToString(backend)
                         << " backend: " << e.what() << std::endl;
                continue;
            }
        }
    }
    
    throw AttentionException(AttentionError::BACKEND_NOT_AVAILABLE,
                            "No suitable backend available");
}
```

## Best Practices

### Performance Optimization

1. **Memory Management**
   - Use memory pools for frequent allocations
   - Reuse tensors when possible
   - Align memory for optimal performance

2. **Backend Selection**
   - Use AUTO backend type for automatic selection
   - Check backend capabilities before using advanced features
   - Implement fallback mechanisms for robust deployment

3. **Configuration Tuning**
   - Adjust block sizes based on problem size
   - Enable mixed precision on supported hardware
   - Use fast math for better performance when accuracy permits

### Error Handling

1. **Exception

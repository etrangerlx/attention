#pragma once

#include <exception>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <sstream>
#include <cstdint>
#include <typeinfo>
#include <functional>
#include <thread>
#include <chrono>

namespace attention_hpc {

// Forward declarations
class StackTrace;
class ErrorContext;

// Error codes enumeration
enum class ErrorCode : uint32_t {
    // Success
    SUCCESS = 0,
    
    // General errors (1000-1999)
    UNKNOWN_ERROR = 1000,
    INTERNAL_ERROR = 1001,
    NOT_IMPLEMENTED = 1002,
    INVALID_STATE = 1003,
    OPERATION_FAILED = 1004,
    TIMEOUT_EXCEEDED = 1005,
    RESOURCE_EXHAUSTED = 1006,
    PERMISSION_DENIED = 1007,
    
    // Parameter and input errors (2000-2999)
    INVALID_PARAMETER = 2000,
    NULL_POINTER = 2001,
    INVALID_INPUT = 2002,
    INVALID_OUTPUT = 2003,
    PARAMETER_OUT_OF_RANGE = 2004,
    INVALID_TENSOR_SHAPE = 2005,
    INVALID_DATA_TYPE = 2006,
    INVALID_MEMORY_LAYOUT = 2007,
    INCOMPATIBLE_DIMENSIONS = 2008,
    INVALID_ATTENTION_PARAMS = 2009,
    
    // Memory errors (3000-3999)
    MEMORY_ALLOCATION_FAILED = 3000,
    MEMORY_DEALLOCATION_FAILED = 3001,
    OUT_OF_MEMORY = 3002,
    INVALID_MEMORY_ACCESS = 3003,
    MEMORY_CORRUPTION = 3004,
    BUFFER_OVERFLOW = 3005,
    BUFFER_UNDERFLOW = 3006,
    MEMORY_ALIGNMENT_ERROR = 3007,
    MEMORY_POOL_EXHAUSTED = 3008,
    
    // Backend errors (4000-4999)
    BACKEND_NOT_AVAILABLE = 4000,
    BACKEND_INITIALIZATION_FAILED = 4001,
    BACKEND_NOT_SUPPORTED = 4002,
    DEVICE_NOT_FOUND = 4003,
    DEVICE_BUSY = 4004,
    DEVICE_ERROR = 4005,
    DRIVER_ERROR = 4006,
    RUNTIME_ERROR = 4007,
    
    // CUDA specific (4100-4199)
    CUDA_ERROR = 4100,
    CUDA_OUT_OF_MEMORY = 4101,
    CUDA_INVALID_DEVICE = 4102,
    CUDA_INVALID_CONTEXT = 4103,
    CUDA_KERNEL_LAUNCH_FAILED = 4104,
    CUDA_INVALID_MEMORY_ACCESS = 4105,
    CUDA_DRIVER_VERSION_MISMATCH = 4106,
    
    // OpenCL specific (4200-4299)
    OPENCL_ERROR = 4200,
    OPENCL_PLATFORM_NOT_FOUND = 4201,
    OPENCL_DEVICE_NOT_FOUND = 4202,
    OPENCL_CONTEXT_CREATION_FAILED = 4203,
    OPENCL_PROGRAM_BUILD_FAILED = 4204,
    OPENCL_KERNEL_EXECUTION_FAILED = 4205,
    OPENCL_MEMORY_OBJECT_ALLOCATION_FAILED = 4206,
    
    // OpenGL specific (4300-4399)
    OPENGL_ERROR = 4300,
    OPENGL_CONTEXT_NOT_AVAILABLE = 4301,
    OPENGL_SHADER_COMPILATION_FAILED = 4302,
    OPENGL_PROGRAM_LINKING_FAILED = 4303,
    OPENGL_BUFFER_CREATION_FAILED = 4304,
    OPENGL_COMPUTE_SHADER_NOT_SUPPORTED = 4305,
    
    // Vulkan specific (4400-4499)
    VULKAN_ERROR = 4400,
    VULKAN_INSTANCE_CREATION_FAILED = 4401,
    VULKAN_DEVICE_NOT_SUITABLE = 4402,
    VULKAN_LOGICAL_DEVICE_CREATION_FAILED = 4403,
    VULKAN_SHADER_MODULE_CREATION_FAILED = 4404,
    VULKAN_PIPELINE_CREATION_FAILED = 4405,
    VULKAN_BUFFER_CREATION_FAILED = 4406,
    VULKAN_MEMORY_ALLOCATION_FAILED = 4407,
    
    // Computation errors (5000-5999)
    COMPUTATION_FAILED = 5000,
    INVALID_COMPUTATION_GRAPH = 5001,
    NUMERICAL_INSTABILITY = 5002,
    CONVERGENCE_FAILED = 5003,
    INVALID_ALGORITHM_CONFIGURATION = 5004,
    WORKSPACE_SIZE_INSUFFICIENT = 5005,
    ATTENTION_COMPUTATION_FAILED = 5006,
    SOFTMAX_COMPUTATION_FAILED = 5007,
    GRADIENT_COMPUTATION_FAILED = 5008,
    
    // I/O and file errors (6000-6999)
    FILE_NOT_FOUND = 6000,
    FILE_ACCESS_DENIED = 6001,
    FILE_WRITE_ERROR = 6002,
    FILE_READ_ERROR = 6003,
    INVALID_FILE_FORMAT = 6004,
    FILE_CORRUPTION = 6005,
    DISK_FULL = 6006,
    
    // Configuration errors (7000-7999)
    INVALID_CONFIGURATION = 7000,
    CONFIGURATION_NOT_FOUND = 7001,
    CONFIGURATION_PARSING_ERROR = 7002,
    INCOMPATIBLE_CONFIGURATION = 7003,
    MISSING_REQUIRED_PARAMETER = 7004,
    
    // Threading and concurrency errors (8000-8999)
    THREAD_CREATION_FAILED = 8000,
    THREAD_JOIN_FAILED = 8001,
    MUTEX_LOCK_FAILED = 8002,
    CONDITION_VARIABLE_ERROR = 8003,
    DEADLOCK_DETECTED = 8004,
    RACE_CONDITION_DETECTED = 8005,
    
    // Network and communication errors (9000-9999)
    NETWORK_ERROR = 9000,
    CONNECTION_FAILED = 9001,
    CONNECTION_TIMEOUT = 9002,
    DATA_TRANSMISSION_ERROR = 9003,
    PROTOCOL_ERROR = 9004
};

// Error severity levels
enum class ErrorSeverity {
    INFO,
    WARNING,
    ERROR,
    CRITICAL,
    FATAL
};

// Stack trace information
struct StackTraceEntry {
    std::string function_name;
    std::string file_name;
    int line_number;
    std::string module_name;
    void* address;
    
    StackTraceEntry() : line_number(0), address(nullptr) {}
    
    StackTraceEntry(const std::string& func, const std::string& file, int line, 
                   const std::string& module = "", void* addr = nullptr)
        : function_name(func), file_name(file), line_number(line), 
          module_name(module), address(addr) {}
    
    std::string to_string() const {
        std::ostringstream oss;
        oss << function_name;
        if (!file_name.empty()) {
            oss << " (" << file_name;
            if (line_number > 0) {
                oss << ":" << line_number;
            }
            oss << ")";
        }
        if (!module_name.empty()) {
            oss << " [" << module_name << "]";
        }
        if (address) {
            oss << " @" << address;
        }
        return oss.str();
    }
};

// Stack trace capture and management
class StackTrace {
public:
    StackTrace();
    explicit StackTrace(size_t skip_frames);
    ~StackTrace() = default;
    
    // Copy and move constructors
    StackTrace(const StackTrace& other) = default;
    StackTrace(StackTrace&& other) noexcept = default;
    StackTrace& operator=(const StackTrace& other) = default;
    StackTrace& operator=(StackTrace&& other) noexcept = default;
    
    // Stack trace operations
    void capture(size_t skip_frames = 0);
    void clear();
    bool empty() const;
    size_t size() const;
    
    // Access stack trace entries
    const std::vector<StackTraceEntry>& get_entries() const;
    const StackTraceEntry& get_entry(size_t index) const;
    
    // String representation
    std::string to_string(size_t max_entries = 0) const;
    std::vector<std::string> to_string_vector() const;
    
    // Static utilities
    static StackTrace current(size_t skip_frames = 1);
    static bool is_supported();
    static void enable_symbolication(bool enable);

private:
    std::vector<StackTraceEntry> entries_;
    
    void capture_native_stack_trace(size_t skip_frames);
    std::string symbolicate_address(void* address) const;
};

// Error context for additional debugging information
class ErrorContext {
public:
    ErrorContext() = default;
    ~ErrorContext() = default;
    
    // Copy and move constructors
    ErrorContext(const ErrorContext& other) = default;
    ErrorContext(ErrorContext&& other) noexcept = default;
    ErrorContext& operator=(const ErrorContext& other) = default;
    ErrorContext& operator=(ErrorContext&& other) noexcept = default;
    
    // Context information management
    void add_context(const std::string& key, const std::string& value);
    void add_context(const std::string& key, int64_t value);
    void add_context(const std::string& key, double value);
    void add_context(const std::string& key, bool value);
    
    template<typename T>
    void add_context(const std::string& key, const T& value) {
        std::ostringstream oss;
        oss << value;
        context_data_[key] = oss.str();
    }
    
    // Context retrieval
    std::string get_context(const std::string& key) const;
    bool has_context(const std::string& key) const;
    void remove_context(const std::string& key);
    void clear_context();
    
    // Get all context data
    const std::unordered_map<std::string, std::string>& get_all_context() const;
    
    // String representation
    std::string to_string() const;
    
    // Predefined context keys
    static const char* FUNCTION_NAME;
    static const char* FILE_NAME;
    static const char* LINE_NUMBER;
    static const char* THREAD_ID;
    static const char* TIMESTAMP;
    static const char* TENSOR_SHAPE;
    static const char* DATA_TYPE;
    static const char* BACKEND_TYPE;
    static const char* DEVICE_NAME;
    static const char* MEMORY_SIZE;
    static const char* BATCH_SIZE;
    static const char* SEQUENCE_LENGTH;
    static const char* NUM_HEADS;
    static const char* HEAD_DIMENSION;

private:
    std::unordered_map<std::string, std::string> context_data_;
};

// Base exception class for all AttentionHPC exceptions
class AttentionException : public std::exception {
public:
    // Constructors
    explicit AttentionException(ErrorCode error_code, const std::string& message = "");
    AttentionException(ErrorCode error_code, const std::string& message, const ErrorContext& context);
    AttentionException(ErrorCode error_code, const std::string& message, const StackTrace& stack_trace);
    AttentionException(ErrorCode error_code, const std::string& message, 
                      const ErrorContext& context, const StackTrace& stack_trace);
    
    // Copy and move constructors
    AttentionException(const AttentionException& other) = default;
    AttentionException(AttentionException&& other) noexcept = default;
    AttentionException& operator=(const AttentionException& other) = default;
    AttentionException& operator=(AttentionException&& other) noexcept = default;
    
    // Destructor
    virtual ~AttentionException() noexcept = default;
    
    // Exception information
    const char* what() const noexcept override;
    virtual const char* get_exception_type() const noexcept;
    
    // Error details
    ErrorCode get_error_code() const noexcept;
    ErrorSeverity get_severity() const noexcept;
    const std::string& get_message() const noexcept;
    const ErrorContext& get_context() const noexcept;
    const StackTrace& get_stack_trace() const noexcept;
    
    // Exception chaining
    void set_inner_exception(std::shared_ptr<AttentionException> inner);
    std::shared_ptr<AttentionException> get_inner_exception() const;
    bool has_inner_exception() const;
    
    // Detailed information
    std::string get_detailed_message() const;
    std::string get_full_error_info() const;
    std::vector<std::string> get_error_chain() const;
    
    // Context manipulation
    void add_context(const std::string& key, const std::string& value);
    template<typename T>
    void add_context(const std::string& key, const T& value) {
        context_.add_context(key, value);
        update_what_message();
    }
    
    // Static factory methods
    static std::shared_ptr<AttentionException> create(ErrorCode error_code, const std::string& message);
    static std::shared_ptr<AttentionException> wrap(const std::exception& e, ErrorCode error_code = ErrorCode::UNKNOWN_ERROR);

protected:
    virtual ErrorSeverity determine_severity() const;
    void update_what_message();

private:
    ErrorCode error_code_;
    std::string message_;
    ErrorContext context_;
    StackTrace stack_trace_;
    std::shared_ptr<AttentionException> inner_exception_;
    mutable std::string what_message_;
};

// Specific exception types

// Parameter and input validation exceptions
class InvalidParameterException : public AttentionException {
public:
    explicit InvalidParameterException(const std::string& parameter_name, const std::string& message = "");
    InvalidParameterException(const std::string& parameter_name, const std::string& message, const ErrorContext& context);
    
    const char* get_exception_type() const noexcept override;
    const std::string& get_parameter_name() const noexcept;

private:
    std::string parameter_name_;
};

class InvalidTensorShapeException : public AttentionException {
public:
    InvalidTensorShapeException(const std::vector<size_t>& expected_shape, 
                               const std::vector<size_t>& actual_shape,
                               const std::string& message = "");
    
    const char* get_exception_type() const noexcept override;
    const std::vector<size_t>& get_expected_shape() const noexcept;
    const std::vector<size_t>& get_actual_shape() const noexcept;

private:
    std::vector<size_t> expected_shape_;
    std::vector<size_t> actual_shape_;
};

class InvalidDataTypeException : public AttentionException {
public:
    InvalidDataTypeException(const std::string& expected_type, const std::string& actual_type,
                           const std::string& message = "");
    
    const char* get_exception_type() const noexcept override;
    const std::string& get_expected_type() const noexcept;
    const std::string& get_actual_type() const noexcept;

private:
    std::string expected_type_;
    std::string actual_type_;
};

// Memory-related exceptions
class MemoryException : public AttentionException {
public:
    explicit MemoryException(ErrorCode error_code, const std::string& message = "");
    MemoryException(ErrorCode error_code, size_t requested_size, const std::string& message = "");
    
    const char* get_exception_type() const noexcept override;
    size_t get_requested_size() const noexcept;

protected:
    ErrorSeverity determine_severity() const override;

private:
    size_t requested_size_;
};

class OutOfMemoryException : public MemoryException {
public:
    explicit OutOfMemoryException(size_t requested_size, size_t available_size = 0);
    OutOfMemoryException(size_t requested_size, size_t available_size, const std::string& message);
    
    const char* get_exception_type() const noexcept override;
    size_t get_available_size() const noexcept;

private:
    size_t available_size_;
};

// Backend-specific exceptions
class BackendException : public AttentionException {
public:
    BackendException(const std::string& backend_name, ErrorCode error_code, const std::string& message = "");
    
    const char* get_exception_type() const noexcept override;
    const std::string& get_backend_name() const noexcept;

private:
    std::string backend_name_;
};

class CudaException : public BackendException {
public:
    CudaException(int cuda_error_code, const std::string& message = "");
    CudaException(int cuda_error_code, const std::string& cuda_function, const std::string& message = "");
    
    const char* get_exception_type() const noexcept override;
    int get_cuda_error_code() const noexcept;
    const std::string& get_cuda_function() const noexcept;

private:
    int cuda_error_code_;
    std::string cuda_function_;
};

class OpenCLException : public BackendException {
public:
    OpenCLException(int opencl_error_code, const std::string& message = "");
    OpenCLException(int opencl_error_code, const std::string& opencl_function, const std::string& message = "");
    
    const char* get_exception_type() const noexcept override;
    int get_opencl_error_code() const noexcept;
    const std::string& get_opencl_function() const noexcept;

private:
    int opencl_error_code_;
    std::string opencl_function_;
};

class VulkanException : public BackendException {
public:
    VulkanException(int vulkan_result, const std::string& message = "");
    VulkanException(int vulkan_result, const std::string& vulkan_function, const std::string& message = "");
    
    const char* get_exception_type() const noexcept override;
    int get_vulkan_result() const noexcept;
    const std::string& get_vulkan_function() const noexcept;

private:
    int vulkan_result_;
    std::string vulkan_function_;
};

// Computation-related exceptions
class ComputationException : public AttentionException {
public:
    explicit ComputationException(ErrorCode error_code, const std::string& message = "");
    ComputationException(ErrorCode error_code, const std::string& operation_name, const std::string& message = "");
    
    const char* get_exception_type() const noexcept override;
    const std::string& get_operation_name() const noexcept;

private:
    std::string operation_name_;
};

class AttentionComputationException : public ComputationException {
public:
    explicit AttentionComputationException(const std::string& message = "");
    AttentionComputationException(const std::string& attention_type, const std::string& message);
    
    const char* get_exception_type() const noexcept override;
    const std::string& get_attention_type() const noexcept;

private:
    std::string attention_type_;
};

// Configuration and state exceptions
class ConfigurationException : public AttentionException {
public:
    explicit ConfigurationException(const std::string& message = "");
    ConfigurationException(const std::string& config_key, const std::string& message);
    
    const char* get_exception_type() const noexcept override;
    const std::string& get_config_key() const noexcept;

private:
    std::string config_key_;
};

class InvalidStateException : public AttentionException {
public:
    explicit InvalidStateException(const std::string& message = "");
    InvalidStateException(const std::string& expected_state, const std::string& actual_state, const std::string& message = "");
    
    const char* get_exception_type() const noexcept override;
    const std::string& get_expected_state() const noexcept;
    const std::string& get_actual_state() const noexcept;

private:
    std::string expected_state_;
    std::string actual_state_;
};

// Utility functions and macros
namespace error_handling_utils {
    // Error code utilities
    std::string error_code_to_string(ErrorCode code);
    ErrorSeverity get_error_severity(ErrorCode code);
    bool is_recoverable_error(ErrorCode code);
    
    // Exception creation helpers
    template<typename ExceptionType, typename... Args>
    std::shared_ptr<ExceptionType> make_exception(Args&&... args) {
        return std::make_shared<ExceptionType>(std::forward<Args>(args)...);
    }
    
    // Exception handling utilities
    void log_exception(const AttentionException& e, const std::string& logger_name = "");
    std::string format_exception_chain(const AttentionException& e);
    void print_exception_info(const AttentionException& e, std::ostream& os = std::cerr);
    
    // Context builders
    ErrorContext build_tensor_context(const std::vector<size_t>& shape, const std::string& data_type);
    ErrorContext build_computation_context(const std::string& operation, size_t batch_size, 
                                         size_t seq_len, size_t num_heads, size_t head_dim);
    ErrorContext build_backend_context(const std::string& backend_name, const std::string& device_name);
    
    // Stack trace utilities
    void print_stack_trace(const StackTrace& trace, std::ostream& os = std::cerr);
    StackTrace capture_stack_trace(size_t skip_frames = 1);
    
    // Exception guards and wrappers
    template<typename Func>
    auto exception_guard(Func&& func, const std::string& operation_name = "") -> decltype(func()) {
        try {
            return func();
        } catch (const AttentionException&) {
            throw; // Re-throw AttentionException as-is
        } catch (const std::exception& e) {
            ErrorContext context;
            if (!operation_name.empty()) {
                context.add_context("operation", operation_name);
            }
            context.add_context("original_exception", e.what());
            throw AttentionException(ErrorCode::UNKNOWN_ERROR, 
                                   "Wrapped standard exception: " + std::string(e.what()), context);
        } catch (...) {
            ErrorContext context;
            if (!operation_name.empty()) {
                context.add_context("operation", operation_name);
            }
            throw AttentionException(ErrorCode::UNKNOWN_ERROR, "Wrapped unknown exception", context);
        }
    }
}

// Convenience macros for exception handling
#define ATTENTION_THROW(error_code, message) \
    do { \
        attention_hpc::ErrorContext __ctx; \
        __ctx.add_context(attention_hpc::ErrorContext::FUNCTION_NAME, __FUNCTION__); \
        __ctx.add_context(attention_hpc::ErrorContext::FILE_NAME, __FILE__); \
        __ctx.add_context(attention_hpc::ErrorContext::LINE_NUMBER, __LINE__); \
        __ctx.add_context(attention_hpc::ErrorContext::THREAD_ID, std::this_thread::get_id()); \
        throw attention_hpc::AttentionException(error_code, message, __ctx, attention_hpc::StackTrace::current()); \
    } while(0)

#define ATTENTION_THROW_IF(condition, error_code, message) \
    do { \
        if (condition) { \
            ATTENTION_THROW(error_code, message); \
        } \
    } while(0)

#define ATTENTION_CHECK_PARAMETER(param, condition, message) \
    do { \
        if (!(condition)) { \
            attention_hpc::ErrorContext __ctx; \
            __ctx.add_context(attention_hpc::ErrorContext::FUNCTION_NAME, __FUNCTION__); \
            __ctx.add_context(attention_hpc::ErrorContext::FILE_NAME, __FILE__); \
            __ctx.add_context(attention_hpc::ErrorContext::LINE_NUMBER, __LINE__); \
            __ctx.add_context("parameter_name", #param); \
            __ctx.add_context("parameter_value", param); \
            throw attention_hpc::InvalidParameterException(#param, message, __ctx); \
        } \
    } while(0)

#define ATTENTION_CHECK_TENSOR_SHAPE(tensor_shape, expected_dims, operation) \
    do { \
        if (tensor_shape.size() != expected_dims) { \
            attention_hpc::ErrorContext __ctx; \
            __ctx.add_context(attention_hpc::ErrorContext::FUNCTION_NAME, __FUNCTION__); \
            __ctx.add_context(attention_hpc::ErrorContext::FILE_NAME, __FILE__); \
            __ctx.add_context(attention_hpc::ErrorContext::LINE_NUMBER, __LINE__); \
            __ctx.add_context("operation", operation); \
            std::vector<size_t> expected_shape(expected_dims, 0); \
            throw attention_hpc::InvalidTensorShapeException(expected_shape, tensor_shape, \
                "Invalid tensor shape for " + std::string(operation)); \
        } \
    } while(0)

#define ATTENTION_WRAP_BACKEND_ERROR(backend_name, error_code, native_error, message) \
    do { \
        attention_hpc::ErrorContext __ctx; \
        __ctx.add_context(attention_hpc::ErrorContext::FUNCTION_NAME, __FUNCTION__); \
        __ctx.add_context(attention_hpc::ErrorContext::FILE_NAME, __FILE__); \
        __ctx.add_context(attention_hpc::ErrorContext::LINE_NUMBER, __LINE__); \
        __ctx.add_context(attention_hpc::ErrorContext::BACKEND_TYPE, backend_name); \
        __ctx.add_context("native_error_code", native_error); \
        throw attention_hpc::BackendException(backend_name, error_code, message, __ctx); \
    } while(0)

#define ATTENTION_TRY_CATCH_RETHROW(operation_name) \
    catch (const attention_hpc::AttentionException& e) { \
        auto new_exception = std::make_shared<attention_hpc::AttentionException>(e); \
        new_exception->add_context("rethrow_location", __FUNCTION__); \
        new_exception->add_context("rethrow_operation", operation_name); \
        throw *new_exception; \
    } catch (const std::exception& e) { \
        attention_hpc::ErrorContext __ctx; \
        __ctx.add_context(attention_hpc::ErrorContext::FUNCTION_NAME, __FUNCTION__); \
        __ctx.add_context("rethrow_operation", operation_name); \
        __ctx.add_context("original_exception", e.what()); \
        throw attention_hpc::AttentionException(attention_hpc::ErrorCode::UNKNOWN_ERROR, \
                                               "Wrapped exception in " + std::string(operation_name), __ctx); \
    }

} // namespace attention_hpc

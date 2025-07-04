#include "opencl_flash_attention.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>

namespace attention_hpc {

// OpenCL error string conversion
const char* get_opencl_error_string(cl_int error) {
    switch (error) {
        case CL_SUCCESS: return "Success";
        case CL_DEVICE_NOT_FOUND: return "Device not found";
        case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE: return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES: return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP: return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH: return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE: return "Build program failure";
        case CL_MAP_FAILURE: return "Map failure";
        case CL_INVALID_VALUE: return "Invalid value";
        case CL_INVALID_DEVICE_TYPE: return "Invalid device type";
        case CL_INVALID_PLATFORM: return "Invalid platform";
        case CL_INVALID_DEVICE: return "Invalid device";
        case CL_INVALID_CONTEXT: return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES: return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE: return "Invalid command queue";
        case CL_INVALID_HOST_PTR: return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT: return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE: return "Invalid image size";
        case CL_INVALID_SAMPLER: return "Invalid sampler";
        case CL_INVALID_BINARY: return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS: return "Invalid build options";
        case CL_INVALID_PROGRAM: return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME: return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION: return "Invalid kernel definition";
        case CL_INVALID_KERNEL: return "Invalid kernel";
        case CL_INVALID_ARG_INDEX: return "Invalid argument index";
        case CL_INVALID_ARG_VALUE: return "Invalid argument value";
        case CL_INVALID_ARG_SIZE: return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS: return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION: return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE: return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE: return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET: return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST: return "Invalid event wait list";
        case CL_INVALID_EVENT: return "Invalid event";
        case CL_INVALID_OPERATION: return "Invalid operation";
        case CL_INVALID_GL_OBJECT: return "Invalid GL object";
        case CL_INVALID_BUFFER_SIZE: return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL: return "Invalid mip level";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "Invalid global work size";
        default: return "Unknown error";
    }
}

// OpenCLContext implementation
OpenCLContext::OpenCLContext() 
    : platform_(nullptr), device_(nullptr), context_(nullptr), initialized_(false) {
}

OpenCLContext::~OpenCLContext() {
    release_context();
}

std::vector<OpenCLPlatformInfo> OpenCLContext::get_available_platforms() {
    std::vector<OpenCLPlatformInfo> platforms;
    
    cl_uint num_platforms = 0;
    cl_int error = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (error != CL_SUCCESS || num_platforms == 0) {
        return platforms;
    }
    
    std::vector<cl_platform_id> platform_ids(num_platforms);
    error = clGetPlatformIDs(num_platforms, platform_ids.data(), nullptr);
    if (error != CL_SUCCESS) {
        return platforms;
    }
    
    for (cl_platform_id platform_id : platform_ids) {
        OpenCLPlatformInfo platform_info;
        platform_info.platform_id = platform_id;
        
        // Get platform information
        char info_buffer[1024];
        size_t info_size;
        
        if (clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
            platform_info.platform_name = std::string(info_buffer, info_size - 1);
        }
        
        if (clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
            platform_info.vendor = std::string(info_buffer, info_size - 1);
        }
        
        if (clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
            platform_info.version = std::string(info_buffer, info_size - 1);
        }
        
        if (clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
            platform_info.profile = std::string(info_buffer, info_size - 1);
        }
        
        if (clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
            std::string extensions_str(info_buffer, info_size - 1);
            std::istringstream iss(extensions_str);
            std::string extension;
            while (iss >> extension) {
                platform_info.extensions.push_back(extension);
            }
        }
        
        // Get devices for this platform
        cl_uint num_devices = 0;
        if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices) == CL_SUCCESS && num_devices > 0) {
            std::vector<cl_device_id> device_ids(num_devices);
            if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_devices, device_ids.data(), nullptr) == CL_SUCCESS) {
                for (cl_device_id device_id : device_ids) {
                    OpenCLDeviceInfo device_info;
                    device_info.device_id = device_id;
                    
                    // Get device information
                    clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_info.device_type), &device_info.device_type, nullptr);
                    
                    if (clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
                        device_info.device_name = std::string(info_buffer, info_size - 1);
                    }
                    
                    if (clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
                        device_info.vendor = std::string(info_buffer, info_size - 1);
                    }
                    
                    if (clGetDeviceInfo(device_id, CL_DEVICE_VERSION, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
                        device_info.version = std::string(info_buffer, info_size - 1);
                    }
                    
                    if (clGetDeviceInfo(device_id, CL_DRIVER_VERSION, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
                        device_info.driver_version = std::string(info_buffer, info_size - 1);
                    }
                    
                    clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_info.global_memory_size), &device_info.global_memory_size, nullptr);
                    clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(device_info.local_memory_size), &device_info.local_memory_size, nullptr);
                    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(device_info.max_work_group_size), &device_info.max_work_group_size, nullptr);
                    clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(device_info.max_compute_units), &device_info.max_compute_units, nullptr);
                    clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(device_info.max_clock_frequency), &device_info.max_clock_frequency, nullptr);
                    
                    // Get work item sizes
                    cl_uint max_work_item_dimensions;
                    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dimensions), &max_work_item_dimensions, nullptr);
                    device_info.max_work_item_sizes.resize(max_work_item_dimensions);
                    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * max_work_item_dimensions, device_info.max_work_item_sizes.data(), nullptr);
                    
                    // Check for precision support
                    cl_device_fp_config fp_config;
                    if (clGetDeviceInfo(device_id, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp_config), &fp_config, nullptr) == CL_SUCCESS) {
                        device_info.supports_double_precision = (fp_config != 0);
                    }
                    
                    if (clGetDeviceInfo(device_id, CL_DEVICE_HALF_FP_CONFIG, sizeof(fp_config), &fp_config, nullptr) == CL_SUCCESS) {
                        device_info.supports_half_precision = (fp_config != 0);
                    }
                    
                    // Get extensions
                    if (clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
                        std::string extensions_str(info_buffer, info_size - 1);
                        std::istringstream iss(extensions_str);
                        std::string extension;
                        while (iss >> extension) {
                            device_info.extensions.push_back(extension);
                        }
                    }
                    
                    platform_info.devices.push_back(device_info);
                }
            }
        }
        
        platforms.push_back(platform_info);
    }
    
    return platforms;
}

bool OpenCLContext::initialize_context(cl_platform_id platform_id, cl_device_id device_id) {
    std::lock_guard<std::mutex> lock(context_mutex_);
    
    if (initialized_) {
        release_context();
    }
    
    platform_ = platform_id;
    device_ = device_id;
    
    cl_int error;
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &error);
    if (error != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context: " << get_opencl_error_string(error) << std::endl;
        return false;
    }
    
    initialized_ = true;
    return true;
}

bool OpenCLContext::initialize_context_from_type(cl_device_type device_type) {
    auto platforms = get_available_platforms();
    
    for (const auto& platform : platforms) {
        for (const auto& device : platform.devices) {
            if (device.device_type & device_type) {
                return initialize_context(platform.platform_id, device.device_id);
            }
        }
    }
    
    std::cerr << "No suitable device found for device type: " << device_type << std::endl;
    return false;
}

void OpenCLContext::release_context() {
    std::lock_guard<std::mutex> lock(context_mutex_);
    
    if (context_) {
        clReleaseContext(context_);
        context_ = nullptr;
    }
    
    platform_ = nullptr;
    device_ = nullptr;
    initialized_ = false;
}

OpenCLDeviceInfo OpenCLContext::get_device_info() const {
    OpenCLDeviceInfo device_info;
    
    if (!initialized_ || !device_) {
        return device_info;
    }
    
    device_info.device_id = device_;
    
    char info_buffer[1024];
    size_t info_size;
    
    clGetDeviceInfo(device_, CL_DEVICE_TYPE, sizeof(device_info.device_type), &device_info.device_type, nullptr);
    
    if (clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
        device_info.device_name = std::string(info_buffer, info_size - 1);
    }
    
    if (clGetDeviceInfo(device_, CL_DEVICE_VENDOR, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
        device_info.vendor = std::string(info_buffer, info_size - 1);
    }
    
    if (clGetDeviceInfo(device_, CL_DEVICE_VERSION, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
        device_info.version = std::string(info_buffer, info_size - 1);
    }
    
    if (clGetDeviceInfo(device_, CL_DRIVER_VERSION, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
        device_info.driver_version = std::string(info_buffer, info_size - 1);
    }
    
    clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_info.global_memory_size), &device_info.global_memory_size, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(device_info.local_memory_size), &device_info.local_memory_size, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(device_info.max_work_group_size), &device_info.max_work_group_size, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(device_info.max_compute_units), &device_info.max_compute_units, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(device_info.max_clock_frequency), &device_info.max_clock_frequency, nullptr);
    
    cl_uint max_work_item_dimensions;
    clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dimensions), &max_work_item_dimensions, nullptr);
    device_info.max_work_item_sizes.resize(max_work_item_dimensions);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * max_work_item_dimensions, device_info.max_work_item_sizes.data(), nullptr);
    
    cl_device_fp_config fp_config;
    if (clGetDeviceInfo(device_, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp_config), &fp_config, nullptr) == CL_SUCCESS) {
        device_info.supports_double_precision = (fp_config != 0);
    }
    
    if (clGetDeviceInfo(device_, CL_DEVICE_HALF_FP_CONFIG, sizeof(fp_config), &fp_config, nullptr) == CL_SUCCESS) {
        device_info.supports_half_precision = (fp_config != 0);
    }
    
    if (clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
        std::string extensions_str(info_buffer, info_size - 1);
        std::istringstream iss(extensions_str);
        std::string extension;
        while (iss >> extension) {
            device_info.extensions.push_back(extension);
        }
    }
    
    return device_info;
}

OpenCLPlatformInfo OpenCLContext::get_platform_info() const {
    OpenCLPlatformInfo platform_info;
    
    if (!initialized_ || !platform_) {
        return platform_info;
    }
    
    platform_info.platform_id = platform_;
    
    char info_buffer[1024];
    size_t info_size;
    
    if (clGetPlatformInfo(platform_, CL_PLATFORM_NAME, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
        platform_info.platform_name = std::string(info_buffer, info_size - 1);
    }
    
    if (clGetPlatformInfo(platform_, CL_PLATFORM_VENDOR, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
        platform_info.vendor = std::string(info_buffer, info_size - 1);
    }
    
    if (clGetPlatformInfo(platform_, CL_PLATFORM_VERSION, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
        platform_info.version = std::string(info_buffer, info_size - 1);
    }
    
    if (clGetPlatformInfo(platform_, CL_PLATFORM_PROFILE, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
        platform_info.profile = std::string(info_buffer, info_size - 1);
    }
    
    if (clGetPlatformInfo(platform_, CL_PLATFORM_EXTENSIONS, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
        std::string extensions_str(info_buffer, info_size - 1);
        std::istringstream iss(extensions_str);
        std::string extension;
        while (iss >> extension) {
            platform_info.extensions.push_back(extension);
        }
    }
    
    return platform_info;
}

cl_mem OpenCLContext::create_buffer(size_t size, cl_mem_flags flags) {
    if (!initialized_ || !context_) {
        return nullptr;
    }
    
    cl_int error;
    cl_mem buffer = clCreateBuffer(context_, flags, size, nullptr, &error);
    if (error != CL_SUCCESS) {
        std::cerr << "Failed to create buffer: " << get_opencl_error_string(error) << std::endl;
        return nullptr;
    }
    
    return buffer;
}

void OpenCLContext::release_buffer(cl_mem buffer) {
    if (buffer) {
        clReleaseMemObject(buffer);
    }
}

void* OpenCLContext::map_buffer(cl_mem buffer, size_t size, cl_map_flags flags) {
    if (!initialized_ || !buffer) {
        return nullptr;
    }
    
    cl_int error;
    void* mapped_ptr = clEnqueueMapBuffer(nullptr, buffer, CL_TRUE, flags, 0, size, 0, nullptr, nullptr, &error);
    if (error != CL_SUCCESS) {
        std::cerr << "Failed to map buffer: " << get_opencl_error_string(error) << std::endl;
        return nullptr;
    }
    
    return mapped_ptr;
}

void OpenCLContext::unmap_buffer(cl_mem buffer, void* mapped_ptr) {
    if (buffer && mapped_ptr) {
        clEnqueueUnmapMemObject(nullptr, buffer, mapped_ptr, 0, nullptr, nullptr);
    }
}

std::string OpenCLContext::get_device_extensions() const {
    if (!initialized_ || !device_) {
        return "";
    }
    
    char info_buffer[4096];
    size_t info_size;
    
    if (clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, sizeof(info_buffer), info_buffer, &info_size) == CL_SUCCESS) {
        return std::string(info_buffer, info_size - 1);
    }
    
    return "";
}

bool OpenCLContext::supports_extension(const std::string& extension) const {
    std::string extensions = get_device_extensions();
    return extensions.find(extension) != std::string::npos;
}

// OpenCLCommandQueue implementation
OpenCLCommandQueue::OpenCLCommandQueue(cl_context context, cl_device_id device)
    : context_(context), device_(device), queue_(nullptr), profiling_enabled_(false), initialized_(false) {
}

OpenCLCommandQueue::~OpenCLCommandQueue() {
    release_queue();
}

bool OpenCLCommandQueue::create_queue(cl_command_queue_properties properties) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    if (initialized_) {
        release_queue();
    }
    
    if (profiling_enabled_) {
        properties |= CL_QUEUE_PROFILING_ENABLE;
    }
    
    cl_int error;
    queue_ = clCreateCommandQueue(context_, device_, properties, &error);
    if (error != CL_SUCCESS) {
        std::cerr << "Failed to create command queue: " << get_opencl_error_string(error) << std::endl;
        return false;
    }
    
    initialized_ = true;
    return true;
}

void OpenCLCommandQueue::release_queue() {
    if (queue_) {
        clReleaseCommandQueue(queue_);
        queue_ = nullptr;
    }
    initialized_ = false;
}

void OpenCLCommandQueue::enqueue_kernel(cl_kernel kernel, size_t global_work_size, 
                                       size_t local_work_size, cl_event* event) {
    if (!initialized_ || !queue_ || !kernel) {
        return;
    }
    
    const size_t* local_size_ptr = (local_work_size > 0) ? &local_work_size : nullptr;
    
    cl_int error = clEnqueueNDRangeKernel(queue_, kernel, 1, nullptr, &global_work_size, 
                                         local_size_ptr, 0, nullptr, event);
    if (error != CL_SUCCESS) {
        std::cerr << "Failed to enqueue kernel: " << get_opencl_error_string(error) << std::endl;
    }
}

void OpenCLCommandQueue::enqueue_kernel_nd(cl_kernel kernel, cl_uint work_dim,
                                          const size_t* global_work_size,
                                          const size_t* local_work_size,
                                          cl_event* event) {
    if (!initialized_ || !queue_ || !kernel) {
        return;
    }
    
    cl_int error = clEnqueueNDRangeKernel(queue_, kernel, work_dim, nullptr, global_work_size, 
                                         local_work_size, 0, nullptr, event);
    if (error != CL_SUCCESS) {
        std::cerr << "Failed to enqueue ND kernel: " << get_opencl_error_string(error) << std::endl;
    }
}

void OpenCLCommandQueue::enqueue_read_buffer(cl_mem buffer, void* host_ptr, size_t size,
                                            size_t offset, cl_event* event) {
    if (!initialized_ || !queue_ || !buffer || !host_ptr) {
        return;
    }
    
    cl_int error = clEnqueueReadBuffer(queue_, buffer, CL_TRUE, offset, size, host_ptr, 
                                      0, nullptr, event);
    if (error != CL_SUCCESS) {
        std::cerr << "Failed to enqueue read buffer: " << get_opencl_error_string(error) << std::endl;
    }
}

void OpenCLCommandQueue::enqueue_write_buffer(cl_mem buffer, const void* host_ptr, size_t size,
                                             size_t offset, cl_event* event) {
    if (!initialized_ || !queue_ || !buffer || !host_ptr) {
        return;
    }
    
    cl_int error = clEnqueueWriteBuffer(queue_, buffer, CL_TRUE, offset, size, host_ptr, 
                                       0, nullptr, event);
    if (error != CL_SUCCESS) {
        std::cerr << "Failed to enqueue write buffer: " << get_opencl_error_string(error) << std::endl;
    }
}

void OpenCLCommandQueue::enqueue_copy_buffer(cl_mem src_buffer, cl_mem dst_buffer, size_t size,
                                            size_t src_offset, size_t dst_offset,
                                            cl_event* event) {
    if (!initialized_ || !queue_ || !src_buffer || !dst_buffer) {
        return;
    }
    
    cl_int error = clEnqueueCopyBuffer(queue_, src_buffer, dst_buffer, src_offset, dst_offset, 
                                      size, 0, nullptr, event);
    if (error != CL_SUCCESS) {
        std::cerr << "Failed to enqueue copy buffer: " << get_opencl_error_string(error) << std::endl;
    }
}

void OpenCLCommandQueue::enqueue_fill_buffer(cl_mem buffer, const void* pattern, size_t pattern_size,
                                            size_t offset, size_t size, cl_event* event) {
    if (!initialized_ || !queue_ || !buffer || !pattern) {
        return;
    }
    
    cl_int error = clEnqueueFillBuffer(queue_, buffer, pattern, pattern_size, offset, size, 
                                      0, nullptr, event);
    if (error != CL_SUCCESS) {
        std::cerr << "Failed to enqueue fill buffer: " << get_opencl_error_string(error) << std::endl;
    }
}

void OpenCLCommandQueue::finish() {
    if (initialized_ && queue_) {
        clFinish(queue_);
    }
}

void OpenCLCommandQueue::flush() {
    if (initialized_ && queue_) {
        clFlush(queue_);
    }
}

void OpenCLCommandQueue::wait_for_events(const std::vector<cl_event>& events) {
    if (!events.empty()) {
        clWaitForEvents(static_cast<cl_uint>(events.size()), events.data());
    }
}

void OpenCLCommandQueue::enable_profiling(bool enable) {
    profiling_enabled_ = enable;
    if (initialized_) {
        // Recreate queue with/without profiling
        cl_command_queue_properties properties = 0;
        if (profiling_enabled_) {
            properties |= CL_QUEUE_PROFILING_ENABLE;
        }
        create_queue(properties);
    }
}

// OpenCLKernelManager implementation
OpenCLKernelManager::OpenCLKernelManager(cl_context context, cl_device_id device)
    : context_(context), device_(device) {
}

OpenCLKernelManager::~OpenCLKernelManager() {
    clear_all_programs();
}

bool OpenCLKernelManager::compile_program_from_source(const std::string& source, 
                                                     const std::string& program_name,
                                                     const std::string& build_options) {
    std::lock_guard<std::mutex> lock(programs_mutex_);
    
    const char* source_ptr = source.c_str();
    size_t source_length = source.length();
    
    cl_int error;
    cl_program program = clCreateProgramWithSource(context_, 1, &source_ptr, &source_length, &error);
    if (error != CL_SUCCESS) {
        std::cerr << "Failed to create program from source: " << get_opencl_error_string(error) << std::endl;
        return false;
    }
    
    if (!build_program(program, build_options)) {
        clReleaseProgram(program);
        return false;
    }
    
    // Release existing program if it exists
    auto it = programs_.find(program_name);
    if (it != programs_.end()) {
        clReleaseProgram(it->second);
    }
    
    programs_[program_name] = program;
    return true;
}

bool OpenCLKernelManager::compile_program_from_file(const std::string& filename,
                                                   const std::string& program_name,
                                                   const std::string& build_options) {
    std::string source = load_source_from_file(filename);
    if (source.empty()) {
        std::cerr << "Failed to load source from file: " << filename << std::endl;
        return false;
    }
    
    return compile_program_from_source(source, program_name, build_options);
}

bool OpenCLKernelManager::load_program_from_binary(const std::vector<unsigned char>& binary,
                                                  const std::string& program_name) {
    std::lock_guard<std::mutex> lock(programs_mutex_);
    
    const unsigned char* binary_ptr = binary.data();
    size_t binary_length = binary.size();
    
    cl_int error;
    cl_program program = clCreateProgramWithBinary(context_, 1, &device_, &binary_length, 
                                                  &binary_ptr, nullptr, &error);
    if (error != CL_SUCCESS) {
        std::cerr << "Failed to create program from binary: " << get_opencl_error_string(error) << std::endl;
        return false;
    }
    
    if (!build_program(program, "")) {
        clReleaseProgram(program);
        return false;
    }
    
    // Release existing program if it exists
    auto it = programs_.find(program_name);
    if (it != programs_.end()) {
        clR

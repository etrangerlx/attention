# Backends module CMakeLists.txt
# This module contains the HPC backend implementations for CUDA, OpenCL, Vulkan, and OpenGL

# Initialize backend sources and headers
set(BACKEND_SOURCES "")
set(BACKEND_HEADERS "")

# CUDA Backend
if(CUDA_FOUND)
    list(APPEND BACKEND_SOURCES
        cuda/cuda_kernels.cu
    )
    list(APPEND BACKEND_HEADERS
        cuda/cuda_flash_attention.hpp
    )
    
    # Create CUDA backend library
    add_library(attention_cuda ${BACKEND_SOURCES})
    
    # Set CUDA-specific properties
    set_target_properties(attention_cuda PROPERTIES
        CUDA_ARCHITECTURES "52;60;61;70;75;80;86"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    
    # CUDA compiler flags
    target_compile_options(attention_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
    
    # Include directories
    target_include_directories(attention_cuda
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include/attention_hpc/backends>
            ${CUDA_INCLUDE_DIRS}
    )
    
    # Link CUDA libraries
    target_link_libraries(attention_cuda
        PUBLIC
            ${CUDA_LIBRARIES}
            ${CUDA_CUBLAS_LIBRARIES}
    )
    
    # Add compile definitions
    target_compile_definitions(attention_cuda
        PUBLIC
            CUDA_ENABLED
        PRIVATE
            $<$<CONFIG:Debug>:DEBUG>
            $<$<CONFIG:Release>:NDEBUG>
    )
endif()

# OpenCL Backend
if(OpenCL_FOUND)
    list(APPEND BACKEND_SOURCES
        opencl/opencl_backend.cpp
    )
    list(APPEND BACKEND_HEADERS
        opencl/opencl_flash_attention.hpp
    )
    
    # Create OpenCL backend library
    add_library(attention_opencl opencl/opencl_backend.cpp)
    
    # Include directories
    target_include_directories(attention_opencl
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include/attention_hpc/backends>
    )
    
    # Link OpenCL libraries
    target_link_libraries(attention_opencl
        PUBLIC
            OpenCL::OpenCL
    )
    
    # Add compile definitions
    target_compile_definitions(attention_opencl
        PUBLIC
            OPENCL_ENABLED
        PRIVATE
            $<$<CONFIG:Debug>:DEBUG>
            $<$<CONFIG:Release>:NDEBUG>
    )
endif()

# Vulkan Backend
if(Vulkan_FOUND)
    list(APPEND BACKEND_SOURCES
        vulkan/vulkan_backend.cpp
    )
    list(APPEND BACKEND_HEADERS
        vulkan/vulkan_flash_attention.hpp
    )
    
    # Create Vulkan backend library
    add_library(attention_vulkan vulkan/vulkan_backend.cpp)
    
    # Include directories
    target_include_directories(attention_vulkan
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include/attention_hpc/backends>
    )
    
    # Link Vulkan libraries
    target_link_libraries(attention_vulkan
        PUBLIC
            Vulkan::Vulkan
    )
    
    # Add compile definitions
    target_compile_definitions(attention_vulkan
        PUBLIC
            VULKAN_ENABLED
        PRIVATE
            $<$<CONFIG:Debug>:DEBUG>
            $<$<CONFIG:Release>:NDEBUG>
    )
endif()

# OpenGL Backend
if(OpenGL_FOUND AND GLEW_FOUND)
    list(APPEND BACKEND_HEADERS
        opengl/opengl_flash_attention.hpp
    )
    
    # Create OpenGL backend library (header-only)
    add_library(attention_opengl INTERFACE)
    
    # Include directories
    target_include_directories(attention_opengl
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include/attention_hpc/backends>
    )
    
    # Link OpenGL libraries
    target_link_libraries(attention_opengl
        INTERFACE
            OpenGL::GL
            OpenGL::GLU
            GLEW::GLEW
    )
    
    # Add compile definitions
    target_compile_definitions(attention_opengl
        INTERFACE
            OPENGL_ENABLED
    )
endif()

# Create combined backends library if any backend is available
set(BACKEND_TARGETS "")
if(CUDA_FOUND)
    list(APPEND BACKEND_TARGETS attention_cuda)
endif()
if(OpenCL_FOUND)
    list(APPEND BACKEND_TARGETS attention_opencl)
endif()
if(Vulkan_FOUND)
    list(APPEND BACKEND_TARGETS attention_vulkan)
endif()
if(OpenGL_FOUND AND GLEW_FOUND)
    list(APPEND BACKEND_TARGETS attention_opengl)
endif()

if(BACKEND_TARGETS)
    # Create interface library to combine all backends
    add_library(attention_backends INTERFACE)
    
    # Link all available backend libraries
    target_link_libraries(attention_backends
        INTERFACE
            ${BACKEND_TARGETS}
    )
    
    # Include directories
    target_include_directories(attention_backends
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include/attention_hpc/backends>
    )
endif()

# Add to parent scope for main library
set(BACKEND_SOURCES ${BACKEND_SOURCES} PARENT_SCOPE)
set(BACKEND_HEADERS ${BACKEND_HEADERS} PARENT_SCOPE)

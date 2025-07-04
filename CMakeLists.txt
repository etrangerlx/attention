cmake_minimum_required(VERSION 3.18)

project(AttentionHPC 
    VERSION 1.0.0
    DESCRIPTION "High-Performance Computing implementations of FlashAttention and PagedAttention"
    LANGUAGES C CXX CUDA
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(MSVC)
    set(CMAKE_CXX_FLAGS "/W4")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /RTC1")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
endif()

# Add custom cmake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find required packages
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# Optional dependencies
find_package(CUDA QUIET)
find_package(OpenCL QUIET)
find_package(Vulkan QUIET)
find_package(OpenGL QUIET)
find_package(GLEW QUIET)

# Define preprocessor macros based on available backends
if(CUDA_FOUND)
    add_definitions(-DHAVE_CUDA)
    message(STATUS "CUDA support enabled")
endif()

if(OpenCL_FOUND)
    add_definitions(-DHAVE_OPENCL)
    message(STATUS "OpenCL support enabled")
endif()

if(Vulkan_FOUND)
    add_definitions(-DHAVE_VULKAN)
    message(STATUS "Vulkan support enabled")
endif()

if(OpenGL_FOUND AND GLEW_FOUND)
    add_definitions(-DHAVE_OPENGL)
    message(STATUS "OpenGL compute shader support enabled")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add subdirectories
add_subdirectory(src)

# Optional subdirectories (build only if dependencies are met)
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        add_subdirectory(tests)
        message(STATUS "Tests enabled")
    else()
        message(WARNING "Google Test not found, tests disabled")
    endif()
endif()

option(BUILD_BENCHMARKS "Build benchmarks" ON)
if(BUILD_BENCHMARKS)
    find_package(benchmark QUIET)
    if(benchmark_FOUND)
        add_subdirectory(benchmarks)
        message(STATUS "Benchmarks enabled")
    else()
        message(WARNING "Google Benchmark not found, benchmarks disabled")
    endif()
endif()

option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
    message(STATUS "Examples enabled")
endif()

# Installation settings
include(GNUInstallDirs)

# Export configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/AttentionHPCConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/AttentionHPCConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AttentionHPC
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/AttentionHPCConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Print configuration summary
message(STATUS "=== AttentionHPC Configuration Summary ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Available backends:")
if(CUDA_FOUND)
    message(STATUS "  - CUDA: YES")
else()
    message(STATUS "  - CUDA: NO")
endif()
if(OpenCL_FOUND)
    message(STATUS "  - OpenCL: YES")
else()
    message(STATUS "  - OpenCL: NO")
endif()
if(Vulkan_FOUND)
    message(STATUS "  - Vulkan: YES")
else()
    message(STATUS "  - Vulkan: NO")
endif()
if(OpenGL_FOUND AND GLEW_FOUND)
    message(STATUS "  - OpenGL: YES")
else()
    message(STATUS "  - OpenGL: NO")
endif()
message(STATUS "==========================================")
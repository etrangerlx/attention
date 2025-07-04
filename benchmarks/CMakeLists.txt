# CMakeLists.txt for AttentionHPC Benchmarks

cmake_minimum_required(VERSION 3.18)

# Set benchmark project name
project(AttentionHPC_Benchmarks)

# Find Google Benchmark package
find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)

# Set C++ standard to match main project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${benchmark_INCLUDE_DIRS})

# Set benchmark data directory
set(BENCHMARK_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")
add_definitions(-DBENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR}")

# Create benchmark data directory if it doesn't exist
file(MAKE_DIRECTORY ${BENCHMARK_DATA_DIR})

# Common benchmark libraries
set(BENCHMARK_LIBRARIES
    benchmark::benchmark
    benchmark::benchmark_main
    Threads::Threads
)

# Add backend-specific libraries if available
if(TARGET attention_hpc_core)
    list(APPEND BENCHMARK_LIBRARIES attention_hpc_core)
endif()

if(CUDA_FOUND)
    list(APPEND BENCHMARK_LIBRARIES ${CUDA_LIBRARIES})
    add_definitions(-DHAVE_CUDA_BENCHMARKS)
endif()

if(OpenCL_FOUND)
    list(APPEND BENCHMARK_LIBRARIES ${OpenCL_LIBRARIES})
    add_definitions(-DHAVE_OPENCL_BENCHMARKS)
endif()

if(Vulkan_FOUND)
    list(APPEND BENCHMARK_LIBRARIES ${Vulkan_LIBRARIES})
    add_definitions(-DHAVE_VULKAN_BENCHMARKS)
endif()

if(OpenGL_FOUND AND GLEW_FOUND)
    list(APPEND BENCHMARK_LIBRARIES ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})
    add_definitions(-DHAVE_OPENGL_BENCHMARKS)
endif()

# Benchmark source files
set(BENCHMARK_SOURCES
    benchmark_attention.cpp
    benchmark_flash_attention.cpp
    benchmark_paged_attention.cpp
    benchmark_backends.cpp
)

# Create individual benchmark executables
foreach(BENCHMARK_SOURCE ${BENCHMARK_SOURCES})
    get_filename_component(BENCHMARK_NAME ${BENCHMARK_SOURCE} NAME_WE)
    add_executable(${BENCHMARK_NAME} ${BENCHMARK_SOURCE})
    target_link_libraries(${BENCHMARK_NAME} ${BENCHMARK_LIBRARIES})
    
    # Set benchmark properties
    set_target_properties(${BENCHMARK_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
    
    # Add custom target to run benchmarks with output formatting
    add_custom_target(run_${BENCHMARK_NAME}
        COMMAND ${CMAKE_BINARY_DIR}/benchmarks/${BENCHMARK_NAME} 
            --benchmark_format=console 
            --benchmark_out=${CMAKE_BINARY_DIR}/benchmarks/${BENCHMARK_NAME}_results.json 
            --benchmark_out_format=json 
            --benchmark_color=true
        DEPENDS ${BENCHMARK_NAME}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running ${BENCHMARK_NAME} benchmark"
    )
endforeach()

# Add combined target to run all benchmarks
add_custom_target(run_all_benchmarks
    COMMENT "Running all benchmarks"
)

# Add dependency to combined target
foreach(BENCHMARK_SOURCE ${BENCHMARK_SOURCES})
    get_filename_component(BENCHMARK_NAME ${BENCHMARK_SOURCE} NAME_WE)
    add_dependencies(run_all_benchmarks run_${BENCHMARK_NAME})
endforeach()

# Generate benchmark datasets
add_custom_target(generate_benchmark_data
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BENCHMARK_DATA_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
            ${CMAKE_CURRENT_SOURCE_DIR}/data_templates
            ${BENCHMARK_DATA_DIR}
    COMMENT "Generating benchmark datasets"
)

# Add custom target to visualize benchmark results
find_program(PYTHON_EXECUTABLE python3)
if(PYTHON_EXECUTABLE)
    add_custom_target(visualize_benchmarks
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/visualize_results.py 
                --input_dir=${CMAKE_BINARY_DIR}/benchmarks
                --output_dir=${CMAKE_BINARY_DIR}/benchmarks/reports
                --format=html,pdf
        DEPENDS run_all_benchmarks
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating benchmark visualization reports"
    )
    
    # Create reports directory
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks/reports)
else()
    message(STATUS "Python not found, benchmark visualization will not be available")
endif()

# Add helper library for benchmark utilities
add_library(benchmark_helpers STATIC
    helpers/benchmark_utils.cpp
    helpers/data_generators.cpp
)

target_include_directories(benchmark_helpers PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers
)

target_link_libraries(benchmark_helpers ${BENCHMARK_LIBRARIES})

# Link benchmark helpers to all benchmark executables
foreach(BENCHMARK_SOURCE ${BENCHMARK_SOURCES})
    get_filename_component(BENCHMARK_NAME ${BENCHMARK_SOURCE} NAME_WE)
    target_link_libraries(${BENCHMARK_NAME} benchmark_helpers)
endforeach()

# Install benchmark executables (optional)
option(INSTALL_BENCHMARKS "Install benchmark executables" OFF)
if(INSTALL_BENCHMARKS)
    install(TARGETS ${BENCHMARK_NAMES}
        RUNTIME DESTINATION bin/benchmarks
    )
    
    install(DIRECTORY ${BENCHMARK_DATA_DIR}
        DESTINATION share/attention_hpc/benchmarks
    )
endif()

# Benchmark configuration summary
message(STATUS "=== Benchmark Configuration Summary ===")
message(STATUS "Google Benchmark found: ${benchmark_FOUND}")
message(STATUS "Benchmark data directory: ${BENCHMARK_DATA_DIR}")
message(STATUS "Available benchmarks:")
foreach(BENCHMARK_SOURCE ${BENCHMARK_SOURCES})
    get_filename_component(BENCHMARK_NAME ${BENCHMARK_SOURCE} NAME_WE)
    message(STATUS "  - ${BENCHMARK_NAME}")
endforeach()
message(STATUS "Run with: make run_all_benchmarks")
message(STATUS "========================================")

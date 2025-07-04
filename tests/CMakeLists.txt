# CMakeLists.txt for AttentionHPC Tests

cmake_minimum_required(VERSION 3.18)

# Set test project name
project(AttentionHPC_Tests)

# Find required packages for testing
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

# Set C++ standard to match main project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${GTEST_INCLUDE_DIRS})

# Set test data directory
set(TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")
add_definitions(-DTEST_DATA_DIR="${TEST_DATA_DIR}")

# Create test data directory if it doesn't exist
file(MAKE_DIRECTORY ${TEST_DATA_DIR})

# Common test libraries
set(TEST_LIBRARIES
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
    Threads::Threads
)

# Add backend-specific libraries if available
if(TARGET attention_hpc_core)
    list(APPEND TEST_LIBRARIES attention_hpc_core)
endif()

if(CUDA_FOUND)
    list(APPEND TEST_LIBRARIES ${CUDA_LIBRARIES})
    add_definitions(-DHAVE_CUDA_TESTS)
endif()

if(OpenCL_FOUND)
    list(APPEND TEST_LIBRARIES ${OpenCL_LIBRARIES})
    add_definitions(-DHAVE_OPENCL_TESTS)
endif()

if(Vulkan_FOUND)
    list(APPEND TEST_LIBRARIES ${Vulkan_LIBRARIES})
    add_definitions(-DHAVE_VULKAN_TESTS)
endif()

if(OpenGL_FOUND AND GLEW_FOUND)
    list(APPEND TEST_LIBRARIES ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})
    add_definitions(-DHAVE_OPENGL_TESTS)
endif()

# Test source files
set(TEST_SOURCES
    test_flash_attention.cpp
    test_paged_attention.cpp
    test_backends.cpp
    test_memory_pool.cpp
    test_tensor.cpp
    test_utils.cpp
)

# Create individual test executables
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} ${TEST_LIBRARIES})
    
    # Set test properties
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
    
    # Add to test suite
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    
    # Set test environment variables
    set_tests_properties(${TEST_NAME} PROPERTIES
        ENVIRONMENT "TEST_DATA_DIR=${TEST_DATA_DIR}"
        TIMEOUT 300
        LABELS "unit_tests"
    )
endforeach()

# Integration tests
set(INTEGRATION_TEST_SOURCES
    integration/test_end_to_end.cpp
    integration/test_performance.cpp
    integration/test_memory_usage.cpp
)

foreach(INTEGRATION_SOURCE ${INTEGRATION_TEST_SOURCES})
    get_filename_component(INTEGRATION_NAME ${INTEGRATION_SOURCE} NAME_WE)
    add_executable(${INTEGRATION_NAME} ${INTEGRATION_SOURCE})
    target_link_libraries(${INTEGRATION_NAME} ${TEST_LIBRARIES})
    
    set_target_properties(${INTEGRATION_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/integration
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
    
    add_test(NAME ${INTEGRATION_NAME} COMMAND ${INTEGRATION_NAME})
    set_tests_properties(${INTEGRATION_NAME} PROPERTIES
        ENVIRONMENT "TEST_DATA_DIR=${TEST_DATA_DIR}"
        TIMEOUT 600
        LABELS "integration_tests"
    )
endforeach()

# Performance benchmark tests (optional)
option(BUILD_PERFORMANCE_TESTS "Build performance benchmark tests" OFF)
if(BUILD_PERFORMANCE_TESTS)
    find_package(benchmark QUIET)
    if(benchmark_FOUND)
        set(BENCHMARK_SOURCES
            benchmarks/benchmark_flash_attention.cpp
            benchmarks/benchmark_paged_attention.cpp
            benchmarks/benchmark_backends.cpp
        )
        
        foreach(BENCHMARK_SOURCE ${BENCHMARK_SOURCES})
            get_filename_component(BENCHMARK_NAME ${BENCHMARK_SOURCE} NAME_WE)
            add_executable(${BENCHMARK_NAME} ${BENCHMARK_SOURCE})
            target_link_libraries(${BENCHMARK_NAME} ${TEST_LIBRARIES} benchmark::benchmark)
            
            set_target_properties(${BENCHMARK_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/benchmarks
                CXX_STANDARD 17
                CXX_STANDARD_REQUIRED ON
            )
            
            add_test(NAME ${BENCHMARK_NAME} COMMAND ${BENCHMARK_NAME})
            set_tests_properties(${BENCHMARK_NAME} PROPERTIES
                ENVIRONMENT "TEST_DATA_DIR=${TEST_DATA_DIR}"
                TIMEOUT 1200
                LABELS "benchmark_tests"
            )
        endforeach()
    else()
        message(WARNING "Google Benchmark not found, performance tests disabled")
    endif()
endif()

# Test helper library
add_library(test_helpers STATIC
    helpers/test_utils.cpp
    helpers/mock_backends.cpp
    helpers/test_data_generator.cpp
    helpers/assertion_helpers.cpp
)

target_include_directories(test_helpers PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers
)

target_link_libraries(test_helpers ${TEST_LIBRARIES})

# Link test helpers to all test executables
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    target_link_libraries(${TEST_NAME} test_helpers)
endforeach()

foreach(INTEGRATION_SOURCE ${INTEGRATION_TEST_SOURCES})
    get_filename_component(INTEGRATION_NAME ${INTEGRATION_SOURCE} NAME_WE)
    target_link_libraries(${INTEGRATION_NAME} test_helpers)
endforeach()

# Custom test targets
add_custom_target(run_unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L unit_tests --output-on-failure
    DEPENDS ${TEST_SOURCES}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running unit tests"
)

add_custom_target(run_integration_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L integration_tests --output-on-failure
    DEPENDS ${INTEGRATION_TEST_SOURCES}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running integration tests"
)

add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests"
)

# Code coverage (if enabled)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -g -O0")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -g -O0")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
        
        add_custom_target(coverage
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
            COMMAND lcov --capture --directory . --output-file coverage.info
            COMMAND lcov --remove coverage.info '/usr/*' --output-file coverage.info
            COMMAND lcov --list coverage.info
            COMMAND genhtml coverage.info --output-directory coverage_html
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating code coverage report"
        )
    else()
        message(WARNING "Code coverage only supported with GCC or Clang")
    endif()
endif()

# Test configuration file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/test_config.h
    @ONLY
)

# Include the generated config header
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Test data generation
add_custom_target(generate_test_data
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_DATA_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
            ${CMAKE_CURRENT_SOURCE_DIR}/test_data 
            ${TEST_DATA_DIR}
    COMMENT "Copying test data files"
)

# Memory testing with Valgrind (Linux only)
if(UNIX AND NOT APPLE)
    find_program(VALGRIND_EXECUTABLE valgrind)
    if(VALGRIND_EXECUTABLE)
        add_custom_target(test_memory
            COMMAND ${CMAKE_CTEST_COMMAND} -T memcheck --output-on-failure
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running memory tests with Valgrind"
        )
        
        set(MEMORYCHECK_COMMAND_OPTIONS "--leak-check=full --show-leak-kinds=all --track-origins=yes --verbose")
    endif()
endif()

# Install test executables (optional)
option(INSTALL_TESTS "Install test executables" OFF)
if(INSTALL_TESTS)
    install(TARGETS ${TEST_NAMES} ${INTEGRATION_NAMES}
        RUNTIME DESTINATION bin/tests
    )
    
    install(DIRECTORY ${TEST_DATA_DIR}
        DESTINATION share/attention_hpc/tests
    )
endif()

# Test summary
message(STATUS "=== Test Configuration Summary ===")
message(STATUS "Google Test found: ${GTEST_FOUND}")
message(STATUS "Test data directory: ${TEST_DATA_DIR}")
message(STATUS "Unit tests: ${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "Integration tests: ${CMAKE_CURRENT_LIST_DIR}/integration")
if(BUILD_PERFORMANCE_TESTS)
    message(STATUS "Performance tests: ENABLED")
else()
    message(STATUS "Performance tests: DISABLED")
endif()
if(ENABLE_COVERAGE)
    message(STATUS "Code coverage: ENABLED")
else()
    message(STATUS "Code coverage: DISABLED")
endif()
message(STATUS "=====================================")

# Examples CMakeLists.txt

# Check if we have the main library target
if(NOT TARGET AttentionHPC::AttentionHPC)
    message(FATAL_ERROR "AttentionHPC library target not found")
endif()

# Basic usage example
add_executable(basic_usage basic_usage.cpp)
target_link_libraries(basic_usage PRIVATE AttentionHPC::AttentionHPC)
target_include_directories(basic_usage PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

# Performance comparison example
add_executable(performance_comparison performance_comparison.cpp)
target_link_libraries(performance_comparison PRIVATE AttentionHPC::AttentionHPC)
target_include_directories(performance_comparison PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

# Set target properties
set_target_properties(basic_usage performance_comparison PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Add examples to install if requested
option(INSTALL_EXAMPLES "Install example executables" OFF)
if(INSTALL_EXAMPLES)
    install(TARGETS basic_usage performance_comparison
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples
    )
endif()

# Add custom target to build all examples
add_custom_target(examples
    DEPENDS basic_usage performance_comparison
    COMMENT "Building all examples"
)

message(STATUS "Examples configured: basic_usage, performance_comparison")

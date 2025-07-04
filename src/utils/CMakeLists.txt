# Utils module CMakeLists.txt
# This module contains header-only utility classes for logging, error handling, and timing

# Define utility headers
set(UTILS_HEADERS
    logger.hpp
    error_handling.hpp
    timer.hpp
)

# Create interface library for header-only utilities
add_library(attention_utils INTERFACE)

# Set target properties
set_target_properties(attention_utils PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${UTILS_HEADERS}"
)

# Include directories
target_include_directories(attention_utils
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/attention_hpc/utils>
)

# Link with required libraries
target_link_libraries(attention_utils
    INTERFACE
        Threads::Threads
)

# Compiler definitions
target_compile_definitions(attention_utils
    INTERFACE
        $<$<CONFIG:Debug>:DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
)

# Add to parent scope for main library
set(UTILS_HEADERS ${UTILS_HEADERS} PARENT_SCOPE)

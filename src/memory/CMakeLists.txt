# Memory module CMakeLists.txt
# This module contains the memory management implementation for efficient memory pooling

# Define memory sources
set(MEMORY_SOURCES
    memory_pool.cpp
)

# Define memory headers
set(MEMORY_HEADERS
    memory_pool.hpp
)

# Create memory library
add_library(attention_memory ${MEMORY_SOURCES} ${MEMORY_HEADERS})

# Set target properties
set_target_properties(attention_memory PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${MEMORY_HEADERS}"
)

# Include directories
target_include_directories(attention_memory
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/attention_hpc/memory>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link with required libraries
target_link_libraries(attention_memory
    PUBLIC
        Threads::Threads
)

# Compiler definitions
target_compile_definitions(attention_memory
    PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
)

# Add to parent scope for main library
set(MEMORY_SOURCES ${MEMORY_SOURCES} PARENT_SCOPE)
set(MEMORY_HEADERS ${MEMORY_HEADERS} PARENT_SCOPE)

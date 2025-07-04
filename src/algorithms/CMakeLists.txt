# Algorithms module CMakeLists.txt
# This module contains the core FlashAttention and PagedAttention algorithm implementations

# Define algorithm sources
set(ALGORITHM_SOURCES
    flash_attention.cpp
    paged_attention.cpp
)

# Define algorithm headers
set(ALGORITHM_HEADERS
    flash_attention.hpp
    paged_attention.hpp
)

# Create algorithms library
add_library(attention_algorithms ${ALGORITHM_SOURCES} ${ALGORITHM_HEADERS})

# Set target properties
set_target_properties(attention_algorithms PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${ALGORITHM_HEADERS}"
)

# Include directories
target_include_directories(attention_algorithms
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/attention_hpc/algorithms>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link with required libraries
target_link_libraries(attention_algorithms
    PUBLIC
        Threads::Threads
)

# Compiler definitions
target_compile_definitions(attention_algorithms
    PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
)

# Add to parent scope for main library
set(ALGORITHM_SOURCES ${ALGORITHM_SOURCES} PARENT_SCOPE)
set(ALGORITHM_HEADERS ${ALGORITHM_HEADERS} PARENT_SCOPE)

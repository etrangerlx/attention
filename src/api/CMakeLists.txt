# API module CMakeLists.txt
# This module contains the core API interfaces and factory implementations

# Define API sources
set(API_SOURCES
    backend_factory.cpp
)

# Define API headers
set(API_HEADERS
    attention_interface.hpp
    backend_factory.hpp
    tensor.hpp
)

# Create API library
add_library(attention_api ${API_SOURCES} ${API_HEADERS})

# Set target properties
set_target_properties(attention_api PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${API_HEADERS}"
)

# Include directories
target_include_directories(attention_api
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/attention_hpc/api>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link with system libraries
target_link_libraries(attention_api
    PUBLIC
        Threads::Threads
)

# Compiler definitions
target_compile_definitions(attention_api
    PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
)

# Add to parent scope for main library
set(API_SOURCES ${API_SOURCES} PARENT_SCOPE)
set(API_HEADERS ${API_HEADERS} PARENT_SCOPE)

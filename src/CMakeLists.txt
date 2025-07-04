# Collect all source files
set(ATTENTION_HPC_SOURCES "")
set(ATTENTION_HPC_HEADERS "")

# Add subdirectories to collect sources
add_subdirectory(api)
add_subdirectory(algorithms)
add_subdirectory(backends)
add_subdirectory(memory)
add_subdirectory(utils)

# API sources
list(APPEND ATTENTION_HPC_SOURCES
    api/backend_factory.cpp
)
list(APPEND ATTENTION_HPC_HEADERS
    api/attention_interface.hpp
    api/backend_factory.hpp
    api/tensor.hpp
)

# Algorithm sources
list(APPEND ATTENTION_HPC_SOURCES
    algorithms/flash_attention.cpp
    algorithms/paged_attention.cpp
)
list(APPEND ATTENTION_HPC_HEADERS
    algorithms/flash_attention.hpp
    algorithms/paged_attention.hpp
)

# Memory management sources
list(APPEND ATTENTION_HPC_SOURCES
    memory/memory_pool.cpp
)
list(APPEND ATTENTION_HPC_HEADERS
    memory/memory_pool.hpp
)

# Utility headers
list(APPEND ATTENTION_HPC_HEADERS
    utils/logger.hpp
    utils/error_handling.hpp
    utils/timer.hpp
)

# Backend sources (conditional)
if(CUDA_FOUND)
    list(APPEND ATTENTION_HPC_SOURCES
        backends/cuda/cuda_kernels.cu
    )
    list(APPEND ATTENTION_HPC_HEADERS
        backends/cuda/cuda_flash_attention.hpp
    )
endif()

if(OpenCL_FOUND)
    list(APPEND ATTENTION_HPC_SOURCES
        backends/opencl/opencl_backend.cpp
    )
    list(APPEND ATTENTION_HPC_HEADERS
        backends/opencl/opencl_flash_attention.hpp
    )
endif()

if(Vulkan_FOUND)
    list(APPEND ATTENTION_HPC_SOURCES
        backends/vulkan/vulkan_backend.cpp
    )
    list(APPEND ATTENTION_HPC_HEADERS
        backends/vulkan/vulkan_flash_attention.hpp
    )
endif()

if(OpenGL_FOUND AND GLEW_FOUND)
    list(APPEND ATTENTION_HPC_HEADERS
        backends/opengl/opengl_flash_attention.hpp
    )
endif()

# Create main library
add_library(AttentionHPC ${ATTENTION_HPC_SOURCES} ${ATTENTION_HPC_HEADERS})

# Set target properties
set_target_properties(AttentionHPC PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${ATTENTION_HPC_HEADERS}"
)

# Include directories
target_include_directories(AttentionHPC
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link required libraries
target_link_libraries(AttentionHPC
    PUBLIC
        Threads::Threads
)

# Conditional linking based on available backends
if(CUDA_FOUND)
    target_link_libraries(AttentionHPC PUBLIC ${CUDA_LIBRARIES})
    target_include_directories(AttentionHPC PUBLIC ${CUDA_INCLUDE_DIRS})
    
    # CUDA-specific compiler flags
    set_property(TARGET AttentionHPC PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75 80 86)
    target_compile_options(AttentionHPC PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

if(OpenCL_FOUND)
    target_link_libraries(AttentionHPC PUBLIC OpenCL::OpenCL)
endif()

if(Vulkan_FOUND)
    target_link_libraries(AttentionHPC PUBLIC Vulkan::Vulkan)
endif()

if(OpenGL_FOUND AND GLEW_FOUND)
    target_link_libraries(AttentionHPC PUBLIC OpenGL::GL OpenGL::GLU GLEW::GLEW)
endif()

# Compiler definitions
target_compile_definitions(AttentionHPC
    PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
)

# Install targets
install(TARGETS AttentionHPC
    EXPORT AttentionHPCTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/attention_hpc
)

# Install headers with directory structure
install(DIRECTORY api algorithms backends memory utils
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/attention_hpc
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# Export targets
install(EXPORT AttentionHPCTargets
    FILE AttentionHPCTargets.cmake
    NAMESPACE AttentionHPC::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AttentionHPC
)

# Create alias for consistent naming
add_library(AttentionHPC::AttentionHPC ALIAS AttentionHPC)

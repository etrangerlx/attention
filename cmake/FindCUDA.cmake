# FindCUDA.cmake - Find CUDA toolkit and libraries
#
# This module defines:
#  CUDA_FOUND - True if CUDA is found
#  CUDA_INCLUDE_DIRS - Include directories for CUDA headers
#  CUDA_LIBRARIES - List of libraries when using CUDA
#  CUDA_TOOLKIT_ROOT_DIR - Root directory of CUDA toolkit
#  CUDA_VERSION - Version of CUDA toolkit
#  CUDA_NVCC_EXECUTABLE - Path to nvcc compiler
#  CUDA_RUNTIME_LIBRARY - CUDA runtime library
#  CUDA_CUBLAS_LIBRARIES - cuBLAS library
#  CUDA_CUFFT_LIBRARIES - cuFFT library
#  CUDA_CURAND_LIBRARIES - cuRAND library
#  CUDA_CUSPARSE_LIBRARIES - cuSPARSE library
#  CUDA_NPP_LIBRARIES - NPP library
#  CUDA_CUDNN_LIBRARY - cuDNN library (if available)

cmake_minimum_required(VERSION 3.18)

# Set default CUDA toolkit root directory candidates
set(CUDA_TOOLKIT_ROOT_DIR_CANDIDATES
    $ENV{CUDA_PATH}
    $ENV{CUDA_HOME}
    $ENV{CUDA_ROOT}
    /usr/local/cuda
    /opt/cuda
    /usr/local/cuda-12.0
    /usr/local/cuda-11.8
    /usr/local/cuda-11.7
    /usr/local/cuda-11.6
    /usr/local/cuda-11.5
    /usr/local/cuda-11.4
    /usr/local/cuda-11.3
    /usr/local/cuda-11.2
    /usr/local/cuda-11.1
    /usr/local/cuda-11.0
    /usr/local/cuda-10.2
    /usr/local/cuda-10.1
    /usr/local/cuda-10.0
)

# Find CUDA toolkit root directory
find_path(CUDA_TOOLKIT_ROOT_DIR
    NAMES bin/nvcc bin/nvcc.exe
    PATHS ${CUDA_TOOLKIT_ROOT_DIR_CANDIDATES}
    DOC "CUDA toolkit root directory"
)

if(CUDA_TOOLKIT_ROOT_DIR)
    # Find CUDA version
    if(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/version.txt")
        file(READ "${CUDA_TOOLKIT_ROOT_DIR}/version.txt" CUDA_VERSION_TEXT)
        string(REGEX MATCH "CUDA Version ([0-9]+\\.[0-9]+)" CUDA_VERSION_MATCH ${CUDA_VERSION_TEXT})
        if(CUDA_VERSION_MATCH)
            set(CUDA_VERSION ${CMAKE_MATCH_1})
        endif()
    elseif(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/version.json")
        file(READ "${CUDA_TOOLKIT_ROOT_DIR}/version.json" CUDA_VERSION_JSON)
        string(REGEX MATCH "\"version\": \"([0-9]+\\.[0-9]+)" CUDA_VERSION_MATCH ${CUDA_VERSION_JSON})
        if(CUDA_VERSION_MATCH)
            set(CUDA_VERSION ${CMAKE_MATCH_1})
        endif()
    endif()

    # Find NVCC compiler
    find_program(CUDA_NVCC_EXECUTABLE
        NAMES nvcc
        PATHS ${CUDA_TOOLKIT_ROOT_DIR}/bin
        DOC "CUDA compiler (nvcc)"
    )

    # Determine library architecture suffix
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(CUDA_LIB_SUFFIX "64")
    else()
        set(CUDA_LIB_SUFFIX "")
    endif()

    # Set library directory paths
    set(CUDA_LIB_DIRS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib${CUDA_LIB_SUFFIX}
        ${CUDA_TOOLKIT_ROOT_DIR}/lib
    )

    # Find CUDA include directory
    find_path(CUDA_INCLUDE_DIR
        NAMES cuda.h cuda_runtime.h
        PATHS ${CUDA_TOOLKIT_ROOT_DIR}/include
        DOC "CUDA include directory"
    )

    # Find CUDA runtime library
    find_library(CUDA_RUNTIME_LIBRARY
        NAMES cudart
        PATHS ${CUDA_LIB_DIRS}
        DOC "CUDA runtime library"
    )

    # Find CUDA driver library
    find_library(CUDA_DRIVER_LIBRARY
        NAMES cuda
        PATHS ${CUDA_LIB_DIRS}
        DOC "CUDA driver library"
    )

    # Find cuBLAS library
    find_library(CUDA_CUBLAS_LIBRARY
        NAMES cublas
        PATHS ${CUDA_LIB_DIRS}
        DOC "cuBLAS library"
    )

    # Find cuFFT library
    find_library(CUDA_CUFFT_LIBRARY
        NAMES cufft
        PATHS ${CUDA_LIB_DIRS}
        DOC "cuFFT library"
    )

    # Find cuRAND library
    find_library(CUDA_CURAND_LIBRARY
        NAMES curand
        PATHS ${CUDA_LIB_DIRS}
        DOC "cuRAND library"
    )

    # Find cuSPARSE library
    find_library(CUDA_CUSPARSE_LIBRARY
        NAMES cusparse
        PATHS ${CUDA_LIB_DIRS}
        DOC "cuSPARSE library"
    )

    # Find NPP library
    find_library(CUDA_NPP_LIBRARY
        NAMES nppc npp
        PATHS ${CUDA_LIB_DIRS}
        DOC "NPP library"
    )

    # Find cuDNN library (optional)
    find_library(CUDA_CUDNN_LIBRARY
        NAMES cudnn
        PATHS ${CUDA_LIB_DIRS}
        DOC "cuDNN library"
    )

    # Set up library lists
    set(CUDA_LIBRARIES ${CUDA_RUNTIME_LIBRARY})
    set(CUDA_CUBLAS_LIBRARIES ${CUDA_CUBLAS_LIBRARY})
    set(CUDA_CUFFT_LIBRARIES ${CUDA_CUFFT_LIBRARY})
    set(CUDA_CURAND_LIBRARIES ${CUDA_CURAND_LIBRARY})
    set(CUDA_CUSPARSE_LIBRARIES ${CUDA_CUSPARSE_LIBRARY})
    set(CUDA_NPP_LIBRARIES ${CUDA_NPP_LIBRARY})

    # Set include directories
    set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIR})

endif()

# Handle standard CMake find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDA
    REQUIRED_VARS CUDA_TOOLKIT_ROOT_DIR CUDA_NVCC_EXECUTABLE CUDA_INCLUDE_DIR CUDA_RUNTIME_LIBRARY
    VERSION_VAR CUDA_VERSION
)

# Create imported targets
if(CUDA_FOUND AND NOT TARGET CUDA::runtime)
    add_library(CUDA::runtime SHARED IMPORTED)
    set_target_properties(CUDA::runtime PROPERTIES
        IMPORTED_LOCATION ${CUDA_RUNTIME_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
    )

    if(CUDA_DRIVER_LIBRARY)
        add_library(CUDA::driver SHARED IMPORTED)
        set_target_properties(CUDA::driver PROPERTIES
            IMPORTED_LOCATION ${CUDA_DRIVER_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
        )
    endif()

    if(CUDA_CUBLAS_LIBRARY)
        add_library(CUDA::cublas SHARED IMPORTED)
        set_target_properties(CUDA::cublas PROPERTIES
            IMPORTED_LOCATION ${CUDA_CUBLAS_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
            INTERFACE_LINK_LIBRARIES CUDA::runtime
        )
    endif()

    if(CUDA_CUFFT_LIBRARY)
        add_library(CUDA::cufft SHARED IMPORTED)
        set_target_properties(CUDA::cufft PROPERTIES
            IMPORTED_LOCATION ${CUDA_CUFFT_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
            INTERFACE_LINK_LIBRARIES CUDA::runtime
        )
    endif()

    if(CUDA_CURAND_LIBRARY)
        add_library(CUDA::curand SHARED IMPORTED)
        set_target_properties(CUDA::curand PROPERTIES
            IMPORTED_LOCATION ${CUDA_CURAND_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
            INTERFACE_LINK_LIBRARIES CUDA::runtime
        )
    endif()

    if(CUDA_CUSPARSE_LIBRARY)
        add_library(CUDA::cusparse SHARED IMPORTED)
        set_target_properties(CUDA::cusparse PROPERTIES
            IMPORTED_LOCATION ${CUDA_CUSPARSE_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
            INTERFACE_LINK_LIBRARIES CUDA::runtime
        )
    endif()

    if(CUDA_NPP_LIBRARY)
        add_library(CUDA::npp SHARED IMPORTED)
        set_target_properties(CUDA::npp PROPERTIES
            IMPORTED_LOCATION ${CUDA_NPP_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
            INTERFACE_LINK_LIBRARIES CUDA::runtime
        )
    endif()

    if(CUDA_CUDNN_LIBRARY)
        add_library(CUDA::cudnn SHARED IMPORTED)
        set_target_properties(CUDA::cudnn PROPERTIES
            IMPORTED_LOCATION ${CUDA_CUDNN_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
            INTERFACE_LINK_LIBRARIES CUDA::runtime
        )
    endif()
endif()

# Set CUDA compiler flags and definitions
if(CUDA_FOUND)
    # Define CUDA availability macro
    add_definitions(-DHAVE_CUDA=1)
    
    # Set CUDA architecture flags
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "52;61;70;75;80;86")
    endif()
    
    # Set CUDA separable compilation
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    
    # Set CUDA standard
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
    endif()
    
    # CUDA compiler flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
    
    # Enable CUDA language
    enable_language(CUDA)
endif()

# Mark variables as advanced
mark_as_advanced(
    CUDA_TOOLKIT_ROOT_DIR
    CUDA_INCLUDE_DIR
    CUDA_RUNTIME_LIBRARY
    CUDA_DRIVER_LIBRARY
    CUDA_CUBLAS_LIBRARY
    CUDA_CUFFT_LIBRARY
    CUDA_CURAND_LIBRARY
    CUDA_CUSPARSE_LIBRARY
    CUDA_NPP_LIBRARY
    CUDA_CUDNN_LIBRARY
    CUDA_NVCC_EXECUTABLE
)

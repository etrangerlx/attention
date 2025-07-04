# FindOpenCL.cmake - Find OpenCL headers and libraries
#
# This module defines:
#  OpenCL_FOUND - True if OpenCL is found
#  OpenCL_INCLUDE_DIRS - Include directories for OpenCL headers
#  OpenCL_LIBRARIES - List of libraries when using OpenCL
#  OpenCL_VERSION - Version of OpenCL (if determinable)
#  OpenCL_VERSION_STRING - Version string of OpenCL
#  OpenCL_VERSION_MAJOR - Major version number
#  OpenCL_VERSION_MINOR - Minor version number
#  OpenCL_PLATFORMS - List of available OpenCL platforms
#  OpenCL_DEVICES - List of available OpenCL devices
#  OpenCL_ICD_LOADER_LIBRARY - OpenCL ICD loader library

cmake_minimum_required(VERSION 3.18)

# Platform-specific search paths
if(WIN32)
    set(OPENCL_ROOT_DIRS
        $ENV{CUDA_PATH}
        $ENV{INTELOCLSDKROOT}
        $ENV{AMDAPPSDKROOT}
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*"
        "C:/Program Files (x86)/Intel/OpenCL SDK"
        "C:/Program Files (x86)/AMD APP SDK"
        "C:/Program Files/AMD APP SDK"
    )
    set(OPENCL_LIB_SEARCH_PATHS
        ${OPENCL_ROOT_DIRS}/lib/x64
        ${OPENCL_ROOT_DIRS}/lib/Win32
        ${OPENCL_ROOT_DIRS}/lib
    )
    set(OPENCL_INC_SEARCH_PATHS
        ${OPENCL_ROOT_DIRS}/include
    )
elseif(APPLE)
    set(OPENCL_LIB_SEARCH_PATHS
        /System/Library/Frameworks/OpenCL.framework
    )
    set(OPENCL_INC_SEARCH_PATHS
        /System/Library/Frameworks/OpenCL.framework/Headers
        /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/OpenCL.framework/Headers
    )
else()
    set(OPENCL_ROOT_DIRS
        $ENV{CUDA_PATH}
        $ENV{INTELOCLSDKROOT}
        $ENV{AMDAPPSDKROOT}
        /usr/local/cuda
        /opt/cuda
        /usr/local
        /usr
        /opt/intel/opencl
        /opt/AMDAPP
        /opt/AMDAPPSDK-3.0
        /opt/AMDAPPSDK-2.9-1
    )
    set(OPENCL_LIB_SEARCH_PATHS
        ${OPENCL_ROOT_DIRS}/lib64
        ${OPENCL_ROOT_DIRS}/lib
        ${OPENCL_ROOT_DIRS}/lib/x86_64
        ${OPENCL_ROOT_DIRS}/lib/x64
        /usr/lib64
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib64/nvidia
        /usr/lib/nvidia
    )
    set(OPENCL_INC_SEARCH_PATHS
        ${OPENCL_ROOT_DIRS}/include
        /usr/include
        /usr/local/include
    )
endif()

# Find OpenCL include directory
find_path(OpenCL_INCLUDE_DIR
    NAMES 
        CL/cl.h 
        OpenCL/cl.h
    PATHS ${OPENCL_INC_SEARCH_PATHS}
    DOC "OpenCL include directory"
)

# Find OpenCL library
if(APPLE)
    find_library(OpenCL_LIBRARY
        NAMES OpenCL
        PATHS ${OPENCL_LIB_SEARCH_PATHS}
        DOC "OpenCL library"
    )
else()
    find_library(OpenCL_LIBRARY
        NAMES 
            OpenCL
            opencl
        PATHS ${OPENCL_LIB_SEARCH_PATHS}
        DOC "OpenCL library"
    )
endif()

# Set OpenCL ICD loader library (same as main library on most platforms)
set(OpenCL_ICD_LOADER_LIBRARY ${OpenCL_LIBRARY})

# Determine OpenCL version from headers
if(OpenCL_INCLUDE_DIR)
    set(OpenCL_VERSION_TEST_FILE "${CMAKE_BINARY_DIR}/CMakeFiles/opencl_version_test.c")
    
    # Create a test file to determine OpenCL version
    file(WRITE ${OpenCL_VERSION_TEST_FILE}
        "#include <stdio.h>\n"
        "#ifdef __APPLE__\n"
        "#include <OpenCL/cl.h>\n"
        "#else\n"
        "#include <CL/cl.h>\n"
        "#endif\n"
        "int main() {\n"
        "#ifdef CL_VERSION_3_0\n"
        "    printf(\"3.0\");\n"
        "#elif defined(CL_VERSION_2_2)\n"
        "    printf(\"2.2\");\n"
        "#elif defined(CL_VERSION_2_1)\n"
        "    printf(\"2.1\");\n"
        "#elif defined(CL_VERSION_2_0)\n"
        "    printf(\"2.0\");\n"
        "#elif defined(CL_VERSION_1_2)\n"
        "    printf(\"1.2\");\n"
        "#elif defined(CL_VERSION_1_1)\n"
        "    printf(\"1.1\");\n"
        "#elif defined(CL_VERSION_1_0)\n"
        "    printf(\"1.0\");\n"
        "#else\n"
        "    printf(\"unknown\");\n"
        "#endif\n"
        "    return 0;\n"
        "}\n"
    )
    
    # Try to compile and run the test
    try_run(OpenCL_VERSION_RUN_RESULT OpenCL_VERSION_COMPILE_RESULT
        ${CMAKE_BINARY_DIR}
        ${OpenCL_VERSION_TEST_FILE}
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${OpenCL_INCLUDE_DIR}"
        RUN_OUTPUT_VARIABLE OpenCL_VERSION_STRING
    )
    
    if(OpenCL_VERSION_COMPILE_RESULT AND OpenCL_VERSION_RUN_RESULT EQUAL 0)
        set(OpenCL_VERSION ${OpenCL_VERSION_STRING})
        
        # Parse major and minor version numbers
        string(REGEX MATCH "([0-9]+)\\.([0-9]+)" OpenCL_VERSION_MATCH ${OpenCL_VERSION_STRING})
        if(OpenCL_VERSION_MATCH)
            set(OpenCL_VERSION_MAJOR ${CMAKE_MATCH_1})
            set(OpenCL_VERSION_MINOR ${CMAKE_MATCH_2})
        endif()
    else()
        set(OpenCL_VERSION "unknown")
        set(OpenCL_VERSION_STRING "unknown")
    endif()
    
    # Clean up test file
    file(REMOVE ${OpenCL_VERSION_TEST_FILE})
endif()

# Create platform detection program
if(OpenCL_INCLUDE_DIR AND OpenCL_LIBRARY)
    set(OpenCL_PLATFORM_TEST_FILE "${CMAKE_BINARY_DIR}/CMakeFiles/opencl_platform_test.c")
    
    file(WRITE ${OpenCL_PLATFORM_TEST_FILE}
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        "#ifdef __APPLE__\n"
        "#include <OpenCL/cl.h>\n"
        "#else\n"
        "#include <CL/cl.h>\n"
        "#endif\n"
        "int main() {\n"
        "    cl_uint num_platforms = 0;\n"
        "    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);\n"
        "    if (err != CL_SUCCESS || num_platforms == 0) {\n"
        "        printf(\"0\");\n"
        "        return 0;\n"
        "    }\n"
        "    cl_platform_id* platforms = malloc(sizeof(cl_platform_id) * num_platforms);\n"
        "    err = clGetPlatformIDs(num_platforms, platforms, NULL);\n"
        "    if (err != CL_SUCCESS) {\n"
        "        free(platforms);\n"
        "        printf(\"0\");\n"
        "        return 0;\n"
        "    }\n"
        "    printf(\"%u\", num_platforms);\n"
        "    free(platforms);\n"
        "    return 0;\n"
        "}\n"
    )
    
    # Try to compile and run platform detection
    try_run(OpenCL_PLATFORM_RUN_RESULT OpenCL_PLATFORM_COMPILE_RESULT
        ${CMAKE_BINARY_DIR}
        ${OpenCL_PLATFORM_TEST_FILE}
        CMAKE_FLAGS 
            "-DINCLUDE_DIRECTORIES:STRING=${OpenCL_INCLUDE_DIR}"
            "-DLINK_LIBRARIES:STRING=${OpenCL_LIBRARY}"
        RUN_OUTPUT_VARIABLE OpenCL_PLATFORM_COUNT
    )
    
    if(OpenCL_PLATFORM_COMPILE_RESULT AND OpenCL_PLATFORM_RUN_RESULT EQUAL 0)
        set(OpenCL_PLATFORMS ${OpenCL_PLATFORM_COUNT})
    else()
        set(OpenCL_PLATFORMS "0")
    endif()
    
    # Clean up test file
    file(REMOVE ${OpenCL_PLATFORM_TEST_FILE})
else()
    set(OpenCL_PLATFORMS "0")
endif()

# Set up variables
if(OpenCL_INCLUDE_DIR AND OpenCL_LIBRARY)
    set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})
    set(OpenCL_LIBRARIES ${OpenCL_LIBRARY})
endif()

# Handle standard CMake find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCL
    REQUIRED_VARS OpenCL_LIBRARY OpenCL_INCLUDE_DIR
    VERSION_VAR OpenCL_VERSION
)

# Create imported targets
if(OpenCL_FOUND AND NOT TARGET OpenCL::OpenCL)
    add_library(OpenCL::OpenCL UNKNOWN IMPORTED)
    set_target_properties(OpenCL::OpenCL PROPERTIES
        IMPORTED_LOCATION ${OpenCL_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${OpenCL_INCLUDE_DIRS}
    )
    
    # Set platform-specific properties
    if(APPLE)
        set_target_properties(OpenCL::OpenCL PROPERTIES
            INTERFACE_LINK_LIBRARIES "-framework OpenCL"
        )
    endif()
endif()

# Set OpenCL availability macro and compiler definitions
if(OpenCL_FOUND)
    add_definitions(-DHAVE_OPENCL=1)
    
    # Add OpenCL version definitions
    if(OpenCL_VERSION_MAJOR AND OpenCL_VERSION_MINOR)
        add_definitions(-DOPENCL_VERSION_MAJOR=${OpenCL_VERSION_MAJOR})
        add_definitions(-DOPENCL_VERSION_MINOR=${OpenCL_VERSION_MINOR})
    endif()
    
    # Add platform count definition
    if(OpenCL_PLATFORMS)
        add_definitions(-DOPENCL_PLATFORM_COUNT=${OpenCL_PLATFORMS})
    endif()
endif()

# Mark variables as advanced
mark_as_advanced(
    OpenCL_INCLUDE_DIR
    OpenCL_LIBRARY
    OpenCL_ICD_LOADER_LIBRARY
)

# Debug information
if(OpenCL_FOUND)
    message(STATUS "Found OpenCL: ${OpenCL_LIBRARY}")
    message(STATUS "OpenCL version: ${OpenCL_VERSION}")
    message(STATUS "OpenCL include dir: ${OpenCL_INCLUDE_DIR}")
    message(STATUS "OpenCL platforms detected: ${OpenCL_PLATFORMS}")
else()
    message(STATUS "OpenCL not found")
endif()

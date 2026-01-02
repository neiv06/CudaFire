cmake_minimum_required(VERSION 3.18)

# Force MSVC on Windows
if(WIN32)
    set(CMAKE_C_COMPILER "cl.exe")
    set(CMAKE_CXX_COMPILER "cl.exe")
endif()

project(WildfireSimulator CUDA CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures for RTX 3080 (Ampere - SM 86)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Source files
set(SOURCES
    src/main.cpp
    src/terrain.cpp
    src/simulation.cu
    src/rothermel.cu
)

# Create executable
add_executable(wildfire_sim ${SOURCES})

# Include directories
target_include_directories(wildfire_sim PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link CUDA runtime (static to avoid DLL issues)
target_link_libraries(wildfire_sim PRIVATE CUDA::cudart_static)

# Enable separable compilation for CUDA
set_target_properties(wildfire_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Compiler flags
if(MSVC)
    target_compile_options(wildfire_sim PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /O2>
    )
    # Suppress unreferenced parameter warnings for stub functions
    target_compile_options(wildfire_sim PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:/wd4100>
    )
endif()

# CUDA compiler flags
target_compile_options(wildfire_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)
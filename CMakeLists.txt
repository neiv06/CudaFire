cmake_minimum_required(VERSION 3.18)
project(WildfireSimulator CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find GDAL
find_package(GDAL REQUIRED)

# Set CUDA architectures (86 for RTX 3080)
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
target_include_directories(wildfire_sim PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${GDAL_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(wildfire_sim PRIVATE 
    CUDA::cudart_static
    ${GDAL_LIBRARY}
)

# CUDA specific flags
set_target_properties(wildfire_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(wildfire_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)

# MSVC specific
if(MSVC)
    target_compile_options(wildfire_sim PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /O2 /wd4100>
    )
endif()
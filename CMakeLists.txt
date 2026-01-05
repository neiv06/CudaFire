cmake_minimum_required(VERSION 3.18)
project(WildfireSimulator CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find GDAL (paths set via command line)
find_package(GDAL REQUIRED)

# Find OpenGL
find_package(OpenGL REQUIRED)

# GLFW - manual path
set(GLFW_DIR "C:/libs/glfw-3.4.bin.WIN64")  # Adjust version number if different
set(GLFW_INCLUDE_DIR "${GLFW_DIR}/include")
set(GLFW_LIBRARY "${GLFW_DIR}/lib-vc2022/glfw3.lib")  # or lib-vc2019 depending on your VS version

# GLM - header only
set(GLM_INCLUDE_DIR "C:/libs/glm")

# Set CUDA architectures (86 for RTX 3080)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Source files
set(SOURCES
    src/main.cpp
    src/terrain.cpp
    src/simulation.cu
    src/rothermel.cu
    src/gl_viewer.cpp
    src/glad.c
)

# Create executable
add_executable(wildfire_sim ${SOURCES})

# Include directories
target_include_directories(wildfire_sim PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${GDAL_INCLUDE_DIR}
    ${GLFW_INCLUDE_DIR}
    ${GLM_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(wildfire_sim PRIVATE 
    CUDA::cudart_static
    ${GDAL_LIBRARY}
    ${GLFW_LIBRARY}
    OpenGL::GL
)

# CUDA flags
set_target_properties(wildfire_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(wildfire_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

if(MSVC)
    target_compile_options(wildfire_sim PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /O2 /wd4100>
    )
endif()
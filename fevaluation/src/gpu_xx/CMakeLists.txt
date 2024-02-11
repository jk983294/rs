cmake_minimum_required(VERSION 3.17)

project(cuda_lib)
#set(CMAKE_CUDA_ARCHITECTURES 65)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)
set(CUDA_NVCC_FLAGS "-O3 -DNDEBUG")
add_definitions("${CUDA_NVCC_FLAGS}")

file( GLOB CUDA_LIB_SOURCES "*.cu")
add_library(gpuxx_static STATIC ${CUDA_LIB_SOURCES})
set_target_properties(gpuxx_static PROPERTIES 
                        CUDA_SEPARABLE_COMPILATION ON
                        CUDA_RESOLVE_DEVICE_SYMBOLS ON
                        POSITION_INDEPENDENT_CODE ON)
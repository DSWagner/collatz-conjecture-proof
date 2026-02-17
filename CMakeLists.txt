cmake_minimum_required(VERSION 3.20)
project(collatz_proof LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "86")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v,-O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount=48")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

add_executable(collatz_proof src/collatz_proof.cu)
target_include_directories(collatz_proof PRIVATE ${CMAKE_SOURCE_DIR}/include)

set_target_properties(collatz_proof PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(WIN32)
    set_target_properties(collatz_proof PROPERTIES CUDA_RUNTIME_LIBRARY Static)
endif()

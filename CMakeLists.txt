cmake_minimum_required(VERSION 3.20)
project(collatz_conjecture_proof LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 3070 = SM 8.6
set(CMAKE_CUDA_ARCHITECTURES "86")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

# MSVC flags -- generator expressions so they only go to cl.exe, NOT nvcc
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/O2>
        $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
        $<$<COMPILE_LANGUAGE:CXX>:/fp:fast>
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4005>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4819>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
    )
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=znver3 -mtune=znver3 -DNDEBUG")
endif()

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v,-O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount=64")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

add_executable(collatz_proof
    src/collatz_proof.cu
)

target_include_directories(collatz_proof PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(collatz_proof PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(WIN32)
    set_target_properties(collatz_proof PROPERTIES CUDA_RUNTIME_LIBRARY Static)
    target_link_libraries(collatz_proof PRIVATE CUDA::cudart_static)
else()
    target_link_libraries(collatz_proof PRIVATE cuda)
endif()

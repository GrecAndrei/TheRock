# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(therock-inference C CXX HIP)

if(NOT THEROCK_ENABLE_HIP_RUNTIME OR NOT THEROCK_ENABLE_BLAS)
  message(STATUS "inference: skipping (requires HIP_RUNTIME and BLAS)")
  return()
endif()

find_package(hip REQUIRED CONFIG)
find_package(hipblas REQUIRED CONFIG PATHS "${THEROCK_DIST_DIR}/rocm/lib/cmake")
find_package(rocblas REQUIRED CONFIG PATHS "${THEROCK_DIST_DIR}/rocm/lib/cmake")
find_package(hipblaslt REQUIRED CONFIG PATHS "${THEROCK_DIST_DIR}/rocm/lib/cmake")

add_library(therock-inference STATIC
  src/hardware/descriptors.c
  src/hardware/hardware_detect_hip.cpp
  src/memory/memory.cpp
  src/memory/prefetch.cpp
  src/executor/kernel_registry.cpp
  src/executor/executor.cpp
  src/kernels/gemv_wave32.hip
  src/kernels/flash_attn_wave32.hip
  src/kernels/flash_attn_hipblaslt.hip
  src/kernels/gemm_hipblaslt.cpp
  src/kernels/ops.hip
)

target_include_directories(therock-inference
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(therock-inference
  PRIVATE
    hip::host
    roc::hipblas
    roc::rocblas
    roc::hipblaslt
)

set_target_properties(therock-inference PROPERTIES
  C_STANDARD 11
  CXX_STANDARD 17
)

install(
  TARGETS therock-inference
  ARCHIVE DESTINATION lib
)

install(
  DIRECTORY include/
  DESTINATION include
)

if(THEROCK_BUILD_TESTING)
  add_subdirectory(tests)
endif()

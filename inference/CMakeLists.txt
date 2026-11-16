# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Use -B build.")
endif()

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx1201" CACHE STRING "Target AMDGPU architectures")
endif()

project(rxinfer LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

set(HIP_PLATFORM "amd")
find_package(hip CONFIG REQUIRED)
find_package(hipblaslt CONFIG QUIET)

option(RXINFER_BUILD_BENCH "Build microbenchmarks and hardware probes" ON)
option(RXINFER_ENABLE_HIPBLASLT "Link hipBLASLt for baseline comparison/prefill fallback" ON)

add_library(rxinfer_core STATIC
    src/arena.cpp
    src/config_json.cpp
    src/loader.cpp
    src/markdown.cpp
    src/sampler.cpp
    src/server.cpp
    src/tokenizer.cpp
    src/tools.cpp
    src/tui.cpp
    src/engine.cpp
    src/kernels/quant_q4_gguf.hip
    src/kernels/quant_r4_gfx12.hip
    src/kernels/norm_rope_ops.hip
    src/kernels/attention_gfx12.hip
    src/kernels/gdn_gfx12.hip
)

set_source_files_properties(
    src/kernels/quant_q4_gguf.hip
    src/kernels/quant_r4_gfx12.hip
    src/kernels/norm_rope_ops.hip
    src/kernels/attention_gfx12.hip
    src/kernels/gdn_gfx12.hip
    PROPERTIES LANGUAGE HIP
)

set_target_properties(rxinfer_core PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
)

target_include_directories(rxinfer_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(rxinfer_core PUBLIC
    hip::host
    hip::device
)

if(TARGET hip::amdhip64)
    target_link_options(rxinfer_core PUBLIC
        "LINKER:-rpath,$<TARGET_FILE_DIR:hip::amdhip64>"
    )
endif()

target_compile_options(rxinfer_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -mno-wavefrontsize64>
)

add_executable(rxinfer
    src/main.cpp
)
target_link_libraries(rxinfer PRIVATE rxinfer_core hip::host)
target_compile_options(rxinfer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
)

if(RXINFER_BUILD_BENCH)
    add_executable(rxinfer_test_io
        bench/test_io.cpp
    )
    target_link_libraries(rxinfer_test_io PRIVATE rxinfer_core hip::host)
    target_compile_options(rxinfer_test_io PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    )
    add_executable(rxinfer_wmma_probe
        bench/wmma_probe.hip
    )
    set_source_files_properties(bench/wmma_probe.hip PROPERTIES LANGUAGE HIP)
    set_target_properties(rxinfer_wmma_probe PROPERTIES
        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    )
    target_link_libraries(rxinfer_wmma_probe PRIVATE rxinfer_core hip::device)
    target_compile_options(rxinfer_wmma_probe PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3 -mno-wavefrontsize64>
    )

    add_executable(rxinfer_bench_quant
        bench/bench_quant.hip
    )
    set_source_files_properties(bench/bench_quant.hip PROPERTIES LANGUAGE HIP)
    set_target_properties(rxinfer_bench_quant PROPERTIES
        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    )
    target_link_libraries(rxinfer_bench_quant PRIVATE rxinfer_core hip::device)
    target_compile_options(rxinfer_bench_quant PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3 -mno-wavefrontsize64>
    )

    add_executable(rxinfer_test_kernels
        bench/test_kernels.hip
    )
    set_source_files_properties(bench/test_kernels.hip PROPERTIES LANGUAGE HIP)
    set_target_properties(rxinfer_test_kernels PROPERTIES
        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    )
    target_link_libraries(rxinfer_test_kernels PRIVATE rxinfer_core hip::device)
    target_compile_options(rxinfer_test_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3 -mno-wavefrontsize64>
    )

    add_executable(rxinfer_bench_e2e
        bench/bench_e2e.hip
    )
    set_source_files_properties(bench/bench_e2e.hip PROPERTIES LANGUAGE HIP)
    set_target_properties(rxinfer_bench_e2e PROPERTIES
        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    )
    target_link_libraries(rxinfer_bench_e2e PRIVATE rxinfer_core hip::device)
    target_compile_options(rxinfer_bench_e2e PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3 -mno-wavefrontsize64>
    )
endif()

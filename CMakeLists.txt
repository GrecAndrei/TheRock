cmake_minimum_required(VERSION 3.22)

project(qwen38_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(hip CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)
pkg_check_modules(ICU REQUIRED IMPORTED_TARGET icu-uc)
include(CTest)

add_executable(qwen38-native
    src/cpu_matvec.cpp
    src/gguf.cpp
    src/kernels.cpp
    src/main.cpp
)
set_source_files_properties(src/cpu_matvec.cpp PROPERTIES
    COMPILE_OPTIONS "-march=native;-mtune=native;-mavx2;-mfma;-mf16c"
)

target_include_directories(qwen38-native PRIVATE src)
target_link_libraries(qwen38-native PRIVATE
    hip::host Threads::Threads PkgConfig::PCRE2 PkgConfig::ICU
)
target_compile_options(qwen38-native PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(qwen38-native PROPERTIES
    BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib"
    INSTALL_RPATH "${CMAKE_PREFIX_PATH}/lib"
)

if(BUILD_TESTING)
    add_executable(qwen38-cpu-self-test
        src/cpu_matvec.cpp
        src/cpu_matvec_test.cpp
        src/gguf.cpp
    )
    target_include_directories(qwen38-cpu-self-test PRIVATE src)
    target_link_libraries(qwen38-cpu-self-test PRIVATE Threads::Threads)
    target_compile_options(qwen38-cpu-self-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(qwen38-cpu-self-test PROPERTIES
        BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib"
    )
    add_test(NAME qwen38-cpu-self-test COMMAND qwen38-cpu-self-test)

    add_executable(qwen38-runtime-options-test tests/runtime_options_test.cpp)
    target_include_directories(qwen38-runtime-options-test PRIVATE src)
    target_compile_options(qwen38-runtime-options-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME qwen38-runtime-options-test COMMAND qwen38-runtime-options-test)

    add_executable(qwen38-attention-tile-test tests/attention_tile_test.cpp)
    target_include_directories(qwen38-attention-tile-test PRIVATE src)
    target_compile_options(qwen38-attention-tile-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME qwen38-attention-tile-test COMMAND qwen38-attention-tile-test)
    add_executable(qwen38-single-job-worker-test tests/single_job_worker_test.cpp)
    target_include_directories(qwen38-single-job-worker-test PRIVATE src)
    target_link_libraries(qwen38-single-job-worker-test PRIVATE Threads::Threads)
    target_compile_options(qwen38-single-job-worker-test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME qwen38-single-job-worker-test COMMAND qwen38-single-job-worker-test)
    set_tests_properties(qwen38-single-job-worker-test PROPERTIES TIMEOUT 30)
    set_target_properties(qwen38-runtime-options-test qwen38-attention-tile-test
        qwen38-single-job-worker-test PROPERTIES
        BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib"
    )
endif()

install(TARGETS qwen38-native RUNTIME DESTINATION bin)

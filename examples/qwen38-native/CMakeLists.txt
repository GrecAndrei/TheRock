cmake_minimum_required(VERSION 3.22)

project(qwen38_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(hip CONFIG REQUIRED)

add_executable(qwen38-native
    src/gguf.cpp
    src/kernels.cpp
    src/main.cpp
)

target_include_directories(qwen38-native PRIVATE src)
target_link_libraries(qwen38-native PRIVATE hip::host)
target_compile_options(qwen38-native PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(qwen38-native PROPERTIES
    BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib"
    INSTALL_RPATH "${CMAKE_PREFIX_PATH}/lib"
)

install(TARGETS qwen38-native RUNTIME DESTINATION bin)

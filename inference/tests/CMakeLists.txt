# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_executable(test_gemv test_gemv.cpp)
target_link_libraries(test_gemv PRIVATE therock-inference hip::host)
target_include_directories(test_gemv PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
add_test(NAME test_gemv COMMAND test_gemv)

add_executable(test_flash_attn test_flash_attn.cpp)
target_link_libraries(test_flash_attn PRIVATE therock-inference hip::host)
target_include_directories(test_flash_attn PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
add_test(NAME test_flash_attn COMMAND test_flash_attn)

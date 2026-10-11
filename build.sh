#!/usr/bin/env bash

set -euo pipefail

therock_dir=${THEROCK_DIR:-/home/alex/TheRock}
sdk_dir=${THEROCK_SDK_DIR:-$therock_dir/build/dist/rocm}
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir=${QWEN_NATIVE_BUILD_DIR:-$source_dir/build}

cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$sdk_dir" \
    -DCMAKE_CXX_COMPILER="$sdk_dir/bin/hipcc"
cmake --build "$build_dir" --parallel

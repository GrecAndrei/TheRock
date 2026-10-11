#!/usr/bin/env bash
set -euo pipefail
workspace_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
check_dir=$(mktemp -d "$workspace_dir/results/host-checks-XXXXXX")
export CARGO_TARGET_DIR="$workspace_dir/build/kyanite-host"
export CARGO_BUILD_JOBS=2
host_cxx=${QWEN_HOST_CXX:-clang++}
printf 'Host-only checks; evidence directory: %s\n' "$check_dir"
python3 "$workspace_dir/scripts/doctor.py" | tee "$check_dir/doctor.json"
cd "$workspace_dir/sources/kyanite"
cargo fmt --all -- --check 2>&1 | tee "$check_dir/cargo-fmt.log"
cargo clippy --offline --locked --workspace --all-targets --all-features -- -D warnings \
    2>&1 | tee "$check_dir/cargo-clippy.log"
cargo test --offline --locked --workspace --all-features \
    2>&1 | tee "$check_dir/cargo-test.log"
native_dir="$workspace_dir/sources/native-baseline"
native_build="$workspace_dir/build/native-host"
mkdir -p "$native_build"
for test_name in runtime_options attention_tile single_job_worker; do
    "$host_cxx" -std=c++20 -O2 -pthread -Wall -Wextra -Wpedantic -Werror \
        -I "$native_dir/src" "$native_dir/tests/${test_name}_test.cpp" \
        -o "$native_build/${test_name}_test" \
        2>&1 | tee "$check_dir/${test_name}-build.log"
    "$native_build/${test_name}_test" 2>&1 | tee "$check_dir/${test_name}.log"
done
"$host_cxx" -std=c++20 -O2 -pthread -march=native -mavx2 -mfma -mf16c \
    -Wall -Wextra -Wpedantic -Werror -I "$native_dir/src" \
    "$native_dir/src/cpu_matvec.cpp" "$native_dir/src/cpu_matvec_test.cpp" \
    "$native_dir/src/gguf.cpp" -o "$native_build/cpu_matvec_test" \
    2>&1 | tee "$check_dir/cpu_matvec-build.log"
"$native_build/cpu_matvec_test" 2>&1 | tee "$check_dir/cpu_matvec.log"
cd "$native_dir"
python3 -m unittest discover -s tests -p 'test_*.py' \
    2>&1 | tee "$check_dir/python-lifecycle.log"
printf 'Host-only checks completed successfully. No inference TPS measured.\n'

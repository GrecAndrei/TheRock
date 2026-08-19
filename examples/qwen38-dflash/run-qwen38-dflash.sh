#!/usr/bin/env bash

set -euo pipefail

llama_cpp_dir=${LLAMA_CPP_DIR:-/home/alex/llama.cpp}
therock_dir=${THEROCK_DIR:-/home/alex/TheRock}
downloads_dir=${DOWNLOADS_DIR:-/home/alex/Downloads}

runner=${LLAMA_RUNNER:-$llama_cpp_dir/build-dflash/bin/llama-speculative-simple}
target_model=${TARGET_MODEL:-$downloads_dir/Qwen3.8-27B-Q4_K_M.gguf}
draft_model=${DRAFT_MODEL:-$downloads_dir/Qwen3.8-27B-DFlash2-Q4_K_M.gguf}

for required_file in "$runner" "$target_model" "$draft_model"; do
    if [[ ! -e "$required_file" ]]; then
        printf 'missing required file: %s\n' "$required_file" >&2
        exit 1
    fi
done

if pgrep -x baloo_file >/dev/null 2>&1; then
    printf '%s\n' 'warning: KDE Baloo is running; large-file indexing can cut inference throughput substantially.' >&2
fi

prompt=${1:-Write a compact C function that computes the greatest common divisor of two positive integers, then explain why it works.}
if (( $# > 0 )); then
    shift
fi

export LD_LIBRARY_PATH="$llama_cpp_dir/build-dflash/bin:$therock_dir/build/dist/rocm/lib:$therock_dir/build/profiler/roctracer/dist/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$runner" \
    -m "$target_model" \
    -md "$draft_model" \
    --spec-type draft-dflash \
    --spec-draft-n-max 7 \
    -ngl 20 \
    -ngld 99 \
    -c "${CONTEXT_SIZE:-1024}" \
    -b 512 \
    -ub 512 \
    -t 16 \
    -td 16 \
    -fa on \
    -n "${N_PREDICT:-256}" \
    --temp 0 \
    --top-k 1 \
    --seed 42 \
    -p "$prompt" \
    "$@"

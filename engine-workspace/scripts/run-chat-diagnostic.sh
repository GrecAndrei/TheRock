#!/usr/bin/env bash
# This bounded request diagnostic is separate from the final 128K protocol.
set -euo pipefail
workspace_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
context_capacity=${1:-131072}
resident_gib=${2:-1.0}
timeout_seconds=${3:-360}
prefill_chunk=${QWEN_NATIVE_PREFILL_CHUNK:-512}
pin_model=${QWEN_NATIVE_PIN_MODEL:-0}
binary=${QWEN_NATIVE_DIAGNOSTIC_BINARY:-$workspace_dir/build/native-engine/qwen38-native}
target=/home/alex/Downloads/Qwen3.8-27B-UD-Q4_K_M.gguf
draft=/home/alex/Downloads/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
input="$workspace_dir/fixtures/real-chat-input.txt"
for path in "$binary" "$target" "$draft" "$input"; do
    test -f "$path"
done
result_dir=$(mktemp -d "$workspace_dir/results/native-real-chat-XXXXXX")
printf 'Request diagnostic, no 128K certification: %s\n' "$result_dir"
python3 "$workspace_dir/scripts/doctor.py" --require-gpu | tee "$result_dir/doctor.json"
sha256sum "$binary" "$target" "$draft" "$input" | tee "$result_dir/inputs.sha256"
printf '%q ' timeout "$timeout_seconds" env \
    QWEN_NATIVE_HYBRID=1 QWEN_NATIVE_CPU_THREADS=20 \
    QWEN_NATIVE_DEVICE_CHECKPOINT=1 QWEN_NATIVE_CPU_Q6_PREDECODE=1 \
    QWEN_NATIVE_CPU_IQ4_TILE_GEMM=0 QWEN_NATIVE_CPU_IQ4_PREDECODE=0 \
    QWEN_NATIVE_CPU_K_TILE_GEMM=0 QWEN_NATIVE_ASYNC_ACTIVATIONS=0 \
    "QWEN_NATIVE_PREFILL_CHUNK=$prefill_chunk" "QWEN_NATIVE_PIN_MODEL=$pin_model" \
    QWEN_NATIVE_KV=q8 QWEN_NATIVE_Q8_TILED=warp16 \
    "$binary" chat "$target" "$draft" "$context_capacity" "$resident_gib" \
    | tee "$result_dir/command.txt"
printf '\n' | tee -a "$result_dir/command.txt"
date --iso-8601=seconds | tee "$result_dir/start.txt"
# Capture both pipeline statuses before restoring errexit, including timeouts.
set +e
timeout "$timeout_seconds" env \
    QWEN_NATIVE_HYBRID=1 QWEN_NATIVE_CPU_THREADS=20 \
    QWEN_NATIVE_DEVICE_CHECKPOINT=1 QWEN_NATIVE_CPU_Q6_PREDECODE=1 \
    QWEN_NATIVE_CPU_IQ4_TILE_GEMM=0 QWEN_NATIVE_CPU_IQ4_PREDECODE=0 \
    QWEN_NATIVE_CPU_K_TILE_GEMM=0 QWEN_NATIVE_ASYNC_ACTIVATIONS=0 \
    "QWEN_NATIVE_PREFILL_CHUNK=$prefill_chunk" "QWEN_NATIVE_PIN_MODEL=$pin_model" \
    QWEN_NATIVE_KV=q8 QWEN_NATIVE_Q8_TILED=warp16 \
    "$binary" chat "$target" "$draft" "$context_capacity" "$resident_gib" \
    < "$input" 2>&1 | tee "$result_dir/chat.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
printf 'timeout_wrapper_exit=%s\nlog_writer_exit=%s\n' \
    "${pipeline_status[0]}" "${pipeline_status[1]}" | tee "$result_dir/exit.txt"
date --iso-8601=seconds | tee "$result_dir/end.txt"
printf 'Wrapper exit is recorded; native child signal is not independently journaled.\n'
if [ "${pipeline_status[1]}" -ne 0 ]; then
    exit "${pipeline_status[1]}"
fi
exit "${pipeline_status[0]}"

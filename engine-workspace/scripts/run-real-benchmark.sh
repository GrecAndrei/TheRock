#!/usr/bin/env bash
set -euo pipefail
workspace_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
label=${1:-baseline}
if [[ ! "$label" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    printf 'Benchmark label must contain letters, numbers, underscores or hyphens.\n' >&2
    exit 2
fi
outputs=${2:-256}
binary=${QWEN_NATIVE_DIAGNOSTIC_BINARY:-$workspace_dir/build/native-engine/qwen38-native}
trial_dir=$(mktemp -d "$workspace_dir/results/decode-${label}-XXXXXX")
cp "$binary" "$trial_dir/executable"
tar -C "$workspace_dir/sources/native-engine" --exclude=.git --exclude=__pycache__ \
    --exclude='*.pyc' -cf "$trial_dir/source.tar" src CMakeLists.txt
sha256sum "$trial_dir/executable" "$trial_dir/source.tar" \
    "$workspace_dir/fixtures/real-benchmark-prompt.txt" | tee "$trial_dir/inputs.sha256"
controls=(
    QWEN_NATIVE_HYBRID=1 QWEN_NATIVE_CPU_THREADS=20 QWEN_NATIVE_DEVICE_CHECKPOINT=1
    QWEN_NATIVE_CPU_Q6_PREDECODE=1 QWEN_NATIVE_CPU_IQ4_TILE_GEMM=0
    QWEN_NATIVE_CPU_IQ4_PREDECODE=0 QWEN_NATIVE_CPU_K_TILE_GEMM=0
    QWEN_NATIVE_ASYNC_ACTIVATIONS=0 QWEN_NATIVE_KV=q8 QWEN_NATIVE_Q8_TILED=warp16
    QWEN_NATIVE_PIN_MODEL=0
    "QWEN_NATIVE_PINNED_STAGING=${QWEN_NATIVE_PINNED_STAGING:-0}"
    "QWEN_NATIVE_PINNED_STAGING_COARSE=${QWEN_NATIVE_PINNED_STAGING_COARSE:-0}"
)
# Capture explicit controls above, then prevent an inherited experimental flag
# (including presence-only flags) from silently changing the next workload.
unset_controls=()
while IFS= read -r control_name; do
    case "$control_name" in
        QWEN_NATIVE_*) unset_controls+=(-u "$control_name") ;;
    esac
done < <(compgen -e)
command=(env "${unset_controls[@]}" "${controls[@]}" "$trial_dir/executable" benchmark-real
    /home/alex/Downloads/Qwen3.8-27B-UD-Q4_K_M.gguf
    /home/alex/Downloads/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
    "$workspace_dir/fixtures/real-benchmark-prompt.txt" 131072 "$outputs" 512 1.0)
printf '%q ' "${command[@]}" | tee "$trial_dir/command.txt"
printf '\n' | tee -a "$trial_dir/command.txt"
date --iso-8601=seconds | tee "$trial_dir/start.txt"
printf 'Real benchmark evidence: %s\n' "$trial_dir"
set +e
timeout 360s "${command[@]}" 2>&1 | tee "$trial_dir/result.log"
statuses=("${PIPESTATUS[@]}")
set -e
printf 'wrapper_exit=%s\nlogger_exit=%s\n' "${statuses[0]}" "${statuses[1]}" \
    | tee "$trial_dir/exit.txt"
date --iso-8601=seconds | tee "$trial_dir/end.txt"
if [ "${statuses[1]}" -ne 0 ]; then exit "${statuses[1]}"; fi
exit "${statuses[0]}"

#!/usr/bin/env bash
set -euo pipefail
workspace_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
git -C "$workspace_dir/sources/kyanite" fetch --quiet origin
git -C "$workspace_dir/references/llama.cpp" fetch --quiet origin
git -C "$workspace_dir/sources/kyanite" remote set-head origin -a
git -C "$workspace_dir/references/llama.cpp" remote set-head origin -a
printf 'Upstream refs fetched; pinned checkouts and pending edits preserved.\n'
git -C "$workspace_dir/sources/kyanite" log -1 --format='%H %s' origin/HEAD
git -C "$workspace_dir/references/llama.cpp" log -1 --format='%H %s' origin/HEAD

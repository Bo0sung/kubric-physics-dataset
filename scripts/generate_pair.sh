#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE_ID="${1:-scene_000001}"
SEED="${2:-100}"
INTERVENTION="${3:-gravity_scale}"
START="${4:-12}"
END="${5:-18}"
shift $(( $# > 5 ? 5 : $# )) || true
if (( $# > 0 )); then
  DOSE=("$@")
else
  DOSE=(0.5)
fi

IMAGE="${KUBRIC_IMAGE:-kubric-physics-dataset:latest}"
GPU_ID="${GPU_ID:-0}"
RENDER_DEVICE="${RENDER_DEVICE:-CUDA}"
OUTPUT_ROOT="outputs/$SCENE_ID"
COMMON=(--seed "$SEED" --frames 48 --fps 24 --step-rate 240 --resolution 256 --samples-per-pixel 32 --render-device "$RENDER_DEVICE")
DOCKER_GPU_ARGS=()
DOCKER_USER_ARGS=()
if [[ "${USE_GPU:-1}" == "1" ]]; then
  DOCKER_GPU_ARGS=(--gpus "device=$GPU_ID")
fi
if ! docker info --format '{{json .SecurityOptions}}' | grep -q 'rootless'; then
  DOCKER_USER_ARGS=(--user "$(id -u):$(id -g)")
  echo "Docker mode: rootful (mapping container writes to $(id -u):$(id -g))"
else
  echo "Docker mode: rootless (container root maps to the current host user)"
fi
mkdir -p "$ROOT/$OUTPUT_ROOT"

run_worker() {
  docker run --rm "${DOCKER_GPU_ARGS[@]}" \
    "${DOCKER_USER_ARGS[@]}" \
    -e PYTHONPATH=/workspace \
    -v "$ROOT:/workspace" \
    "$IMAGE" /usr/bin/python3 workers/freefall_worker.py "$@"
}

run_worker --variant normal --output-dir "$OUTPUT_ROOT/normal" "${COMMON[@]}"
run_worker --variant violation --output-dir "$OUTPUT_ROOT/violation" "${COMMON[@]}" \
  --intervention "$INTERVENTION" --violation-start "$START" --violation-end "$END" --dose "${DOSE[@]}"

docker run --rm "${DOCKER_USER_ARGS[@]}" \
  -e PYTHONPATH=/workspace -v "$ROOT:/workspace" "$IMAGE" \
  /usr/bin/python3 tools/compute_pair_metrics.py \
  --normal "$OUTPUT_ROOT/normal/state.npz" \
  --violation "$OUTPUT_ROOT/violation/state.npz" \
  --violation-start "$START" \
  --output "$OUTPUT_ROOT/pair_metrics.json"

docker run --rm "${DOCKER_USER_ARGS[@]}" \
  -e PYTHONPATH=/workspace -v "$ROOT:/workspace" "$IMAGE" \
  /usr/bin/python3 tools/validate_pair.py "$OUTPUT_ROOT" --violation-start "$START"

echo "Generated $ROOT/$OUTPUT_ROOT"

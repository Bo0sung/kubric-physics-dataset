#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COUNT="${1:-10}"
START_SEED="${2:-1000}"
INTERVENTION="${3:-gravity_scale}"
START_FRAME="${4:-12}"
END_FRAME="${5:-18}"
shift $(( $# > 5 ? 5 : $# )) || true
if (( $# > 0 )); then
  DOSE=("$@")
else
  DOSE=(0.5)
fi

if ! [[ "$COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "COUNT must be a positive integer" >&2
  exit 2
fi

for ((index=0; index<COUNT; index++)); do
  seed=$((START_SEED + index))
  scene_id="$(printf 'scene_%06d' "$seed")"
  GPU_ID="${GPU_ID:-0}" \
    KUBRIC_IMAGE="${KUBRIC_IMAGE:-kubric-physics-dataset:latest}" \
    USE_GPU="${USE_GPU:-1}" \
    RENDER_DEVICE="${RENDER_DEVICE:-CUDA}" \
    bash "$ROOT/scripts/generate_pair.sh" \
      "$scene_id" "$seed" "$INTERVENTION" "$START_FRAME" "$END_FRAME" "${DOSE[@]}"
done

echo "Generated $COUNT paired scenes under $ROOT/outputs"

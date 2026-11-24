#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing coco/train2017}"
MODEL="${MODEL:-qwen3}"
BITS="${BITS:-4}"
SCALE_PATH="${SCALE_PATH:-runs/${MODEL}_W4A${BITS}/scales.npz}"
if [[ -e "$SCALE_PATH" || -e "${SCALE_PATH%.npz}.json" ]]; then
  echo "Refusing to overwrite scales: $SCALE_PATH" >&2; exit 1
fi
args=(--model "$MODEL" --bits "$BITS" --data-root "$DATA_ROOT" --layers "${LAYERS:-0:36}" --output "$SCALE_PATH")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "$CHECKPOINT")
[[ -z "${CALIB_MANIFEST:-}" ]] || args+=(--manifest "$CALIB_MANIFEST")
exec "${PYTHON:-python}" main_quant.py "${args[@]}" "$@"

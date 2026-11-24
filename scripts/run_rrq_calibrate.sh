#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing coco/train2017}"
MODEL="${MODEL:-qwen3_vl_8b}"
W_BITS="${W_BITS:-4}"
A_BITS="${A_BITS:-4}"
SCALE_PATH="${SCALE_PATH:-runs/${MODEL}_W${W_BITS}A${A_BITS}/scales.npz}"
if [[ -e "$SCALE_PATH" || -e "${SCALE_PATH%.npz}.json" ]]; then
  echo "Refusing to overwrite scales: $SCALE_PATH" >&2; exit 1
fi
args=(--model "$MODEL" --w-bits "$W_BITS" --a-bits "$A_BITS" --data-root "$DATA_ROOT" --layers "${LAYERS:-all}" --output "$SCALE_PATH")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "$CHECKPOINT")
[[ -z "${CALIB_MANIFEST:-}" ]] || args+=(--manifest "$CALIB_MANIFEST")
exec "${PYTHON:-python}" main_quant.py "${args[@]}" "$@"

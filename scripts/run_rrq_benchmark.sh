#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${COST_MANIFEST:?Set COST_MANIFEST to the fixed-input JSONL file}"
MODEL="${MODEL:-qwen3_vl_8b}"
W_BITS="${W_BITS:-4}"
A_BITS="${A_BITS:-4}"
args=(--model "$MODEL" --w-bits "$W_BITS" --a-bits "$A_BITS" --manifest "$COST_MANIFEST" --data-root "$DATA_ROOT" --output "${OUTPUT_PATH:-runs/${MODEL}_W${W_BITS}A${A_BITS}/cost}" --tokens "${TOKENS:-128}" --repeats "${REPEATS:-3}")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "$CHECKPOINT")
[[ -z "${SCALE_PATH:-}" ]] || args+=(--scales "$SCALE_PATH")
exec "${PYTHON:-python}" scripts/benchmark.py "${args[@]}" "$@"

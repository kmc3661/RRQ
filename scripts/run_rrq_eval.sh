#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${EVAL_MANIFEST:?Set EVAL_MANIFEST to a task-formatted JSONL file}"
MODEL="${MODEL:-qwen3_vl_8b}"
W_BITS="${W_BITS:-4}"
A_BITS="${A_BITS:-4}"
args=(--model "$MODEL" --w-bits "$W_BITS" --a-bits "$A_BITS" --manifest "$EVAL_MANIFEST" --data-root "$DATA_ROOT" --output "${OUTPUT_PATH:-runs/${MODEL}_W${W_BITS}A${A_BITS}/responses.jsonl}" --backend "${BACKEND:-accuracy}")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "$CHECKPOINT")
[[ -z "${SCALE_PATH:-}" ]] || args+=(--scales "$SCALE_PATH")
exec "${PYTHON:-python}" scripts/evaluate.py "${args[@]}" "$@"

#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${EVAL_MANIFEST:?Set EVAL_MANIFEST to a task-formatted JSONL file}"
MODEL="${MODEL:-qwen3}"
BITS="${BITS:-4}"
args=(--model "$MODEL" --bits "$BITS" --manifest "$EVAL_MANIFEST" --data-root "$DATA_ROOT" --output "${OUTPUT_PATH:-runs/${MODEL}_W4A${BITS}/responses.jsonl}" --backend "${BACKEND:-accuracy}")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "$CHECKPOINT")
[[ -z "${SCALE_PATH:-}" ]] || args+=(--scales "$SCALE_PATH")
exec "${PYTHON:-python}" scripts/evaluate.py "${args[@]}" "$@"

#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${COST_MANIFEST:?Set COST_MANIFEST to the fixed-input JSONL file}"
MODEL="${MODEL:-qwen3}"
BITS="${BITS:-4}"
args=(--model "$MODEL" --bits "$BITS" --manifest "$COST_MANIFEST" --data-root "$DATA_ROOT" --output "${OUTPUT_PATH:-runs/${MODEL}_W4A${BITS}/cost}" --tokens "${TOKENS:-128}" --repeats "${REPEATS:-3}")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "$CHECKPOINT")
[[ -z "${SCALE_PATH:-}" ]] || args+=(--scales "$SCALE_PATH")
exec "${PYTHON:-python}" scripts/benchmark.py "${args[@]}" "$@"

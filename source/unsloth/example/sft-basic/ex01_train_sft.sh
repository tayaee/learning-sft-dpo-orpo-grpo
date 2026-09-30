#!/usr/bin/env bash
# ex01_train_sft.sh — step 01: quick SFT test with typical defaults.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_MODEL="${BASE_MODEL:-LiquidAI/LFM2.5-1.2B-Instruct}"
DATASET="${DATASET:-mlabonne/FineTome-100k}"
OUTPUT_REPO="${OUTPUT_REPO:-tayaee/my-model-test}"
MAX_STEPS="${MAX_STEPS:-60}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
EVAL_SPLIT="${EVAL_SPLIT:-0.0}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"

ARGS=(--base-model "$BASE_MODEL" --dataset "$DATASET" --output-repo "$OUTPUT_REPO"
  --eval-split "$EVAL_SPLIT" --num-samples "$NUM_SAMPLES")
if [[ -n "$NUM_EPOCHS" ]]; then
  ARGS+=(--num-epochs "$NUM_EPOCHS")
else
  ARGS+=(--max-steps "$MAX_STEPS")
fi

exec uv run ./ex01_train_sft.py "${ARGS[@]}" "$@"

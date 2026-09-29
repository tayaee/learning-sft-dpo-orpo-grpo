#!/usr/bin/env bash
# run-epochs.sh — Unsloth SFT 예제의 epoch 기반 학습 wrapper.
# 원본 명령 (unsloth_sft_example.py docstring):
#   uv run unsloth_sft_example.py --dataset mlabonne/FineTome-100k \
#       --num-epochs 1 --eval-split 0.2 --output-repo your-username/model-finetuned
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_MODEL="${BASE_MODEL:-LiquidAI/LFM2.5-1.2B-Instruct}"
DATASET="${DATASET:-mlabonne/FineTome-100k}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
EVAL_SPLIT="${EVAL_SPLIT:-0.2}"
OUTPUT_REPO="${OUTPUT_REPO:?OUTPUT_REPO 미지정 (예: OUTPUT_REPO=your-username/model-finetuned $0)}"

exec uv run ./unsloth_sft_example.py \
  --base-model "$BASE_MODEL" \
  --dataset "$DATASET" \
  --num-epochs "$NUM_EPOCHS" \
  --eval-split "$EVAL_SPLIT" \
  --output-repo "$OUTPUT_REPO" \
  "$@"

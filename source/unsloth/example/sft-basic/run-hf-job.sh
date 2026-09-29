#!/usr/bin/env bash
# run-hf-job.sh — Unsloth SFT 예제의 HF Jobs 실행 wrapper (1 epoch + eval).
# 원본 명령 (unsloth_sft_example.py docstring):
#   hf jobs uv run unsloth_sft_example.py \
#       --flavor a10g-small --secrets HF_TOKEN --timeout 4h \
#       -- --dataset mlabonne/FineTome-100k --num-epochs 1 \
#          --eval-split 0.2 --output-repo your-username/model-finetuned
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

FLAVOR="${FLAVOR:-a10g-small}"
TIMEOUT="${TIMEOUT:-4h}"
BASE_MODEL="${BASE_MODEL:-LiquidAI/LFM2.5-1.2B-Instruct}"
DATASET="${DATASET:-mlabonne/FineTome-100k}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
EVAL_SPLIT="${EVAL_SPLIT:-0.2}"
OUTPUT_REPO="${OUTPUT_REPO:?OUTPUT_REPO 미지정 (예: OUTPUT_REPO=your-username/model-finetuned $0)}"

exec hf jobs uv run ./unsloth_sft_example.py \
  --flavor "$FLAVOR" --secrets HF_TOKEN --timeout "$TIMEOUT" \
  -- --base-model "$BASE_MODEL" \
     --dataset "$DATASET" \
     --num-epochs "$NUM_EPOCHS" \
     --eval-split "$EVAL_SPLIT" \
     --output-repo "$OUTPUT_REPO" \
     "$@"

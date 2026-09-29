#!/usr/bin/env bash
# run-steps.sh — Unsloth SFT 예제의 step 기반 학습 wrapper (빠른 테스트용).
# 원본 명령 (unsloth_sft_example.py docstring):
#   uv run unsloth_sft_example.py --dataset mlabonne/FineTome-100k \
#       --max-steps 500 --output-repo your-username/model-finetuned
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_MODEL="${BASE_MODEL:-LiquidAI/LFM2.5-1.2B-Instruct}"
DATASET="${DATASET:-mlabonne/FineTome-100k}"
MAX_STEPS="${MAX_STEPS:-500}"
OUTPUT_REPO="${OUTPUT_REPO:?OUTPUT_REPO 미지정 (예: OUTPUT_REPO=your-username/model-test $0)}"

exec uv run ./unsloth_sft_example.py \
  --base-model "$BASE_MODEL" \
  --dataset "$DATASET" \
  --max-steps "$MAX_STEPS" \
  --output-repo "$OUTPUT_REPO" \
  "$@"

#!/usr/bin/env bash
# 23-postprocess.sh [mini|full] — Stage 2c wrapper. 생성 CSV 후처리→SFT JSONL.
# 입력: generated-<mode>.csv → 출력: synthetic-<mode>.jsonl
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
p5_log "in=$P5_DATASETS/generated-$MODE.csv out=$P5_DATASETS/synthetic-$MODE.jsonl"
uv run "$P5_ROOT/23-postprocess.py" --mode "$MODE"

echo ---- result ----
ls -lh "$P5_DATASETS/synthetic-$MODE.jsonl"
wc -l "$P5_DATASETS/synthetic-$MODE.jsonl"

#!/usr/bin/env bash
# 51-quant-qlora-fp8.sh [mini|full] [calib] — Stage 5b wrapper. QLoRA-merged→FP8 static quant.
# 입력: synthetic-qlora-<mode>-single-merged → 출력: ...-merged-fp8
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
CALIB="${2:-$CALIB_N}"
: "${CALIB_FILE:=$P5_DATASETS/gsm8k-calibration-256.jsonl}"
OUT="$P5_MODELS/synthetic-qlora-$MODE-single-merged-fp8"
p5_log "calib=$CALIB src=$CALIB_FILE out=$OUT"
(set -x; uv run "$P5_ROOT/51-quant-qlora-fp8.py" --mode "$MODE" --calib "$CALIB" --calib-file "$CALIB_FILE")

echo ---- result ----
ls -l "$OUT"

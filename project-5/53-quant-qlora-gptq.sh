#!/usr/bin/env bash
# 53-quant-qlora-gptq.sh [mini|full] [calib] — Stage 5d wrapper. QLoRA-merged→GPTQ 4bit-g128.
# 입력: synthetic-qlora-<mode>-single-merged → 출력: ...-merged-gptq
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
CALIB="${2:-$CALIB_N}"
: "${CALIB_FILE:=$P5_DATASETS/gsm8k-calibration-256.jsonl}"
OUT="$P5_MODELS/synthetic-qlora-$MODE-single-merged-gptq"
p5_log "calib=$CALIB src=$CALIB_FILE out=$OUT"
uv run "$P5_ROOT/53-quant-qlora-gptq.py" --mode "$MODE" --calib "$CALIB" --calib-file "$CALIB_FILE"

echo ---- result ----
ls -l "$OUT"

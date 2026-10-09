#!/usr/bin/env bash
# 40-quant-fft-awq.sh [mini|full] [calib] — Stage 4a wrapper. FFT→AWQ 4bit.
# 입력: synthetic-fft-<mode>-single → 출력: synthetic-fft-<mode>-single-awq
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
CALIB="${2:-$CALIB_N}"
: "${CALIB_FILE:=$P5_DATASETS/gsm8k-calibration-256.jsonl}"
OUT="$P5_MODELS/synthetic-fft-$MODE-single-awq"
p5_log "calib=$CALIB src=$CALIB_FILE out=$OUT"
uv run "$P5_ROOT/40-quant-fft-awq.py" --mode "$MODE" --calib "$CALIB" --calib-file "$CALIB_FILE"

echo ---- result ----
ls -l "$OUT"

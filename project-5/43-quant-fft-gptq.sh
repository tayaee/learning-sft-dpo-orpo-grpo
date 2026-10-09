#!/usr/bin/env bash
# 43-quant-fft-gptq.sh [mini|full] [calib] — Stage 4d wrapper. FFT→GPTQ 4bit-g128.
# 입력: synthetic-fft-<mode>-single → 출력: synthetic-fft-<mode>-single-gptq
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
CALIB="${2:-$CALIB_N}"
: "${CALIB_FILE:=$P5_DATASETS/gsm8k-calibration-256.jsonl}"
OUT="$P5_MODELS/synthetic-fft-$MODE-single-gptq"
p5_log "calib=$CALIB src=$CALIB_FILE out=$OUT"
(set -x; uv run "$P5_ROOT/43-quant-fft-gptq.py" --mode "$MODE" --calib "$CALIB" --calib-file "$CALIB_FILE")

echo ---- result ----
ls -l "$OUT"

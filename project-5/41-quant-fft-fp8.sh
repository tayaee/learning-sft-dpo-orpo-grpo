#!/usr/bin/env bash
# 41-quant-fft-fp8.sh [mini|full] [calib] — Stage 4b wrapper. FFT→FP8 static quant.
# 입력: synthetic-fft-<mode>-single → 출력: synthetic-fft-<mode>-single-fp8
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
CALIB="${2:-$CALIB_N}"
OUT="$P5_MODELS/synthetic-fft-$MODE-single-fp8"
p5_log "calib=$CALIB out=$OUT"
uv run "$P5_ROOT/41-quant-fft-fp8.py" --mode "$MODE" --calib "$CALIB"

echo ---- result ----
ls -l "$OUT"

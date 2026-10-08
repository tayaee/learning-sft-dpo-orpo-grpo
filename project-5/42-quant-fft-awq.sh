#!/usr/bin/env bash
# 42-quant-fft-awq.sh [mini|full] [calib] — Stage 4c wrapper. FFT→AWQ 4bit.
# 입력: synthetic-fft-<mode>-single → 출력: synthetic-fft-<mode>-single-awq
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
CALIB="${2:-$CALIB_N}"
OUT="$P5_MODELS/synthetic-fft-$MODE-single-awq"
p5_log "calib=$CALIB out=$OUT"
uv run "$P5_ROOT/42-quant-fft-awq.py" --mode "$MODE" --calib "$CALIB"

echo ---- result ----
ls -l "$OUT"

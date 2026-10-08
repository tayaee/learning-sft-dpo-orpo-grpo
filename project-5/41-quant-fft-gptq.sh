#!/usr/bin/env bash
# 41-quant-fft-gptq.sh [mini|full] [calib] — Stage 4b wrapper. FFT→GPTQ 4bit-g128.
# 입력: synthetic-fft-<mode>-single → 출력: synthetic-fft-<mode>-single-gptq
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
CALIB="${2:-$CALIB_N}"
OUT="$P5_MODELS/synthetic-fft-$MODE-single-gptq"
p5_log "calib=$CALIB out=$OUT"
uv run "$P5_ROOT/41-quant-fft-gptq.py" --mode "$MODE" --calib "$CALIB"

echo ---- result ----
ls -l "$OUT"

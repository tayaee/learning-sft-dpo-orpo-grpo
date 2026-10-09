#!/usr/bin/env bash
# 40-quant-fft-awq.sh [mini|full] [calib] — Stage 4a wrapper. FFT→AWQ 4bit.
# 입력: synthetic-fft-<mode>-single → 출력: synthetic-fft-<mode>-single-awq
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
CALIB="${2:-$CALIB_N}"
: "${CALIB_FILE:=$P5_DATASETS/gsm8k-calibration-256.jsonl}"
SRC="$P5_MODELS/synthetic-fft-$MODE-single"
OUT="$P5_MODELS/synthetic-fft-$MODE-single-awq"
p5_require "$SRC/config.json" "$CALIB_FILE"
if [ -f "$OUT/config.json" ] && p5_fresh "$OUT" "$SRC" "$CALIB_FILE"; then
  p5_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -l "$OUT")
  exit 0
fi
p5_log "calib=$CALIB src=$CALIB_FILE out=$OUT"
(set -x; uv run "$P5_ROOT/40-quant-fft-awq.py" --mode "$MODE" --calib "$CALIB" --calib-file "$CALIB_FILE")

echo ---- result ----
(set -x; ls -l "$OUT")

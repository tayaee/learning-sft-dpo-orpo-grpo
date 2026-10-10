#!/usr/bin/env bash
# 43-quant-fft-gptq.sh [mini|full] [calib] (STRAT env, 기본 single) — Stage 4d wrapper. FFT→GPTQ 4bit-g128.
# 입력: synthetic-fft-<mode>-$STRAT → 출력: synthetic-fft-<mode>-$STRAT-gptq
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
STRAT="${STRAT:-single}"
case "$STRAT" in single|ddp|fsdp) ;; *) echo "STRAT must be single|ddp|fsdp" >&2; exit 1;; esac
CALIB="${2:-$CALIB_N}"
: "${CALIB_FILE:=$P5_DATASETS/gsm8k-calibration-256.jsonl}"
SRC="$P5_MODELS/synthetic-fft-$MODE-$STRAT"
OUT="$P5_MODELS/synthetic-fft-$MODE-$STRAT-gptq"
p5_require "$SRC/config.json" "$CALIB_FILE"
if [ -f "$OUT/config.json" ] && p5_fresh "$OUT/config.json" "$SRC/config.json" "$CALIB_FILE"; then
  p5_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -l "$OUT")
  exit 0
fi
p5_log "calib=$CALIB src=$CALIB_FILE out=$OUT"
(set -x; uv run "$P5_ROOT/43-quant-fft-gptq.py" --mode "$MODE" --strat "$STRAT" --calib "$CALIB" --calib-file "$CALIB_FILE")

echo ---- result ----
(set -x; ls -l "$OUT")

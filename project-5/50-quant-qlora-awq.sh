#!/usr/bin/env bash
# 50-quant-qlora-awq.sh [mini|full] [calib] — Stage 5a wrapper. QLoRA-merged→AWQ 4bit.
# 입력: synthetic-qlora-<mode>-single-merged → 출력: ...-merged-awq
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
CALIB="${2:-$CALIB_N}"
: "${CALIB_FILE:=$P5_DATASETS/gsm8k-calibration-256.jsonl}"
SRC="$P5_MODELS/synthetic-qlora-$MODE-single-merged"
OUT="$P5_MODELS/synthetic-qlora-$MODE-single-merged-awq"
p5_require "$SRC/config.json" "$CALIB_FILE"
if [ -f "$OUT/config.json" ] && p5_fresh "$OUT" "$SRC" "$CALIB_FILE"; then
  p5_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -l "$OUT")
  exit 0
fi
p5_log "calib=$CALIB src=$CALIB_FILE out=$OUT"
(set -x; uv run "$P5_ROOT/50-quant-qlora-awq.py" --mode "$MODE" --calib "$CALIB" --calib-file "$CALIB_FILE")

echo ---- result ----
(set -x; ls -l "$OUT")

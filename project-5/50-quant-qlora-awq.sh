#!/usr/bin/env bash
# 50-quant-qlora-awq.sh [mini|full] [calib] — Stage 5a wrapper. QLoRA-merged→AWQ 4bit.
# 입력: synthetic-qlora-<mode>-single-merged → 출력: ...-merged-awq
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
CALIB="${2:-$CALIB_N}"
OUT="$P5_MODELS/synthetic-qlora-$MODE-single-merged-awq"
p5_log "calib=$CALIB out=$OUT"
uv run "$P5_ROOT/50-quant-qlora-awq.py" --mode "$MODE" --calib "$CALIB"

echo ---- result ----
ls -l "$OUT"

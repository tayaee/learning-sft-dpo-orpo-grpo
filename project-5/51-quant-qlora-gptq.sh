#!/usr/bin/env bash
# 51-quant-qlora-gptq.sh [mini|full] [calib] — Stage 5b wrapper. QLoRA-merged→GPTQ 4bit-g128.
# 입력: synthetic-qlora-<mode>-single-merged → 출력: ...-merged-gptq
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
CALIB="${2:-$CALIB_N}"
OUT="$P5_MODELS/synthetic-qlora-$MODE-single-merged-gptq"
p5_log "calib=$CALIB out=$OUT"
uv run "$P5_ROOT/51-quant-qlora-gptq.py" --mode "$MODE" --calib "$CALIB"

echo ---- result ----
ls -l "$OUT"

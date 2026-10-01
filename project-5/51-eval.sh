#!/usr/bin/env bash
# 51-eval.sh [mini|full] [target...] — Stage 5b wrapper. target 기본 all.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
TARGETS="${2:-all}"
[ "$TARGETS" = all ] && TARGETS="base fft qlora gptq awq fp8"
N="$EVAL_N"  # mini 10 / full 0=전체 (common.env)
p5_log "tp=$TP targets=$TARGETS n=$N"
for t in $TARGETS; do
  uv run "$P5_ROOT/51-eval.py" --mode "$MODE" --target "$t" --n "$N" --tp "$TP"
done

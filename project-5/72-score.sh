#!/usr/bin/env bash
# 72-score.sh [mini|full] — Stage 7c wrapper. eval-<mode>/*.jsonl 채점표 출력.
# (72-score.py 자체가 결과표를 stdout에 찍는다.)
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
p5_log "evaldir=$P5_OUTPUTS/eval-$MODE"
uv run "$P5_ROOT/72-score.py" --mode "$MODE"

echo ---- result ----
ls -l "$P5_OUTPUTS/eval-$MODE/"

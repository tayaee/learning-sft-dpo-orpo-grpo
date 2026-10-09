#!/usr/bin/env bash
# 73-score.sh [mini|full] — Stage 7c wrapper. eval-<mode>/*.jsonl 채점표 출력.
# (73-score.py 자체가 결과표를 stdout에 찍고 score.json 아티팩트를 남긴다.)
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
EVALDIR="$P5_OUTPUTS/eval-$MODE"
SCORE="$EVALDIR/score.json"
p5_require "$EVALDIR"
_inputs=()
for _f in "$EVALDIR"/*.jsonl; do [ -e "$_f" ] && _inputs+=("$_f"); done
if [ -f "$SCORE" ] && [ "${#_inputs[@]}" -gt 0 ] && p5_fresh "$SCORE" "${_inputs[@]}"; then
  p5_log "skip: fresh $SCORE (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; cat "$SCORE")
  exit 0
fi
p5_log "evaldir=$EVALDIR"
uv run "$P5_ROOT/73-score.py" --mode "$MODE"

echo ---- result ----
(set -x; ls -l "$P5_OUTPUTS/eval-$MODE/")

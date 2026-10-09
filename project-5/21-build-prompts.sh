#!/usr/bin/env bash
# 21-build-prompts.sh [mini|full] — Stage 2b wrapper. 후보→template 프롬프트 생성.
# 출력: prompts-<mode>.parquet. PUSH=1이면 Hub(tayaee/alpaca_syntheticdatagen_prompt-<mode>) 업로드.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
ARGS=()
[ "${PUSH:-0}" = 1 ] && ARGS+=(--push)
IN="$P5_DATASETS/candidates-$MODE.parquet"
OUT="$P5_DATASETS/prompts-$MODE.parquet"
p5_require "$IN"
if [ "${PUSH:-0}" != 1 ] && p5_fresh "$OUT" "$IN"; then
  p5_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -lh "$OUT")
  exit 0
fi
p5_log "push=${PUSH:-0}"
uv run "$P5_ROOT/21-build-prompts.py" --mode "$MODE" "${ARGS[@]}"

echo ---- result ----
(set -x; ls -lh "$P5_DATASETS/prompts-$MODE.parquet")

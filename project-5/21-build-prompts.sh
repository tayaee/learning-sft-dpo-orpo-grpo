#!/usr/bin/env bash
# 21-build-prompts.sh [mini|full] — Stage 2b wrapper. 후보→template 프롬프트 생성.
# 출력: prompts-<mode>.parquet. PUSH=1이면 Hub(tayaee/alpaca_syntheticdatagen_prompt-<mode>) 업로드.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
ARGS=()
[ "${PUSH:-0}" = 1 ] && ARGS+=(--push)
p5_log "push=${PUSH:-0}"
uv run "$P5_ROOT/21-build-prompts.py" --mode "$MODE" "${ARGS[@]}"

echo ---- result ----
ls -lh "$P5_DATASETS/prompts-$MODE.parquet"

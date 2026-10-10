#!/usr/bin/env bash
# 32-merge-lora.sh [mini|full] [base-model] (STRAT env, 기본 single) — Stage 3c wrapper. QLoRA 어댑터 병합.
# 입력: synthetic-qlora-<mode>-$STRAT → 출력: synthetic-qlora-<mode>-$STRAT-merged
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
STRAT="${STRAT:-single}"
case "$STRAT" in single|ddp|fsdp) ;; *) echo "STRAT must be single|ddp|fsdp" >&2; exit 1;; esac
BASE="${2:-$BASE_MODEL}"
SRC="$P5_MODELS/synthetic-qlora-$MODE-$STRAT"
OUT="$P5_MODELS/synthetic-qlora-$MODE-$STRAT-merged"
p5_require "$SRC/adapter_config.json"
if [ -f "$OUT/config.json" ] && p5_fresh "$OUT/config.json" "$SRC/adapter_config.json"; then
  p5_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -l "$OUT")
  exit 0
fi
mkdir -p "$OUT"
p5_log "base=$BASE out=$OUT"
(set -x; uv run "$P5_ROOT/32-merge-lora.py" --mode "$MODE" --strat "$STRAT" --base "$BASE")

echo ---- result ----
(set -x; ls -l "$OUT")

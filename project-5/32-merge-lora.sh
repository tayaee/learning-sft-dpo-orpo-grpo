#!/usr/bin/env bash
# 32-merge-lora.sh [mini|full] [base-model] — Stage 3c wrapper. QLoRA 어댑터 병합.
# 입력: synthetic-qlora-<mode>-single → 출력: synthetic-qlora-<mode>-single-merged
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
BASE="${2:-$BASE_MODEL}"
SRC="$P5_MODELS/synthetic-qlora-$MODE-single"
OUT="$P5_MODELS/synthetic-qlora-$MODE-single-merged"
p5_require "$SRC/adapter_config.json"
if [ -f "$OUT/config.json" ] && p5_fresh "$OUT/config.json" "$SRC/adapter_config.json"; then
  p5_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -l "$OUT")
  exit 0
fi
mkdir -p "$OUT"
p5_log "base=$BASE out=$OUT"
(set -x; uv run "$P5_ROOT/32-merge-lora.py" --mode "$MODE" --base "$BASE")

echo ---- result ----
(set -x; ls -l "$OUT")

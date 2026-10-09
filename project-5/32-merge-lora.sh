#!/usr/bin/env bash
# 32-merge-lora.sh [mini|full] [base-model] — Stage 3c wrapper. QLoRA 어댑터 병합.
# 입력: synthetic-qlora-<mode>-single → 출력: synthetic-qlora-<mode>-single-merged
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
BASE="${2:-$BASE_MODEL}"
OUT="$P5_MODELS/synthetic-qlora-$MODE-single-merged"
mkdir -p "$OUT"
p5_log "base=$BASE out=$OUT"
(set -x; uv run "$P5_ROOT/32-merge-lora.py" --mode "$MODE" --base "$BASE")

echo ---- result ----
ls -l "$OUT"

#!/usr/bin/env bash
# 23-postprocess.sh [mini|full] — Stage 2c wrapper. 생성 CSV 후처리→SFT JSONL.
# 입력: generated-<mode>.csv → 출력: synthetic-<mode>.jsonl
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
p5_repro_echo
IN="$P5_DATASETS/generated-$MODE.csv"
OUT="$P5_DATASETS/synthetic-$MODE.jsonl"
p5_require "$IN"
if p5_fresh "$OUT" "$IN"; then
  p5_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -lh "$OUT")
  (set -x; wc -l "$OUT")
  exit 0
fi
p5_log "in=$IN out=$OUT"
uv run "$P5_ROOT/23-postprocess.py" --mode "$MODE"

echo ---- result ----
(set -x; ls -lh "$P5_DATASETS/synthetic-$MODE.jsonl")
(set -x; wc -l "$P5_DATASETS/synthetic-$MODE.jsonl")

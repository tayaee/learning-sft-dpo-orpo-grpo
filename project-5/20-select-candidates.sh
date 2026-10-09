#!/usr/bin/env bash
# 20-select-candidates.sh [mini|full] [seed] — Stage 2a wrapper. Alpaca→GSM8K 유사 후보 선별.
# 출력: alpaca-embeddings.parquet (캐시) + candidates-<mode>.parquet
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
SEED="${2:-1}"
IN="$P5_DATASETS/gsm8k-train.jsonl"
OUT="$P5_DATASETS/candidates-$MODE.parquet"
EMBCACHE="$P5_DATASETS/alpaca-embeddings.parquet"
p5_require "$IN"
if p5_fresh "$OUT" "$IN"; then
  p5_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -lh "$EMBCACHE")
  (set -x; ls -lh "$OUT")
  exit 0
fi
p5_log "seed=$SEED"
(set -x; uv run "$P5_ROOT/20-select-candidates.py" --mode "$MODE" --seed "$SEED")

echo ---- result ----
(set -x; ls -lh "$P5_DATASETS/alpaca-embeddings.parquet") 
(set -x; ls -lh "$P5_DATASETS/candidates-$MODE.parquet")

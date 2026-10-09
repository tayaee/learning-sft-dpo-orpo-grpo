#!/usr/bin/env bash
# 20-select-candidates.sh [mini|full] [seed] — Stage 2a wrapper. Alpaca→GSM8K 유사 후보 선별.
# 출력: alpaca-embeddings.parquet (캐시) + candidates-<mode>.parquet
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
SEED="${2:-1}"
p5_log "seed=$SEED"
uv run "$P5_ROOT/20-select-candidates.py" --mode "$MODE" --seed "$SEED"

echo ---- result ----
ls -lh "$P5_DATASETS/alpaca-embeddings.parquet" 
ls -lh "$P5_DATASETS/candidates-$MODE.parquet"

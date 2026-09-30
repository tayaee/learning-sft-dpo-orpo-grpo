#!/usr/bin/env bash
# 30-fft-train.sh [mini|full] [single|ddp|fsdp] — Stage 3a. 합성데이터 FFT.
# 원본: train_FFT.sh. --train_dir → synthetic-<mode>.jsonl
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
p5_resolve_strat "${2:-single}"

OUT="$P5_MODELS/synthetic-fft-$MODE-$STRAT"
p5_log "train=$P5_DATASETS/synthetic-$MODE.jsonl out=$OUT strat=$STRAT world=$WORLD accum=$ACCUM"

"${LAUNCH[@]}" "$P5_ROOT/10-train-entry.py" \
  --model "$BASE_MODEL" --train "$P5_DATASETS/synthetic-$MODE.jsonl" \
  --out "$OUT" --epochs "$EPOCHS" --mode "$MODE" \
  --strategy "$STRAT" --accum "$ACCUM"

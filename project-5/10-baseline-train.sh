#!/usr/bin/env bash
# 10-baseline-train.sh [mini|full] [single|ddp|fsdp] — Stage 1. GSM8K 베이스라인.
# 원본: train_basic.sh (torchrun nproc=4 + FSDP full_shard).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
p5_resolve_strat "${2:-single}"

OUT="$P5_MODELS/base-gsm8k-$MODE-$STRAT"
p5_log "model=$BASE_MODEL out=$OUT epochs=$EPOCHS strat=$STRAT world=$WORLD accum=$ACCUM eff_batch=$((MICRO_BATCH * ACCUM * WORLD))"

# STUB: 원본 main.py+trainer436 자리에 modern SFT (TRL SFTTrainer) 진입 예정.
# 동일 하이퍼파라미터: lr 1e-5, cosine, warmup 0.03, micro_batch 2,
# accum=$ACCUM (effective 64 유지), tok/tgt 512, bf16.
# fsdp 전략일 때만 --fsdp "full_shard auto_wrap" 추가 (구현 시).
"${LAUNCH[@]}" "$P5_ROOT/10-train-entry.py" \
  --model "$BASE_MODEL" --train "$P5_DATASETS/gsm8k-train.jsonl" \
  --out "$OUT" --epochs "$EPOCHS" --mode "$MODE" \
  --strategy "$STRAT" --accum "$ACCUM"

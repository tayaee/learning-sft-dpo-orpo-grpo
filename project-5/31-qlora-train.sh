#!/usr/bin/env bash
# 31-qlora-train.sh [mini|full] [single|ddp|fsdp] — Stage 3b. 합성데이터 QLoRA.
# 전략 선택: $2 > $STRAT(환경) > single.
# 원본: train_QLoRA.sh. 포인트: 8bit 블록 비활성 + 4bit nf4 double_quant,
# target=q/k/v/o/gate/down/up(+embed,lm_head), 산출물은 어댑터만.
# NOTE: 원본 강의는 FSDP+QLoRA 불가 → DDP로 동작. fsdp 전략은 학습용으로만 수행.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
p5_resolve_strat "${2:-${STRAT:-single}}"

OUT="$P5_MODELS/synthetic-qlora-$MODE-$STRAT"
IN="$P5_DATASETS/synthetic-$MODE.jsonl"
p5_require "$IN"
if [ -f "$OUT/adapter_config.json" ] && p5_fresh "$OUT/adapter_config.json" "$IN"; then
  p5_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -l "$OUT")
  exit 0
fi
mkdir -p "$OUT"  # 출력 부모 사전 생성 (공유FS 동시 makedirs race 회피)
p5_log "train=$P5_DATASETS/synthetic-$MODE.jsonl out=$OUT (adapter only) strat=$STRAT world=$WORLD accum=$ACCUM"

(set -x; "${LAUNCH[@]}" "$P5_ROOT/10-train-entry.py" --peft \
  --model "$BASE_MODEL" --train "$P5_DATASETS/synthetic-$MODE.jsonl" \
  --out "$OUT" --epochs "$EPOCHS" --mode "$MODE" \
  --strategy "$STRAT" --accum "$ACCUM" \
  --max_rows "$GSM_ROWS")

echo ---- result ----
(set -x; ls -l "$OUT")

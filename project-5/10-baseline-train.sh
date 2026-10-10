#!/usr/bin/env bash
# 10-baseline-train.sh [mini|full] [single|ddp|fsdp] — Stage 1. GSM8K 베이스라인.
# 전략 선택: $2 > $STRAT(환경) > single.
# 원본: train_basic.sh (torchrun nproc=4 + FSDP full_shard).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
p5_resolve_strat "${2:-${STRAT:-single}}"
# 분산 전략은 2x 필수 — 1x에서 돌리면 solo 성공을 분산 성공으로 착각한다 (실측).
# ddp는 resolve가 이미 거부, fsdp 1x 기능검증도 이 환경(1노드=1GPU)에선 무의미.
if [ "$STRAT" != single ] && [ "$PROFILE" != "2x" ]; then
  echo "REFUSE: strat=$STRAT needs INFRA=dgx-spark-2x (현재 PROFILE=$PROFILE)" >&2
  exit 1
fi
p5_repro_echo

OUT="$P5_MODELS/base-gsm8k-$MODE-$STRAT"
IN="$P5_DATASETS/gsm8k-train.jsonl"
p5_require "$IN"
if [ -f "$OUT/config.json" ] && p5_fresh "$OUT/config.json" "$IN"; then
  p5_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -l "$OUT")
  exit 0
fi
# 출력 디렉토리는 rank0만 생성 (양쪽 동시 mkdir -p가 공유FS에서 EEXIST로 죽는
# 레이스 실측). rank1은 OUT에 쓰지 않는다 (중간 ckpt·최종 저장 모두 rank0-only).
if [ "${NODE_RANK:-0}" = "0" ]; then mkdir -p "$OUT"; fi
p5_log "model=$BASE_MODEL out=$OUT epochs=$EPOCHS strat=$STRAT world=$WORLD accum=$ACCUM eff_batch=$((MICRO_BATCH * ACCUM * WORLD))"
# rank별 학습 로그 (tmux 유실 대비 — grep rank1 logs/* 로 확인).
TRAIN_LOG="$P5_ROOT/logs/train-$MODE-$STRAT-rank${NODE_RANK:-0}.log"
echo "== $(date -u +%FT%TZ) $(hostname) rank=${NODE_RANK:-0} world=$WORLD strat=$STRAT ==" > "$TRAIN_LOG"
p5_log "log: $TRAIN_LOG"

# WORLD==1 solo 학습은 중복 실행 금지 (같은 OUT 저장 레이스 — 실측).
# 분산(WORLD=2)은 양쪽이 다 돌아야 해서 락 없음.
[ "$WORLD" = "1" ] && p5_lock "$OUT" nowait
# STUB: 원본 main.py+trainer436 자리에 modern SFT (TRL SFTTrainer) 진입 예정.
# 동일 하이퍼파라미터: lr 1e-5, cosine, warmup 0.03, micro_batch 2,
# accum=$ACCUM (effective 64 유지), tok/tgt 512, bf16.
# fsdp 전략일 때만 --fsdp "full_shard auto_wrap" 추가 (구현 시).
(set -x; "${LAUNCH[@]}" "$P5_ROOT/10-train-entry.py" \
  --model "$BASE_MODEL" --train "$P5_DATASETS/gsm8k-train.jsonl" \
  --out "$OUT" --epochs "$EPOCHS" --mode "$MODE" \
  --strategy "$STRAT" --accum "$ACCUM" --expect-world "$WORLD" \
  --max_rows "$GSM_ROWS") 2>&1 | sed -u "s/^/[rank${NODE_RANK:-0}]: /" | tee -a "$TRAIN_LOG"

echo ---- result ----
(set -x; ls -l "$OUT")

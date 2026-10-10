#!/usr/bin/env bash
# 30-fft-train.sh [mini|full] [single|ddp|fsdp] — Stage 3a. 합성데이터 FFT.
# 전략 선택: $2 > $STRAT(환경) > single.
# 원본: train_FFT.sh. --train_dir → synthetic-<mode>.jsonl
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

OUT="$P5_MODELS/synthetic-fft-$MODE-$STRAT"
IN="$P5_DATASETS/synthetic-$MODE.jsonl"
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
p5_log "train=$P5_DATASETS/synthetic-$MODE.jsonl out=$OUT strat=$STRAT world=$WORLD accum=$ACCUM"
# rank별 학습 로그 (tmux 유실 대비 — 매 줄 [rankN] 접두, grep rank1 logs/* 로 확인).
TRAIN_LOG="$P5_ROOT/logs/train-$MODE-$STRAT-rank${NODE_RANK:-0}.log"
echo "== $(date -u +%FT%TZ) $(hostname) rank=${NODE_RANK:-0} world=$WORLD strat=$STRAT ==" > "$TRAIN_LOG"
p5_log "log: $TRAIN_LOG"

(set -x; "${LAUNCH[@]}" "$P5_ROOT/10-train-entry.py" \
  --model "$BASE_MODEL" --train "$P5_DATASETS/synthetic-$MODE.jsonl" \
  --out "$OUT" --epochs "$EPOCHS" --mode "$MODE" \
  --strategy "$STRAT" --accum "$ACCUM" --expect-world "$WORLD" \
  --max_rows "$GSM_ROWS") 2>&1 | sed -u "s/^/[rank${NODE_RANK:-0}]: /" | tee -a "$TRAIN_LOG"

echo ---- result ----
(set -x; ls -l "$OUT")

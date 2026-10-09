#!/usr/bin/env bash
# 30-fft-train.sh [mini|full] [single|ddp|fsdp] — Stage 3a. 합성데이터 FFT.
# 전략 선택: $2 > $STRAT(환경) > single.
# 원본: train_FFT.sh. --train_dir → synthetic-<mode>.jsonl
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
p5_resolve_strat "${2:-${STRAT:-single}}"

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

(set -x; "${LAUNCH[@]}" "$P5_ROOT/10-train-entry.py" \
  --model "$BASE_MODEL" --train "$P5_DATASETS/synthetic-$MODE.jsonl" \
  --out "$OUT" --epochs "$EPOCHS" --mode "$MODE" \
  --strategy "$STRAT" --accum "$ACCUM" \
  --max_rows "$GSM_ROWS")

echo ---- result ----
(set -x; ls -l "$OUT")

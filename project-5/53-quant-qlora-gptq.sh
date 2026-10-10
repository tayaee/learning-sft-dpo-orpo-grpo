#!/usr/bin/env bash
# 53-quant-qlora-gptq.sh [mini|full] [calib] (STRAT env, 기본 single) — Stage 5d wrapper. QLoRA-merged→GPTQ 4bit-g128.
# 입력: synthetic-qlora-<mode>-$STRAT-merged → 출력: ...-merged-gptq
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
STRAT="${STRAT:-single}"
case "$STRAT" in single|ddp|fsdp) ;; *) echo "STRAT must be single|ddp|fsdp" >&2; exit 1;; esac
p5_repro_echo
CALIB="${2:-$CALIB_N}"
: "${CALIB_FILE:=$P5_DATASETS/gsm8k-calibration-256.jsonl}"
SRC="$P5_MODELS/synthetic-qlora-$MODE-$STRAT-merged"
OUT="$P5_MODELS/synthetic-qlora-$MODE-$STRAT-merged-gptq"
p5_lock "$OUT"
p5_require "$SRC/config.json" "$CALIB_FILE"
# 마커 파일끼리 비교: 양자화 save()는 기존 파일을 제자리 덮어쓰기해서
# 디렉토리 mtime이 안 바뀌므로, 디렉토리끼리 비교하면 항상 stale이 된다.
if [ -f "$OUT/config.json" ] && p5_fresh "$OUT/config.json" "$SRC/config.json" "$CALIB_FILE"; then
  p5_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -l "$OUT")
  exit 0
fi
p5_log "calib=$CALIB src=$CALIB_FILE out=$OUT"
(set -x; uv run "$P5_ROOT/53-quant-qlora-gptq.py" --mode "$MODE" --strat "$STRAT" --calib "$CALIB" --calib-file "$CALIB_FILE")

echo ---- result ----
(set -x; ls -l "$OUT")

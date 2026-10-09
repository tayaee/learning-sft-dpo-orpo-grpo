#!/usr/bin/env bash
# 71-eval.sh [mini|full] [target...] — Stage 7b wrapper. target 기본 all.
# all = base + fft/qlora FP 2종 + HF 양자화 6종 (GGUF 10종 제외, 72-eval-gguf.sh 별도 평가).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
TARGETS="${2:-all}"
[ "$TARGETS" = all ] && TARGETS="base fft qlora fft-gptq fft-awq fft-fp8 qlora-gptq qlora-awq qlora-fp8"
N="$EVAL_N"  # mini 10 / full 0=전체 (common.env)
TEST="$P5_DATASETS/gsm8k-test.jsonl"
p5_require "$TEST"
# 71-eval.py resolve()의 bash 미러 — base(HF)는 로컬 타임스탬프 없음
target_model() {
  case "$1" in
    base) echo "" ;;
    fft) echo "$P5_MODELS/synthetic-fft-$MODE-single" ;;
    qlora) echo "$P5_MODELS/synthetic-qlora-$MODE-single-merged" ;;
    fft-*) echo "$P5_MODELS/synthetic-fft-$MODE-single-${1#fft-}" ;;
    qlora-*) echo "$P5_MODELS/synthetic-qlora-$MODE-single-merged-${1#qlora-}" ;;
    gptq|awq|fp8) echo "$P5_MODELS/synthetic-fft-$MODE-single-$1" ;;
    *) echo "" ;;
  esac
}
p5_log "tp=$TP targets=$TARGETS n=$N"
for t in $TARGETS; do
  out="$P5_OUTPUTS/eval-$MODE/$t.jsonl"
  m="$(target_model "$t")"
  if [ -z "$m" ]; then
    if [ "${FORCE:-0}" != 1 ] && [ -f "$out" ]; then p5_log "skip: fresh $out (base/HF, FORCE=1 to rebuild)"; continue; fi
  elif [ -f "$m/config.json" ] && [ -f "$out" ] && p5_fresh "$out" "$m/config.json" "$TEST"; then
    p5_log "skip: fresh $out (FORCE=1 to rebuild)"; continue
  fi
  (set -x; uv run "$P5_ROOT/71-eval.py" --mode "$MODE" --target "$t" --n "$N" --tp "$TP")
done

echo ---- result ----
(set -x; ls -l "$P5_OUTPUTS/eval-$MODE/")
(set -x; wc -l "$P5_OUTPUTS/eval-$MODE/"*.jsonl)

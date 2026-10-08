#!/usr/bin/env bash
# 71-eval.sh [mini|full] [target...] — Stage 7b wrapper. target 기본 all.
# all = base + fft/qlora FP 2종 + HF 양자화 6종 (GGUF 2종 제외, llama.cpp 별도 평가).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
TARGETS="${2:-all}"
[ "$TARGETS" = all ] && TARGETS="base fft qlora fft-gptq fft-awq fft-fp8 qlora-gptq qlora-awq qlora-fp8"
N="$EVAL_N"  # mini 10 / full 0=전체 (common.env)
p5_log "tp=$TP targets=$TARGETS n=$N"
for t in $TARGETS; do
  uv run "$P5_ROOT/71-eval.py" --mode "$MODE" --target "$t" --n "$N" --tp "$TP"
done

echo ---- result ----
ls -l "$P5_OUTPUTS/eval-$MODE/"
wc -l "$P5_OUTPUTS/eval-$MODE/"*.jsonl

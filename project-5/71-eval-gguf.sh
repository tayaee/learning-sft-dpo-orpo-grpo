#!/usr/bin/env bash
# 71-eval-gguf.sh [mini|full] [target...] — Stage 7b-gguf wrapper. llama-cli greedy GSM8K 추론.
# target 기본 all = GGUF 10종 (fft/qlora × {q8_0,q6_k,q5_k_m,q4_k_m,q3_k_m}).
# 전제: llama-cli 빌드 ($LLAMACPP/build/bin/llama-cli). 없으면 전 타깃 SKIP (rc=0).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
TARGETS="${2:-all}"
[ "$TARGETS" = all ] && TARGETS="fft-gguf-q8_0 fft-gguf-q6_k fft-gguf-q5_k_m fft-gguf-q4_k_m fft-gguf-q3_k_m qlora-gguf-q8_0 qlora-gguf-q6_k qlora-gguf-q5_k_m qlora-gguf-q4_k_m qlora-gguf-q3_k_m"
N="$EVAL_N"  # mini 10 / full 0=전체 (common.env)
LLAMACPP="${LLAMACPP:-$HOME/git/llama.cpp}"
if [ ! -x "$LLAMACPP/build/bin/llama-cli" ]; then
  p5_log "WARN: no llama-cli, GGUF eval SKIP (build llama.cpp llama-cli target)"
  ls -l "$P5_OUTPUTS/eval-$MODE/" 2>/dev/null || true
  exit 0
fi
p5_log "targets=$TARGETS n=$N"
for t in $TARGETS; do
  (set -x; uv run "$P5_ROOT/71-eval-gguf.py" --mode "$MODE" --target "$t" --n "$N")
done

echo ---- result ----
ls -l "$P5_OUTPUTS/eval-$MODE/"
wc -l "$P5_OUTPUTS/eval-$MODE/"*.jsonl

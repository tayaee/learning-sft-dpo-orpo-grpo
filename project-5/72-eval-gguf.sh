#!/usr/bin/env bash
# 72-eval-gguf.sh [mini|full] [target...] — Stage 7b-gguf wrapper. llama-cli greedy GSM8K 추론.
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
  (set -x; ls -l "$P5_OUTPUTS/eval-$MODE/" 2>/dev/null || true)
  exit 0
fi
TEST="$P5_DATASETS/gsm8k-test.jsonl"
p5_require "$TEST"
# <target> → GGUF 파일 (72-eval-gguf.py TARGETS 규칙 미러)
gguf_file() {
  case "$1" in
    fft-gguf-*) echo "$P5_MODELS/synthetic-fft-$MODE-single-gguf/model-${1#fft-gguf-}.gguf" ;;
    qlora-gguf-*) echo "$P5_MODELS/synthetic-qlora-$MODE-single-merged-gguf/model-${1#qlora-gguf-}.gguf" ;;
  esac
}
p5_log "targets=$TARGETS n=$N"
for t in $TARGETS; do
  out="$P5_OUTPUTS/eval-$MODE/$t.jsonl"
  g="$(gguf_file "$t")"
  if [ -n "$g" ] && [ -f "$g" ] && [ -f "$out" ] && p5_fresh "$out" "$g" "$TEST"; then
    p5_log "skip: fresh $out (FORCE=1 to rebuild)"; continue
  fi
  (set -x; uv run "$P5_ROOT/72-eval-gguf.py" --mode "$MODE" --target "$t" --n "$N")
done

echo ---- result ----
(set -x; ls -l "$P5_OUTPUTS/eval-$MODE/")
(set -x; wc -l "$P5_OUTPUTS/eval-$MODE/"*.jsonl)

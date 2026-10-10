#!/usr/bin/env bash
# 71-eval.sh [mini|full] [target...] (STRAT env, 기본 single) — Stage 7b wrapper. target 기본 all.
# all = base + fft/qlora FP 2종 + HF 양자화 6종 (GGUF 10종 제외, 72-eval-gguf.sh 별도 평가).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
STRAT="${STRAT:-single}"
case "$STRAT" in single|ddp|fsdp) ;; *) echo "STRAT must be single|ddp|fsdp" >&2; exit 1;; esac
TARGETS="${2:-all}"
[ "$TARGETS" = all ] && TARGETS="base fft qlora fft-gptq fft-awq fft-fp8 qlora-gptq qlora-awq qlora-fp8"
N="$EVAL_N"  # mini 10 / full 0=전체 (common.env)
TEST="$P5_DATASETS/gsm8k-test.jsonl"
p5_require "$TEST"
: "${VLLM_MAX_NUM_SEQS:=8}"
# 71-eval.py resolve()의 bash 미러 — base(HF)는 로컬 타임스탬프 없음
target_model() {
  case "$1" in
    base) echo "" ;;
    fft) echo "$P5_MODELS/synthetic-fft-$MODE-$STRAT" ;;
    qlora) echo "$P5_MODELS/synthetic-qlora-$MODE-$STRAT-merged" ;;
    fft-*) echo "$P5_MODELS/synthetic-fft-$MODE-$STRAT-${1#fft-}" ;;
    qlora-*) echo "$P5_MODELS/synthetic-qlora-$MODE-$STRAT-merged-${1#qlora-}" ;;
    gptq|awq|fp8) echo "$P5_MODELS/synthetic-fft-$MODE-$STRAT-$1" ;;
    *) echo "" ;;
  esac
}
p5_log "tp=$TP targets=$TARGETS n=$N gpu_mem_util=${VLLM_GPU_MEM_UTIL:-0.8} max_num_seqs=$VLLM_MAX_NUM_SEQS"
for t in $TARGETS; do
  out="$P5_OUTPUTS/eval-$MODE-$STRAT/$t.jsonl"
  m="$(target_model "$t")"
  if [ -z "$m" ]; then
    if [ "${FORCE:-0}" != 1 ] && [ -f "$out" ]; then p5_log "skip: fresh $out (base/HF, FORCE=1 to rebuild)"; continue; fi
  elif [ -f "$m/config.json" ] && [ -f "$out" ] && p5_fresh "$out" "$m/config.json" "$TEST"; then
    p5_log "skip: fresh $out (FORCE=1 to rebuild)"; continue
  fi
  (set -x; uv run "$P5_ROOT/71-eval.py" --mode "$MODE" --strat "$STRAT" --target "$t" --n "$N" --tp "$TP" \
    --gpu-mem-util "${VLLM_GPU_MEM_UTIL:-0.8}" --max-num-seqs "$VLLM_MAX_NUM_SEQS")
done

echo ---- result ----
(set -x; ls -l "$P5_OUTPUTS/eval-$MODE-$STRAT/")
(set -x; wc -l "$P5_OUTPUTS/eval-$MODE-$STRAT/"*.jsonl)

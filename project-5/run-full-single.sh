#!/usr/bin/env bash
# run-full-single.sh — full 전 파이프라인 원샷 (spark1 단독, 약 10시간 내외).
# Stage 0→7: setup → baseline FFT → 합성데이터(선별→프롬프트→생성→후처리)
# → 합성 FFT + QLoRA + merge → PTQ 8종(FFT 4 + QLoRA-merged 4)
# → PPL 게이트 → vLLM eval + score.
# NOTE: 22-teacher 합성생성(10k, 약 5~6시간)이 가장 오래 걸리므로 밤샘 실행 권장.
# 제외: 74-upload-hf (Hub 업로드), 80-serve-vllm (상주 서버), 81-infer-examples
#   → 필요시 맨 아래 주석 명령으로 개별 실행.
# 사용: ./run-full-single.sh
#   SKIP_SETUP=1 ./run-full-single.sh   # 00-setup 생략 (양 노드 1회 수행済み)
#   STEPS=setup,baseline ./run-full-single.sh  # 일부 스텝만 실행 (아래 목록)
# 스텝명: setup baseline syndata fft qlora merge quant ppl eval (기본 STEPS=all)
# 스텝별 소요시간: stdout + logs/timing-full-single.log
set -euo pipefail
# 프리셋은 run-full-single.inc에 (수동 실행: source ./run-full-single.inc 후 단계 스크립트 인자 없이 실행)
source "$(dirname "$0")/run-full-single.inc"
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
source "$ROOT/config/common.env" "$MODE"
p5_repro_echo

# ---- 스텝 타이밍 (stdout + TIMING_LOG) ----
TIMING_LOG="$ROOT/logs/timing-$MODE-$STRAT.log"
mkdir -p "$ROOT/logs"
RUN_T0=$(date +%s)
p5_step() { # <스텝명> <명령...> — 시작/종료/소요초를 stdout과 TIMING_LOG에 기록
  local _name="$1"; shift
  local _t0 _dt _rc=0
  _t0=$(date +%s)
  p5_log "step start: $_name"
  "$@" || _rc=$?
  _dt=$(($(date +%s) - _t0))
  p5_log "step done: $_name elapsed=${_dt}s rc=$_rc"
  # 0s 성공(스킵)은 미기록 — 로그가 스킵으로 뒤덮여 실측을 찾기 힘듦. 실패는 항상 기록.
  if [ "$_rc" -ne 0 ] || [ "$_dt" -gt 0 ]; then
    printf '%s\t%s\t%s\t%ss\trc=%d\n' "$(date -u +%FT%TZ)" "$MODE-$STRAT" "$_name" "$_dt" "$_rc" >>"$TIMING_LOG"
  fi
  return $_rc
}
want_step() { # <스텝명> — STEPS=all(기본)이면 전부, 아니면 지정된 것만 실행
  case ",${STEPS:-all}," in *,all,*|*,"$1",*) return 0 ;; *) return 1 ;; esac
}
run_done() { # EXIT trap — 전체 소요시간 기록 (성공/실패 무관)
  p5_log "run-$MODE-$STRAT end rc=$? total=$(( $(date +%s) - RUN_T0 ))s (timing: $TIMING_LOG)"
}
trap 'P5_RC=$?; p5_repro; run_done' EXIT

p5_log "run-$MODE-$STRAT start (steps=${STEPS:-all})"

if [ "${SKIP_SETUP:-0}" != 1 ] && want_step setup; then
  p5_step "00-setup" ./00-setup.sh
fi

want_step baseline && p5_step "10-baseline-$STRAT" ./10-baseline-train.sh "$MODE" "$STRAT"

if want_step syndata; then
  p5_step "20-select-candidates" ./20-select-candidates.sh "$MODE"
  p5_step "21-build-prompts" ./21-build-prompts.sh "$MODE"
  p5_step "22-teacher-generate" ./22-teacher-to-generate-syn-data.sh "$MODE"
  p5_step "23-postprocess" ./23-postprocess.sh "$MODE"
fi

want_step fft && p5_step "30-fft-$STRAT" ./30-fft-train.sh "$MODE" "$STRAT"
want_step qlora && p5_step "31-qlora-$STRAT" ./31-qlora-train.sh "$MODE" "$STRAT"
want_step merge && p5_step "32-merge-lora" ./32-merge-lora.sh "$MODE"

if want_step quant; then
  p5_step "40-quant-fft-awq" ./40-quant-fft-awq.sh "$MODE"
  p5_step "41-quant-fft-fp8" ./41-quant-fft-fp8.sh "$MODE"
  p5_step "43-quant-fft-gptq" ./43-quant-fft-gptq.sh "$MODE"
  p5_step "50-quant-qlora-awq" ./50-quant-qlora-awq.sh "$MODE"
  p5_step "51-quant-qlora-fp8" ./51-quant-qlora-fp8.sh "$MODE"
  p5_step "53-quant-qlora-gptq" ./53-quant-qlora-gptq.sh "$MODE"
  if [ -x "${LLAMACPP:-$HOME/git/llama.cpp}/build/bin/llama-quantize" ]; then
    p5_step "42-quant-fft-gguf" ./42-quant-fft-gguf.sh "$MODE"
    p5_step "52-quant-qlora-gguf" ./52-quant-qlora-gguf.sh "$MODE"
  else
    p5_log "WARN: no llama-quantize, GGUF SKIP (run 42/52 after build)"
  fi
fi

want_step ppl && p5_step "60-measure-ppl" ./60-measure-ppl.sh "$MODE"
if want_step eval; then
  p5_step "71-eval" ./71-eval.sh "$MODE"
  p5_step "72-eval-gguf" ./72-eval-gguf.sh "$MODE"
  p5_step "73-score" ./73-score.sh "$MODE"
fi

p5_log "run-$MODE-$STRAT done"
# --- 선택 (수동) ---
# ./74-upload-hf.sh full
# ./80-serve-vllm.sh full fft            # 터미널1 (상주)
# ./81-infer-examples.sh tayaee/Llama-3.2-1B-math-fft-full  # 터미널2

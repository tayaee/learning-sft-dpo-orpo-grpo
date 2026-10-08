#!/usr/bin/env bash
# run-full-fsdp.sh — full FSDP 원샷 (1x 동작학습 기본, 2x 실분산 겸용).
# 1노드 학습용 (기본, 메모리 이득 없음·동작 학습):
#   ./run-full-fsdp.sh                     # MASTER_ADDR·NODE_RANK 불필요
# 2노드 실분산 (spark1+spark2에서 같은 명령, rank0 먼저):
#   ./run-full-fsdp.sh  # INFRA=dgx-spark-2x export 시, 양 노드에서 실행
# 분산 대상: 학습만 (10-baseline, 30-fft, 31-qlora).
# 2x에서는 합성데이터·전략비교·merge·PTQ·PPL·평가는 rank0만 수행 (TP=1).
# rank1은 rank0의 synthetic-full.jsonl 생성을 기다렸다가 학습에 합류한다
# (22-teacher 10k 생성에 약 5~6시간, 대기 상한 9시간).
# NOTE: FSDP+QLoRA는 강의에서 비권장 → 31-qlora는 DDP 동작으로 학습용 수행.
# 2x 저장: DCP sharded save + rank0 consolidation (10-train-entry.py).
#   giant NCCL gather를 fabric에 던지지 않음 (mini-2x 검증됨: 양 노드 rc=0).
# 하류(merge→평가)는 -single 산출물이 있어야 진행한다:
#   없으면 FSDP 학습 + 12-compare까지만 수행하고 종료 (먼저 ./run-full-single.sh 실행).
# 사용: SKIP_SETUP=1 ./run-full-fsdp.sh                # 00-setup 생략
#   STEPS=setup,baseline ./run-full-fsdp.sh            # 최소 파이프라인 (분산 동작 확인용)
# 자동 검출(2x일 때만): MASTER_ADDR는 CX7(spark1-p1-r0→p1-r1→LAN 순,
# /etc/hosts+ping 확인), NODE_RANK는 호스트명(spark1→0, spark2→1, …) 기준.
# 스텝명: setup baseline syndata fft qlora compare merge quant ppl eval (기본 STEPS=all)
# 스텝별 소요시간: stdout + logs/timing-full-fsdp.log
set -euo pipefail
MODE=full; STRAT=fsdp
export INFRA="${INFRA:-dgx-spark-1x}"
DIST=0; [ "$INFRA" = "dgx-spark-2x" ] && DIST=1
if [ "$DIST" = "1" ]; then
  # MASTER_ADDR 자동 검출 (미지정 시): CX7 fabric 우선, /etc/hosts·getent 기준, ping 확인
  if [ -z "${MASTER_ADDR:-}" ]; then
    for _h in spark1-p1-r0 spark1-p1-r1 spark1.local; do
      _ip="$(getent hosts "$_h" 2>/dev/null | awk '{print $1; exit}')"
      [ -z "$_ip" ] && continue
      if ! command -v ping >/dev/null 2>&1 || ping -c1 -W1 "$_ip" >/dev/null 2>&1; then
        MASTER_ADDR="$_h"
        echo "[$MODE][$STRAT] MASTER_ADDR auto-detected: $_h ($_ip)"
        break
      fi
    done
    unset _h _ip
  fi
  : "${MASTER_ADDR:?MASTER_ADDR 검출 실패 — 수동 지정 필요 (예: MASTER_ADDR=spark1-p1-r0)}"
  # NODE_RANK 자동 검출 (미지정 시): 호스트명 기준
  if [ -z "${NODE_RANK:-}" ]; then
    case "$(hostname)" in
      spark1*) NODE_RANK=0 ;;
      spark2*) NODE_RANK=1 ;;
      spark3*) NODE_RANK=2 ;;
      spark4*) NODE_RANK=3 ;;
      *) : "${NODE_RANK:?NODE_RANK 검출 실패 — 수동 지정 필요 (예: NODE_RANK=0|1)}" ;;
    esac
    export NODE_RANK
    echo "[$MODE][$STRAT] NODE_RANK auto-detected: $NODE_RANK ($(hostname))"
  fi
fi
: "${NODE_RANK:=0}"
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
source "$ROOT/config/common.env" "$MODE"

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
  printf '%s\t%s\t%s\t%ss\trc=%d\n' "$(date -u +%FT%TZ)" "$MODE-$STRAT" "$_name" "$_dt" "$_rc" >>"$TIMING_LOG"
  return $_rc
}
want_step() { # <스텝명> — STEPS=all(기본)이면 전부, 아니면 지정된 것만 실행
  case ",${STEPS:-all}," in *,all,*|*,"$1",*) return 0 ;; *) return 1 ;; esac
}
run_done() { # EXIT trap — 전체 소요시간 기록 (성공/실패 무관)
  p5_log "run-$MODE-$STRAT end rc=$? total=$(( $(date +%s) - RUN_T0 ))s (timing: $TIMING_LOG)"
}
trap run_done EXIT

is_rank0() { [ "$NODE_RANK" = "0" ]; }
wait_file() { # <path> <timeout_s> — 공유디스크에 파일이 생길 때까지 대기
  local p="$1" t="${2:-32400}" waited=0
  while [ ! -s "$p" ]; do
    sleep 30; waited=$((waited + 30))
    if [ "$waited" -ge "$t" ]; then p5_log "ERROR: wait timeout (${t}s): $p"; exit 1; fi
  done
}

p5_log "run-$MODE-$STRAT start (infra=$INFRA rank=$NODE_RANK steps=${STEPS:-all})"

if [ "${SKIP_SETUP:-0}" != 1 ] && want_step setup; then
  p5_step "00-setup" ./00-setup.sh
fi

want_step baseline && p5_step "10-baseline-$STRAT" ./10-baseline-train.sh "$MODE" "$STRAT"

if [ "$DIST" = "0" ] || is_rank0; then
  export TP=1  # 추론계(vLLM 생성·평가)는 노드 단독 1GPU로 실행 (2x에서 필수)
  if want_step syndata; then
    p5_step "20-select-candidates" uv run 20-select-candidates.py --mode "$MODE"
    p5_step "21-build-prompts" uv run 21-build-prompts.py --mode "$MODE"
    p5_step "22-teacher-generate" ./22-teacher-to-generate-syn-data.sh "$MODE"
    p5_step "23-postprocess" uv run 23-postprocess.py --mode "$MODE"
  fi
else
  wait_file "$P5_DATASETS/synthetic-$MODE.jsonl" 32400
fi

want_step fft && p5_step "30-fft-$STRAT" ./30-fft-train.sh "$MODE" "$STRAT"
want_step qlora && p5_step "31-qlora-$STRAT" ./31-qlora-train.sh "$MODE" "$STRAT"

if [ "$DIST" = "1" ] && ! is_rank0; then
  p5_log "run-$MODE-$STRAT done (rank=$NODE_RANK, train-only)"; exit 0
fi
export TP=1

want_step compare && p5_step "12-compare" ./12-compare-strategies.sh "$MODE"

if [ ! -d "$P5_MODELS/synthetic-fft-$MODE-single" ] || [ ! -d "$P5_MODELS/synthetic-qlora-$MODE-single" ]; then
  p5_log "downstream SKIP: no -single output, run ./run-$MODE-single.sh first"
  exit 0
fi

want_step merge && p5_step "32-merge-lora" uv run 32-merge-lora.py --mode "$MODE"

if want_step quant; then
  if [ -x "${LLAMACPP:-$HOME/git/llama.cpp}/build/bin/llama-quantize" ]; then
    p5_step "40-quant-fft-gguf" ./40-quant-fft-gguf.sh "$MODE"
    p5_step "50-quant-qlora-gguf" ./50-quant-qlora-gguf.sh "$MODE"
  else
    p5_log "WARN: no llama-quantize, GGUF SKIP (run 40/50 after build)"
  fi
  p5_step "41-quant-fft-gptq" uv run 41-quant-fft-gptq.py --mode "$MODE"
  p5_step "42-quant-fft-awq" uv run 42-quant-fft-awq.py --mode "$MODE"
  p5_step "43-quant-fft-fp8" uv run 43-quant-fft-fp8.py --mode "$MODE"
  p5_step "51-quant-qlora-gptq" uv run 51-quant-qlora-gptq.py --mode "$MODE"
  p5_step "52-quant-qlora-awq" uv run 52-quant-qlora-awq.py --mode "$MODE"
  p5_step "53-quant-qlora-fp8" uv run 53-quant-qlora-fp8.py --mode "$MODE"
fi

want_step ppl && p5_step "60-measure-ppl" ./60-measure-ppl.sh "$MODE"
if want_step eval; then
  p5_step "71-eval" ./71-eval.sh "$MODE"
  p5_step "72-score" uv run 72-score.py --mode "$MODE"
fi

p5_log "run-$MODE-$STRAT done"
# --- 선택 (수동, 2x면 rank0) ---
# ./73-upload-hf.sh full
# ./80-serve-vllm.sh full fft            # 터미널1 (상주)
# uv run 81-infer-examples.py --model tayaee/Llama-3.2-1B-math-fft-full  # 터미널2

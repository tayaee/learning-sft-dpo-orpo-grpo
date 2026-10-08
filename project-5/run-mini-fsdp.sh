#!/usr/bin/env bash
# run-mini-fsdp.sh — mini FSDP 원샷 (1x 동작학습 기본, 2x 실분산 겸용).
# 1노드 학습용 (기본, 메모리 이득 없음·동작 학습):
#   ./run-mini-fsdp.sh
# 2노드 실분산 (spark1+spark2에서 같은 명령, rank0 먼저):
#   spark1: INFRA=dgx-spark-2x MASTER_ADDR=spark1-p1-r0 NODE_RANK=0 ./run-mini-fsdp.sh
#   spark2: INFRA=dgx-spark-2x MASTER_ADDR=spark1-p1-r0 NODE_RANK=1 ./run-mini-fsdp.sh
# 분산 대상: 학습만 (10-baseline, 30-fft, 31-qlora).
# 2x에서는 합성데이터·전략비교·merge·PTQ·PPL·평가는 rank0만 수행 (TP=1).
# NOTE: FSDP+QLoRA는 강의에서 비권장 → 31-qlora는 DDP 동작으로 학습용 수행.
# 하류(merge→평가)는 -single 산출물이 있어야 진행한다:
#   없으면 FSDP 학습 + 12-compare까지만 수행하고 종료 (먼저 ./run-mini-single.sh 실행).
# 사용: SKIP_SETUP=1 ./run-mini-fsdp.sh   # 00-setup 생략
set -euo pipefail
MODE=mini; STRAT=fsdp
export INFRA="${INFRA:-dgx-spark-1x}"
DIST=0; [ "$INFRA" = "dgx-spark-2x" ] && DIST=1
if [ "$DIST" = "1" ]; then
  : "${MASTER_ADDR:?MASTER_ADDR=spark1-p1-r0 (또는 spark1 IP) export 필요}"
  : "${NODE_RANK:?NODE_RANK=0(spark1) | 1(spark2) export 필요}"
fi
: "${NODE_RANK:=0}"
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
source "$ROOT/config/common.env" "$MODE"

is_rank0() { [ "$NODE_RANK" = "0" ]; }
wait_file() { # <path> <timeout_s> — 공유디스크에 파일이 생길 때까지 대기
  local p="$1" t="${2:-3600}" waited=0
  while [ ! -s "$p" ]; do
    sleep 30; waited=$((waited + 30))
    if [ "$waited" -ge "$t" ]; then p5_log "ERROR: 대기 초과 (${t}s): $p"; exit 1; fi
  done
}

p5_log "run-$MODE-$STRAT start (infra=$INFRA rank=$NODE_RANK)"

[ "${SKIP_SETUP:-0}" = 1 ] || ./00-setup.sh

./10-baseline-train.sh "$MODE" "$STRAT"

if [ "$DIST" = "0" ] || is_rank0; then
  export TP=1  # 추론계(vLLM 생성·평가)는 노드 단독 1GPU로 실행 (2x에서 필수)
  uv run 20-select-candidates.py --mode "$MODE"
  uv run 21-build-prompts.py --mode "$MODE"
  ./22-teacher-to-generate-syn-data.sh "$MODE"
  uv run 23-postprocess.py --mode "$MODE"
else
  wait_file "$P5_DATASETS/synthetic-$MODE.jsonl" 3600
fi

./30-fft-train.sh "$MODE" "$STRAT"
./31-qlora-train.sh "$MODE" "$STRAT"

if [ "$DIST" = "1" ] && ! is_rank0; then
  p5_log "run-$MODE-$STRAT done (rank=$NODE_RANK, 학습만 참여)"; exit 0
fi
export TP=1

./12-compare-strategies.sh "$MODE"

if [ ! -d "$P5_MODELS/synthetic-fft-$MODE-single" ] || [ ! -d "$P5_MODELS/synthetic-qlora-$MODE-single" ]; then
  p5_log "하류 SKIP: -single 산출물 없음 → 먼저 ./run-$MODE-single.sh 실행 (FSDP 학습+비교 완료)"
  exit 0
fi

uv run 32-merge-lora.py --mode "$MODE"

if [ -x "${LLAMACPP:-$HOME/git/llama.cpp}/build/bin/llama-quantize" ]; then
  ./40-quant-fft-gguf.sh "$MODE"
  ./50-quant-qlora-gguf.sh "$MODE"
else
  p5_log "WARN: llama-quantize 없음 → GGUF 2종 SKIP (빌드 후 40/50 개별 실행)"
fi
uv run 41-quant-fft-gptq.py --mode "$MODE"
uv run 42-quant-fft-awq.py --mode "$MODE"
uv run 43-quant-fft-fp8.py --mode "$MODE"
uv run 51-quant-qlora-gptq.py --mode "$MODE"
uv run 52-quant-qlora-awq.py --mode "$MODE"
uv run 53-quant-qlora-fp8.py --mode "$MODE"

./60-measure-ppl.sh "$MODE"
./71-eval.sh "$MODE"
uv run 72-score.py --mode "$MODE"

p5_log "run-$MODE-$STRAT done"
# --- 선택 (수동, 2x면 rank0) ---
# ./73-upload-hf.sh mini
# ./80-serve-vllm.sh mini fft            # 터미널1 (상주)
# uv run 81-infer-examples.py --model tayaee/p5-1B-math-fft-mini  # 터미널2

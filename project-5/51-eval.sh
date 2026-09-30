#!/usr/bin/env bash
# 51-eval.sh [mini|full] [target...] — Stage 5b. vLLM greedy GSM8K 추론.
# 원본: evaluation/gen_math_greedy.py (temp 0, max 512, prompt_no_input).
# target: base|fft|qlora|gptq|awq|fp8|all (기본 all). 양자화 타깃은 quant flag 부여.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
TARGETS="${2:-all}"

N=50; [ "$MODE" = full ] && N=0  # 0=전체
p5_log "tp=$TP targets=$TARGETS n=$N"

# STUB: 타깃별 LLM 경로/quant 매핑 후 gen 호출. FFT/QLoRA 타깃은 -single 산출물.
# 테스트 데이터: $P5_DATASETS/gsm8k-test.jsonl (repo project-5/data에 벤더링됨).
#   base  → $BASE_MODEL (HF)
#   fft   → $P5_MODELS/synthetic-fft-$MODE-single
#   qlora → $P5_MODELS/synthetic-qlora-$MODE-single-merged
#   gptq  → ...-single-gptq  (quantization=gptq)
#   awq   → ...-single-awq   (quantization=awq)
#   fp8   → ...-single-fp8   (quantization=fp8)
# 출력: $P5_OUTPUTS/eval-<mode>/<target>.jsonl (kd_data 포함)
echo "STUB: 평가 본문 미구현 (원본 evaluation/gen_math_greedy.py 이식 예정)"

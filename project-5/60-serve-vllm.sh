#!/usr/bin/env bash
# 60-serve-vllm.sh [mini|full] [target] — Stage 6b. vLLM OpenAI-호환 서빙.
# target: fft|qlora|gptq|awq|fp8|gguf (기본 fft).
# SOURCE=local|hf (기본 local; hf면 tayaee/* repo로 서빙). PORT 기본 8000.
# 예: SOURCE=hf ./60-serve-vllm.sh mini gptq
#     61-infer-examples.py --model tayaee/p5-1B-math-gptq-mini (별도 터미널)
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
TARGET="${2:-fft}"
SOURCE="${SOURCE:-local}"
PORT="${PORT:-8000}"

case "$TARGET" in
  fft)   SUF="synthetic-fft-$MODE-single"; QUANT="" ;;
  qlora) SUF="synthetic-qlora-$MODE-single-merged"; QUANT="" ;;
  gptq)  SUF="synthetic-fft-$MODE-single-gptq"; QUANT="gptq" ;;
  awq)   SUF="synthetic-fft-$MODE-single-awq"; QUANT="awq" ;;
  fp8)   SUF="synthetic-fft-$MODE-single-fp8"; QUANT="fp8" ;;
  gguf)  SUF="synthetic-fft-$MODE-single-gguf"; QUANT="gguf" ;;
  *) echo "target: fft|qlora|gptq|awq|fp8|gguf" >&2; exit 1 ;;
esac

# HF repo명 규칙은 53-upload-hf.py TARGETS와 동일
REPO="tayaee/p5-1B-math-${TARGET}-${MODE}"
if [ "$SOURCE" = "hf" ]; then MODEL="$REPO"; else MODEL="$P5_MODELS/$SUF"; fi
[ "$TARGET" = "gguf" ] && [ "$SOURCE" = "local" ] && MODEL="$MODEL/model-q8_0.gguf"

p5_log "serving model=$MODEL quant=${QUANT:-none} tp=$TP port=$PORT"
exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name "$REPO" \
  --tensor-parallel-size "$TP" \
  ${QUANT:+--quantization "$QUANT"} \
  --max-model-len 2048 \
  --gpu-memory-utilization "${GPU_UTIL:-0.9}" \
  --trust-remote-code

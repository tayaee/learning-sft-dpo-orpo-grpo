#!/usr/bin/env bash
# 80-serve-vllm.sh [mini|full] [target] (STRAT env, 기본 single) — Stage 8b. vLLM OpenAI-호환 서빙.
# target: fft|qlora|fft-gptq|fft-awq|fft-fp8|qlora-gptq|qlora-awq|qlora-fp8|fft-gguf|qlora-gguf (기본 fft).
# SOURCE=local|hf (기본 local; hf면 tayaee/* repo로 서빙). PORT 기본 8000.
# 예: SOURCE=hf ./80-serve-vllm.sh mini fft-gptq
#     QTYPE=q4_k_m ./80-serve-vllm.sh mini fft-gguf  # GGUF 변종 선택 (기본 q8_0)
#     81-infer-examples.py --model tayaee/Llama-3.2-1B-math-fft-gptq-mini (별도 터미널)
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
STRAT="${STRAT:-single}"
case "$STRAT" in single|ddp|fsdp) ;; *) echo "STRAT must be single|ddp|fsdp" >&2; exit 1;; esac
TARGET="${2:-fft}"
SOURCE="${SOURCE:-local}"
PORT="${PORT:-8000}"

case "$TARGET" in
  fft)   SUF="synthetic-fft-$MODE-$STRAT"; QUANT="" ;;
  qlora) SUF="synthetic-qlora-$MODE-$STRAT-merged"; QUANT="" ;;
  fft-gptq|gptq)  SUF="synthetic-fft-$MODE-$STRAT-gptq"; QUANT="gptq"; TARGET="fft-gptq" ;;
  # llmcompressor 산출물(AWQ/FP8)은 compressed-tensors 포맷 (71-eval.py와 동일)
  fft-awq|awq)   SUF="synthetic-fft-$MODE-$STRAT-awq"; QUANT="compressed-tensors"; TARGET="fft-awq" ;;
  fft-fp8|fp8)   SUF="synthetic-fft-$MODE-$STRAT-fp8"; QUANT="compressed-tensors"; TARGET="fft-fp8" ;;
  qlora-gptq)  SUF="synthetic-qlora-$MODE-$STRAT-merged-gptq"; QUANT="gptq" ;;
  qlora-awq)   SUF="synthetic-qlora-$MODE-$STRAT-merged-awq"; QUANT="compressed-tensors" ;;
  qlora-fp8)   SUF="synthetic-qlora-$MODE-$STRAT-merged-fp8"; QUANT="compressed-tensors" ;;
  fft-gguf|gguf)  SUF="synthetic-fft-$MODE-$STRAT-gguf"; QUANT="gguf"; TARGET="fft-gguf" ;;
  qlora-gguf)  SUF="synthetic-qlora-$MODE-$STRAT-merged-gguf"; QUANT="gguf" ;;
  *) echo "target: fft|qlora|fft-gptq|fft-awq|fft-fp8|qlora-gptq|qlora-awq|qlora-fp8|fft-gguf|qlora-gguf" >&2; exit 1 ;;
esac

# HF repo명 규칙은 74-upload-hf.py repo_name()과 동일 (single 동결, full 무표기).
# models-tree.txt 설계 결정 3 참조.
MODEL_SLUG="${BASE_MODEL##*/}"
REPO="tayaee/${MODEL_SLUG}-math-${TARGET}"
[ "$STRAT" != single ] && REPO="$REPO-$STRAT"
[ "$MODE" != full ] && REPO="$REPO-$MODE"
if [ "$SOURCE" = "hf" ]; then MODEL="$REPO"; else MODEL="$P5_MODELS/$SUF"; fi
case "$TARGET" in *-gguf) [ "$SOURCE" = "local" ] && MODEL="$MODEL/model-${QTYPE:-q8_0}.gguf" ;; esac

p5_log "serving model=$MODEL quant=${QUANT:-none} tp=$TP port=$PORT"
# FLASH_ATTN 고정 + flashinfer sampler off: flashinfer JIT(sampling 커널) 빌드가
# GB10에서 깨짐 (ninja 실패 → Engine core init 실패)
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER=0
set -x
exec "$VLLM_BIN" serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name "$REPO" \
  --tensor-parallel-size "$TP" \
  ${QUANT:+--quantization "$QUANT"} \
  --max-model-len 2048 \
  --gpu-memory-utilization "${GPU_UTIL:-0.9}" \
  --trust-remote-code

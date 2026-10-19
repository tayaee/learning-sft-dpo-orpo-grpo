#!/usr/bin/env bash
# 62-infer-examples.sh [model] [n] — Stage 8c wrapper. 서빙 중 모델에 추론 예제 요청.
# 인자 없이 실행하면 복사-붙여넣기용 실행 명령 20줄(mini 10+full 10)을 보여준다.
# 예: ./62-infer-examples.sh                              # 실행 명령 목록만 출력
#     ./62-infer-examples.sh tayaee/Llama-3.2-1B-math-fft-mini 3
#     DRY_RUN=1 ./62-infer-examples.sh tayaee/Llama-3.2-1B-math-fft-mini (프롬프트만 출력)
#     BASE_URL=http://localhost:8000/v1 ./62-infer-examples.sh <model>
# (추론 텍스트 자체가 결과이므로 별도 ls 없음.)
set -euo pipefail
if [ $# -eq 0 ]; then
  source "$(dirname "$0")/config/common.env" mini
  STRAT="${STRAT:-single}"
  MODEL_SLUG="${BASE_MODEL##*/}"
  echo "copy & paste (terminal2, 60 기동 후):"
  for m in mini full; do
    for t in fft fft-awq fft-fp8 fft-gguf fft-gptq qlora qlora-awq qlora-fp8 qlora-gguf qlora-gptq; do
      r="tayaee/${MODEL_SLUG}-math-${t}"
      [ "$STRAT" != single ] && r="$r-$STRAT"
      [ "$m" != full ] && r="$r-$m"
      echo "./62-infer-examples.sh $r 3"
    done
  done
  exit 0
fi
source "$(dirname "$0")/config/common.env" "${1:+mini}"
MODEL="${1:?usage: $0 <model> [n]}"
N="${2:-3}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
ARGS=()
[ "${DRY_RUN:-0}" = 1 ] && ARGS+=(--dry-run)
p5_repro_echo
p5_log "model=$MODEL base_url=$BASE_URL n=$N"
echo + uv run "$P5_ROOT/62-infer-examples.py" --model "$MODEL" --base-url "$BASE_URL" --n "$N" "${ARGS[@]}"
uv run "$P5_ROOT/62-infer-examples.py" --model "$MODEL" --base-url "$BASE_URL" --n "$N" "${ARGS[@]}"

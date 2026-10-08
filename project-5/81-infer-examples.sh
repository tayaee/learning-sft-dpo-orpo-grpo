#!/usr/bin/env bash
# 81-infer-examples.sh <model> [n] — Stage 8c wrapper. 서빙 중 모델에 추론 예제 요청.
# 예: ./81-infer-examples.sh tayaee/Llama-3.2-1B-math-fft-mini 3
#     DRY_RUN=1 ./81-infer-examples.sh tayaee/Llama-3.2-1B-math-fft-mini (프롬프트만 출력)
#     BASE_URL=http://localhost:8000/v1 ./81-infer-examples.sh <model>
# (추론 텍스트 자체가 결과이므로 별도 ls 없음.)
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:+mini}"
MODEL="${1:?usage: $0 <model> [n]}"
N="${2:-3}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
ARGS=()
[ "${DRY_RUN:-0}" = 1 ] && ARGS+=(--dry-run)
p5_log "model=$MODEL base_url=$BASE_URL n=$N"
uv run "$P5_ROOT/81-infer-examples.py" --model "$MODEL" --base-url "$BASE_URL" --n "$N" "${ARGS[@]}"

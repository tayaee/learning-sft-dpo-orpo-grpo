#!/usr/bin/env bash
# 22-generate.sh [mini|full] — Stage 2b. vLLM 합성 생성.
# 원본: data_gen_vllm.py (tp=4, max 8192, temp 0.5/top_p 0.8/top_k 5/rep 1.05).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"

MAXLEN=4096; [ "$MODE" = full ] && MAXLEN=8192
p5_log "teacher=$TEACHER_MODEL tp=$TP maxlen=$MAXLEN n=$SYNTH_N"
uv run "$P5_ROOT/22-generate.py" --mode "$MODE" --tp "$TP" --maxlen "$MAXLEN" \
  --teacher "$TEACHER_MODEL" --n "$SYNTH_N"

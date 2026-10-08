#!/usr/bin/env bash
# 40-quant-fft-gguf.sh [mini|full] — Stage 4a. FFT → llama.cpp GGUF F16 → Q8_0.
# 원본: convert_hf_to_gguf.py + quantize (vocab assert 수동패치 불필요, 2026 빌드).
# 입력: FFT **single** 결과 ($P5_MODELS/synthetic-fft-<mode>-single). 전략별 산출물 중
# single만 하류에서 사용한다 (ddp/fsdp는 12-compare-strategies.sh 비교용).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"

SRC="$P5_MODELS/synthetic-fft-$MODE-single"
DST="$P5_MODELS/synthetic-fft-$MODE-single-gguf"
LLAMACPP="${LLAMACPP:-$HOME/git/llama.cpp}"
p5_log "src=$SRC dst=$DST llamacpp=$LLAMACPP"
mkdir -p "$DST"

[ -x "$LLAMACPP/build/bin/llama-quantize" ] || { p5_log "llama-quantize build required (see cheatsheet/PROCEDURE)"; exit 1; }
[ -f "$LLAMACPP/convert_hf_to_gguf.py" ] || { p5_log "convert_hf_to_gguf.py missing"; exit 1; }
"$VENV_BIN/python" "$LLAMACPP/convert_hf_to_gguf.py" "$SRC" --outfile "$DST/model-f16.gguf"
"$LLAMACPP/build/bin/llama-quantize" "$DST/model-f16.gguf" "$DST/model-q8_0.gguf" Q8_0
p5_log "OK: $DST/model-q8_0.gguf"

echo ---- result ----
ls -lh "$DST"

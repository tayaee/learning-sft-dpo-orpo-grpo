#!/usr/bin/env bash
# 52-quant-qlora-gguf.sh [mini|full] — Stage 5c. QLoRA-merged → GGUF F16 → Q8_0.
# 입력: 32-merge-lora.py 결과 ($P5_MODELS/synthetic-qlora-<mode>-single-merged).
# 출력: $P5_MODELS/synthetic-qlora-<mode>-single-merged-gguf/model-q8_0.gguf
# FFT용 42-quant-fft-gguf.sh와 동일 플로우, 입력만 merged.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"

SRC="$P5_MODELS/synthetic-qlora-$MODE-single-merged"
DST="$P5_MODELS/synthetic-qlora-$MODE-single-merged-gguf"
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

#!/usr/bin/env bash
# 42-quant-fft-gguf.sh [mini|full] [QTYPE...] — Stage 4c. FFT → GGUF F16 → 5종 양자화.
# 원본: convert_hf_to_gguf.py + quantize (vocab assert 수동패치 불필요, 2026 빌드).
# 입력: FFT $STRAT 결과 ($P5_MODELS/synthetic-fft-<mode>-$STRAT). (STRAT env, 기본 single)
# 출력: $DST/model-{q8_0,q6_k,q5_k_m,q4_k_m,q3_k_m}.gguf (+ 중간물 model-f16.gguf).
#   ./42-quant-fft-gguf.sh mini                       # 5종 전부
#   QTYPES="Q8_0 Q4_K_M" ./42-quant-fft-gguf.sh mini  # 부분 실행
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
STRAT="${STRAT:-single}"
case "$STRAT" in single|ddp|fsdp) ;; *) echo "STRAT must be single|ddp|fsdp" >&2; exit 1;; esac
shift || true
if [ "$#" -gt 0 ]; then QTYPES="$*"; else QTYPES="${QTYPES:-Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M}"; fi

SRC="$P5_MODELS/synthetic-fft-$MODE-$STRAT"
DST="$P5_MODELS/synthetic-fft-$MODE-$STRAT-gguf"
LLAMACPP="${LLAMACPP:-$HOME/git/llama.cpp}"
p5_require "$SRC/config.json"
p5_log "src=$SRC dst=$DST llamacpp=$LLAMACPP qtypes=$QTYPES"
mkdir -p "$DST"

[ -x "$LLAMACPP/build/bin/llama-quantize" ] || { p5_log "llama-quantize build required (see cheatsheet/PROCEDURE)"; exit 1; }
[ -f "$LLAMACPP/convert_hf_to_gguf.py" ] || { p5_log "convert_hf_to_gguf.py missing"; exit 1; }
if p5_fresh "$DST/model-f16.gguf" "$SRC/config.json"; then
  p5_log "skip: fresh $DST/model-f16.gguf"
else
  (set -x; "$VENV_BIN/python" "$LLAMACPP/convert_hf_to_gguf.py" "$SRC" --outfile "$DST/model-f16.gguf")
fi
for q in $QTYPES; do
  out="model-$(echo "$q" | tr '[:upper:]' '[:lower:]').gguf"
  if p5_fresh "$DST/$out" "$DST/model-f16.gguf"; then p5_log "skip: fresh $out"; continue; fi
  (set -x; "$LLAMACPP/build/bin/llama-quantize" "$DST/model-f16.gguf" "$DST/$out" "$q")
done
p5_log "OK: $DST/model-*.gguf"

echo ---- result ----
(set -x; ls -lh "$DST")

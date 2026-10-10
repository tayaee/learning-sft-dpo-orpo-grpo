#!/usr/bin/env bash
# 52-quant-qlora-gguf.sh [mini|full] [QTYPE...] — Stage 5c. QLoRA-merged → GGUF F16 → 5종 양자화.
# 입력: 32-merge-lora.py 결과 ($P5_MODELS/synthetic-qlora-<mode>-$STRAT-merged). (STRAT env, 기본 single)
# 출력: $DST/model-{q8_0,q6_k,q5_k_m,q4_k_m,q3_k_m}.gguf (+ 중간물 model-f16.gguf).
# FFT용 42-quant-fft-gguf.sh와 동일 플로우, 입력만 merged.
#   ./52-quant-qlora-gguf.sh mini                      # 5종 전부
#   QTYPES="Q8_0 Q4_K_M" ./52-quant-qlora-gguf.sh mini # 부분 실행
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
STRAT="${STRAT:-single}"
case "$STRAT" in single|ddp|fsdp) ;; *) echo "STRAT must be single|ddp|fsdp" >&2; exit 1;; esac
p5_repro_echo
shift || true
if [ "$#" -gt 0 ]; then QTYPES="$*"; else QTYPES="${QTYPES:-Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M}"; fi
echo "+ FORCE=1 MODE=$MODE STRAT=$STRAT $0 $QTYPES"

SRC="$P5_MODELS/synthetic-qlora-$MODE-$STRAT-merged"
DST="$P5_MODELS/synthetic-qlora-$MODE-$STRAT-merged-gguf"
LLAMACPP="${LLAMACPP:-$HOME/git/llama.cpp}"
p5_require "$SRC/config.json"
p5_log "src=$SRC dst=$DST llamacpp=$LLAMACPP qtypes=$QTYPES"
mkdir -p "$DST"

[ -x "$LLAMACPP/build/bin/llama-quantize" ] || { p5_log "llama-quantize build required (see cheatsheet/PROCEDURE)"; exit 1; }
[ -f "$LLAMACPP/convert_hf_to_gguf.py" ] || { p5_log "convert_hf_to_gguf.py missing"; exit 1; }
if p5_fresh "$DST/model-f16.gguf" "$SRC/config.json"; then
  p5_log "skip: fresh $DST/model-f16.gguf"
else
  # 원자 교체: convert 중단 시 잘린 f16이 fresh로 보여 이후 quantize가 매번 깨지는
  # 함정 방지 (42 실측). tmp는 freshness 검사 대상이 아니라 다음 실행이 재변환한다.
  (set -x; "$VENV_BIN/python" "$LLAMACPP/convert_hf_to_gguf.py" "$SRC" --outfile "$DST/model-f16.gguf.tmp" \
    && mv "$DST/model-f16.gguf.tmp" "$DST/model-f16.gguf")
fi
for q in $QTYPES; do
  out="model-$(echo "$q" | tr '[:upper:]' '[:lower:]').gguf"
  if p5_fresh "$DST/$out" "$DST/model-f16.gguf"; then p5_log "skip: fresh $out"; continue; fi
  (set -x; "$LLAMACPP/build/bin/llama-quantize" "$DST/model-f16.gguf" "$DST/$out" "$q")
done
p5_log "OK: $DST/model-*.gguf"

echo ---- result ----
(set -x; ls -lh "$DST")

#!/usr/bin/env bash
# 60-measure-ppl.sh [mini|full] [targets] — Stage 6 wrapper. PPL 측정 + 판정.
# targets: all|base,fft,qlora,fft-gptq,... (콤마 구분, 기본 all).
# 기준: 부모 FP 대비 Δ<0.3 Accept / 0.3~1.0 Conditional(GSM8K 2차) / ≥1.0 Discard.
# GGUF 10종은 llama-perplexity로 측정 (미빌드 시 해당 타깃만 SKIP).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
N="${PPL_N:-32}"
TARGETS="${2:-all}"
SUMMARY="$P5_OUTPUTS/ppl-$MODE/summary.json"
if [ "${FORCE:-0}" != 1 ] && [ -f "$SUMMARY" ]; then
  # 모델 디렉토리 중 SUMMARY보다新しい 것이 하나도 없으면 스킵
  if [ -z "$(find "$P5_MODELS" -maxdepth 1 -newer "$SUMMARY" 2>/dev/null | head -1)" ]; then
    p5_log "skip: fresh $SUMMARY (FORCE=1 to rebuild)"
    echo ---- result ----
    (set -x; ls -l "$P5_OUTPUTS/ppl-$MODE/")
    (set -x; cat "$SUMMARY")
    exit 0
  fi
fi
p5_log "ppl_n=$N targets=$TARGETS"
(set -x; uv run "$P5_ROOT/60-measure-ppl.py" --mode "$MODE" --n "$N" --targets "$TARGETS")

echo ---- result ----
(set -x; ls -l "$P5_OUTPUTS/ppl-$MODE/")
(set -x; cat "$P5_OUTPUTS/ppl-$MODE/summary.json")

#!/usr/bin/env bash
# 60-measure-ppl.sh [mini|full] [targets] — Stage 6 wrapper. PPL 측정 + 판정.
# targets: all|base,fft,qlora,fft-gptq,... (콤마 구분, 기본 all).
# 기준: 부모 FP 대비 Δ<0.3 Accept / 0.3~1.0 Conditional(GSM8K 2차) / ≥1.0 Discard.
# GGUF 2종은 transformers PPL 불가 → SKIP (llama.cpp perplexity 별도).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
N="${PPL_N:-32}"
TARGETS="${2:-all}"
p5_log "ppl_n=$N targets=$TARGETS"
uv run "$P5_ROOT/60-measure-ppl.py" --mode "$MODE" --n "$N" --targets "$TARGETS"

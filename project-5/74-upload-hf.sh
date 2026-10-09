#!/usr/bin/env bash
# 74-upload-hf.sh [mini|full] [targets] — Stage 7a. HF Hub 업로드 (tayaee/*).
# targets: all|fft|qlora|fft-gptq|...|qlora-gguf (콤마 구분 가능, 기본 all).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
(set -x; uv run "$P5_ROOT/74-upload-hf.py" --mode "$MODE" --targets "${2:-all}")

echo ---- result ----
p5_log "uploaded targets=${2:-all} (see URLs above)"

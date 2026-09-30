#!/usr/bin/env bash
# 53-upload-hf.sh [mini|full] [targets] — Stage 6a. HF Hub 업로드 (tayaee/*).
# targets: all|fft|qlora|gptq|awq|fp8|gguf (콤마 구분 가능, 기본 all).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"
uv run "$P5_ROOT/53-upload-hf.py" --mode "$MODE" --targets "${2:-all}"

#!/usr/bin/env bash
# ex02_save_model.sh — step 02: local save from ex01 adapter.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${MODEL:-LiquidAI/LFM2.5-1.2B-Instruct}"
ADAPTER="${ADAPTER:-unsloth-output}"
OUT_DIR="${OUT_DIR:-gguf-output}"
MERGED_DIR="${MERGED_DIR:-merged-output}"
QUANT="${QUANT:-all}"

ARGS=(--model "$MODEL" --adapter "$ADAPTER" --out-dir "$OUT_DIR" --merged-dir "$MERGED_DIR" --quant "$QUANT")

exec uv run ./ex02_save_model.py "${ARGS[@]}" "$@"

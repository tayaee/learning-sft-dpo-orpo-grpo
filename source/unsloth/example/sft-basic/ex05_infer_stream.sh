#!/usr/bin/env bash
# ex05_infer_stream.sh — step 05: streaming inference on ex02 GGUF.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${MODEL:-LiquidAI/LFM2.5-1.2B-Instruct}"
ADAPTER="${ADAPTER:-}"
GGUF_DIR="${GGUF_DIR:-gguf-output}"
QUANT="${QUANT:-5}"
PROMPT="${PROMPT:-What is 1+1?}"
SYSTEM="${SYSTEM:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

ARGS=(--model "$MODEL" --quant "$QUANT" --prompt "$PROMPT" --max-new-tokens "$MAX_NEW_TOKENS")
[[ -n "$ADAPTER" ]] && ARGS+=(--adapter "$ADAPTER")
if [[ -n "$GGUF_DIR" ]]; then ARGS+=(--gguf-dir "$GGUF_DIR"); else ARGS+=(--no-gguf); fi
[[ -n "$SYSTEM" ]] && ARGS+=(--system "$SYSTEM")

exec uv run ./ex05_infer_stream.py "${ARGS[@]}" "$@"

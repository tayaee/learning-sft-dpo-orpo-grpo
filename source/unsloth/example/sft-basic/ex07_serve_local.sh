#!/usr/bin/env bash
# ex07_serve_local.sh — step 07: vLLM serve ex02 local output.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="${MODEL_DIR:-merged-output}"
GGUF_DIR="${GGUF_DIR:-}"
QUANT="${QUANT:-4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

ARGS=(--model-dir "$MODEL_DIR" --quant "$QUANT" --host "$HOST" --port "$PORT" --run)
[[ -n "$GGUF_DIR" ]] && ARGS+=(--gguf-dir "$GGUF_DIR")

exec uv run ./ex07_serve_local.py "${ARGS[@]}" "$@"

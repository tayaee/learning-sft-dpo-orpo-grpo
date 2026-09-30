#!/usr/bin/env bash
# ex06_serve_hf.sh — step 06: vLLM serve ex03 Hub repo.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

REPO="${REPO:-tayaee/my-model-gguf}"
QUANT="${QUANT:-4}"
FILENAME="${FILENAME:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

ARGS=(--repo "$REPO" --quant "$QUANT" --host "$HOST" --port "$PORT" --run)
[[ -n "$FILENAME" ]] && ARGS+=(--filename "$FILENAME")

exec uv run ./ex06_serve_hf.py "${ARGS[@]}" "$@"

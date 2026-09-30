#!/usr/bin/env bash
# ex03_upload_model.sh — step 03: upload ex02 GGUF files.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

GGUF_DIR="${GGUF_DIR:-gguf-output}"
REPO="${REPO:-tayaee/my-model-gguf}"
QUANT="${QUANT:-all}"

exec uv run ./ex03_upload_model.py --gguf-dir "$GGUF_DIR" --repo "$REPO" --quant "$QUANT" "$@"

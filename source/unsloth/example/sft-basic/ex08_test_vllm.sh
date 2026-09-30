#!/usr/bin/env bash
# ex08_test_vllm.sh — step 08: inference test vs ex06/ex07 server.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-tayaee/my-model-gguf}"
QUANT="${QUANT:-all}"
PROMPT="${PROMPT:-What is 1+1?}"

exec uv run ./ex08_test_vllm.py --base-url "$BASE_URL" --model "$MODEL" --quant "$QUANT" --prompt "$PROMPT" "$@"

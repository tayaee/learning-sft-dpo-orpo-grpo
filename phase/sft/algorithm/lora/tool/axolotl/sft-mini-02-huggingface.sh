#!/usr/bin/env bash
# sft-02-huggingface.sh: Hugging Face CLI 로그인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ ! -f "$ROOT_DIR/pyproject.toml" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR/library/function:$ROOT_DIR/library/script${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/library/script/scripts_common.sh"
parse_flags "$@"
set -euo pipefail

echo "input: (interactive token)"
echo "output: ~/.cache/huggingface/token"

set -x
hf auth login
set +x

echo "input: (interactive token)"
echo "output: ~/.cache/huggingface/token"

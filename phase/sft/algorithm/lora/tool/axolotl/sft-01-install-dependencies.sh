#!/usr/bin/env bash
# sft-01-install-dependencies.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ ! -f "$ROOT_DIR/pyproject.toml" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR/library/function:$ROOT_DIR/library/script${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/library/script/scripts_common.sh"
parse_flags "$@"
set -euo pipefail

echo "input: pyproject.toml, uv.lock"
echo "output: .venv"

do_sync() {
  UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu130}" uv sync
}

_make ".venv" "pyproject.toml" "uv.lock" -- do_sync

echo "input: pyproject.toml, uv.lock"
echo "output: .venv"

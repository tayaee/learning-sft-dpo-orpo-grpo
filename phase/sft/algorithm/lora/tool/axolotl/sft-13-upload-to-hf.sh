#!/bin/bash
# sft-13: HF Hub 업로드 (평가 완료 후)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ ! -f "$ROOT_DIR/pyproject.toml" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR/library/function:$ROOT_DIR/library/script${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/library/script/scripts_common.sh"
parse_flags "$@"
MODE=$(require_mode "${1:-full}" "$0" "$@") || exit 1

if [ "$MODE" = "mini" ]; then
  SRC=./data/sft-mini-out/merged
  REPO=tayaee/Qwen2.5-1.5B-Korean-SFT-mini
else
  SRC=./data/sft-full-out/merged
  REPO=tayaee/Qwen2.5-1.5B-Korean-SFT
fi

echo "input: $SRC"
echo "output: Hugging Face Hub ($REPO)"

if [ ! -d "$SRC" ]; then
  echo "ERROR: $SRC 가 없습니다. 먼저 sft-06 + sft-07 을 --mode $MODE 로 실행하세요." >&2
  exit 1
fi

set -x
uv run hf auth login --token $HF_TOKEN
uv run hf auth whoami
uv run hf upload "$REPO" "$SRC" . --commit-message "Upload Qwen2.5-1.5B Korean SFT model ($MODE)"
set +x

echo "input: $SRC"
echo "output: Hugging Face Hub ($REPO)"

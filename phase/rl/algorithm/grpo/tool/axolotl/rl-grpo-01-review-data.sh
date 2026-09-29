#!/bin/bash
# rl-grpo-01: GRPO 학습 데이터 검증
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ ! -f "$ROOT_DIR/pyproject.toml" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR/library/function:$ROOT_DIR/library/script${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/library/script/scripts_common.sh"
parse_flags "$@"
MODE=$(require_mode "${1:-}" "$0" "$@") || exit 1
SAMPLE="data/grpo-$MODE-out/sample_rl-grpo.jsonl"

echo "input: $SAMPLE"
echo "output: (stdout)"

if [ ! -f "$SAMPLE" ]; then
  echo "ERROR: $SAMPLE 가 없습니다. 파일을 준비하거나 다른 모드를 선택하세요." >&2
  exit 1
fi

set -x
wc -l "$SAMPLE"
head -1 "$SAMPLE" | uv run python -m json.tool | head -10
set +x

echo "input: $SAMPLE"
echo "output: (stdout)"

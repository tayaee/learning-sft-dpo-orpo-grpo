#!/bin/bash
# rl-orpo-06: ORPO 모델 추론 테스트
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ ! -f "$ROOT_DIR/pyproject.toml" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR/library/function:$ROOT_DIR/library/script${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/library/script/scripts_common.sh"
parse_flags "$@"
MODE=$(require_mode "${1:-}" "$0" "$@") || exit 1

if [ "$MODE" = "mini" ]; then
  MODEL=./data/orpo-mini-out/merged
else
  MODEL=./data/orpo-full-out/merged
fi

echo "input: $MODEL"
echo "output: (stdout)"

set -x
uv run "$ROOT_DIR/library/script/query_rl_orpo.py" --mode "$MODE" "방정식 x^2 + 5x + 6 = 0 의 해를 구하시오."
set +x

echo "input: $MODEL"
echo "output: (stdout)"

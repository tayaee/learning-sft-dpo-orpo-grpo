#!/bin/bash
# grpo2: GRPO reward 함수 검증
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ ! -f "$ROOT_DIR/pyproject.toml" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR/library/function:$ROOT_DIR/library/script${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/library/script/scripts_common.sh"
parse_flags "$@"

echo "input: library/function/reward_fn.py"
echo "output: (stdout)"

set -x
PYTHONPATH="$ROOT_DIR/library/function${PYTHONPATH:+:$PYTHONPATH}" uv run python -c "import reward_fn; print('reward functions:', [f for f in dir(reward_fn) if f.endswith('_reward')])"
cat "$ROOT_DIR/library/function/reward_fn.py"
set +x

echo "input: library/function/reward_fn.py"
echo "output: (stdout)"

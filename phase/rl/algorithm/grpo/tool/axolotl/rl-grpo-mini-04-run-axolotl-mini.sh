#!/bin/bash
# rl-grpo-04 mini: GRPO mini 학습
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ ! -f "$ROOT_DIR/pyproject.toml" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR/library/function:$ROOT_DIR/library/script${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/library/script/scripts_common.sh"
parse_flags "$@"

CFG=phase/rl/algorithm/grpo/tool/axolotl/config/qwen2.5-1.5b-rl-grpo-mini.yaml
LOG=logs/rl-grpo-mini.log
OUT_DIR=./data/grpo-mini-out/adapter
DATA_IN="data/grpo-mini-in/sample_rl-grpo.jsonl"

echo "input: $CFG, $DATA_IN, library/function/reward_fn.py"
echo "output: $OUT_DIR, $LOG"

mkdir -p logs

do_train() {
  uv run axolotl train "$CFG" 2>&1 | tee "$LOG"
}

_make "$OUT_DIR" "$CFG" "$DATA_IN" "library/function/reward_fn.py" -- do_train

echo "input: $CFG, $DATA_IN, library/function/reward_fn.py"
echo "output: $OUT_DIR, $LOG"

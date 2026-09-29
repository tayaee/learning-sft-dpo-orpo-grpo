#!/bin/bash
# rl-orpo-08: SFT vs ORPO 정량 비교 — 휴리스틱 reward 평균 점수
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ ! -f "$ROOT_DIR/pyproject.toml" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR/library/function:$ROOT_DIR/library/script${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/library/script/scripts_common.sh"
parse_flags "$@"
MODE=$(require_mode "${1:-}" "$0" "$@") || exit 1

if [ "$MODE" = "mini" ]; then
  N=10
  SFT_MODEL=./data/sft-mini-out/merged
  ORPO_MODEL=./data/orpo-mini-out/merged
else
  N=50
  SFT_MODEL=./data/sft-full-out/merged
  ORPO_MODEL=./data/orpo-full-out/merged
fi
OUT="outputs/eval_results/orpo-reward-$MODE.jsonl"

echo "input: $SFT_MODEL, $ORPO_MODEL, library/function/reward_fn.py"
echo "output: $OUT"

mkdir -p outputs/eval_results

do_eval() {
  uv run python "$ROOT_DIR/library/script/eval_orpo_reward.py" --mode "$MODE" --num-prompts "$N" \
    --out "$OUT"
}

_make "$OUT" "$SFT_MODEL" "$ORPO_MODEL" "library/function/reward_fn.py" -- do_eval

echo "input: $SFT_MODEL, $ORPO_MODEL, library/function/reward_fn.py"
echo "output: $OUT"

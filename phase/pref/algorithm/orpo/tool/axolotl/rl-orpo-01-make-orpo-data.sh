#!/bin/bash
# rl-orpo-01: ORPO 선호 데이터 생성 (mini / full 선택 — 반드시 지정)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ ! -f "$ROOT_DIR/pyproject.toml" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR/library/function:$ROOT_DIR/library/script${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/library/script/scripts_common.sh"
parse_flags "$@"
MODE=$(require_mode "${1:-}" "$0" "$@") || exit 1

if [ "$MODE" = "mini" ]; then
  N=20
else
  N=1000
fi

FROM_DPO="data/dpo-$MODE-out/train_rl-dpo.jsonl"
OUT="data/orpo-$MODE-out/train_rl-orpo.jsonl"

echo "input: $FROM_DPO"
echo "output: $OUT"

if [ ! -f "$FROM_DPO" ]; then
  echo "ERROR: $FROM_DPO 가 없습니다. 먼저 phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-01-make-rl-dpo-data.sh $MODE 실행하세요." >&2
  exit 1
fi

mkdir -p "data/orpo-$MODE-out"

_make "$OUT" "$FROM_DPO" -- \
  uv run python "$ROOT_DIR/library/script/make_rl_orpo_data.py" \
    --from-dpo "$FROM_DPO" \
    --num-prompts "$N" \
    --out "$OUT"

echo "input: $FROM_DPO"
echo "output: $OUT"

#!/bin/bash
# rl-orpo-07: SFT vs ORPO 정성 비교 (동일 질문 side-by-side)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ ! -f "$ROOT_DIR/pyproject.toml" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR/library/function:$ROOT_DIR/library/script${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/library/script/scripts_common.sh"
parse_flags "$@"
MODE=$(require_mode "${1:-}" "$0" "$@") || exit 1

if [ "$MODE" = "mini" ]; then
  SFT_MODEL=./data/sft-mini-out/merged
  ORPO_MODEL=./data/orpo-mini-out/merged
else
  SFT_MODEL=./data/sft-full-out/merged
  ORPO_MODEL=./data/orpo-full-out/merged
fi

echo "input: $SFT_MODEL, $ORPO_MODEL"
echo "output: (stdout)"

Q="피타고라스 정리를 증명하시오."

echo "===== SFT ($MODE) ====="
set -x
uv run "$ROOT_DIR/library/script/query_sft.py" --mode "$MODE" "$Q"
set +x
echo "===== ORPO ($MODE) ====="
set -x
uv run "$ROOT_DIR/library/script/query_rl_orpo.py" --mode "$MODE" "$Q"
set +x

echo "input: $SFT_MODEL, $ORPO_MODEL"
echo "output: (stdout)"

#!/bin/bash
# rl-grpo-08: 4 모델 비교 — 동일 질문(피타고라스 정리)을 모든 모델에 던져 정성 비교
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
  DPO_MODEL=./data/dpo-mini-out/merged
  GRPO_MODEL=./data/grpo-mini-out/merged
else
  SFT_MODEL=./data/sft-full-out/merged
  DPO_MODEL=./data/dpo-full-out/merged
  GRPO_MODEL=./data/grpo-full-out/merged
fi

echo "input: Qwen/Qwen2.5-1.5B-Instruct, $SFT_MODEL, $DPO_MODEL, $GRPO_MODEL"
echo "output: (stdout)"

Q="피타고라스 정리를 증명하시오."
set -x
uv run "$ROOT_DIR/library/script/query_base.py"       "$Q"
uv run "$ROOT_DIR/library/script/query_sft.py" --mode "$MODE" "$Q"
uv run "$ROOT_DIR/library/script/query_rl_dpo.py" --mode "$MODE"     "$Q"
uv run "$ROOT_DIR/library/script/query_rl_grpo.py" --mode "$MODE"    "$Q"
set +x

echo "input: Qwen/Qwen2.5-1.5B-Instruct, $SFT_MODEL, $DPO_MODEL, $GRPO_MODEL"
echo "output: (stdout)"

#!/bin/bash
# rl-dpo-01: DPO 선호 데이터 생성 (mini / full 선택 — 반드시 지정)
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
  K=2
  SFT_MODEL=./data/sft-mini-out/merged
else
  N=1000
  K=4
  SFT_MODEL=./data/sft-full-out/merged
fi

OUT="data/dpo-$MODE-out/train_rl-dpo.jsonl"
SAMPLE="data/dpo-$MODE-out/sample_rl-dpo.jsonl"

echo "input: $SFT_MODEL, Qwen/Qwen2.5-1.5B-Instruct"
echo "output: $OUT, $SAMPLE"

if [ ! -d "$SFT_MODEL" ]; then
  echo "ERROR: $SFT_MODEL 가 없습니다. 먼저 sft 학습+merge 를 --mode $MODE 로 완료하세요." >&2
  exit 1
fi

ensure_train_jsonl || exit 1

mkdir -p "data/dpo-$MODE-out" logs

_make "$OUT" "$SFT_MODEL" -- \
  uv run python "$ROOT_DIR/library/script/make_rl_dpo_data.py" \
    --sft-model "$SFT_MODEL" \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --num-prompts "$N" \
    --samples-per-prompt "$K" \
    --out "$OUT" \
    --sample-out "$SAMPLE"

echo "input: $SFT_MODEL, Qwen/Qwen2.5-1.5B-Instruct"
echo "output: $OUT, $SAMPLE"

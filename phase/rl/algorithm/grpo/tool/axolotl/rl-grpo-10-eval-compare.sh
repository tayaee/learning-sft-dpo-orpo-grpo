#!/bin/bash
# rl-grpo-10: 이전 모델(SFT merge) vs 새 모델(GRPO merge) 비교 평가 — 2개 평가 + 비교표
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ ! -f "$ROOT_DIR/pyproject.toml" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR/library/function:$ROOT_DIR/library/script${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/library/script/scripts_common.sh"
parse_flags "$@"
MODE=$(require_mode "${1:-}" "$0" "$@") || exit 1

GENERAL_TASKS=tinyArc,tinyHellaswag,tinyMMLU,tinyWinogrande
KOREAN_TASKS=kobest_copa,kobest_hellaswag

if [ "$MODE" = "mini" ]; then
  PREV_MODEL=./data/sft-mini-out/merged   # 이전 모델 = SFT(mini)
  NEW_MODEL=./data/grpo-mini-out/merged
  PREV_LABEL="SFT(mini)"
  NEW_LABEL="GRPO(mini)"
else
  PREV_MODEL=./data/sft-full-out/merged         # 이전 모델 = SFT(full)
  NEW_MODEL=./data/grpo-full-out/merged
  PREV_LABEL="SFT(full)"
  NEW_LABEL="GRPO(full)"
fi
OUT=outputs/lm_eval_results/rl-grpo-$MODE
TABLE="$OUT/comparison-table.md"

echo "input: $PREV_MODEL, $NEW_MODEL"
echo "output: $OUT, $TABLE"

[ -d "$PREV_MODEL" ] || { echo "ERROR: $PREV_MODEL 없음 — 먼저 sft-06 + sft-07 을 --mode $MODE 로 실행하세요." >&2; exit 1; }
[ -d "$NEW_MODEL" ] || { echo "ERROR: $NEW_MODEL 없음 — 먼저 rl-grpo-04 + rl-grpo-06 을 --mode $MODE 로 실행하세요." >&2; exit 1; }

run_eval () {  # $1=model  $2=tasks  $3=outdir
  mkdir -p "$OUT"
  uv run lm_eval \
    --model hf \
    --model_args pretrained="$1",dtype=bfloat16 \
    --tasks "$2" \
    --apply_chat_template \
    --batch_size 8 \
    --output_path "$OUT/$3"
}

do_compare() {
  run_eval "$PREV_MODEL" "$GENERAL_TASKS" prev-general
  run_eval "$NEW_MODEL"  "$GENERAL_TASKS" new-general
  run_eval "$PREV_MODEL" "$KOREAN_TASKS"  prev-korean
  run_eval "$NEW_MODEL"  "$KOREAN_TASKS"  new-korean

  uv run python "$ROOT_DIR/library/script/eval_compare_table.py" "$OUT" \
    --labels "prev=$PREV_LABEL,new=$NEW_LABEL" | tee "$TABLE"
}

_make "$TABLE" "$PREV_MODEL" "$NEW_MODEL" -- do_compare

echo "input: $PREV_MODEL, $NEW_MODEL"
echo "output: $OUT, $TABLE"

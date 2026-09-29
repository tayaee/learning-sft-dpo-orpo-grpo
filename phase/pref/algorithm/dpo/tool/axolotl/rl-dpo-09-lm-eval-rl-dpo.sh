#!/bin/bash
# rl-dpo-09: DPO 모델 lm-eval (한국어 + 영어 tasks)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ ! -f "$ROOT_DIR/pyproject.toml" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR/library/function:$ROOT_DIR/library/script${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/library/script/scripts_common.sh"
parse_flags "$@"
MODE=$(require_mode "${1:-}" "$0" "$@") || exit 1

if [ "$MODE" = "mini" ]; then
  MODEL=./data/dpo-mini-out/merged
else
  MODEL=./data/dpo-full-out/merged
fi

OUT=./outputs/lm_eval_results/rl-dpo-$MODE

echo "input: $MODEL"
echo "output: $OUT"

if [ ! -d "$MODEL" ]; then
  echo "ERROR: $MODEL 가 없습니다. 먼저 dpo 학습+merge 를 --mode $MODE 로 완료하세요." >&2
  exit 1
fi

do_eval() {
  uv run lm_eval \
    --model hf \
    --model_args pretrained="$MODEL",dtype=bfloat16 \
    --tasks kobest_hellaswag,kobest_copa,kmmlu,hellaswag,arc_easy,piqa,winogrande \
    --apply_chat_template \
    --batch_size 8 \
    --limit 100 \
    --output_path "$OUT"
}

_make "$OUT" "$MODEL" -- do_eval

echo "input: $MODEL"
echo "output: $OUT"

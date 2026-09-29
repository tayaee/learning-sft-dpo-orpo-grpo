#!/bin/bash
# rl-dpo-02: 같은 prompt 에 대한 chosen vs rejected 비교
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ ! -f "$ROOT_DIR/pyproject.toml" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR/library/function:$ROOT_DIR/library/script${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/library/script/scripts_common.sh"
parse_flags "$@"
MODE=$(require_mode "${1:-}" "$0" "$@") || exit 1
SAMPLE="data/dpo-$MODE-out/sample_rl-dpo.jsonl"
DATA="data/dpo-$MODE-out/train_rl-dpo.jsonl"

echo "input: $SAMPLE, $DATA"
echo "output: (stdout)"

if [ ! -f "$SAMPLE" ]; then
  echo "ERROR: $SAMPLE 가 없습니다. 먼저 phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-01-make-rl-dpo-data.sh $MODE 실행하세요." >&2
  exit 1
fi

set -x
head -1 "$SAMPLE" | uv run python -c "
import json, sys
d = json.loads(sys.stdin.read())
print('=== PROMPT ===')
print(d['prompt'][:200])
print('\n=== CHOSEN ===')
print(d['chosen'][:400])
print('\n=== REJECTED ===')
print(d['rejected'][:400])
"
wc -l "$SAMPLE" "$DATA"
set +x

echo "input: $SAMPLE, $DATA"
echo "output: (stdout)"

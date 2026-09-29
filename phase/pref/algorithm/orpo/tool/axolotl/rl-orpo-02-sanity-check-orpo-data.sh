#!/bin/bash
# rl-orpo-02: ORPO 데이터 sanity check — chosen vs rejected 눈으로 비교
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ ! -f "$ROOT_DIR/pyproject.toml" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR/library/function:$ROOT_DIR/library/script${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/library/script/scripts_common.sh"
parse_flags "$@"
MODE=$(require_mode "${1:-}" "$0" "$@") || exit 1
DATA="data/orpo-$MODE-out/train_rl-orpo.jsonl"

if [ "$MODE" = "mini" ]; then
  CFG="phase/pref/algorithm/orpo/tool/axolotl/config/qwen2.5-1.5b-rl-orpo-mini.yaml"
else
  CFG="phase/pref/algorithm/orpo/tool/axolotl/config/qwen2.5-1.5b-rl-orpo.yaml"
fi

echo "input: $DATA, $CFG"
echo "output: (stdout)"

if [ ! -f "$DATA" ]; then
  echo "ERROR: $DATA 가 없습니다. 먼저 phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-01-make-orpo-data.sh $MODE 실행하세요." >&2
  exit 1
fi

set -x
head -1 "$DATA" | uv run python -c "
import json, sys
d = json.loads(sys.stdin.read())
print('=== PROMPT ===')
print(d['prompt'][:200])
print('\n=== CHOSEN (assistant) ===')
print(d['chosen'][-1]['content'][:400])
print('\n=== REJECTED (assistant) ===')
print(d['rejected'][-1]['content'][:400])
assert d['chosen'][0]['role'] == 'user' and d['chosen'][1]['role'] == 'assistant'
assert len(d['chosen']) == len(d['rejected']) == 2
print('\n[ok] message-list format valid')
"
wc -l "$DATA"
cat "$CFG"
set +x

echo "input: $DATA, $CFG"
echo "output: (stdout)"

#!/bin/bash
# probe_r1_reasoning.sh — OpenRouter reasoning/content 분리 프로브 실행 래퍼
# stage 파이프라인이 아니라 공용 관측 도구이므로 library/script에 상주.
# 사용법: library/script/probe_r1_reasoning.sh ["질문..."] [--model ...] [--out ...] [--no-save]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ ! -f "$ROOT_DIR/pyproject.toml" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$ROOT_DIR" || exit 1

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "ERROR: OPENROUTER_API_KEY 환경 변수가 설정되지 않았습니다." >&2
  echo "  export OPENROUTER_API_KEY=... 후 다시 실행하세요." >&2
  exit 1
fi

if [ $# -eq 0 ]; then
  set -- "한 유리잔은 5달러인데 두 번째 잔마다 40%를 할인해 준다고 한다. 16잔을 사려면 총 얼마가 필요한가?"
fi

echo "input: (OpenRouter API, model via --model)"
echo "output: (stdout + outputs/reasoning-probe/*.jsonl)"

set -x
uv run --script "$ROOT_DIR/library/script/probe_r1_reasoning.py" "$@"
set +x

echo "input: (OpenRouter API, model via --model)"
echo "output: (stdout + outputs/reasoning-probe/*.jsonl)"

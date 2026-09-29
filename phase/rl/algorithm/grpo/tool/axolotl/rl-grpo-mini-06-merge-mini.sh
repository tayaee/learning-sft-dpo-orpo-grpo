#!/bin/bash
# rl-grpo-06-merge-mini.sh — rl-grpo-06-merge.sh 의 mini 모드 래퍼 (인자는 그대로 전달)
exec "$(dirname "$0")/rl-grpo-06-merge.sh" mini "$@"

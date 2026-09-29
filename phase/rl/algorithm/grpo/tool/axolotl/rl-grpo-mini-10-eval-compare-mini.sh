#!/bin/bash
# rl-grpo-10-eval-compare-mini.sh — rl-grpo-10-eval-compare.sh 의 mini 모드 래퍼 (인자는 그대로 전달)
exec "$(dirname "$0")/rl-grpo-10-eval-compare.sh" mini "$@"

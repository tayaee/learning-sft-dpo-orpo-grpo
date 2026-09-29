#!/bin/bash
# rl-grpo-08-pythagorean-theorem-mini.sh — rl-grpo-08-pythagorean-theorem.sh 의 mini 모드 래퍼 (인자는 그대로 전달)
exec "$(dirname "$0")/rl-grpo-08-pythagorean-theorem.sh" mini "$@"

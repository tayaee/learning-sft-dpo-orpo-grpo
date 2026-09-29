#!/bin/bash
# rl-grpo-01-review-data-mini.sh — rl-grpo-01-review-data.sh 의 mini 모드 래퍼 (인자는 그대로 전달)
exec "$(dirname "$0")/rl-grpo-01-review-data.sh" mini "$@"

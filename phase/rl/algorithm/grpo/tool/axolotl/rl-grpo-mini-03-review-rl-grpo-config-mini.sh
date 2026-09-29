#!/bin/bash
# rl-grpo-03-review-rl-grpo-config-mini.sh — rl-grpo-03-review-rl-grpo-config.sh 의 mini 모드 래퍼 (인자는 그대로 전달)
exec "$(dirname "$0")/rl-grpo-03-review-rl-grpo-config.sh" mini "$@"

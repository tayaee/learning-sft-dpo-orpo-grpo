#!/bin/bash
# rl-grpo-09-lm-eval-rl-grpo-mini.sh — rl-grpo-09-lm-eval-rl-grpo.sh 의 mini 모드 래퍼 (인자는 그대로 전달)
exec "$(dirname "$0")/rl-grpo-09-lm-eval-rl-grpo.sh" mini "$@"

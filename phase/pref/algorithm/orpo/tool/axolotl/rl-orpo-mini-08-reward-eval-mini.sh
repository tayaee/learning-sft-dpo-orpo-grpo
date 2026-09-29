#!/bin/bash
# rl-orpo-08-reward-eval-mini.sh — rl-orpo-08-reward-eval.sh 의 mini 모드 래퍼 (인자는 그대로 전달)
exec "$(dirname "$0")/rl-orpo-08-reward-eval.sh" mini "$@"

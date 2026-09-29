#!/bin/bash
# rl-orpo-09-lm-eval-rl-orpo-mini.sh — rl-orpo-09-lm-eval-rl-orpo.sh 의 mini 모드 래퍼 (인자는 그대로 전달)
exec "$(dirname "$0")/rl-orpo-09-lm-eval-rl-orpo.sh" mini "$@"

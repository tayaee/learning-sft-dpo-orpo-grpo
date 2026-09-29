#!/bin/bash
# rl-orpo-01-make-orpo-data-mini.sh — rl-orpo-01-make-orpo-data.sh 의 mini 모드 래퍼 (인자는 그대로 전달)
exec "$(dirname "$0")/rl-orpo-01-make-orpo-data.sh" mini "$@"

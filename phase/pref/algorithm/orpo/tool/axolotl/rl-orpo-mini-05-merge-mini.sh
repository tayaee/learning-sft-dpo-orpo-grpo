#!/bin/bash
# rl-orpo-05-merge-mini.sh — rl-orpo-05-merge.sh 의 mini 모드 래퍼 (인자는 그대로 전달)
exec "$(dirname "$0")/rl-orpo-05-merge.sh" mini "$@"

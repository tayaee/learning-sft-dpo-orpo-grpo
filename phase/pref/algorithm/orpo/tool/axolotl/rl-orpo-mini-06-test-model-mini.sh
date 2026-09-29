#!/bin/bash
# rl-orpo-06-test-model-mini.sh — rl-orpo-06-test-model.sh 의 mini 모드 래퍼 (인자는 그대로 전달)
exec "$(dirname "$0")/rl-orpo-06-test-model.sh" mini "$@"

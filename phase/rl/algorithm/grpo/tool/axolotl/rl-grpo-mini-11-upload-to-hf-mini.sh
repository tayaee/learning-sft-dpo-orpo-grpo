#!/bin/bash
# rl-grpo-mini-11-upload-to-hf-mini.sh — rl-grpo-11-upload-to-hf.sh 의 mini 모드 래퍼
exec "$(dirname "$0")/rl-grpo-11-upload-to-hf.sh" mini "$@"

#!/bin/bash
# rl-dpo-mini-11-upload-to-hf-mini.sh — rl-dpo-11-upload-to-hf.sh 의 mini 모드 래퍼
exec "$(dirname "$0")/rl-dpo-11-upload-to-hf.sh" mini "$@"

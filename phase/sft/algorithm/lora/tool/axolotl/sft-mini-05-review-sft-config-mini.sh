#!/bin/bash
# sft-05-review-sft-config-mini.sh — sft-05-review-sft-config.sh 의 mini 모드 래퍼 (인자는 그대로 전달)
exec "$(dirname "$0")/sft-05-review-sft-config.sh" mini "$@"

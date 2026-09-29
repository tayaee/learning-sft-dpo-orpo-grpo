#!/bin/bash
# rl-dpo-02-sanity-check-rl-dpo-data-mini.sh — rl-dpo-02-sanity-check-rl-dpo-data.sh 의 mini 모드 래퍼 (인자는 그대로 전달)
exec "$(dirname "$0")/rl-dpo-02-sanity-check-rl-dpo-data.sh" mini "$@"

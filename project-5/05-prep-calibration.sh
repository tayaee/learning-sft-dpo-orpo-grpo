#!/usr/bin/env bash
# 05-prep-calibration.sh [n] [seed] — gsm8k-train → gsm8k-calibration-<n>.jsonl.
# AWQ 예제 정석(256) 기본. 40/50-awq가 이 파일을 캘리브레이션으로 읽는다.
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${MODE:-mini}"
N="${1:-256}"
SEED="${2:-42}"
export UV_NO_SYNC=1; unset VIRTUAL_ENV
uv run "$P5_ROOT/05-prep-calibration.py" --n "$N" --seed "$SEED"

echo ---- result ----
ls -lh "$P5_DATASETS/gsm8k-calibration-$N.jsonl"
wc -l "$P5_DATASETS/gsm8k-calibration-$N.jsonl"

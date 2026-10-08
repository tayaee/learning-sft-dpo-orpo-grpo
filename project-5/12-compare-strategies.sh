#!/usr/bin/env bash
# 12-compare-strategies.sh [mini|full] — Stage 1 산출물 3전략 비교.
# single/ddp/fsdp 체크포인트의 safetensors 해시를 대조해
# "결과가 똑같은가?"를 직접 확인한다. config.json은 동일해야 정상,
# 가중치 해시는 effective batch를 맞춰도 bit-identical이 아닐 수 있다
# (allreduce vs reduce-scatter 합산 순서, dataloader 샤딩 차이).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"

for s in single ddp fsdp; do
  d="$P5_MODELS/base-gsm8k-$MODE-$s"
  echo "== $s ($d)"
  [ -d "$d" ] || { echo "  (missing, train that strategy first)"; continue; }
  cmp -s "$d/config.json" "$P5_MODELS/base-gsm8k-$MODE-single/config.json" 2>/dev/null \
    && echo "  config: same as single" || echo "  config: differ/uncomparable"
  (cd "$d" && sha256sum ./*.safetensors 2>/dev/null | head -5) || echo "  (no safetensors)"
done

#!/usr/bin/env bash
# infra/dgx-spark-1x/env.sh — DGX Spark 1대 기본 환경.
# 사용: source infra/dgx-spark-1x/env.sh  (또는 각 스크립트가 자동 반영)
# NOTE: Blackwell sm_120 에서는 flash-attn 빌드 실패 → axolotl yaml은
# flash_attention: false (SDPA fallback) 유지. Hopper용으로 옮기면
# infra/<target>/overlay.yaml 로 뒤집을 것.

export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu130}"

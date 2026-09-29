#!/usr/bin/env bash
# infra/runpod-h100-1x/env.sh — RunPod H100 1x (x86_64, CUDA 12).
# 사용: source infra/runpod-h100-1x/env.sh
# NOTE: pyproject의 aarch64 전용 핀(required-environments) 해제 + torch cu124 계열 필요.

export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu124}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# RunPod 네트워크 볼륨 영속 캐시:
if [ -d /workspace ]; then
  export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.cache/huggingface}"
fi

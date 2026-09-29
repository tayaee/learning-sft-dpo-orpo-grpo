#!/usr/bin/env bash
# infra/colab-t4/env.sh — Colab free (T4) 세션용 값 모음.
# NOTE: Colab 셀의 `!source`는 서브셸이라 env가 남지 않음.
# 렌처 스크립트에서 직접 export하거나 `%env`로 옮겨 쓸 것. 아래는 참조값.
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu124}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/content/.cache/huggingface}"
# NOTE: 세션 종료 시 로컬 디스크 소멸.
# output은 Drive(`/content/drive/MyDrive/...`) 지정 권장. pyproject aarch64 핀 해제 필요.

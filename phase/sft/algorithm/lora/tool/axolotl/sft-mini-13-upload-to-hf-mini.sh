#!/bin/bash
# sft-mini-13-upload-to-hf-mini.sh — sft-13-upload-to-hf.sh 의 mini 모드 래퍼
exec "$(dirname "$0")/sft-13-upload-to-hf.sh" mini "$@"

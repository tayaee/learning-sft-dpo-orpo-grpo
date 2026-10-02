#!/usr/bin/env bash
# 00-setup.sh — 양 노드에서 1회씩 실행. 학습 없음.
set -euo pipefail
source "$(dirname "$0")/config/common.env"

p5_log "repo=$REPO_ROOT infra=$INFRA python=$(python3 --version 2>&1)"

echo + uv sync
uv sync 2>/dev/null || p5_log "WARN: root uv sync 실패 — 개별 venv로 진행"

echo + uv pip install -r "$P5_ROOT/requirements-p5.txt"
uv pip install -r "$P5_ROOT/requirements-p5.txt" || p5_log "WARN: p5 의존성 일부 실패 — 로그 확인"

# llmcompressor는 resolver 충돌 회피를 위해 --no-deps로 별도 설치 (상세: requirements-p5.txt NOTE)

echo + uv pip install --no-deps "llmcompressor==0.14.0"
uv pip install --no-deps "llmcompressor==0.14.0" 2>/dev/null || p5_log "WARN: llmcompressor 설치 실패 — 42/43 건너뜀"

# torchvision/torchaudio: LLM 파이프라인 불필요 + cu130 aarch64 휠 깨짐
# (torchvision::nms 부재 → transformers 임포트 오염). 있으면 제거.

echo + uv pip uninstall -y torchvision torchaudio 
uv pip uninstall -y torchvision torchaudio 2>/dev/null | tail -1 || true

mkdir -p "$P5_SHARED"/{datasets,models,outputs,hf-cache,vllm-cache}

p5_log "원본 업스트림 확인: $MASKED"
if [ ! -d "$MASKED/.git" ]; then
  git clone https://github.com/changyuchen347/maskedthought "$MASKED" \
    || p5_log "WARN: 클론 실패 — 오프라인이면 수동 클론 후 재실행"
else
  p5_log "maskedthought 존재 — 참조용 (실행은 이 repo 파일만 사용)"
fi

p5_log "벤더링 데이터 → 공용 datasets 복사 (없을 때만)"
[ -f "$P5_DATASETS/gsm8k-train.jsonl" ] || cp "$P5_DATA/gsm8k-train.jsonl" "$P5_DATASETS/gsm8k-train.jsonl"
[ -f "$P5_DATASETS/gsm8k-test.jsonl" ] || cp "$P5_DATA/gsm8k-test.jsonl" "$P5_DATASETS/gsm8k-test.jsonl"

p5_log "HF 로그인 확인 (Llama gated repo 필요)"

(set -x; "$HF_BIN" auth whoami 2>/dev/null) || echo "-> 'uv run hf auth login' 실행 필요 (토큰: huggingface.co)"

p5_log "GPU 확인"
nvidia-smi -L 2>/dev/null || echo "-> nvidia-smi 없음. DGX OS에서 확인"
(set -x; "$VENV_BIN/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())" 2>/dev/null || true)

p5_log "OK. shared=$P5_SHARED (spark1/spark2 동일 경로인지 ls로 대조)"

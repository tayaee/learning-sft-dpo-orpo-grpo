#!/usr/bin/env bash
# 10-setup.sh — 양 노드에서 1회씩 실행. 학습 없음.
set -euo pipefail
cd "$(dirname "$0")"  # uv가 project-5/pyproject.toml을 프로젝트로 인식하도록 고정
source "$(dirname "$0")/config/common.env"
p5_repro_echo

p5_log "repo=$REPO_ROOT infra=$INFRA python=$(python3 --version 2>&1)"

echo + uv sync
uv sync 2>/dev/null || p5_log "WARN: root uv sync 실패 — 개별 venv로 진행"

# llmcompressor는 resolver 충돌 회피를 위해 --no-deps로 별도 설치 (상세: pyproject.toml NOTE)

echo + uv pip install --no-deps "llmcompressor==0.14.0"
uv pip install --no-deps "llmcompressor==0.14.0" 2>/dev/null || p5_log "WARN: llmcompressor install failed, skip 42/43"

# compressed-tensors==0.19.0: llmcompressor==0.14.0이 요구. pyproject에 함께
# 적으면 vllm(==0.17.0 요구)과 unsatisfiable이 되므로 여기서 별도 설치.
# (--no-deps 설치라 의존성을 안 맞춰줌. vLLM AWQ/FP8 로딩 동작은 50-eval에서 확인)
echo + uv pip install "compressed-tensors==0.19.0"
uv pip install "compressed-tensors==0.19.0" 2>/dev/null | tail -1 || p5_log "WARN: compressed-tensors install failed, skip 42/43"

# torchvision은 pyproject.toml에 고정됨 (uv sync가 설치).
# torchaudio는 여전히 불필요 → 제거 유지.

echo + uv pip uninstall -y torchaudio
uv pip uninstall -y torchaudio 2>/dev/null | tail -1 || true

mkdir -p "$P5_SHARED"/{datasets,models,outputs,hf-cache,vllm-cache}

p5_log "upstream check: $MASKED"
if [ ! -d "$MASKED/.git" ]; then
  git clone https://github.com/changyuchen347/maskedthought "$MASKED" \
    || p5_log "WARN: clone failed, clone manually if offline"
else
  p5_log "maskedthought present, reference only"
fi

p5_log "copy vendored data to shared datasets (if missing, via .tmp+rename)"
for _bn in gsm8k-train.jsonl gsm8k-test.jsonl; do
  if [ ! -f "$P5_DATASETS/$_bn" ]; then
    cp "$P5_DATA/$_bn" "$P5_DATASETS/$_bn.tmp" && mv "$P5_DATASETS/$_bn.tmp" "$P5_DATASETS/$_bn"
  fi
done
unset _bn

p5_log "HF login check (Llama gated)"

(set -x; "$HF_BIN" auth whoami 2>/dev/null) || echo "-> 'uv run hf auth login' required"

p5_log "GPU check"
nvidia-smi -L 2>/dev/null || echo "-> no nvidia-smi, check on DGX OS"
(set -x; "$VENV_BIN/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())" 2>/dev/null || true)

p5_log "OK. shared=$P5_SHARED (compare ls on spark1/spark2)"

echo ---- result ----
(set -x; ls -lh "$P5_DATASETS/")

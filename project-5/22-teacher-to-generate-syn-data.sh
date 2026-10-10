#!/usr/bin/env bash
# 22-generate.sh [mini|full] — Stage 2b. vLLM 합성 생성.
# 원본: data_gen_vllm.py (tp=4, max 8192, temp 0.5/top_p 0.8/top_k 5/rep 1.05).
set -euo pipefail
source "$(dirname "$0")/config/common.env" "${1:-mini}"

MAXLEN=8192
# VLLM_GPU_MEM_UTIL: DGX Spark(통합메모리 128GB, OS/Xorg가 ~13GB 점유)에서는
# 0.9(109.5GB 요구)가 기동 시 여유분(108.3GB)을 초과해 EngineCore 기동 실패.
# 기본 0.8으로 낮춤. 필요시 VLLM_GPU_MEM_UTIL=0.85 ... 로 오버라이드.
# VLLM_MAX_NUM_SEQS=8: mini/full 통일. 8k 컨텍스트 시퀀스당 KV ~1GB,
# 8개 동시 = 8GB 수준이라 서빙급 선점(80GB)은 불필요.
: "${VLLM_MAX_NUM_SEQS:=8}"
p5_repro_echo
IN="$P5_DATASETS/prompts-$MODE.parquet"
OUT="$P5_DATASETS/generated-$MODE.csv"
p5_require "$IN"
if p5_fresh "$OUT" "$IN"; then
  p5_log "skip: fresh $OUT (FORCE=1 to rebuild)"
  echo ---- result ----
  (set -x; ls -lh "$OUT")
  (set -x; wc -l "$OUT")
  exit 0
fi
p5_log "teacher=$TEACHER_MODEL tp=$TP maxlen=$MAXLEN n=$SYNTH_N gpu_mem_util=${VLLM_GPU_MEM_UTIL:-0.8} max_num_seqs=$VLLM_MAX_NUM_SEQS"
(set -x; uv run "$P5_ROOT/22-teacher-to-generate-syn-data.py" --mode "$MODE" --tp "$TP" --maxlen "$MAXLEN" \
  --teacher "$TEACHER_MODEL" --n "$SYNTH_N" --gpu-mem-util "${VLLM_GPU_MEM_UTIL:-0.8}" \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS")

echo ---- result ----
(set -x; ls -lh "$P5_DATASETS/generated-$MODE.csv")
(set -x; wc -l "$P5_DATASETS/generated-$MODE.csv")

#!/usr/bin/env bash
# infra/dgx-spark-2x/env.sh — DGX Spark 2대 분산 기본 환경.
# 사용: source infra/dgx-spark-2x/env.sh
# NOTE: SoC는 1x와 동일 (aarch64, Blackwell sm_120, 대당 unified 128GB).

export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu130}"

# 분산 랑데부 (axolotl/torchrun 실행 시점에 실제 값 지정):
: "${MASTER_ADDR:=}"   # 예: export MASTER_ADDR=192.168.1.10
: "${MASTER_PORT:=29500}"
export MASTER_ADDR MASTER_PORT

# 첫 분산 실행 점검: NCCL_DEBUG=INFO <train-cmd>

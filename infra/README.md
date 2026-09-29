# infra/ — 실행 환경 오버레이

레시피(`phase/.../tool/.../base.yaml`)는 고정. 환경별 차이
(정밀도·어텐션·배치·분산·경로)만 `infra/<infra>/`에서 덮어쓴다.

```
infra/<infra>/env.sh        # env (예: UV_TORCH_BACKEND, HF cache)
infra/<infra>/overlay.yaml  # (필요시) axolotl yaml override 조각
```

| infra | 대상 | 핵심 diff (base=`phase/.../config/*.yaml` 대비) |
|---|---|---|
| `dgx-spark-1x` | DGX Spark 1대 (aarch64, Blackwell sm_120, unified 128GB) | 현 기본값. `flash_attention: false` (SDPA), cu130 |
| `dgx-spark-2x` | DGX Spark 2대 (대당 사양은 1x와 동일, 합 256GB 분산) | `grad_accum 8→4` (effective 32 유지) + deepspeed/fsdp 택1, `MASTER_ADDR/PORT` |
| `runpod-h100-1x` | RunPod H100 1x (x86_64, Hopper sm_90 80GB, CUDA 12) | `flash_attention: true`, cu124, `seq 2048`·`micro_batch 8` 예시, `/workspace` 캐시 |
| `colab-t4` | Colab free (T4 Turing sm_75 16GB, ephemeral) | `fp16` (bf16 불가), QLoRA 전환, `seq 512`·`micro_batch 1x32`, liger off, Drive 출력 권장 |

x86_64 타깃(runpod/colab) 실행 전제: `pyproject.toml`의 aarch64 전용 핀
(`required-environments`) 해제 + torch cu124 계열. 이 전제가 없으면
`uv` 설치 단계에서 실패한다.

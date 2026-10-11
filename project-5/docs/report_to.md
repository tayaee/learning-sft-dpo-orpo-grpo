# report_to — 동일 잡 5종 동시 리포팅

동일 full-ddp 잡에 `tensorboard,mlflow,trackio,dvclive,wandb` 5개를 동시에 켜서 보고서를 한 번에 비교하기 위한 문서.
개별 옵션 상세는 `docs/reporting.<name>.howto.md` 참조.

## 0. 사전 체크 (학습 전)

```bash
uv run --no-sync check_report_to.py --report_to all
uv run --no-sync check_report_to.py --report_to tensorboard,mlflow,trackio,dvclive,wandb --out $OUT
```

`--no-sync` 필수 (plain `uv run`은 `uv sync`로 .venv를 갈아엎음 — 학습 중 실행 금지).
종료 코드 0=전체 활성, 1=MISSING/UNKNOWN 있음. `all`은 설치된 연동에만 보고됨.

## 1. 코드 변경 (완료)

* `10-train-entry.py`: `--report_to` CSV 인자 추가 (`"a,b"` → `["a","b"]`, `"none"` 유지).
* `30-fft-train.sh` / `31-qlora-train.sh`: `--report_to "${P5_REPORT_TO:-none}"` 전달. 기본값 `none`이라 기존 동작 불변.

## 2. 설치 (spark1/spark2 양쪽)

```bash
uv pip install tensorboard mlflow "trackio>=0.21.1" dvclive wandb
.venv/bin/python -c "from transformers.integrations.integration_utils import get_available_reporting_integrations; print(get_available_reporting_integrations())"
# 기대: ['dvclive', 'mlflow', 'tensorboard', 'wandb', 'trackio'] 포함
```

## 3. 환경변수 (양 노드 동일)

```bash
export P5_REPORT_TO="tensorboard,mlflow,trackio,dvclive,wandb"
export MLFLOW_TRACKING_URI="file:/rosenas/data/AIML/project-5-shared/outputs/mlruns"
export MLFLOW_EXPERIMENT_NAME="p5-sft"
export WANDB_PROJECT="p5-sft"
wandb login   # wandb만 유일하게 인증 필요. offline 비교면 WANDB_MODE=offline
```

## 4. 기존 full-ddp 중단 후 재실행

`report_to`는 Trainer 생성 시 고정이라 진행 중 잡에는 적용 불가. 중단 후 재실행 필수:

```bash
# 양 노드에서 실행 중인 31-qlora-train.sh / torchrun 확인 후 종료
# rank0 + rank1 각각
./31-qlora-train.sh full ddp
```

OUT은 기존과 동일 (`$P5_MODELS/synthetic-qlora-full-ddp`). 이어 highly? 처음부터 재학습됨 — 중간 ckpt(`save_strategy=epoch`)는 남지 않음에 유의.

## 5. 보고서 위치 대조표 (동일 잡)

| 연동 | 산출물 | 확인 명령 |
|---|---|---|
| tensorboard | `$OUT/runs/` | `tensorboard --logdir $OUT --port 6006` |
| mlflow | `$MLFLOW_TRACKING_URI` | `mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI" --port 5000` |
| trackio | `./trackio/` | `trackio show --project project-5` |
| dvclive | `./dvclive/` | `cat dvclive/metrics.tsv` |
| wandb | cloud/offline `wandb/` | wandb.ai 대시보드 또는 `wandb sync --sync-all` |

## 6. 트러블슈팅

* 설치 안 된 연동은 조용히 스킵됨 (`get_available_reporting_integrations()`에 없음). 5개 다 안 나오면 양 노드 설치 상태 확인.
* **설치됐는데 import가 깨지는 연동(trackio 실측)은 Trainer 생성 시점에 잡 전체를 죽임.** `check_report_to.py`가 실제 import까지 검증해 `[BROKEN]`으로 표시. 예: trackio<0.21.1 + huggingface_hub>=1 (`CommitOperationAdd` ImportError). 처방: 해당 연동을 report_to에서 제외하고 재실행, 패키지 업그레이드는 별도 검증 후.
* wandb 미로그인 상태에서 online 모드면 잡 실패 가능 — 먼저 `wandb login` 또는 `WANDB_MODE=offline`.
* dvclive+mlflow는 cwd 기준 기록이라 실행 디렉토리 고정 권장.
* 패키지 추가 후에도 `--report_to` 미지정(`none`)이면 아무 파일도 안 생김. 정상.

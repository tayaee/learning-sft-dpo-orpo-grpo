# reporting.mlflow — 로컬 MLflow 트래킹

## 1. 설치 (양 노드 동일하게)

```bash
uv pip install mlflow
```

## 2. 학습에 전달

```bash
# 트래킹 저장소 지정 (공유FS 권장 — 양 rank가 같은 곳을 봄)
export MLFLOW_TRACKING_URI="file:/rosenas/data/AIML/project-5-shared/outputs/mlruns"
export MLFLOW_EXPERIMENT_NAME="p5-sft"
P5_REPORT_TO=mlflow ./31-qlora-train.sh full ddp
```

미지정 시 기본 `./mlruns` (실행 cwd)에 기록되므로, 비교 목적이면 위처럼 절대경로 권장.

## 3. 보고서 확인

```bash
mlflow ui --backend-store-uri "file:/rosenas/data/AIML/project-5-shared/outputs/mlruns" --port 5000
# 브라우저: http://<학습노드>:5000
```

지표: `train/loss`, `train/lr`, `train/epoch` 등. 파라미터(lr/epochs/accum)도 함께 기록됨.

## 4. DDP 주의

* rank0만 기록. `MLFLOW_TRACKING_URI`는 양 노드 동일 export 필수.
* `DISABLE_MLFLOW_INTEGRATION=TRUE`가 설정돼 있으면 transformers가 mlflow를 무시 — 미설정 확인.

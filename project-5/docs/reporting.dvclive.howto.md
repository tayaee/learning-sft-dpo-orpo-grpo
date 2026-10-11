# reporting.dvclive — git 친화 경량 로거

## 1. 설치 (양 노드 동일하게)

```bash
uv pip install dvclive
```

## 2. 학습에 전달

```bash
P5_REPORT_TO=dvclive ./31-qlora-train.sh full ddp
```

모델 아티팩트까지 dvclive에 남기려면 (기본 OFF):

```bash
HF_DVCLIVE_LOG_MODEL=TRUE P5_REPORT_TO=dvclive ./31-qlora-train.sh full ddp
```

## 3. 보고서 확인

실행 cwd 아래 `dvclive/` 생성 (`metrics.tsv`, `params.yaml`).

```bash
cat dvclive/metrics.tsv | head
dvc plots show dvclive/metrics.tsv  # dvc 설치 시
```

git 비교용: `dvclive/`를 커밋하면 잡별 지표 diff가 가능. 동일 OUT 5종 비교 시에는 디렉토리가 겹치지 않게 런별 cwd 분리 권장.

## 4. DDP 주의

* rank0만 기록. 공유FS에서 두 rank가 같은 `dvclive/`를 쓰지 않도록 rank0-only 저장 유지 (현재 구조 그대로 사용).

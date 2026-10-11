# reporting.tensorboard — 로컬 TensorBoard

## 1. 설치 (양 노드 동일하게)

```bash
uv pip install tensorboard
```

확인:

```bash
.uvvenv/bin/python -c "import tensorboard; print(tensorboard.__version__)" 2>/dev/null || \
.venv/bin/python -c "import tensorboard; print(tensorboard.__version__)"
# repo root에서 실행. spark1/spark2 양쪽에서 수행
```

## 2. 학습에 전달

`10-train-entry.py --report_to` CSV. 셸 래퍼는 `P5_REPORT_TO`를 그대로 전달:

```bash
# 단일 지정
P5_REPORT_TO=tensorboard ./31-qlora-train.sh full ddp
# 직접 지정 (디버그용)
.venv/bin/python project-5/10-train-entry.py --train <jsonl> --out /tmp/x --report_to tensorboard --dry_run
```

## 3. 보고서 확인

이벤트 파일: `$OUT/runs/` 아래 생성 (rank0만 기록).

```bash
tensorboard --logdir /rosenas/data/AIML/project-5-shared/models/synthetic-qlora-full-ddp --port 6006
# 브라우저: http://<학습노드>:6006
```

## 4. DDP 주의

* rank0만 SummaryWriter 생성. rank1에 tensorboard 미설치여도 동작은 하나, 양 노드 동일 설치 권장.
* `logging_steps=200`이라 짧은 mini런은 점이 몇 개만 찍힘. 정상.

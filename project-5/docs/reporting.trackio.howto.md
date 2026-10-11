# reporting.trackio — HF 경량 대시보드 (로컬)

## 1. 설치 (양 노드 동일하게)

```bash
uv pip install "trackio>=0.21.1"
```

## 2. 학습에 전달

```bash
P5_REPORT_TO=trackio ./31-qlora-train.sh full ddp
```

외부 Space 공유를 쓰지 않는 한 로그인·키 불필요. 기본 project=`huggingface`, 로컬 디렉토리에만 기록.

## 3. 보고서 확인

```bash
trackio show --project huggingface
# 또는
python -m trackio show --project huggingface
```

로컬 `./trackio/` 아래 run 기록. Hub Space로 실시간 공유하려면 `SFTConfig(trackio_space_id=...)` 추가가 필요 — 현재 `10-train-entry.py`는 미지원, 로컬 비교용으로 사용.

## 4. DDP 주의

* rank0만 기록. 양 노드 동일 버전 설치 권장.

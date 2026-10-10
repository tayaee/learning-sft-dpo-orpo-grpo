# runlog — project-5 실행 시간 기록 (logs/timing-*.log 실측 집계)

## run-*.sh 요약

| 스크립트 | 범위 | 실행 위치 | 실행시간 |
|---|---|---|---|
| run-all-all.sh | 아래 6종 고정 순서로 순차 실행 | 양 노드 (같은 명령) | 약 32h 00m (추정, 미실측) |
| run-mini-single.sh | mini 전 파이프라인 (Stage 0→7) | spark1 단독 | 약 0h 55m (실측 합계) |
| run-mini-ddp.sh | mini 2노드 DDP (학습만 분산, 나머지는 rank0) | 양 노드 | 약 0h 35m (실측, 30/31 수동분 제외) |
| run-mini-fsdp.sh | mini FSDP 2노드 (ddp와 동일 구조) | 양 노드 | 약 0h 38m (실측, 32 수동분 제외) |
| run-full-single.sh | full 전 파이프라인 (Stage 0→7) | spark1 단독 | 약 10h 00m (추정, 미실측) |
| run-full-ddp.sh | full 2노드 DDP | 양 노드 | 약 10h 00m (추정, 미실측) |
| run-full-fsdp.sh | full FSDP (동일) | 1x spark1 / 2x 양 노드 | 약 10h 00m (추정, 미실측) |

- 공통: 스텝 타이밍(stdout + `logs/timing-*.log`), `STEPS=` 부분 실행,
  `MASTER_ADDR`·`NODE_RANK` 자동 검출, `SKIP_SETUP=1` 생략.
  run-all-all 전용: `RUNS=` 일부만 실행, single 2종 rank0 전용 게이팅.
- 하류(merge→평가)는 같은 전략 산출물 기준으로 진행 (콤보별 독립).

- 시간 표기: 분 단위 (1분 미만은 `<1m`). 대표 성공런 기준.
- full 계열은 아직 완주 없음 (아래 pending).

## run-mini-single.sh (spark1 단독, 완주)

| 스텝 | 시간 | 비고 |
|---|---|---|
| 00-setup | <1m | 2회目以降 (초회는 수 분) |
| 10-baseline-single | 3m | 256행·4스텝 |
| 20-select-candidates | <1m | 임베딩 캐시 적중 시 |
| 21-build-prompts | <1m | |
| 22-teacher-generate | 8m | 64프롬프트, TP=1, 8B 상주 후 |
| 23-postprocess | <1m | |
| 30-fft-single | 3m | |
| 31-qlora-single | 3m | |
| 32-merge-lora | 1m | |
| 40-quant-fft-awq | 1m | llmcompressor 0.14.0 + compressed-tensors 0.19.0 |
| 41-quant-fft-fp8 | <1m | |
| 42-quant-fft-gguf | 1m | convert+quantize (Q8_0 1종 실측, 5종 전량 ~2분, f16 기변환·q8_0 skip 시) |
| 43-quant-fft-gptq | 2m | calib 2 |
| 50-quant-qlora-awq | 1m | |
| 51-quant-qlora-fp8 | <1m | |
| 52-quant-qlora-gguf | 2m | Q8_0 1종 실측, 5종 전량 ~2분 (f16 기변환·q8_0 skip 시) |
| 53-quant-qlora-gptq | 2m | |
| 60-measure-ppl | 2m | 9타깃×32텍스트 실측, 19타깃 시 증가 (GGUF 10종은 llama-perplexity) |
| 71-eval | 11m | 9타깃×10문항 (타깃당 ~1m) |
| 72-eval-gguf | 미측정 | 10타깃×10문항, llama-completion CPU (PROCEDURE 추정 ~30분) |
| 73-score | <1m | |
| **합계** | **약 43m** | PROCEDURE §3의 약 30분보다 김 (eval·teacher 실측 반영) |

PPL 게이트 (mini): base/fft/qlora 9.867 동일, quant Δ +0.12~+0.99
→ fft-fp8·qlora-fp8 Accept, 나머지 Conditional, Discard 없음.

## 분산 baseline (STEPS=baseline 실측)

| 스크립트·스텝 | spark1 (rank0) | spark2 (rank1) |
|---|---|---|
| run-mini-ddp.sh / 10-baseline-ddp | 6m (rank1 대기 ~3m 포함) | 4m |
| run-mini-fsdp.sh / 10-baseline-fsdp (2노드) | 3m | 3m |
| run-mini-fsdp.sh / 10-baseline-fsdp (1x) | — | 4m (spark2 단독) |

## 실패 기록 (원인→수정, 재실행 성공)

| 시각(Z) | 현상 | 원인 | 수정 |
|---|---|---|---|
| 10:24 | 20-select 실패 | detached 실행 `uv` 미해결 | common.env PATH 보완 |
| 10:28 | baseline 실패 | TRL `trackio` import 충돌 (hub 1.33) | `create_model_card` no-op |
| 10:37 | 22-teacher 실패 (<1m) | TP=2 단일GPU 불가 | TP=1 고정 |
| 10:50 | 22-teacher 실패 (4m) | vLLM이 `torchvision` 필수인데 setup에서 제거 | 00-setup 제거→설치로 전환 |
| 11:16 | 32-merge 실패 (<1m) | `optimum` 1.16.2 vs gptqmodel 요구 ≥1.24 | requirements 핀 + 양 venv 2.3.0 |
| 11:52 | 42-awq 실패 (<1m) | llmcompressor 0.14.0 vs compressed-tensors 0.17.0 | 0.19.0 고정 (1m 성공) |
| 12:15 | 60-ppl 실패 (2m) | GGUF SKIP행 `delta: None` 포맷 크래시 | None 가드 (2m 성공) |
| 12:54 | mini-ddp baseline 실패 (15m) | rank1 합류 실패 (1/2 join 타임아웃) | 조사 중 — 아래 참조 |

## 랑데부 이슈 (2026-10-08 13:4xZ 현재진행)

- spark2 rank1이 spark1:29500 store에 합류 못함. TCP 연결 자체는 즉시 성공, loopback rendezvous 정상, GPU·NCCL 정상.
- rank1은 무출력 재시도, rank0는 15m 대기 후 타임아웃. strace상 `ECONNREFUSED` 재시도 루프 관측.
- GLOO 단독 테스트도 동일 증상 → NCCL/IB 문제가 아니라 join 경로 문제로 좁혀짐.
- 다음 단계: 클린 포트(2960x)에서 즉시 페어링 재시험.

## pending

- run-full-single.sh / run-full-ddp.sh / run-full-fsdp.sh (미실행)
- run-mini-ddp.sh 전체 (baseline 이후 미진행 — 랑데부 해결 후)
- run-mini-fsdp.sh 전체 (DCP 저장 검증됨: 양 노드 rc=0, 3m/3m)

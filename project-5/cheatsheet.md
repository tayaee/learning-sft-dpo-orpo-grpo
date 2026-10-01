# cheatsheet — 4 시나리오 실행 순서

- master = **spark1**, slave = **spark2**.
- 분산(C·D) 학습은 양 노드에서 **같은 명령**을 각자의 `NODE_RANK`로 실행한다.
  rank 0(spark1)을 먼저 띄우고 spark2를 이어 붙인다.
- 아래 표는 2컬럼이다. 왼쪽 셀 = spark1에 입력, 오른쪽 셀 = spark2에 입력.
- 설명 행(`# …`)과 명령 행(`` `…` ``)은 분리되어 있다. 한 줄에 명령어 하나씩.
  양쪽에서 실행하는 명령은 양쪽 셀에 반복했다.
- `공통 준비`는 4 시나리오 공통이며 spark1·spark2 각각 1회 수행한다.

## 주소 정리 (실측)

| 용도 | spark1 | spark2 |
|---|---|---|
| LAN (관리·SSH, FQDN) | `spark1.local` | `spark2.local` |
| CX7 fabric (분산 학습용, spark1에서 ping 확인) | `spark1-p1-r0` = 192.168.102.1 | `spark2-p1-r0` = 192.168.102.2 |
| CX7 예비 링크 | `spark1-p1-r1` = 192.168.103.1 | `spark2-p1-r1` = 192.168.103.2 |

- `MASTER_ADDR`는 CX7명(`spark1-p1-r0`) 권장. 호스트명 해석 실패 시에만 IP 직접 지정.

## 공통 준비

| spark1 | spark2 |
|---|---|
| # 0-1. repo 클론 (양 노드 동일) | # 0-1. repo 클론 (양 노드 동일) |
| `git clone <repo-url> ~/git/learning-sft-and-rl` | `git clone <repo-url> ~/git/learning-sft-and-rl` |
| `cd ~/git/learning-sft-and-rl/project-5` | `cd ~/git/learning-sft-and-rl/project-5` |
| # 0-2. 환경 셋업 (양 노드 동일) | # 0-2. 환경 셋업 (양 노드 동일) |
| `./00-setup.sh` | `./00-setup.sh` |
| # 0-2b. 직접 `uv run`을 쓸 셸이면 필수 (sync 되돌림 방지) | # 0-2b. 직접 `uv run`을 쓸 셸이면 필수 (sync 되돌림 방지) |
| `export UV_NO_SYNC=1` | `export UV_NO_SYNC=1` |
| # 0-3. HF 로그인 (양 노드 동일, Llama gated) | # 0-3. HF 로그인 (양 노드 동일, Llama gated) |
| `hf auth login` | `hf auth login` |
| # 0-4. 공유디스크 대조 (양 노드 동일 출력이어야 함) | # 0-4. 공유디스크 대조 (양 노드 동일 출력이어야 함) |
| `ls /rosenas/data/AIML/project-5-shared/datasets/` | `ls /rosenas/data/AIML/project-5-shared/datasets/` |
| # 0-5. 노드·링크 확인 (양 노드 동일, spark1=192.168.102.1) | # 0-5. 노드·링크 확인 (양 노드 동일, spark1=192.168.102.1) |
| `hostname` | `hostname` |
| `getent hosts spark1-p1-r0` | `getent hosts spark1-p1-r0` |
| # 0-6. 랑데부 export (C·D 전용, RANK만 다름, 호스트명 권장) | # 0-6. 랑데부 export (C·D 전용, RANK만 다름, 호스트명 권장) |
| `export MASTER_ADDR=spark1-p1-r0 MASTER_PORT=29500 NODE_RANK=0` | `export MASTER_ADDR=spark1-p1-r0 MASTER_PORT=29500 NODE_RANK=1` |

## A. mini-single — 파이프라인 점검, spark1만 (~30분, PROCEDURE.md §3 실측표)

| spark1 | spark2 |
|---|---|
| # A는 spark1 단독 (아래 전부 spark1) | # spark2: 실행 없음 |
| # 1. 베이스라인 FFT | |
| `./10-baseline-train.sh mini single` | |
| # 2a. 후보 선별 | |
| `uv run 20-select-candidates.py --mode mini` | |
| # 2b. 프롬프트 생성 | |
| `uv run 21-build-prompts.py --mode mini` | |
| # 2b. 합성 생성 | |
| `./22-generate.sh mini` | |
| # 2c. 후처리 | |
| `uv run 23-postprocess.py --mode mini` | |
| # 3. 합성 FFT | |
| `./30-fft-train.sh mini single` | |
| # 3. QLoRA | |
| `./31-qlora-train.sh mini single` | |
| # 4. GGUF Q8_0 | |
| `./40-quant-gguf.sh mini` | |
| # 4. GPTQ | |
| `uv run 41-quant-gptq.py --mode mini` | |
| # 4. AWQ | |
| `uv run 42-quant-awq.py --mode mini` | |
| # 4. FP8 | |
| `uv run 43-quant-fp8.py --mode mini` | |
| # 5. LoRA 병합 | |
| `uv run 50-merge-lora.py --mode mini` | |
| # 5. 평가 | |
| `./51-eval.sh mini` | |
| # 5. 채점 | |
| `uv run 52-score.py --mode mini` | |
| # 6. 업로드 (선택) | |
| `./53-upload-hf.sh mini` | |
| # 6. 서빙 (선택, 별도 터미널 유지) | |
| `./60-serve-vllm.sh mini fft` | |
| # 6. 추론 예제 (선택, 서빙 기동 후 다른 터미널) | |
| `uv run 61-infer-examples.py --model tayaee/p5-1B-math-fft-mini` | |

## B. full-single — 실전 1노드, spark1만 (~10시간 내외, 추정)

| spark1 | spark2 |
|---|---|
| # B는 spark1 단독 (아래 전부 spark1) | # spark2: 실행 없음 |
| # 1. 베이스라인 FFT | |
| `./10-baseline-train.sh full single` | |
| # 2a. 후보 선별 | |
| `uv run 20-select-candidates.py --mode full` | |
| # 2b. 프롬프트 생성 | |
| `uv run 21-build-prompts.py --mode full` | |
| # 2b. 합성 생성 (가장 오래 걸림, 밤에 권장) | |
| `./22-generate.sh full` | |
| # (10k ≈ 5~6시간: mini 실측 200개/8.3분 기준 외삽) | |
| # 2c. 후처리 | |
| `uv run 23-postprocess.py --mode full` | |
| # 3. 합성 FFT | |
| `./30-fft-train.sh full single` | |
| # 3. QLoRA | |
| `./31-qlora-train.sh full single` | |
| # 4. GGUF Q8_0 | |
| `./40-quant-gguf.sh full` | |
| # 4. GPTQ | |
| `uv run 41-quant-gptq.py --mode full` | |
| # 4. AWQ | |
| `uv run 42-quant-awq.py --mode full` | |
| # 4. FP8 | |
| `uv run 43-quant-fp8.py --mode full` | |
| # 5. LoRA 병합 | |
| `uv run 50-merge-lora.py --mode full` | |
| # 5. 평가 | |
| `./51-eval.sh full` | |
| # 5. 채점 | |
| `uv run 52-score.py --mode full` | |
| # 6. 업로드 (선택) | |
| `./53-upload-hf.sh full` | |
| # 6. 서빙 (선택, 별도 터미널 유지) | |
| `./60-serve-vllm.sh full fft` | |
| # 6. 추론 예제 (선택, 서빙 기동 후 다른 터미널) | |
| `uv run 61-infer-examples.py --model tayaee/p5-1B-math-fft-full` | |

## C. full-ddp — 2노드 DDP, spark1 master + spark2 slave

| spark1 | spark2 |
|---|---|
| # 1. 베이스라인 DDP (rank 0 먼저 기동) | # 1. 베이스라인 DDP (rank 0 기동 후 실행) |
| `INFRA=dgx-spark-2x ./10-baseline-train.sh full ddp` | `INFRA=dgx-spark-2x ./10-baseline-train.sh full ddp` |
| # 2a. 후보 선별 (spark1 단독) | # spark2: 대기 |
| `uv run 20-select-candidates.py --mode full` | |
| # 2b. 프롬프트 생성 (spark1 단독) | # spark2: 대기 |
| `uv run 21-build-prompts.py --mode full` | |
| # 2b. 합성 생성 (spark1 단독, INFRA 없이 1x 기본) | # spark2: 대기 |
| `./22-generate.sh full` | |
| # 2c. 후처리 (spark1 단독) | # spark2: 대기 |
| `uv run 23-postprocess.py --mode full` | |
| # 3. 합성 FFT DDP (rank 0 먼저 기동) | # 3. 합성 FFT DDP (rank 0 기동 후 실행) |
| `INFRA=dgx-spark-2x ./30-fft-train.sh full ddp` | `INFRA=dgx-spark-2x ./30-fft-train.sh full ddp` |
| # 3. QLoRA DDP (rank 0 먼저 기동) | # 3. QLoRA DDP (rank 0 기동 후 실행) |
| `INFRA=dgx-spark-2x ./31-qlora-train.sh full ddp` | `INFRA=dgx-spark-2x ./31-qlora-train.sh full ddp` |
| # 4. GGUF Q8_0 (spark1 단독) | # spark2: 대기 |
| `./40-quant-gguf.sh full` | |
| # 4. GPTQ (spark1 단독) | # spark2: 대기 |
| `uv run 41-quant-gptq.py --mode full` | |
| # 4. AWQ (spark1 단독) | # spark2: 대기 |
| `uv run 42-quant-awq.py --mode full` | |
| # 4. FP8 (spark1 단독) | # spark2: 대기 |
| `uv run 43-quant-fp8.py --mode full` | |
| # 5. LoRA 병합 (spark1 단독) | # spark2: 대기 |
| `uv run 50-merge-lora.py --mode full` | |
| # 5. 평가 (spark1 단독) | # spark2: 대기 |
| `./51-eval.sh full` | |
| # 5. 채점 (spark1 단독) | # spark2: 대기 |
| `uv run 52-score.py --mode full` | |
| # 6. 업로드 (spark1 단독, 선택) | # spark2: 대기 |
| `./53-upload-hf.sh full` | |
| # 6. 서빙 (spark1 단독, 선택, 별도 터미널 유지) | # spark2: 대기 |
| `./60-serve-vllm.sh full fft` | |
| # 6. 추론 예제 (spark1 단독, 선택, 서빙 기동 후 다른 터미널) | # spark2: 대기 |
| `uv run 61-infer-examples.py --model tayaee/p5-1B-math-fft-full` | |

## D. full-fsdp — 2노드 FSDP, spark1 master + spark2 slave

| spark1 | spark2 |
|---|---|
| # 1. 베이스라인 FSDP (rank 0 먼저 기동) | # 1. 베이스라인 FSDP (rank 0 기동 후 실행) |
| `INFRA=dgx-spark-2x ./10-baseline-train.sh full fsdp` | `INFRA=dgx-spark-2x ./10-baseline-train.sh full fsdp` |
| # 2a. 후보 선별 (spark1 단독) | # spark2: 대기 |
| `uv run 20-select-candidates.py --mode full` | |
| # 2b. 프롬프트 생성 (spark1 단독) | # spark2: 대기 |
| `uv run 21-build-prompts.py --mode full` | |
| # 2b. 합성 생성 (spark1 단독, INFRA 없이 1x 기본) | # spark2: 대기 |
| `./22-generate.sh full` | |
| # 2c. 후처리 (spark1 단독) | # spark2: 대기 |
| `uv run 23-postprocess.py --mode full` | |
| # 3. 합성 FFT FSDP (rank 0 먼저 기동) | # 3. 합성 FFT FSDP (rank 0 기동 후 실행) |
| `INFRA=dgx-spark-2x ./30-fft-train.sh full fsdp` | `INFRA=dgx-spark-2x ./30-fft-train.sh full fsdp` |
| # 3. QLoRA FSDP (rank 0 먼저 기동, 학습용 — 강의는 비권장) | # 3. QLoRA FSDP (rank 0 기동 후 실행, 학습용) |
| `INFRA=dgx-spark-2x ./31-qlora-train.sh full fsdp` | `INFRA=dgx-spark-2x ./31-qlora-train.sh full fsdp` |
| # 4. GGUF Q8_0 (spark1 단독) | # spark2: 대기 |
| `./40-quant-gguf.sh full` | |
| # 4. GPTQ (spark1 단독) | # spark2: 대기 |
| `uv run 41-quant-gptq.py --mode full` | |
| # 4. AWQ (spark1 단독) | # spark2: 대기 |
| `uv run 42-quant-awq.py --mode full` | |
| # 4. FP8 (spark1 단독) | # spark2: 대기 |
| `uv run 43-quant-fp8.py --mode full` | |
| # 5. LoRA 병합 (spark1 단독) | # spark2: 대기 |
| `uv run 50-merge-lora.py --mode full` | |
| # 5. 평가 (spark1 단독) | # spark2: 대기 |
| `./51-eval.sh full` | |
| # 5. 채점 (spark1 단독) | # spark2: 대기 |
| `uv run 52-score.py --mode full` | |
| # 참고. 1x 단독 FSDP 동작 학습 (spark1만) | # spark2: 실행 없음 |
| `./10-baseline-train.sh mini fsdp` | |

## 전략 비교 — spark1에서 실행

| spark1 | spark2 |
|---|---|
| # 해시 대조 (config 동일 여부 + safetensors 해시) | # spark2: 실행 없음 |
| `./12-compare-strategies.sh mini` | |
| `./12-compare-strategies.sh full` | |

- 3전략은 effective batch 64·seed 동일 조건에서도 bit-identical이 아닐 수 있다
  (allreduce vs reduce-scatter 합산 순서, dataloader 샤딩 차이).
- 하류(Stage 4~6) 입력은 항상 `-single` 산출물을 쓴다.
- `INFRA=`는 명령 앞에 붙이는 방식이다. 셸 전체에 export했다면 추론계
  (22/51/60) 실행 시 `TP=1`을 앞에 붙인다.
- `meta-llama/Llama-3.2-1B`는 gated 403이 나면 tayaee 계정으로 라이선스 승인 후
  사용한다. smoke는 `BASE_MODEL=unsloth/Llama-3.2-1B` 미러로 돌렸다
  (teacher `meta-llama/Llama-3.1-8B-Instruct`는 토큰으로 바로 접근됨).

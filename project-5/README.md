# project-5 — Capstone: GSM8K SFT + PTQ (DGX Spark 1x/2x)

양자화 강의 Part 5(MaskedThought 기반) 실습을 DGX Spark에서 재현하는
파이프라인. 원본 업스트림은 `~/git/maskedthought`에 클론해 참조만 하고,
실행에 필요한 것은 이 디렉토리(`data/`, `assets/`, 스크립트)에 다 들어 있다.
상세 절차는 `PROCEDURE.md`.

## 실행 규칙 (repo 관례 준수)

- 첫 인자 `mini | full` (기본 `mini`): mini=파이프라인 점검용 축소,
  full=강의 스케일. 모드별 산출물이 덮어쓰지 않게 분리된다.
- infra 선택: `INFRA=dgx-spark-1x | dgx-spark-2x` (기본 1x).
  `PROFILE=1x | 2x` 약칭도 된다. `infra/<INFRA>/env.sh`를 그대로 쓴다.
- 이 repo는 spark1·spark2 양쪽에 같은 경로(`~/git/learning-sft-and-rl`)로
  clone 하고, `/rosenas/data/AIML/`을 양쪽에서 같이 본다.
- 공용 산출물: `/rosenas/data/AIML/project-5-shared/{datasets,models,outputs,hf-cache,vllm-cache}`
- HF 업로드는 `tayaee/*` (원본 `DopeorNope/*` 대체).
- Python 3.12, `uv`, `UV_TORCH_BACKEND=cu130` (root `mise.toml`/`pyproject.toml` 준수).

```bash
# 0) 환경 (양 노드에서 1회)
./00-setup.sh

# 1) 베이스라인 FFT (GSM8K, Llama-3.2-1B) — 전략 3종 비교
./10-baseline-train.sh mini single          # 기준 (하류 입력용)
INFRA=dgx-spark-2x ./10-baseline-train.sh mini ddp     # 2노드 DDP (MASTER_ADDR 필요)
./10-baseline-train.sh mini fsdp            # 1GPU FSDP (학습용)
INFRA=dgx-spark-2x ./10-baseline-train.sh mini fsdp    # 2노드 FSDP (실분산)
./12-compare-strategies.sh mini             # 해시 대조
# 30/31도 동일하게 [mini|full] [single|ddp|fsdp]를 받는다 (출력 ...-<mode>-<strat>)

# 2) 합성데이터: 선별 → 프롬프트 → 생성 → 후처리
uv run 20-select-candidates.py --mode mini
uv run 21-build-prompts.py --mode mini
./22-teacher-to-generate-syn-data.sh mini
uv run 23-postprocess.py --mode mini

# 3) 합성 SFT: FFT vs QLoRA + merge
./30-fft-train.sh mini
./31-qlora-train.sh mini
uv run 32-merge-lora.py --mode mini        # → synthetic-qlora-mini-single-merged/ (bf16)

# 4) PTQ-FFT (택1~전체): AWQ-4bit / FP8 / GGUF Q8_0 / GPTQ-4bit
uv run 40-quant-fft-awq.py --mode mini
uv run 41-quant-fft-fp8.py --mode mini
./42-quant-fft-gguf.sh mini
uv run 43-quant-fft-gptq.py --mode mini

# 5) PTQ-QLoRA (택1~전체, 입력=merged): 동일 5종 → HF 6종 + GGUF 10종, 양자화 총 16종
uv run 50-quant-qlora-awq.py --mode mini
uv run 51-quant-qlora-fp8.py --mode mini
./52-quant-qlora-gguf.sh mini
uv run 53-quant-qlora-gptq.py --mode mini

# 6) PPL 게이트 (base + FP 2종 + 양자화 16종, Δ<0.3 Accept / 0.3~1.0 Conditional / ≥1.0 Discard)
./60-measure-ppl.sh mini                 # → outputs/ppl-mini/ (GGUF 10종은 llama-perplexity)

# 7) 평가 (HF 9종 + GGUF 10종)
./71-eval.sh mini
./72-eval-gguf.sh mini   # llama-completion 필요, 미빌드 시 SKIP
uv run 73-score.py --mode mini

# 8) HF 업로드 → vLLM 서빙 → 추론 예제
./74-upload-hf.sh mini            # → tayaee/Llama-3.2-1B-math-*-mini (targets 지정 가능)
./80-serve-vllm.sh mini fft      # 터미널1: OpenAI-호환 서버 (:8000)
uv run 81-infer-examples.py --model tayaee/Llama-3.2-1B-math-fft-mini  # 터미널2
# SOURCE=hf ./80-serve-vllm.sh mini fft-gptq  # Hub repo 직접 서빙 예시
```

## 분산 학습 (ray/k3s)

- L1 torchrun: 위 스크립트가 `INFRA=dgx-spark-2x`일 때
  `--nnodes=2 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR` 로 분기
  (rank0=spark1, `MASTER_ADDR`는 spark1 IP).
- L2 ray: `distributed/ray-10-baseline.py` (workers=1|2).
- L3 k3s: `distributed/k8s-job-torchrun.yaml` (스켈레톤, 공용경로 마운트).

## 학습 전략 3종 (single / ddp / fsdp)

- `single`: 단일 GPU 단일 프로세스. 산출물 `...-<mode>-single` — **Stage 4·5·7 하류 입력**.
- `ddp`: 2노드 DDP. `INFRA=dgx-spark-2x` + `MASTER_ADDR`(spark1 IP) 필수.
- `fsdp`: FSDP full_shard. 1x에서는 1proc으로 동작만 학습(메모리 이득 없음),
  2x에서는 2proc 실분산.
- effective batch는 월드 크기에 맞춰 자동 동일화
  (micro 2 × accum × world = 64: single accum 32, 2proc accum 16).
- **결과가 똑같은가?** effective batch·seed가 같으면 정확도는 거의 같지만
  bit-identical은 아니다. DDP allreduce와 FSDP reduce-scatter+allgather의
  부동소수점 합산 순서가 다르고, dataloader 샤딩으로 배치 구성도 달라진다.
  `./12-compare-strategies.sh`로 해시를 직접 대조해 확인할 것.
  ddp/fsdp 산출물은 비교용으로 보관하고 하류에서는 single만 쓴다.

## 파일-원본 대응

| 여기 | 원본 (강의 시나리오 / `~/git/maskedthought`) |
|---|---|
| `data/gsm8k-train.jsonl` (7,473행) | 강의 제공 학습 데이터 (벤더링) |
| `data/gsm8k-test.jsonl` (1,319행) | 강의 제공 평가 데이터 (벤더링) |
| `10-baseline-train.sh` | `train_basic.sh` + `main.py` |
| `20-select-candidates.py` | `datagen.ipynb` 전반 |
| `21-build-prompts.py` + `assets/template.txt` | `template.txt` + notebook 후반 |
| `22-teacher-to-generate-syn-data.sh/py` | `data_gen_vllm.py` |
| `23-postprocess.py` | `data_check.ipynb` |
| `30-fft-train.sh` | `train_FFT.sh` |
| `31-qlora-train.sh` | `train_QLoRA.sh` |
| `32-merge-lora.py` | `save_new_vocab_model.ipynb` 계열 |
| `40~43-quant-fft-*` | `quantizaton_math.ipynb` (+FP8) — FFT 입력 4종 |
| `50~53-quant-qlora-*` | 신규: 동일 5종, QLoRA-merged 입력 (HF 6종 + GGUF 10종, 양자화 총 16종) |
| `60-measure-ppl.sh/py` | 신규: PPL 게이트 (Accept/Conditional/Discard) |
| `71-eval.sh` / `73-score.py` | `evaluation/` 3종 |
| `74-upload-hf.sh/py` | 신규: Hub 업로드 (`tayaee/*`) |
| `80-serve-vllm.sh` | 신규: vLLM OpenAI-호환 서빙 |
| `81-infer-examples.py` | 신규: 서빙 추론 예제 (GSM8K N개 + 한국어 1개) |

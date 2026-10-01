# PROCEDURE — Capstone: GSM8K 수학 추론 SFT + PTQ (DGX Spark 1x/2x)

> 원본: 양자화 강의 Part 5 시나리오(2024년, 4x discrete GPU 서버 기준).
> 참조용 업스트림: `~/git/maskedthought` (`00-setup.sh`가 없으면 클론).
> 실행에 필요한 강의 전용 파일(`gsm8k-*.jsonl`, `template.txt`)은
> `project-5/data/`·`assets/`에 벤더링되어 있어 이 디렉토리만으로 완결된다.
> 표준: `mise` + `uv` + `python@3.12` (repo root 준수, `UV_TORCH_BACKEND=cu130`).
> HF 업로드는 강의의 `DopeorNope` 대신 **`tayaee`** 사용.
> 실행 선택: `INFRA=dgx-spark-1x|dgx-spark-2x` (기본 1x, `PROFILE=1x|2x` 약칭),
> 첫 인자 `mini|full` (기본 mini). 공통 설정은 `config/common.env`.
> 분산학습: torchrun 기본 + `ray`로 multi-node + `k3s` Job까지 단계별 학습.
> 공유디스크: `/rosenas/data/AIML/project-5-shared/{datasets,models,outputs,hf-cache,vllm-cache}` (spark1·spark2 공통 경로).

## 0. 전체 파이프라인 (5 Stage)

```
[0. 환경] → [1. 베이스라인 FFT/GSM8K] → [2. 합성데이터 생성] → [3. 합성데이터 SFT: FFT + QLoRA]
  → [4. PTQ: llama.cpp Q8_0 / GPTQ-4bit / AWQ-4bit / FP8] → [5. 평가·Inference: vLLM greedy GSM8K]
```

| Stage | 스크립트 | 입력 → 출력 |
|---|---|---|
| 0. 환경 | `00-setup.sh` | 업스트림 클론 + 의존성 + 공용 datasets 복사 |
| 1. 베이스라인 | `10-baseline-train.sh` (+`10-train-entry.py`) | `gsm8k-train.jsonl` → `base-gsm8k-<mode>-<strat>/` (Llama-3.2-1B) |
| 2a. 후보 선별 | `20-select-candidates.py` | Alpaca 52k + GSM8K → cosine 유사도 → 약 12k 후보 |
| 2b. 프롬프트+생성 | `21-build-prompts.py`, `22-generate.sh/py` | 후보 → `template.txt` 프롬프트 10k → Llama-3.1-8B-Instruct(vLLM) → `generated-<mode>.csv` (+invalid 재생성 루프) |
| 2c. 후처리 | `23-postprocess.py` | 생성 CSV → Q/A split, 패턴 제거 → `synthetic-<mode>.jsonl` (`source`/`target` 리스트형) |
| 3. 합성 SFT | `30-fft-train.sh` / `31-qlora-train.sh` | `synthetic-<mode>.jsonl` → `synthetic-fft-…/` (FFT) + `synthetic-qlora-…/` (어댑터만) |
| 4. PTQ | `40-quant-gguf.sh`, `41/42/43-quant-*.py` | FFT **single** 결과 → GGUF Q8_0 / GPTQ-4bit-g128 / AWQ-4bit-g128 / **FP8(w8a8)** |
| 5. 평가 | `50-merge-lora.py`, `51-eval.sh`, `52-score.py` | base / FFT / QLoRA-merged / GPTQ / AWQ / FP8 → `eval-<mode>/` → 정답 추출 (`####` / `The answer is` / OpenAI-mini 보조) |
| 6. 배포 | `53-upload-hf.sh/py`, `60-serve-vllm.sh`, `61-infer-examples.py` | 산출물 → `tayaee/*` 업로드 → vLLM OpenAI-호환 서빙 → 추론 예제 |

핵심 포인트 (강의에서 강조, 구현에 반영):
- 로그의 `gram loss/acc`는 커스텀 트레이너 용어 — **CE loss 하락만 볼 것**.
- 학습 프롬프트 형식과 캘리브레이션/평가 프롬프트 형식을 **반드시 일치**시킬 것.
- QLoRA 저장물은 **어댑터만** 저장됨. vLLM 평가는 merge 필요 (`resize_token_embeddings` 필수 — `<mask>`/`[PAD]` 추가로 vocab 128259).
- `df.sample()` 후 반드시 `reset_index(drop=True)` (인덱스 미스매치 방지).
- 강의의 `datasets==2.0.0` 다운그레이드 루틴은 2026 스택에서 불필요 (아래 표).

## 1. Stage별 절차

### Stage 0. 환경 (`00-setup.sh`, 양 노드 1회)
1. root `uv sync` + `requirements-p5.txt` 설치.
2. `~/git/maskedthought` 없으면 `https://github.com/changyuchen347/maskedthought` 클론 (참조용).
3. `data/gsm8k-{train,test}.jsonl` → `$P5_SHARED/datasets/` 복사 (없을 때만).
4. `hf auth login` (Llama gated), `nvidia-smi` + torch CUDA 확인.

### Stage 1. 베이스라인 FFT
- 하이퍼파라미터 (강의 `train_basic.sh` 그대로): Llama-3.2-1B, epoch 3(full)/1(mini),
  lr 1e-5, cosine, warmup 0.03, micro 2, tok/tgt 512(→`--max_len 1024`), seed 1, 200스텝 로그, epoch 저장.
- DGX 수정: `nproc_per_node=1`, FSDP는 `fsdp` 전략일 때만, `gradient_checkpointing=True`, bf16.
- 전략 3종: `./10-baseline-train.sh [mini|full] [single|ddp|fsdp]`
  (ddp는 2x+`MASTER_ADDR` 필수). 출력 `...-<mode>-<strat>`, 하류는 `-single`만 사용.

### Stage 2a. 임베딩 선별 (`20-select-candidates.py`)
1. `sentence-transformers/all-mpnet-base-v2` 로드.
2. Alpaca(`tatsu-lab/alpaca`, 52k): `cri = instruction + output`, 500개씩 배치 인코딩(약 105 배치).
3. GSM8K 임베딩과 cosine_similarity → GSM8K당 상위 1000개 중 랜덤 100개 → 약 12k.
4. HF push는 `tayaee/alpaca_syntheticdatagen_prompt-<mode>`.

### Stage 2b. 프롬프트 생성 + vLLM 합성 (`21`, `22`)
1. 후보에서 2개씩 샘플 → `instruction(+input)` 합쳐 `assets/template.txt`의 `{dg_instruct}/{dg_output}` 포맷 → `prompt` 컬럼 (full 10k).
2. teacher `meta-llama/Llama-3.1-8B-Instruct` + `apply_chat_template` →
   `SamplingParams(temp 0.5, top_p 0.8, top_k 5, rep_penalty 1.05, max 2048)`.
   `tensor_parallel_size` = TP (1x→1, 2x→2), `max_model_len` 4096(mini)/8192(full).
3. `Transformed Domain Question/Answer` + `The answer is` 미포함 행 = invalid → 분리 후 invalid만 재생성 루프.

### Stage 2c. 후처리 → `synthetic-<mode>.jsonl` (`23-postprocess.py`)
Q/A 마커 기준 split + 케이스별 예외 + 잔여 패턴 제거 + `rstrip` →
`{'source':[q], 'target':[a]}` json-lines. Stage 3 SFT와 Stage 4 캘리브레이션 입력.

### Stage 3. 합성 SFT — FFT vs QLoRA
- FFT (`30-fft-train.sh`): Stage 1과 동일 설정, train만 `synthetic-<mode>.jsonl`.
- QLoRA (`31-qlora-train.sh`): 4bit nf4 double_quant, LoRA r16/a32/d0.05,
  target `q/k/v/o/gate/down/up` (+`embed/lm_head`는 modules_to_save).
  FSDP+QLoRA 조합은 강의대로 비권장 (fsdp 전략은 학습용으로만 수행).
- 결과물은 어댑터만 → Stage 5 전 merge 필수.

### Stage 4. PTQ (입력: FFT **single** 결과)
1. **llama.cpp** (`40`): `convert_hf_to_gguf.py` → FP16 GGUF → `llama-quantize Q8_0`.
   강의의 vocab assert 수동패치는 2026 빌드에서 불필요.
2. **GPTQ** (`41`, `gptqmodel`): 4bit-g128, damp 0.1 + synthetic 캘리브 토크나이즈 →
   `save_quantized` + tokenizer 동봉.
3. **AWQ** (`42`, `llm-compressor`): 4bit-g128 GEMM + `calib_data=text_lst` →
   **`model.to('cpu')` 후 저장** + tokenizer 동봉.
4. **FP8** (`43`, 신규, `llm-compressor` w8a8 e4m3, Blackwell 네이티브):
   mini 캘리브 4 / full 64. 평가는 vLLM `quantization='fp8'`.

### Stage 5. Inference·평가
1. QLoRA merge (`50`): base 로드 → 특수토큰 동일 추가 →
   **`resize_token_embeddings`** → `merge_and_unload()` → `*-single-merged/` (resize 생략시 size-mismatch).
2. vLLM greedy (`51`): `SamplingParams(temp 0, max 512)`, `prompt_no_input` 포맷.
   타깃: base / fft / qlora(-merged) / gptq / awq / fp8. 양자화 타깃은 quant flag 부여.
3. 채점 (`52`): `#### <숫자>` 추출, invalid율/acc/BLEU. base처럼 `The answer is`를 안 뱉는
   모델은 CSV 덤프 후 OpenAI `gpt-4o-mini`로 답만 재추출 (강의 방식, 선택).

## Stage 6. 업로드·서빙·추론 예제 (신규)
1. 업로드 (`53`): `TARGETS` 매핑(fft/qlora/gptq/awq/fp8 폴더, gguf 단일파일)대로
   `tayaee/p5-1B-math-<target>-<mode>` 생성+업로드. 로컬 산출물 없으면 SKIP.
2. 서빙 (`60`): `vllm serve <model> --served-model-name <repo> --tensor-parallel-size $TP`
   + 타깃별 `--quantization` (gptq/awq/fp8/gguf), `--max-model-len 2048`,
   `--gpu-memory-utilization 0.9` (env `GPU_UTIL`로 조정). `SOURCE=hf`면 Hub repo 직접 서빙.
3. 추론 예제 (`61`): gsm8k-test 앞 N개 + 한국어 1문제를 학습과 동일 `prompt_no_input`
   포맷으로 completions API 요청 (`temperature 0`, chat template이 없어 chat API는 400).
   서버는 별도 터미널에서 실행.

## 2. 버전표 (2024 강의 → 본 repo)

| 구성 | 2024 강의 | 본 repo | 비고 |
|---|---|---|---|
| Python | 3.11 conda `mass` | **3.12** (`uv`, root `mise.toml`) | 개별설치 루틴 불필요 |
| torch | 2.1.0/2.5.1 cu118/cu124 x86 | **cu130 aarch64** (root `pyproject`, `UV_TORCH_BACKEND=cu130`) | GB10 sm_120, `flash_attention: false`(SDPA, `infra/` 관례) |
| transformers / trl / peft | 4.46.3 / 구버전 | **5.17.0 / 1.8 / 0.19** (smoke 실측) | `load_in_4bit`+`bnb_4bit_*` 키 동일, peft Bloom shim 필요 |
| datasets | 2.0.0 핀 ↔ 3.2.0 혼용 | **4.x 통일** | 궁합 패치 불필요 |
| bitsandbytes | 4bit QLoRA용 | **0.49** (ARM 이슈시 **torchao** 대체) | root에 `torchao` 이미 있음 |
| sentence-transformers / sklearn | 개별설치 | **pip 통합설치** | conda/pip 분리 불필요 |
| vLLM | 0.2.2 / 0.5.5 | **0.30.0** (smoke 실측, aarch64) | AWQ/GPTQ는 `gptqmodel`/`llmcompressor` 경로, flashinfer sampler off 필요 |
| auto-gptq / autoawq | 원본 라이브러리 | **`gptqmodel` / `llm-compressor`** (둘 다 deprecated 후속) | 흐름 동일 |
| llama.cpp | assert 수동패치 | **최신 빌드 (패치 불필요)** | `Q8_0` 절차 동일 |
| 모델·데이터 | Llama-3.2-1B / 3.1-8B-Instruct, Alpaca, `DopeorNope/*` | **동일** + HF id **`tayaee/*`**, 데이터는 `data/` 벤더링 | gated 로그
...[truncated 911 chars]
## 3. mini 실측 시간표 (spark2, GB10 idle 기준)

> 실측 = 타이머·로그로 확인한 값, 추정 = 이전 실행 경과로부터의 유추.
> mini 경량화 이후(GSM 256행·합성 64개·평가 10개·캘리브 2) 합계 **약 30분**.
> 구 mini(전량·200개·50개) 기준으로는 약 80~100분이었다.

| Stage | 스크립트 | mini 시간 | 구분 | 비고 |
|---|---|---|---|---|
| 0 | `00-setup.sh` | ~10분 | 추정 | 초회 패키지 다운로드 포함. 2회目以降 수십초 |
| 1 | `10-baseline` | ~3.5분 | 실측 | 256행·4스텝. 전량 7473행 시 20.4분 실측(train_runtime 1224s) |
| 2a | `20-select` | 15초 | 실측 | 임베딩 캐시 적중 시. 첫 실행은 52k 인코딩 수 분 추가 |
| 2b | `21-build` | 1초 | 실측 | 200프롬프트 포맷 |
| 2b | `22-generate` | ~3분 | 추정 | 64프롬프트. 200개 시 8.3분 실측(500s, 재생성 3회). 8B teacher 상주 후 기준, 첫 다운로드는 15GB 별도(~25분) |
| 2c | `23-postprocess` | ~10초 | 추정 | 188 kept / 12 dropped |
| 3 | `30-fft` | 2.2분 | 실측 | train_runtime 133.5s, 3스텝, loss 1.50 |
| 3 | `31-qlora` | 2.3분 | 실측 | train_runtime 139.1s, 3스텝, loss 1.62 |
| 4 | `40-gguf` | ~2분 | 추정 | quantize 34초 실측 + convert. llama.cpp CPU 빌드 별도 ~10분(1회) |
| 4 | `41-gptq` | ~4분 | 추정 | 1B 로드 + 4캘리브 + 저장 |
| 4 | `42-awq` | ~6분 | 추정 | oneshot smoothing + 112모듈 compress + 저장 |
| 4 | `43-fp8` | ~5분 | 추정 | 42와 동형 |
| 5 | `50-merge` | ~3분 | 추정 | 1B 로드 + 병합 + 저장 |
| 5 | `51-eval` | ~9분 | 추정 | 6타깃×10개 (1타깃 85초 실측). 50개 시 ~20분 |
| 5 | `52-score` | ~5초 | 추정 | 6타깃 전부 acc 0.000 (3스텝 undertraining, §1 Stage 5 참조) |
| 6 | `53-upload` | ~10분 | 추정 | ~14GB 업로드, 회선依存 |
| 6 | `60-serve` | ~3분 | 실측 | 기동~`/v1/models` UP (1B 모델 상주 후) |
| 6 | `61-infer` | ~30초 | 추정 | 3문항 completions 왕복 |

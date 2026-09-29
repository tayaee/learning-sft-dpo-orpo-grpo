# SFT Demo — Qwen2.5-1.5B-Instruct + 한국어 CoT 추론

> **목적**: Qwen/Qwen2.5-1.5B-Instruct(베이스, 로컬 캐시) 모델을
> `train.jsonl` (50k 한국어 추론) 데이터로 **LoRA SFT** 한 뒤,
> **query_base.py vs query_sft.py** 로 두 모델의 출력 차이를 직접 비교.

---

## 0. 환경 확인 (30초)

```bash
cd /home/user1/git/learning-sft-and-rl

# GPU / CUDA
nvidia-smi -L                                # NVIDIA GB10 (Blackwell)
nvcc --version | head -4                     # 13.0.x

# Python / torch / axolotl
uv run python -c "import torch,axolotl; print(torch.__version__, torch.version.cuda, axolotl.__version__)"
# 2.12.1+cu130 13.0 0.18.0

# 로컬 모델 캐시 확인
ls ~/.cache/huggingface/hub/ | grep Qwen2.5-1.5B-Instruct
# models--Qwen--Qwen2.5-1.5B-Instruct
```

> 모든 명령은 **uv run** 으로 실행 (`.venv` 자동 활성화).

---

> **모드 규약**: 학습/merge/추론 스크립트는 `mini` 또는 `full` 을 **반드시 인자로 지정**해야 합니다.
> - `mini`: 파이프라인 점검용 (소량 데이터 + max_steps 3~10, 수분 내 완주)
> - `full`: 실제 학습 (기존과 동일한 full 설정)
> 예: `phase/sft/algorithm/lora/tool/axolotl/sft-mini-06-run-axolotl-mini.sh`(또는 `phase/sft/algorithm/lora/tool/axolotl/sft-06-run-axolotl.sh mini`) → `phase/sft/algorithm/lora/tool/axolotl/sft-mini-07-merge-mini.sh` → `uv run library/script/query_sft.py --mode mini "..."`
> 모드 지정 스크립트에는 `<script>-mini.sh` / `<script>-full.sh` 래퍼가 준비되어 있다.

## 1. 베이스 모델 sanity check (1분)

```bash
# Qwen2.5-1.5B-Instruct 모델이 로딩되는지 확인
uv run library/script/query_base.py "방정식 x^2 + 5x + 6 = 0 의 해를 구하시오."
```

**기대 출력**: 짧고 일반적인 답변,  단계가 없음.

---

## 2. 데이터 sanity check (30초)

```bash
# train.jsonl 포맷: {"messages": [{"role":"user","content":"..."},
#                                  {"role":"assistant","content":"..."}]}
head -1 train.jsonl | uv run python -m json.tool | head -10
wc -l train.jsonl               # 50000
```

> **mini 모드 학습용 데이터**는 root `train.jsonl` 의 앞 200건을 자동 추출하여
> `data/sft-mini-out/train-sft.jsonl` 로 저장한다. (mini 스크립트가 자동 처리)

---

## 3. axolotl config 검증 (1분)

```bash
cat phase/sft/algorithm/lora/tool/axolotl/config/qwen2.5-1.5b-sft.yaml | head -20
```

핵심 파라미터:
- `base_model: Qwen/Qwen2.5-1.5B-Instruct` (HF 캐시 자동 사용)
- `flash_attention: false` (sm_120 미지원 → SDPA fallback)
- `lora_r: 16, lora_alpha: 32`
- `micro_batch_size: 4, gradient_accumulation_steps: 8` → effective batch 32
- `num_epochs: 1, sequence_len: 1024, sample_packing: true`

---

## 4. SFT 학습 (50–80분)

```bash
cd /home/user1/git/learning-sft-and-rl

# 모니터링 (별도 터미널)
watch -n 5 nvidia-smi

# 학습 시작 (full 모드)
phase/sft/algorithm/lora/tool/axolotl/sft-06-run-axolotl.sh
# 또는: uv run axolotl train phase/sft/algorithm/lora/tool/axolotl/config/qwen2.5-1.5b-sft.yaml 2>&1 | tee logs/sft.log
```

**체크 포인트**:
- 첫 10 step 안에 GPU 메모리 ~14–18 GB 점유
- `logging_steps: 10` → loss 가 10 step마다 출력
- 최종 loss 1.5–2.0 대로 떨어지면 정상
- 완료 시 `data/sft-full-out/adapter/` 에 **LoRA 어댑터만** 저장됨
  (`adapter_config.json`, `adapter_model.safetensors`)
- merge된 모델은 별도 단계에서 생성 (Stage 5 참고)

서버 로그 위치 (필요시):
```bash
mkdir -p logs
tail -f logs/sft.log                  # 다른 터미널에서
```

---

## 5. LoRA Merge (30초)

> axolotl의 `train`은 LoRA 어댑터만 저장하므로 **merge 단계가 별도로 필요**합니다.
> merge된 base+LoRA 모델이 있어야 `query_sft.py`로 추론할 수 있습니다.

```bash
cd /home/user1/git/learning-sft-and-rl

phase/sft/algorithm/lora/tool/axolotl/sft-07-merge.sh
# 또는:
# uv run axolotl merge-lora phase/sft/algorithm/lora/tool/axolotl/config/qwen2.5-1.5b-sft.yaml \
#   --lora-model-dir ./data/sft-full-out/adapter \
#   --output-dir    ./data/sft-full-out
```

**체크 포인트**:
- 명령어 옵션명은 `--lora-model-dir`, `--output-dir` (언더스코어 아닌 대시) — fire CLI 규약
- axolotl은 항상 `output_dir` 하위에 `merged/` 폴더를 만들어 저장
  → 최종 경로: `./data/sft-full-out/merged/`
- 로그 끝에 `Applied LoRA to N/338 tensors` 형태의 메시지가 나오면 성공 (보통 20–30초)

생성된 폴더 확인:
```bash
ls data/sft-full-out/merged/   # config.json, model.safetensors, tokenizer.* 등
```

---

## 6. SFT 모델 검증 (1분)

```bash
# merge 폴더 확인
ls data/sft-full-out/merged/

# 추론
uv run library/script/query_sft.py --mode full "방정식 x^2 + 5x + 6 = 0 의 해를 구하시오."
```

**기대 출력**:  블록 + 단계별 풀이 +  ###  결론. 베이스보다 길고 구조적.

---

## 7. 두 모델 비교 (5분)

```bash
# 베이스
uv run library/script/query_base.py "다음 명제의 대우를 쓰시오: 모든 소수는 홀수이다."

# SFT
uv run library/script/query_sft.py --mode full "다음 명제의 대우를 쓰시오: 모든 소수는 홀수이다."
```

또는 REPL 모드로 연속 비교:

```bash
uv run library/script/query_base.py
# [you] 삼각형 ABC의 넓이가 30이고 밑변이 10일 때 높이는?
# [you] f(x)=x^3-3x+2의 극값을 구하시오.
# [you] quit
```

```bash
uv run library/script/query_sft.py --mode full
# 동일한 질문 반복
```

---

## 8. lm-eval 정량 평가 (선택, 5분/모델)

정성 비교와 별개로 표준 benchmark 로 두 모델의 점수를 측정.
각 task 당 100 samples 제한 (`--limit 100`) 으로 sanity check 용도.
전체 데이터 평가가 필요하면 스크립트의 `--limit 100` 줄을 제거.

```bash
# BASE 점수
phase/sft/algorithm/lora/tool/axolotl/sft-10-lm-eval-base.sh

# SFT 점수
phase/sft/algorithm/lora/tool/axolotl/sft-11-lm-eval-sft.sh        # mini 면 phase/sft/algorithm/lora/tool/axolotl/sft-mini-11-lm-eval-sft-mini.sh → outputs/lm_eval_results/sft-{mode}/
```

평가는 다음을 포함:
- 한국어: `kobest_hellaswag`, `kobest_copa`, `kmmlu`
- 영어: `hellaswag`, `arc_easy`, `piqa`, `winogrande`

결과는 `outputs/lm_eval_results/<base|sft>/` 에 저장됨.

> 참고: chat template 적용 모델의 경우 `kmmlu` 가 raw 점수보다 낮게
> 나오는 경향이 있음 (선지 형식 차이). 정성 비교와 함께 봐야 함.

---

## 9. 차이 요약

| 모델 |  블록 | 단계별 풀이 | 한국어 수학 용어 | 결론 명료성 |
|------|-------|-----------|----------------|-----------|
| BASE | 없음 | 약함 | 보통 | 보통 |
| SFTED | 있음 | 강함 | 자연스러움 | 결론 ### 섹션 |

---

## 10. 다음 단계

이제 **RL(DPO)** 로 같은 모델을 더 다듬고 싶다면 → [`phase/pref/algorithm/dpo/tool/axolotl/README.md`](phase/pref/algorithm/dpo/tool/axolotl/README.md) 참고.

---

## 트러블슈팅

| 증상 | 해결 |
|------|------|
| `flash-attn build failed` | 이미 `flash_attention: false` 설정. 무시. |
| `CUDA OOM` | `micro_batch_size: 4` → `2` 로 줄이기 |
| `torch==2.12.1 not found` | `UV_TORCH_BACKEND=cu130` 확인 (mise.toml) |
| `loss 가 8.0 에서 안 떨어짐` | `learning_rate: 2e-4` → `1e-4` 로 줄이기 |
| `data/sft-full-out/merged` 가 없음 | train은 merge를 자동으로 만들지 않음. **Stage 5 `axolotl merge-lora` 명령 실행 필수** (`lora_model_dir`는 merge 출력 경로가 아니라 어댑터 **입력** 경로임) |

---

## 한 줄 요약

```bash
uv run library/script/query_base.py "Q"  \
  &&  phase/sft/algorithm/lora/tool/axolotl/sft-06-run-axolotl.sh  \
  &&  phase/sft/algorithm/lora/tool/axolotl/sft-07-merge.sh  \
  &&  uv run library/script/query_sft.py --mode full "Q"
```
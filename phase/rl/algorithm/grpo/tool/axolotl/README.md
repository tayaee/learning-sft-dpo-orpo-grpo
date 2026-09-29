# GRPO Demo — Group Relative Policy Optimization on SFT'd Qwen2.5-1.5B

> **목적**: SFT'd 모델(`data/sft-full-out/merged`)을 시작점으로,
> **rule-based reward 함수** 만으로 **GRPO** 학습해 응답 구조 / 포맷 /
> 학술 용어 사용을 강화.
> 사전 조건: [`phase/sft/algorithm/lora/tool/axolotl/README.md`](phase/sft/algorithm/lora/tool/axolotl/README.md) Stage 4 완료 (SFT merge 폴더 존재).

---

## 0. GRPO 의 의의 — DPO 와 어떻게 다른가?

| 비교축 | DPO | GRPO |
|--------|-----|------|
| **신호** | (prompt, chosen, rejected) | (prompt) + reward function |
| **데이터** | 인간 선호 라벨 필요 | 라벨 불필요, 보상 함수만 |
| **Reference model** | 필요 (KL anchor) | 선택 (beta=0 이면 없음) |
| **Baseline** | reference KL | **group-relative** (그룹 내 평균) |
| **Value model** | 없음 | 없음 (PPO 와 차이) |
| **State-of-the-art** | (구세대) | **DeepSeek-R1, Qwen3-Thinking, o1** (2025) |
| **학습 안정성** | 선호 품질에 민감 | 그룹 정규화로 안정 |
| **확장성** | 선호 데이터 한계 | sample 수 ∝ 품질 |

### GRPO 가 "더 낫다" 고 말할 수 있는 경우

| 시나리오 | 추천 | 이유 |
|----------|------|------|
| **검증 가능한 정답** (수학, 코딩, 과학) | **GRPO** | rule-based reward 가 정답 일치 여부로 즉시 산출 |
| **자기 개선 / exploration** | **GRPO** | N개 sample 중 좋은 것 강화 → 모델이 스스로 패턴 발견 |
| **subjective / 스타일** (톤, 해로움, 친절함) | DPO | 인간 선호가 곧 정답 |
| **인간 라벨 보유** | DPO | 라벨 쓰지 않으면 낭비 |
| **GPU 적음** | DPO | GRPO 는 N× 추론 비쌈 |
| **검증 불가** | DPO | GRPO 는 보상 함수 자체가 정의 불가 |

### GRPO가 DPO 보다 "일반적으로 더 낫다" ?

**아닙니다.** 둘은 **다른 가정** 하에 작동합니다:

- **DPO** : "이 응답 vs 저 응답" 쌍이 주는 **상대적** 신호. **쌍 라벨** 필요.
- **GRPO** : "응답의 절대 점수" (보상). **점수 함수** 필요.

> 핵심: GRPO 가 2025–2026 SOTA 인 이유는 **검증 가능한 task** (수학·코딩)에서
> 절대 우위라서. **subjective task** 에선 DPO/RLHF 가 여전히 우위.

### 본 데모에서의 의의

본 데이터(`amphora/korean-reasoning-small`)는 **검증 가능한 정답이 없는 자유 추론**.
따라서 GRPO 는 **휴리스틱 보상** (포맷, 구조, 학술 용어) 만 사용.
→ 효과가 "보상 추종" 위주라 DPO 와 비교해서 **국소적으로 우위이지만 절대 우위는 아님**.

진짜 GRPO 우위를 보려면 **GSM8K, MATH, HumanEval** 처럼
`\boxed{42}` 같은 검증 가능 정답이 있는 데이터셋 사용 권장.

---

> **모드 규약**: 학습/merge/추론 스크립트는 `mini` 또는 `full` 을 **반드시 인자로 지정**해야 합니다.
> - `mini`: 파이프라인 점검용 (소량 데이터 + max_steps 3~10, 수분 내 완주)
> - `full`: 실제 학습 (기존과 동일한 full 설정)
> 예: `phase/sft/algorithm/lora/tool/axolotl/sft-06-run-axolotl-mini.sh`(또는 `phase/sft/algorithm/lora/tool/axolotl/sft-06-run-axolotl.sh mini`) → `phase/sft/algorithm/lora/tool/axolotl/sft-07-merge-mini.sh` → `uv run library/script/query_sft.py --mode mini "..."`
> 모드 지정 스크립트에는 `<script>-mini.sh` / `<script>-full.sh` 래퍼가 준비되어 있다.

## 1. 보상 함수 정의 (이미 작성됨)

`library/function/reward_fn.py` 에 정의한 4개 휴리스틱:

| 함수 | 점수 (0–1) | 기반 |
|------|-----------|------|
| `format_reward` | 0.5 |  블록 존재 |
| `structure_reward` | 0.3 / 0.5 |  ###  헤더 개수, 결론 단정 |
| `length_reward` | -0.5 ~ 1.0 | 응답 길이 (300–1500자) |
| `keyword_reward` | 0 ~ 1.0 | 한국어 학술 용어 등장 횟수 |

가중치 (`qwen2.5-1.5b-rl-grpo.yaml`):
```yaml
reward_weights: [0.3, 0.3, 0.2, 0.2]   # format / structure / length / keyword
```

확인:
```bash
uv run python -c "import reward_fn; print('reward functions:', [f for f in dir(reward_fn) if f.endswith('_reward')])"
```

---

## 2. GRPO 데이터 (이미 작성됨)

`data/{stage}-{mode}-out/sample_rl-grpo.jsonl` — GRPO 형식 (`{"prompt": "..."}` 만).
모드별 디렉터리(`data/{stage}-mini-out/`, `data/{stage}-full-out/`)에 두고, config 가 해당 모드 디렉터리를 읽는다.

```bash
phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-01-review-data.sh mini        # 또는 full — 모드별 파일 검증
```

전체 50k 로 확장하려면:
```bash
uv run python -c "
import json, random
random.seed(42)
with open('train.jsonl') as f:
    rows = [json.loads(l) for l in f]
random.shuffle(rows)
with open('data/grpo-full-out/train_rl-grpo.jsonl', 'w', encoding='utf-8') as f:
    for r in rows[:5000]:
        f.write(json.dumps({'prompt': r['messages'][0]['content']}, ensure_ascii=False) + '\n')
print('wrote 5000 GRPO prompts to data/grpo-full-out/train_rl-grpo.jsonl')
"
# 그 다음 phase/rl/algorithm/grpo/tool/axolotl/config/qwen2.5-1.5b-rl-grpo.yaml 의 datasets.train_files 를 train_rl-grpo.jsonl 로 변경
```

---

## 3. GRPO config 검증 (1분)

```bash
cat phase/rl/algorithm/grpo/tool/axolotl/config/qwen2.5-1.5b-rl-grpo.yaml | head -30
```

핵심 파라미터:
- `rl: grpo`, `beta: 0.04` (작은 KL 제약)
- `trl.num_generations: 4` (그룹 크기)
- `trl.temperature: 0.9` (응답 다양성)
- `trl.use_vllm: false` (vLLM 미설치).  빠른 학습 원할 시 `uv add vllm` 후 `true`
- `learning_rate: 1e-6` (GRPO 매우 낮음, DPO 보다 5× 낮음)
- `max_steps: 30` (GPU 메모리 보호, 더 길게 가능)

---

## 4. GRPO 학습 (30–60분, use_vllm=false)

```bash
cd /home/user1/git/learning-sft-and-rl

# 모니터링 (별도 터미널)
watch -n 5 nvidia-smi

# 학습 시작
uv run axolotl train phase/rl/algorithm/grpo/tool/axolotl/config/qwen2.5-1.5b-rl-grpo.yaml 2>&1 | tee logs/rl-grpo.log
```

**체크 포인트** (`logging_steps: 1` 이라 매 step 출력):
- `rewards/format_reward/mean` > 0.15 within 20 steps
- `rewards/structure_reward/mean` > 0.15
- `reward_std` > 0 (그룹 내 variance 존재 ⇒ GRPO 신호 살아있음)
- `entropy` 0.05–0.5 (collapse 했으면 0 근처)
- `grad_norm` 0.001–1.0
- 완료 시 `./data/grpo-full-out/adapter/` 에 **LoRA 어댑터만** 저장됨
- merge된 모델은 Stage 5에서 별도 생성 (`./data/grpo-full-out/merged/merged/`)

> **데모 데이터 (20 prompts) 주의**: `data/{stage}-{mode}-out/sample_rl-grpo.jsonl` 만으로 학습 시
> 데이터가 20개뿐이라 `max_steps=30` 이상은 같은 prompt 를 반복 사용하게 됨.
> 본 데모는 `max_steps: 30` 으로 시연 (전체 50k → 500 prompts 로 확장하려면
> Stage 2 의 스크립트 참조.

### vLLM 으로 10× 빠르게 (선택)

```bash
# vLLM 설치 (시간 좀 걸림)
uv add 'vllm>=0.10.2,<0.13.0'

# config 수정
# phase/rl/algorithm/grpo/tool/axolotl/config/qwen2.5-1.5b-rl-grpo.yaml:
#   trl.use_vllm: true
# 트레이너 실행 전 vLLM 서버 별도 실행
# 터미널 1:
uv run axolotl vllm-serve phase/rl/algorithm/grpo/tool/axolotl/config/qwen2.5-1.5b-rl-grpo.yaml
# 터미널 2:
uv run axolotl train phase/rl/algorithm/grpo/tool/axolotl/config/qwen2.5-1.5b-rl-grpo.yaml
```

---

## 5. LoRA merge (1분)

> axolotl `train`은 LoRA 어댑터만 저장하므로 **merge 단계가 별도로 필요**합니다.
> merge된 base+LoRA 모델이 있어야 `query_rl_grpo.py`로 추론할 수 있습니다.

```bash
cd /home/user1/git/learning-sft-and-rl
phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-06-merge.sh full
```

- axolotl은 항상 `output_dir` 하위에 `merged/` 폴더를 만들어 저장
  → 최종 경로: `./data/grpo-full-out/merged/merged/`

확인:
```bash
ls data/grpo-full-out/merged/merged/   # config.json, model.safetensors, tokenizer.* 등
```

---

## 6. GRPO 모델 검증 (1분)

```bash
# merge 폴더 확인
ls data/grpo-full-out/merged/merged/

# 추론
uv run library/script/query_rl_grpo.py --mode full "방정식 x^2 + 5x + 6 = 0 의 해를 구하시오."
```

**기대 출력**:  블록 +  ###  헤더 + 결론 단정 패턴이 **강화**됨.

---

## 7. 네 모델 비교 (5분)

| 모델 | 스크립트 | 경로 |
|------|----------|------|
| BASE | `query_base.py` | `Qwen/Qwen2.5-1.5B-Instruct` |
| SFT  | `query_sft.py` | `./data/sft-full-out/merged` |
| DPO | `query_rl_dpo.py` | `./data/dpo-full-out/merged` |
| GRPO | `query_rl_grpo.py` | `./data/grpo-full-out/merged/merged` |

```bash
phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-08-pythagorean-theorem.sh full   # mini 면 full 대신 mini
# Q="피타고라스 정리를 증명하시오."
# uv run library/script/query_base.py    "$Q"
# uv run library/script/query_sft.py --mode full     "$Q"
# uv run library/script/query_rl_dpo.py --mode full  "$Q"
# uv run library/script/query_rl_grpo.py --mode full "$Q"
```

---

## 8. 차이 요약

| 모델 |  블록 |  ###  헤더 | 결론 단정 | 학술 용어 | 길이 통제 |
|------|-------|-----------|---------|----------|---------|
| BASE | 약함 | 약함 | 약함 | 보통 | 보통 |
| SFT  | 강함 | 강함 | 강함 | 자연스러움 | 다양 |
| DPO  | 강함 | 강함 | 강함 | 자연스러움 | **더 일관** |
| GRPO | 강함 | **더 강함** | **더 강함** | **더 자연스러움** | **보상 함수가 300–1500 유도** |

DPO vs GRPO 비교:
- **DPO** : 사람이 만든 (chosen, rejected) 쌍 → **취향 / 스타일** 같은 subjective signal
- **GRPO** : rule-based reward → **검증 가능 / 절대 점수** 가 있는 objective signal

---

## 9. lm-eval 정량 평가 (선택, 5분)

정성 비교와 별개로 GRPO 모델의 표준 benchmark 점수를 측정.
각 task 당 100 samples 제한 (`--limit 100`) 으로 sanity check 용도.

```bash
phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-09-lm-eval-rl-grpo.sh full      # mini 면 full 대신 mini
```

평가 tasks:
- 한국어: `kobest_hellaswag`, `kobest_copa`, `kmmlu`
- 영어: `hellaswag`, `arc_easy`, `piqa`, `winogrande`

결과는 `outputs/lm_eval_results/rl-grpo-full/` (mini 는 `rl-grpo-mini/`) 에 저장됨.
비교용 BASE/SFT/DPO 점수는 [`phase/sft/algorithm/lora/tool/axolotl/README.md`](phase/sft/algorithm/lora/tool/axolotl/README.md) Stage 8 과 [`phase/pref/algorithm/dpo/tool/axolotl/README.md`](phase/pref/algorithm/dpo/tool/axolotl/README.md) Stage 8 의 결과와 함께 봐야 함.

> 참고: chat template 적용 모델의 경우 `kmmlu` 가 raw 점수보다 낮게
> 나오는 경향이 있음 (선지 형식 차이). 정성 비교와 함께 봐야 함.

---

## 10. 한 줄 요약

```bash
# SFT 까지 끝났다면
phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-04-run-axolotl.sh full && \
  phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-06-merge.sh full && \
  uv run library/script/query_rl_grpo.py --mode full "Q"
```

---

## 트러블슈팅

| 증상 | 해결 |
|------|------|
| `rewards/format_reward` 0.0 | 학습 시작 전 SFT'd 모델이  블록 생성 잘 하는지 확인 |
| `reward_std = 0` | temperature 0.9 → 1.0, num_generations 4 → 8 |
| `entropy` 0.05 미만 (collapse) | learning_rate 더 낮추기, beta 높이기 |
| `oom` | `max_completion_length: 512` → `256`, `num_generations: 4` → `2` |
| vLLM 연결 실패 | `use_vllm: false` 로 두고 transformers generation |
| `outputs/...-merge` 안 생김 | train 로그 확인. **Stage 5 `axolotl merge-lora` 명령 실행 필수** (train은 LoRA 어댑터만 저장, merge 별도 단계). 또한 `lora_model_dir` 는 merge 출력 경로가 아니라 어댑터 **입력** 경로임 |
| `library/function/reward_fn.py` 에서 TypeError (string indices) | TRL 버전이 completion 을 str 로 전달하는 경우. reward 함수에서 dict/str 모두 처리하도록 수정 (예: `_extract_text` helper) |

---

## 부록: GRPO 가 작동하는 핵심 알고리즘

```
for each prompt:
    generate G responses (G = num_generations, e.g., 4)
    score each response with reward_funcs → rewards[i] (총합)
    compute normalized advantage:
        A_i = (rewards[i] - mean(rewards)) / std(rewards)
    policy update:
        L = - min( ratio_i * A_i, clip(ratio_i, 1-ε, 1+ε) * A_i )  + β KL(π || π_ref)
```

핵심: **group 내 표준편차** 로 advantage 추정 → reference model / value model
**둘 다 필요 없음**. PPO 보다 단순, DPO (pairwise) 보다 절대 점수 활용.

---

## 다음 단계

이 4 모델 비교가 끝났다면 → Stage 9 (`phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-09-lm-eval-rl-grpo.sh full`) 로 정량 평가.

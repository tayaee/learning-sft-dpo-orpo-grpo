# RL Demo — DPO on top of SFT'd Qwen2.5-1.5B

> **목적**: SFT'd 모델(`data/sft-full-out/merged`)을 시작점으로,
> **합성 선호 데이터** 로 **DPO** 를 추가 학습해 응답 일관성·구조를 더 강화.
> 사전 조건: [`phase/sft/algorithm/lora/tool/axolotl/README.md`](phase/sft/algorithm/lora/tool/axolotl/README.md) Stage 4 완료 (SFT merge 폴더 존재).

---

## 0. RL 알고리즘 선택 — DPO

| 알고리즘 | 데이터 | 복잡도 | 비고 |
|----------|--------|--------|------|
| **DPO** | (prompt, chosen, rejected) | 낮음 | reference model 1개, 가장 진입 쉬움 |
| ORPO | (prompt, chosen, rejected) | 낮음 | reference model 없음 |
| GRPO | (prompt) + 보상함수 | 중간 | N개 sample, KL 없이 group-relative |
| PPO | (prompt) + reward model | 높음 | 별도 reward model 필요 |

여기선 **DPO** 를 사용합니다. 이유:
- SFT'd 모델을 시작점으로 사용 가능
- reference model 은 베이스 모델 (`Qwen/Qwen2.5-1.5B-Instruct`) 자동 지정
- 합성 선호 데이터를 스크립트(`make_rl_dpo_data.py`)로 만들 수 있음

---

> **모드 규약**: 학습/merge/추론 스크립트는 `mini` 또는 `full` 을 **반드시 인자로 지정**해야 합니다.
> - `mini`: 파이프라인 점검용 (소량 데이터 + max_steps 3~10, 수분 내 완주)
> - `full`: 실제 학습 (기존과 동일한 full 설정)
> 예: `phase/sft/algorithm/lora/tool/axolotl/sft-06-run-axolotl-mini.sh`(또는 `phase/sft/algorithm/lora/tool/axolotl/sft-06-run-axolotl.sh mini`) → `phase/sft/algorithm/lora/tool/axolotl/sft-07-merge-mini.sh` → `uv run library/script/query_sft.py --mode mini "..."`
> 모드 지정 스크립트에는 `<script>-mini.sh` / `<script>-full.sh` 래퍼가 준비되어 있다.

## 1. 선호 데이터 생성 (10–30분)

```bash
cd /home/user1/git/learning-sft-and-rl
# 권장: 드라이버 스크립트 사용 (모드별 SFT 모델/경로 자동 선택)
phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-01-make-rl-dpo-data.sh full     # → data/dpo-full-out/train_rl-dpo.jsonl (+ sample)
phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-01-make-rl-dpo-data.sh mini    # → data/dpo-mini-out/train_rl-dpo.jsonl (+ sample)

# 직접 실행할 경우 (full 예시):
# 베이스 + SFT'd 모델 로드 → 1000 prompts × 4 samples K=4 → 1000 pairs
uv run python library/script/make_rl_dpo_data.py \
  --sft-model ./data/sft-full-out/merged \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --num-prompts 1000 \
  --samples-per-prompt 4 \
  --out data/dpo-full-out/train_rl-dpo.jsonl \
  --sample-out data/dpo-full-out/sample_rl-dpo.jsonl
```

**생성 원리**:
1. `train.jsonl`에서 N=1000 prompt 샘플링
2. 각 prompt마다 SFT'd 모델에서 K=4개 응답 (T=0.7 sampling)
3. 각 응답에 점수 부여 (  블록,  ###  구조, 적정 길이, 결론 단정)
4. 점수 최고 = chosen, 점수 최저 = rejected
5. 베이스 모델 출력 (T=0.0) 도 rejected 후보로 추가

**출력**:
- `data/{stage}-{mode}-out/train_rl-dpo.jsonl` — DPO 학습용 (full 1000 / mini 50 pairs)
- `data/{stage}-{mode}-out/sample_rl-dpo.jsonl` — 디버깅용 샘플

확인:
```bash
wc -l data/dpo-full-out/train_rl-dpo.jsonl data/dpo-full-out/sample_rl-dpo.jsonl
head -1 data/dpo-full-out/sample_rl-dpo.jsonl | uv run python -m json.tool | head -10
```

---

## 2. 합성 데이터 품질 체크 (1분)

```bash
# 같은 prompt 에 대한 chosen vs rejected 비교 (phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-02-sanity-check-rl-dpo-data.sh <mode>)
head -1 data/dpo-full-out/sample_rl-dpo.jsonl | uv run python -c "
import json, sys
d = json.loads(sys.stdin.read())
print('=== PROMPT ===')
print(d['prompt'][:200])
print('\n=== CHOSEN ===')
print(d['chosen'][:400])
print('\n=== REJECTED ===')
print(d['rejected'][:400])
"
```

**기대**: `chosen`은  블록 +  ###  구조, `rejected`는 짧고 평평한 응답.

---

## 3. DPO config 검증 (1분)

```bash
cat phase/pref/algorithm/dpo/tool/axolotl/config/qwen2.5-1.5b-rl-dpo.yaml | head -20
```

핵심 파라미터:
- `base_model: ./data/sft-full-out/merged` (SFT'd 모델에서 시작)
- `rl: dpo`, `dpo_beta: 0.1`, `dpo_loss_type: sigmoid`
- `ref_model: Qwen/Qwen2.5-1.5B-Instruct` (베이스 모델이 reference)
- `learning_rate: 5e-6` (DPO는 SFT보다 10–20× 낮은 LR)
- `micro_batch_size: 2, gradient_accumulation_steps: 8` → effective batch 16

---

## 4. DPO 학습 (30–60분)

```bash
cd /home/user1/git/learning-sft-and-rl

# 모니터링 (별도 터미널)
watch -n 5 nvidia-smi

# 학습 시작
uv run axolotl train phase/pref/algorithm/dpo/tool/axolotl/config/qwen2.5-1.5b-rl-dpo.yaml 2>&1 | tee logs/rl-dpo.log
```

**체크 포인트**:
- DPO loss  시작 ~0.69 → 점진적 감소
- `reward_margin` (chosen - rejected 보상 차이)  양수로 증가 ⇒ 정상
- DPO 는 SFT보다 훨씬 빠름 (1k pairs × 1 epoch)
- 완료 시 `./data/dpo-full-out/adapter/` 에 **LoRA 어댑터만** 저장됨
- merge된 모델은 Stage 5에서 별도 생성 (`./data/dpo-full-out/merged/`)

---

## 5. LoRA merge (1분)

> axolotl `train`은 LoRA 어댑터만 저장하므로 **merge 단계가 별도로 필요**합니다.
> merge된 base+LoRA 모델이 있어야 `query_rl_dpo.py`로 추론할 수 있습니다.

```bash
cd /home/user1/git/learning-sft-and-rl
phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-06-merge.sh full
```

- axolotl은 항상 `output_dir` 하위에 `merged/` 폴더를 만들어 저장
  → 최종 경로: `./data/dpo-full-out/merged/`

확인:
```bash
ls data/dpo-full-out/merged/merged/   # config.json, model.safetensors, tokenizer.* 등
```

---

## 6. RL 모델 검증 (1분)

```bash
# merge 폴더 확인
ls data/dpo-full-out/merged/merged/

# 추론
uv run library/script/query_rl_dpo.py --mode full "방정식 x^2 + 5x + 6 = 0 의 해를 구하시오."
```

**기대 출력**: SFT'd 보다 더 일관된  ###  구조 응답.

---

## 7. 세 모델 비교 (5분)

| 모델 | 스크립트 | 경로 |
|------|----------|------|
| BASE | `query_base.py` | `Qwen/Qwen2.5-1.5B-Instruct` |
| SFTED | `query_sft.py` | `./data/sft-full-out/merged` |
| RL | `query_rl_dpo.py` | `./data/dpo-full-out/merged` |

```bash
# 3-way 동일 질문 비교
Q="피타고라스 정리를 증명하시오."
uv run library/script/query_base.py  "$Q"
uv run library/script/query_sft.py --mode full "$Q"
uv run library/script/query_rl_dpo.py --mode full    "$Q"
```

REPL 모드:
```bash
uv run library/script/query_base.py    # 동일 질문을 3 모델에 반복
uv run library/script/query_sft.py --mode full
uv run library/script/query_rl_dpo.py --mode full
```

---

## 8. lm-eval 정량 평가 (선택, 3분)

정성 비교와 별개로 DPO 모델의 표준 benchmark 점수를 측정.
각 task 당 100 samples 제한 (`--limit 100`) 으로 sanity check 용도.

```bash
phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-09-lm-eval-rl-dpo.sh full      # mini 면 phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-09-lm-eval-rl-dpo.sh mini
```

평가 tasks:
- 한국어: `kobest_hellaswag`, `kobest_copa`, `kmmlu`
- 영어: `hellaswag`, `arc_easy`, `piqa`, `winogrande`

결과는 `outputs/lm_eval_results/rl-dpo-full/` (mini 는 `rl-dpo-mini/`) 에 저장됨.
비교용 BASE/SFT 점수는 [`phase/sft/algorithm/lora/tool/axolotl/README.md`](phase/sft/algorithm/lora/tool/axolotl/README.md) Stage 8 의 `sft-10`, `sft-11` 결과와 함께 봐야 함.

> 참고: chat template 적용 모델의 경우 `kmmlu` 가 raw 점수보다 낮게
> 나오는 경향이 있음 (선지 형식 차이). 정성 비교와 함께 봐야 함.

---

## 9. 차이 요약

| 모델 |  블록 |  단계 구조 | 한국어 일관성 | 응답 일관성 |
|------|-------|-----------|-------------|-----------|
| BASE | 약함 | 약함 | 보통 | 보통 |
| SFTED | 강함 | 강함 | 자연스러움 | 보통 |
| RL    | 강함 | 강함 | 자연스러움 | **더 일관** |

DPO 의 효과:
- SFT'd 모델이 가끔 짧게 끊기거나  ###  헤더 빠뜨린 응답 → DPO 가 페널티
- 베이스식 평평한 응답 → DPO 가 페널티
- 구조적이고 일관된 응답 → DPO 가 보상

---

## 10. 한 줄 요약

```bash
# SFT 가 끝났다면
uv run python library/script/make_rl_dpo_data.py && \
  uv run axolotl train phase/pref/algorithm/dpo/tool/axolotl/config/qwen2.5-1.5b-rl-dpo.yaml && \
  phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-06-merge.sh full && \
  uv run library/script/query_rl_dpo.py --mode full
```

---

## 트러블슈팅

| 증상 | 해결 |
|------|------|
| `ref_model` 로딩 실패 | `HF_HUB_OFFLINE=1` 유지 + 로컬 캐시 확인 |
| DPO loss 가 0.69 그대로 | learning_rate 너무 낮음 → `1e-5` 로 |
| `reward_margin` 음수 | 데이터 품질 문제 → `make_rl_dpo_data.py` 의 `--num-prompts` 늘리기 |
| `data/dpo-full-out/merged` 없음 | train 로그 확인. **Stage 5 `axolotl merge-lora` 명령 실행 필수** (train은 LoRA 어댑터만 저장, merge 별도 단계). 또한 `lora_model_dir` 는 merge 출력 경로가 아니라 어댑터 **입력** 경로임 |
| 두 모델 출력이 똑같음 | DPO lr/epoch 부족 |

---

## 부록: GRPO (rule-based reward) 으로 업그레이드

수학 정답이 명확한 문제라면 rule-based 보상 함수로 GRPO 가능:

```python
# library/function/reward_fn.py
def math_reward(prompt, response, ground_truth):
    import re
    m = re.search(r"\\boxed\{([^}]+)\}", response)
    if not m: return 0.0
    pred = m.group(1).strip()
    return 1.0 if pred == ground_truth else 0.0
```

```yaml
# phase/rl/algorithm/grpo/tool/axolotl/config/qwen2.5-1.5b-rl-grpo.yaml (대략)
rl: grpo
reward_fn: path/to/reward_fn.py
```

> 본 데모 데이터(`amphora/korean-reasoning-small`)는 정답을 로 추출하기 어려운
> 일반 추론 데이터라 DPO 가 더 적합합니다. 수학 전용 데이터셋이라면 GRPO 추천.

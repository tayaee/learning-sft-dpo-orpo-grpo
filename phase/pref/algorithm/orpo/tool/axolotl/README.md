# RL Demo — ORPO on top of SFT'd Qwen2.5-1.5B

> **목적**: SFT'd 모델(`data/sft-full-out/merged`)을 시작점으로
> **ORPO** 를 학습해 응답 일관성·구조를 강화. DPO 와 같은 선호 데이터로
> reference model 없이 한 번에 학습하는 알고리즘.
> 사전 조건: [`phase/sft/algorithm/lora/tool/axolotl/README.md`](phase/sft/algorithm/lora/tool/axolotl/README.md) Stage 4 완료 (SFT merge 폴더 존재).
> DPO 데이터(`data/{stage}-{mode}-out/train_rl-dpo.jsonl`)가 이미 있으면 데이터 재생성 없이 바로 시작 가능.

---

## 0. ORPO vs DPO — 왜 ORPO 인가

| | DPO | **ORPO** |
|---|---|---|
| 데이터 | (prompt, chosen, rejected) | 동일 |
| reference model | 필요 (메모리 2×) | **불필요** |
| 손실 | KL 제약 + 선호 loss | SFT(NLL) + odds-ratio loss |
| 학습 LR | 매우 낮음 (5e-6) | 비교적 높음 (1e-5) |
| 핵심 하이퍼파라미터 | `dpo_beta` | `orpo_alpha` (선호 loss 가중치) |

ORPO 는 NLL(SFT 성격)과 선호 손실을 동시에 최적화하므로,
"SFT 를 더 강하게 이어가면서 선호 방향으로 밀어붙이는" 학습이 된다.

**Smoke test 와 Real run 두 가지 모드**를 제공한다:

| 모드 | 데이터 | steps | 소요 | 목적 |
|------|--------|-------|------|------|
| **mini** | 20 pairs | max_steps 3 | ~5–10분 | 파이프라인 점검 (데이터/학습/merge/추론 전 경로) |
| **full** | 1000 pairs × 2 epoch | ~125 steps | ~1–2시간 | 실질적 지능 향상 측정 |

---

## A. Smoke Test — 파이프라인 점검 (~10분)

### A-1. mini 데이터 생성

DPO 데이터가 있으면 재사용(권장), 없으면 먼저 DPO 데이터를 만든다.

```bash
cd /home/user1/git/learning-sft-and-rl

# DPO 데이터가 없다면: phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-01-make-rl-dpo-data.sh mini
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-01-make-orpo-data.sh mini    # → data/orpo-mini-out/train_rl-orpo.jsonl
```

### A-2. 데이터 sanity check

```bash
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-02-sanity-check-orpo-data.sh mini
```

**기대**: chosen 은 `` 블록 + `###` 구조 + 결론 단정, rejected 는 평평하거나 짧은 응답.
포맷은 passthrough 로 전달되는 메시지 리스트(TRL conversational) 포맷
(`chosen`/`rejected` 가 `[user, assistant]` 배열). mini 파일은 20줄.

### A-3. mini 학습 (3 step)

```bash
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-04-run-axolotl.sh mini        # phase/pref/algorithm/orpo/tool/axolotl/config/qwen2.5-1.5b-rl-orpo-mini.yaml
```

**체크 포인트**:
- `orpo_loss`, `rewards/chosen`, `rewards/rejected`,
  `rewards/logps/rejected` 등의 로그가 step 마다 찍히는지
- `log_odds_ratio` 가 음수에서 0 방향으로 움직이는지 (완벽히 양수까지 안 가도 됨)
- 완료 후 `data/orpo-mini-out/adapter/` 에 어댑터 저장 확인

### A-4. mini merge + 추론

```bash
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-05-merge.sh mini              # data/orpo-mini-out/merged/
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-06-test-model.sh mini         # 간단한 추론 한 번
```

여기까지 에러 없이 돌면 파이프라인 전 경로(데이터→학습→merge→추론)가 정상이다.

---

## B. Real Run — 실질적 지능 향상 (~1–2시간)

### B-1. full 데이터 (1000 pairs)

```bash
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-01-make-orpo-data.sh full      # data/orpo-full-out/train_rl-orpo.jsonl = 1000 lines
wc -l data/orpo-full-out/train_rl-orpo.jsonl      # 1000 확인
```

> 직접 rejection sampling 으로 새로 만들고 싶으면:
> `uv run python library/script/make_rl_orpo_data.py --generate --num-prompts 1000 \
>   --out data/orpo-full-out/train_rl-orpo.jsonl`
> (SFT merge 모델 필요, 수십 분 소요)

### B-2. full 학습

```bash
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-04-run-axolotl.sh full         # logs/rl-orpo.log 에 tee 됨
```

**핵심 파라미터** (`phase/pref/algorithm/orpo/tool/axolotl/config/qwen2.5-1.5b-rl-orpo.yaml`):
- `orpo_alpha: 1.0` — odds-ratio loss 가중치 (axolotl 이 trl 의 `beta` 로 전달)
- `learning_rate: 1e-5` — DPO(5e-6)보다 높음. ORPO 는 ref-model KL 제약이 없어
  drift 위험이 적지만, 너무 올리면 SFT 스타일이 붕괴되므로 2× 이상 올리지 말 것
- `num_epochs: 2`, effective batch 16 → ~125 steps

**체크 포인트**:
- `rewards/chosen` 상승, `rewards/rejected` 하강 → 분리가 벌어지면 정상
- `log_odds_ratio` 점진 상승
- chosen/rejected reward 차이(rewards margin)가 계속 0 이면 lr 문제 → `2e-5` 로

### B-3. merge + 추론

```bash
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-05-merge.sh full
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-06-test-model.sh full
uv run library/script/query_rl_orpo.py --mode full "피타고라스 정리를 증명하시오."
```

### B-4. SFT vs ORPO side-by-side (정성)

```bash
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-07-compare-models.sh full
```

### B-5. 정량 평가 ① 휴리스틱 reward (5–10분)

```bash
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-08-reward-eval.sh full         # 50 prompts, 결과 jsonl 저장됨
```

같은 프롬프트에 대해 SFT 모델 vs ORPO 모델의 greedy 응답을
선호 데이터 채점 기준(`make_rl_dpo_data.score`)으로 채점해 평균을 낸다.

**기대 출력 예**:
```
SFT  평균 reward : +2.10
ORPO 평균 reward : +2.65
개선폭           : +0.55
win/tie/loss     : 31/12/7
```

win rate 이 절반을 넘고 평균이 오르면, 선호 방향으로 실제 개선이 된 것이다.
(mini 로도 `--mode mini --num-prompts 10` 실행 가능하지만 3-step 학습이라 개선 폭은 미미하다.)

### B-6. 정량 평가 ② lm-eval (선택, ~3분)

```bash
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-09-lm-eval-rl-orpo.sh full
```

tasks: `kobest_hellaswag, kobest_copa, kmmlu, hellaswag, arc_easy, piqa, winogrande`
결과는 `outputs/lm_eval_results/rl-orpo-full/` (mini 는 `rl-orpo-mini/`). sft-demo 의 base/SFT 점수와 나란히 비교.
> 주의: 선호 정렬 학습은 benchmark 점수를 낮출 수도 있다(스타일 변화). 그래서
> B-5 의 reward 기반 평가가 이 데모의 1차 지표다.

---

## C. 파일 구성

| 파일 | 역할 |
|------|------|
| `make_rl_orpo_data.py` | ORPO 데이터 생성 (DPO 재사용 or 직접 rejection sampling) |
| `phase/pref/algorithm/orpo/tool/axolotl/config/qwen2.5-1.5b-rl-orpo.yaml` | full run config |
| `phase/pref/algorithm/orpo/tool/axolotl/config/qwen2.5-1.5b-rl-orpo-mini.yaml` | mini test config |
| `orpo_passthrough_plugin.py` | axolotl 0.18 에 빠진 orpo/passthrough 전략 등록 플러그인 |
| `rl-orpo-01..09-*.sh` | 단계별 드라이버 — 전부 `$1=mini\|full` 필수. 데이터는 모드별 디렉터리(`data/{stage}-mini-out/`, `data/{stage}-full-out/`)로 분리. 각 스크립트마다 `<script>-mini.sh`/`<script>-full.sh` 래퍼 있음 |
| `query_rl_orpo.py` | ORPO merge 모델 추론 (--mode mini/full) |
| `eval_orpo_reward.py` | SFT vs ORPO 휴리스틱 reward 정량 비교 |

## D. 한 줄 요약

```bash
# 점검 (10분)
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-01-make-orpo-data.sh mini && phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-04-run-axolotl.sh mini && \
  phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-05-merge.sh mini && phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-06-test-model.sh mini

# 실전 (1–2시간)
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-01-make-orpo-data.sh full && phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-04-run-axolotl.sh full && \
  phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-05-merge.sh full && phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-08-reward-eval.sh full
```

---

## E. axolotl 0.18 ORPO 관련 검증 노트 (mini test 로 확인)

1. **데이터 포맷**: `chosen`/`rejected` 는 `[user, assistant]` 메시지 리스트.
   TRL ORPOTrainer 가 conversational 포맷으로 인식해 chat template 을 적용한다.
2. **`type: passthrough` + 플러그인**: axolotl 의 RL 데이터 경로는 전략을
   `axolotl.prompt_strategies.orpo.<module>` 에서 찾는데, ORPO 패키지에는
   passthrough 가 없다. → `orpo_passthrough_plugin.py` 가 이를 등록한다.
3. **`skip_prepare_dataset: true` 필수**: axolotl 사전처리의 길이 필터는
   chosen/rejected 를 평문(str)으로 가정해 conversational 포맷에서 깨진다.
   생략하면 TRL 이 직접 토크나이즈하므로 문제 없음.
4. **`path` 에 단일 파일 지정**: `train_files` 라는 키는 axolotl 에 없다(무시됨).
   `path: data` 처럼 디렉터리를 주면 폴더의 모든 jsonl 이 로드되어 스키마 충돌.
5. 학습 로그 정상 신호: `rewards/margins`, `log_odds_ratio`, `log_odds_chosen`,
   `nll_loss` 가 스텝마다 기록되면 정상.

## 트러블슈팅

| 증상 | 해결 |
|------|------|
| `rl: orpo` 미지원 에러 | axolotl ≥ 0.18 확인 (`uv run axolotl --version`) |
| OOM (DGX Spark 통합메모리) | mini 처럼 `micro_batch_size: 1` 로, `sequence_len: 768` 로 축소 |
| rewards margin 이 안 벌어짐 | `learning_rate` 를 `2e-5` 로, 또는 `orpo_alpha` 를 `2.0` 까지 |
| ORPO 결과가 오히려 나빠짐 | `orpo_alpha` 과대 → 0.5 로 낮추거나 epoch 1 로 |
| chosen == rejected 데이터 다수 | DPO 데이터 재생성 (`--generate` 모드) |
| merge 후 추론에서 모델 없음 에러 | `--mode` 플래그가 mini/merge 모드와 일치하는지 확인 |

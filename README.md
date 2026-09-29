# Learning SFT & RL with Axolotl

A hands-on study repo for running **SFT, DPO, GRPO, and ORPO** on Qwen2.5-1.5B-Instruct
with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), targeting Korean
chain-of-thought reasoning data. Each stage builds on the previous one:

```
Base (Qwen2.5-1.5B-Instruct)
   │
   ▼  SFT  ─────────────────────► [phase/sft/algorithm/lora/tool/axolotl/README.md]
   │
   ├─▼  DPO  (from SFT)  ──────► [phase/pref/algorithm/dpo/tool/axolotl/README.md]
   │
   ├─▼  GRPO (from SFT)  ──────► [phase/rl/algorithm/grpo/tool/axolotl/README.md]
   │
   └─▼  ORPO (from SFT)  ──────► [phase/pref/algorithm/orpo/tool/axolotl/README.md]
```

## 1. SFT — Supervised Fine-Tuning
How to run a LoRA SFT on top of a base Instruct model with Axolotl, then merge
the adapter into a single deployable checkpoint.
→ See **[`phase/sft/algorithm/lora/tool/axolotl/README.md`](phase/sft/algorithm/lora/tool/axolotl/README.md)** — base sanity check, data sanity,
config review, train, LoRA merge, side-by-side inference, lm-eval.

## 2. DPO — Direct Preference Optimization
How to run a DPO pass on top of the SFT'd model using `(prompt, chosen, rejected)`
triples, then merge and compare.
→ See **[`phase/pref/algorithm/dpo/tool/axolotl/README.md`](phase/pref/algorithm/dpo/tool/axolotl/README.md)** — DPO data prep, Axolotl DPO config,
train, merge, 4-model comparison (Base vs SFT vs DPO), lm-eval.

## 3. GRPO — Group Relative Policy Optimization
How to run GRPO on top of the SFT'd model using only `(prompt)` + rule-based
reward functions (no preference labels), then merge and compare.
→ See **[`phase/rl/algorithm/grpo/tool/axolotl/README.md`](phase/rl/algorithm/grpo/tool/axolotl/README.md)** — reward function design,
Axolotl GRPO config, train, merge, 4-model comparison, lm-eval.

## 4. ORPO — Odds-Ratio Preference Optimization
How to run ORPO on top of the SFT'd model using the same `(prompt, chosen, rejected)`
pairs as DPO — but **without a reference model**. Provides two modes:
→ See **[`phase/pref/algorithm/orpo/tool/axolotl/README.md`](phase/pref/algorithm/orpo/tool/axolotl/README.md)**.
- **mini test**: 20 pairs × `max_steps: 3` (~10분) — 파이프라인 점검용
- **full run**: 1000 pairs × 2 epochs (~1–2시간) — 휴리스틱 reward 기반 정량 평가 포함

## Stage diagrams

각 스테이지는 **mini**(파이프라인 점검)와 **full**(실전 학습) 두 흐름으로 실행한다.
흐름마다 데이터·로그·출력 모델 디렉터리가 분리되어 있으므로 어느 순서로 실행해도
서로 덮어쓰지 않는다. 아래 다이어그램의 각 박스에는 프로세스별 in/out 을 표시했고,
바로 아래 코드 블록을 위에서부터 copy & paste 하면 그대로 따라간다.
(검토 전용 스크립트 — `*review*.sh`, `*sanity-check*.sh` — 는 산출물이 없어 생략)

> 공통 선행 조건: 모든 RL 스테이지(DPO/GRPO/ORPO)는 **같은 모드의 SFT merge 모델**이 필요하다.
> mini 흐름은 `data/sft-mini-out/`, full 흐름은 `data/sft-full-out/` 의 산출물을 사용한다.

---

### SFT

#### mini flow (~10분)

```
[in]  Qwen/Qwen2.5-1.5B-Instruct (base)
[in]  train.jsonl ──(앞 200건 자동 추출)──▶ data/sft-mini-out/train-sft.jsonl
   │
   ├─ phase/sft/algorithm/lora/tool/axolotl/sft-mini-06-run-axolotl-mini.sh    Axolotl LoRA train
   │    in : phase/sft/algorithm/lora/tool/axolotl/config/qwen2.5-1.5b-sft-mini.yaml + data/sft-mini-in/train-sft.jsonl
   │    out: data/sft-mini-out/adapter/                       (LoRA adapter)
   │
   ├─ phase/sft/algorithm/lora/tool/axolotl/sft-mini-07-merge-mini.sh           axolotl merge-lora
   │    in : data/sft-mini-out/adapter/
   │    out: data/sft-mini-out/merged/                         (HF model)
   │
   ├─ uv run library/script/query_sft.py --mode mini       추론 테스트
   │    in : data/sft-mini-out/merged/
   │
   └─ phase/sft/algorithm/lora/tool/axolotl/sft-mini-11-lm-eval-sft-mini.sh     lm-eval (선택)
        in : data/sft-mini-out/merged/
        out: outputs/lm_eval_results/sft-mini/
```

```bash
phase/sft/algorithm/lora/tool/axolotl/sft-mini-06-run-axolotl-mini.sh
phase/sft/algorithm/lora/tool/axolotl/sft-mini-07-merge-mini.sh
uv run library/script/query_sft.py --mode mini "방정식 x^2 + 5x + 6 = 0 의 해를 구하시오."
# (선택) phase/sft/algorithm/lora/tool/axolotl/sft-mini-11-lm-eval-sft-mini.sh
```

#### full flow (~수 시간)

```
[in]  Qwen/Qwen2.5-1.5B-Instruct (base)
[in]  train.jsonl (50k Korean CoT, chat format)
   │
   ├─ phase/sft/algorithm/lora/tool/axolotl/sft-06-run-axolotl.sh               Axolotl LoRA train
   │    in : phase/sft/algorithm/lora/tool/axolotl/config/qwen2.5-1.5b-sft.yaml + train.jsonl
   │    out: data/sft-full-out/adapter/                       (LoRA adapter)
   │
   ├─ phase/sft/algorithm/lora/tool/axolotl/sft-07-merge.sh                      axolotl merge-lora
   │    in : data/sft-full-out/adapter/
   │    out: data/sft-full-out/merged/                        (HF model)
   │
   ├─ uv run library/script/query_sft.py --mode full        추론 테스트
   │    in : data/sft-full-out/merged/
   │
   ├─ phase/sft/algorithm/lora/tool/axolotl/sft-13-upload-to-hf.sh               HF Hub 업로드 (선택)
   │    in : data/sft-full-out/merged/
   │    out: tayaee/Qwen2.5-1.5B-Korean-SFT
   │
   └─ phase/sft/algorithm/lora/tool/axolotl/sft-11-lm-eval-sft.sh                lm-eval (선택)
        in : data/sft-full-out/merged/
        out: outputs/lm_eval_results/sft-full/
```

```bash
phase/sft/algorithm/lora/tool/axolotl/sft-06-run-axolotl.sh
phase/sft/algorithm/lora/tool/axolotl/sft-07-merge.sh
uv run library/script/query_sft.py --mode full "방정식 x^2 + 5x + 6 = 0 의 해를 구하시오."
# (선택) phase/sft/algorithm/lora/tool/axolotl/sft-13-upload-to-hf.sh
# (선택) phase/sft/algorithm/lora/tool/axolotl/sft-11-lm-eval-sft.sh
```

---

### DPO (from SFT)

#### mini flow (~30분, SFT-mini merge 필요)

```
[in]  SFT-mini merge 모델  data/sft-mini-out/merged/
[in]  train.jsonl (프롬프트 소스) + Qwen/Qwen2.5-1.5B-Instruct (rejected 후보)
   │
   ├─ phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-01-make-rl-dpo-data-mini.sh   rejection sampling 으로 선호 쌍 생성
   │    in : SFT-mini merge 모델 + base 모델
   │    out: data/dpo-mini-out/train_rl-dpo.jsonl  (50 pairs)
   │         data/dpo-mini-out/sample_rl-dpo.jsonl (디버깅용 샘플)
   │
   ├─ phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-04-run-axolotl-mini.sh        Axolotl DPO train
   │    in : phase/pref/algorithm/dpo/tool/axolotl/config/qwen2.5-1.5b-rl-dpo-mini.yaml + data/dpo-mini-in/train_rl-dpo.jsonl
   │    out: data/dpo-mini-out/adapter/                          (LoRA adapter)
   │
   ├─ phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-06-merge-mini.sh              axolotl merge-lora
   │    in : data/dpo-mini-out/adapter/
   │    out: data/dpo-mini-out/merged/
   │
   ├─ phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-05-test-models-mini.sh        단일 추론 테스트
   │    in : data/dpo-mini-out/merged/
   │
   └─ phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-09-lm-eval-rl-dpo-mini.sh     lm-eval (선택)
        in : data/dpo-mini-out/merged/
        out: outputs/lm_eval_results/rl-dpo-mini/
```

```bash
phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-01-make-rl-dpo-data-mini.sh
phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-04-run-axolotl-mini.sh
phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-06-merge-mini.sh
phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-05-test-models-mini.sh "방정식 x^2 + 5x + 6 = 0 의 해를 구하시오."
# (선택) phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-09-lm-eval-rl-dpo-mini.sh
```

#### full flow (~1–3시간, SFT-full merge 필요)

```
[in]  SFT-full merge 모델  data/sft-full-out/merged/
[in]  train.jsonl (프롬프트 소스) + Qwen/Qwen2.5-1.5B-Instruct (rejected 후보)
   │
   ├─ phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-01-make-rl-dpo-data.sh
   │    in : SFT-full merge 모델 + base 모델
   │    out: data/dpo-full-out/train_rl-dpo.jsonl  (1000 pairs)
   │         data/dpo-full-out/sample_rl-dpo.jsonl
   │
   ├─ phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-04-run-axolotl.sh
   │    in : phase/pref/algorithm/dpo/tool/axolotl/config/qwen2.5-1.5b-rl-dpo.yaml + data/dpo-full-in/train_rl-dpo.jsonl
   │    out: data/dpo-full-out/adapter/                            (LoRA adapter)
   │
   ├─ phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-06-merge.sh
   │    in : data/dpo-full-out/adapter/
   │    out: data/dpo-full-out/merged/
   │
   ├─ phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-05-test-models.sh            단일 추론 테스트
   │    in : data/dpo-full-out/merged/
   │
   └─ phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-09-lm-eval-rl-dpo.sh         lm-eval (선택)
        in : data/dpo-full-out/merged/
        out: outputs/lm_eval_results/rl-dpo-full/
```

```bash
phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-01-make-rl-dpo-data.sh
phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-04-run-axolotl.sh
phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-06-merge.sh
phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-05-test-models.sh "피타고라스 정리를 증명하시오."
# (선택) phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-09-lm-eval-rl-dpo.sh
```

---

### GRPO (from SFT)

> 사전 준비: `data/grpo-mini-out/sample_rl-grpo.jsonl` (20 prompts),
> `data/grpo-full-out/sample_rl-grpo.jsonl` (확장 버전) — prompt-only JSONL 을 직접 준비한다.

#### mini flow (~20분, SFT-mini merge 필요)

```
[in]  SFT-mini merge 모델  data/sft-mini-out/merged/
[in]  data/grpo-mini-out/sample_rl-grpo.jsonl (prompt only) + library/function/reward_fn.py
   │
   ├─ phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-mini-04-run-axolotl-mini.sh       Axolotl GRPO train
   │    in : phase/rl/algorithm/grpo/tool/axolotl/config/qwen2.5-1.5b-rl-grpo-mini.yaml
   │          + data/grpo-mini-in/sample_rl-grpo.jsonl + library/function/reward_fn.py
   │    out: data/grpo-mini-out/adapter/                         (LoRA adapter)
   │
   ├─ phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-mini-06-merge-mini.sh             axolotl merge-lora
   │    in : data/grpo-mini-out/adapter/
   │    out: data/grpo-mini-out/merged/
   │
   ├─ phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-mini-05-test-model-mini.sh        단일 추론 테스트
   │    in : data/grpo-mini-out/merged/
   │
   └─ phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-mini-09-lm-eval-rl-grpo-mini.sh   lm-eval (선택)
        in : data/grpo-mini-out/merged/
        out: outputs/lm_eval_results/rl-grpo-mini/
```

```bash
phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-mini-04-run-axolotl-mini.sh
phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-mini-06-merge-mini.sh
phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-mini-05-test-model-mini.sh "방정식 x^2 + 5x + 6 = 0 의 해를 구하시오."
# (선택) phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-mini-09-lm-eval-rl-grpo-mini.sh
```

#### full flow (~1–2시간+, SFT-full merge 필요)

```
[in]  SFT-full merge 모델  data/sft-full-out/merged/
[in]  data/grpo-full-out/sample_rl-grpo.jsonl (prompt only) + library/function/reward_fn.py
   │
   ├─ phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-04-run-axolotl.sh
   │    in : phase/rl/algorithm/grpo/tool/axolotl/config/qwen2.5-1.5b-rl-grpo.yaml
   │          + data/grpo-full-in/sample_rl-grpo.jsonl + library/function/reward_fn.py
   │    out: data/grpo-full-out/adapter/                         (LoRA adapter)
   │
   ├─ phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-06-merge.sh
   │    in : data/grpo-full-out/adapter/
   │    out: data/grpo-full-out/merged/
   │
   ├─ phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-05-test-model.sh            단일 추론 테스트
   │    in : data/grpo-full-out/merged/
   │
   └─ phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-09-lm-eval-rl-grpo.sh       lm-eval (선택)
        in : data/grpo-full-out/merged/
        out: outputs/lm_eval_results/rl-grpo-full/
```

```bash
phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-04-run-axolotl.sh
phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-06-merge.sh
phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-05-test-model.sh "방정식 x^2 + 5x + 6 = 0 의 해를 구하시오."
# (선택) phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-09-lm-eval-rl-grpo.sh
```

---

### ORPO (from SFT, reference model 불필요)

#### mini flow (~15분, DPO-mini 데이터 필요)

```
[in]  data/dpo-mini-out/train_rl-dpo.jsonl (DPO-mini 선호 쌍 재사용)
   │
   ├─ phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-01-make-orpo-data-mini.sh    DPO → ORPO 메시지 리스트 포맷 변환
   │    in : data/dpo-mini-out/train_rl-dpo.jsonl
   │    out: data/orpo-mini-out/train_rl-orpo.jsonl (20 pairs)
   │
   ├─ phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-04-run-axolotl-mini.sh       Axolotl ORPO train
   │    in : phase/pref/algorithm/orpo/tool/axolotl/config/qwen2.5-1.5b-rl-orpo-mini.yaml + data/orpo-mini-in/train_rl-orpo.jsonl
   │    out: data/orpo-mini-out/adapter/                            (LoRA adapter)
   │
   ├─ phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-05-merge-mini.sh             axolotl merge-lora
   │    in : data/orpo-mini-out/adapter/
   │    out: data/orpo-mini-out/merged/
   │
   ├─ phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-06-test-model-mini.sh        단일 추론 테스트
   │    in : data/orpo-mini-out/merged/
   │
   └─ phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-08-reward-eval-mini.sh       SFT vs ORPO reward 정량 비교 (선택)
        in : SFT-mini merge + ORPO-mini merge 모델
        out: outputs/eval_results/orpo-reward-mini.jsonl
```

```bash
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-01-make-orpo-data-mini.sh
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-04-run-axolotl-mini.sh
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-05-merge-mini.sh
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-06-test-model-mini.sh
# (선택) phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-08-reward-eval-mini.sh
```

#### full flow (~1–2시간, DPO-full 데이터 필요)

```
[in]  data/dpo-full-out/train_rl-dpo.jsonl (DPO-full 선호 쌍 재사용)
   │
   ├─ phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-01-make-orpo-data.sh
   │    in : data/dpo-full-out/train_rl-dpo.jsonl
   │    out: data/orpo-full-out/train_rl-orpo.jsonl (1000 pairs)
   │
   ├─ phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-04-run-axolotl.sh
   │    in : phase/pref/algorithm/orpo/tool/axolotl/config/qwen2.5-1.5b-rl-orpo.yaml + data/orpo-full-in/train_rl-orpo.jsonl
   │    out: data/orpo-full-out/adapter/                            (LoRA adapter)
   │
   ├─ phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-05-merge.sh
   │    in : data/orpo-full-out/adapter/
   │    out: data/orpo-full-out/merged/
   │
   ├─ phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-06-test-model.sh            단일 추론 테스트
   │    in : data/orpo-full-out/merged/
   │
   ├─ phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-08-reward-eval.sh           SFT vs ORPO reward 정량 비교
   │    in : SFT-full merge + ORPO-full merge 모델
   │    out: outputs/eval_results/orpo-reward-full.jsonl
   │
   └─ phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-09-lm-eval-rl-orpo.sh       lm-eval (선택)
        in : data/orpo-full-out/merged/
        out: outputs/lm_eval_results/rl-orpo-full/
```

```bash
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-01-make-orpo-data.sh
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-04-run-axolotl.sh
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-05-merge.sh
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-06-test-model.sh
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-08-reward-eval.sh
# (선택) phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-09-lm-eval-rl-orpo.sh
```


## Repository layout

```
phase/<phase>/algorithm/<algo>/tool/<tool>/   실행 파이프라인 (분류자=내용물 규칙)
├── sft/lora/axolotl/          SFT LoRA 파이프라인 (config/*.yaml 포함, 구 data/*-config 승격)
├── pref/dpo/axolotl/          DPO 파이프라인
├── pref/orpo/axolotl/         ORPO 파이프라인
├── rl/grpo/axolotl/           GRPO 파이프라인
└── eval/lm-eval/python/       공용 평가 안내 (stage별 평가 스크립트는 각 stage에 상주)
source/<source>/example/<example>/            남의 예제 원본 보관 (손대지 않음)
├── unsloth/sft-basic/         Unsloth SFT skill 예제 + wrapper 3종 (ORIGIN.md 참조)
└── axolotl/<collection>/      Axolotl 벤더별 참조 yaml (구 examples/)
library/<kind>/<name>/         공용 자산
├── script/                    query_*.py, make_rl_*.py, eval_*.py, scripts_common.sh
├── function/                  reward_fn.py, orpo_passthrough_plugin.py
└── config/lm_eval/            lm-eval 용 YAML (구 configs/lm_eval)
infra/<infra>/                실행 환경 오버레이 (현 기본값: dgx-spark-1x)
data/
├── train.jsonl.gz                           SFT 학습용 원천 (root train.jsonl로 해제, 변경 없음)
├── {sft,dpo,orpo,grpo}-{full,mini}-{in,out}/   stage × mode 별 산출물 (gitignore)
│     in/      : 입력 데이터 (이전 stage 의 out 또는 외부 소스에 대한 symbolic link)
│     out/     : 이 stage 가 생성한 모든 산출물 (jsonl + adapter/ + merged/ + dataset_prepared/)
└── ...
outputs/lm_eval_results/        평가 산출물 (학습 산출물은 data/{stage}-{mode}-out/ 로 이동)

run-mini-all.sh               전 stage mini 드라이버 (root 상주, cross-stage라 예외)
phase/.../README.md           각 stage Step-by-step study notes (구 *-demo.md)
```

## Quick start

모든 학습/데이터 스크립트는 첫 인자로 `mini | full` 을 반드시 받으며,
데이터(`data/{stage}-{mode}-out/`)·로그·출력 모델이 모드별로 분리되어 있어
어떤 순서로 실행해도 서로 덮어쓰지 않는다.

mini 흐름용 래퍼도 있다 (예: `phase/sft/algorithm/lora/tool/axolotl/sft-mini-06-run-axolotl-mini.sh` ≡ `phase/sft/algorithm/lora/tool/axolotl/sft-06-run-axolotl.sh mini`).

```bash
# 1) SFT
phase/sft/algorithm/lora/tool/axolotl/sft-06-run-axolotl.sh full
phase/sft/algorithm/lora/tool/axolotl/sft-07-merge.sh full
uv run library/script/query_sft.py --mode full "방정식 x^2 + 5x + 6 = 0 의 해를 구하시오."

# 2) DPO (requires SFT merge output of the same mode)
phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-01-make-rl-dpo-data.sh full      # → data/dpo-full-out/train_rl-dpo.jsonl
phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-04-run-axolotl.sh full
phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-06-merge.sh full
uv run library/script/query_rl_dpo.py --mode full "피타고라스 정리를 증명하시오."

# 3) GRPO (requires SFT merge output)
phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-04-run-axolotl.sh full          # data/grpo-full-in/sample_rl-grpo.jsonl 사용
phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-06-merge.sh full
uv run library/script/query_rl_grpo.py --mode full "방정식 x^2 + 5x + 6 = 0 의 해를 구하시오."

# 4) ORPO — mini 으로 점검 후 full 실행 (requires DPO data of the same mode)
phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-01-make-rl-dpo-data-mini.sh     # → data/dpo-mini-out/train_rl-dpo.jsonl
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-01-make-orpo-data-mini.sh      # → data/orpo-mini-out/train_rl-orpo.jsonl
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-04-run-axolotl-mini.sh && \
  phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-05-merge-mini.sh && phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-06-test-model-mini.sh
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-01-make-orpo-data.sh full       # → data/orpo-full-out/train_rl-orpo.jsonl
phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-04-run-axolotl.sh full && phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-05-merge.sh full && \
  phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-08-reward-eval.sh full

# 5) 4-model side-by-side (같은 모드의 merge 모델끼리 비교)
phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-08-pythagorean-theorem.sh full
```

Each `phase/.../README.md` contains the full walkthrough with troubleshooting, expected
checkpoints, and parameter rationale. Read the README file for the stage you are
on before running its scripts.

### 모델 비교 평가 (eval-compare)

새 모델을 만든 뒤 **이전 모델 vs 새 모델** 에 대해 2개의 빠른 평가를 돌리고
비교표를 생성한다. 전부 loglikelihood 방식이라 수 분 내 완료된다.

- 평가 ① General 지능: `tinyArc`, `tinyHellaswag`, `tinyMMLU`, `tinyWinogrande` (TinyBenchmarks 100문제씩)
- 평가 ② Korean: `kobest_copa`, `kobest_hellaswag`
- 비교 대상: SFT=Base, DPO/GRPO/ORPO=같은 모드의 SFT merge 모델

```bash
phase/sft/algorithm/lora/tool/axolotl/sft-mini-12-eval-compare-mini.sh        # Base vs SFT-mini
phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-10-eval-compare.sh full      # SFT(full) vs DPO(full)
# → outputs/lm_eval_results/<stage>-<mode>/comparison-table.md 에 표 저장
```
> 참고: 선호 정렬(DPO/ORPO)은 벤치마크 점수를 낮출 수 있다(스타일 변화).
> 같은 조건(`--apply_chat_template`)으로 이전/새 모델을 함께 돌리므로 Δ 값을 보자.

## Requirements

- NVIDIA GPU (Blackwell / sm_120 tested with `uv` + `UV_TORCH_BACKEND=cu130`)
- `uv` package manager
- Python 3.12, CUDA 13.0 toolchain
- ~50 GB disk for base model + training outputs
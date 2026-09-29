# Unsloth SFT basic example (참조용)

Axolotl 파이프라인(`phase/sft/algorithm/lora/tool/axolotl/`)과 같은
LoRA SFT를 Unsloth 백엔드로 수행하는 skill 예제다. 학습용 참조본이며,
Qwen2.5-1.5B 한국어 CoT 파이프라인과는 데이터·모델이 다르다.

## 실행

```bash
cd source/unsloth/example/sft-basic
OUTPUT_REPO=your-username/model-finetuned ./run-epochs.sh   # full (1 epoch + eval)
OUTPUT_REPO=your-username/model-test ./run-steps.sh         # quick test (500 steps)
OUTPUT_REPO=your-username/model-finetuned ./run-hf-job.sh   # HF Jobs (a10g-small)
```

모든 값은 env로 override 가능 (`BASE_MODEL`, `DATASET`, `NUM_EPOCHS` /
`MAX_STEPS`, `EVAL_SPLIT`, `FLAVOR`, `TIMEOUT`). 추가 인자는 `--` 뒤가
아니라 wrapper 뒤에 그대로 붙이면 예제 스크립트로 전달된다 (`"$@"`).

## 요구사항

- CUDA GPU (스크립트 내 `check_cuda()`가 없으면 종료)
- `HF_TOKEN` (Hub 업로드용. 없으면 경고 후 진행)
- 로컬 산출물: `./unsloth-output/` (gitignore)

## Axolotl 파이프라인과의 차이

- Unsloth 최적화 (~60% VRAM 절감 주장), `trl.SFTTrainer` + `train_on_responses_only`
- 기본 모델: `LiquidAI/LFM2.5-1.2B-Instruct`, 기본 데이터: `mlabonne/FineTome-100k`
- 이 리포의 `mini|full` 모드 규약·`data/` 3분리 구조를 따르지 않음 (skill 원본 그대로)

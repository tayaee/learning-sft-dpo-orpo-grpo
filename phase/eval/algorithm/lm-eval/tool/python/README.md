# lm-eval 비교평가 (공용)

Stage별 평가 스크립트(`*-eval-compare*.sh`, `*-lm-eval*.sh`)는 각
`phase/<phase>/algorithm/<algo>/tool/axolotl/`에 둔다 (평가 대상 모델이
그 stage 산출물이므로). 여기는 공용 자산만 둔다:

- `library/config/lm_eval/` — lm-eval YAML
- `library/script/eval_compare_table.py` — 비교표 생성

비교 대상 규칙: SFT=Base, DPO/GRPO/ORPO=같은 모드의 SFT merge 모델.
자세한 task 목록·임계값은 각 stage README의 평가 섹션을 볼 것.

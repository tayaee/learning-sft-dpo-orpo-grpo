# ORIGIN — source/unsloth/example/sft-basic

- 출처: unsloth skill에 포함된 SFT 예제 (`unsloth_sft_example.py`)
- 반입일: 2026-09-28
- 상태: 원본 verbatim 보관. 직접 수정하지 말고, 변형이 필요하면
  `phase/sft/...` 파이프라인으로 승격해서 다룰 것.
- 의존성: 파일 상단 PEP 723 블록이 선언 (`uv run` 즉시 실행, venv 불필요)

핵심 사실:

- 16비트 LoRA SFT 전용 스크립트다 (`load_in_16bit=True`,
  `load_in_4bit=False`, `load_in_8bit=False`).
- 양자화 export가 없다 (`save_pretrained_gguf` / Q4_K_M / BNB-4bit 호출 없음).
  저장은 LoRA 어댑터 또는 `merged_16bit` push뿐이다.
- GGUF 양자화 export를 붙이게 되면 그 산출물은 양자화 리포의
  `family/container/technique/gguf/tool/unsloth/` 관할이다.

Wrapper 매핑 (예제 docstring의 `uv run` 명령 1:1):

| wrapper | 원본 명령 |
|---|---|
| `run-epochs.sh` | epoch 기반 학습 (`--num-epochs 1 --eval-split 0.2`, full 권장) |
| `run-steps.sh` | step 기반 학습 (`--max-steps 500`, 빠른 테스트용) |
| `run-hf-job.sh` | HF Jobs 실행 (`--flavor a10g-small ...`, 1 epoch + eval) |

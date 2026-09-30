#!/usr/bin/env python3
"""41-quant-gptq.py — Stage 4b. GPTQ 4bit-g128 (damp 0.1).
원본: quantizaton_math.ipynb 전반 (auto-gptq) → gptqmodel로 교체.
캘리브: synthetic-<mode>.jsonl에서 CALIB_N개, 학습 프롬프트 템플릿 적용.

  uv run 41-quant-gptq.py --mode mini|full
"""
import argparse, os

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")


def main(mode: str, calib: int):
    # 입력: synthetic-fft-<mode>-single (전략별 산출물 중 single만 하류 사용).
    # TODO: gptqmodel QuantizeConfig(bits=4, group_size=128, damp_percent=0.1)
    # TODO: synthetic 캘리브 토크나이즈 → quantize → save_quantized + tokenizer 저장
    # TODO: push_to_hub(f"tayaee/1B-math-gptq-{mode}") (선택)
    print(f"[p5][{mode}] calib={calib} -> {SHARED}/models/synthetic-fft-{mode}-single-gptq")
    print("STUB: GPTQ 본문 미구현")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument("--calib", type=int, default=None)
    a = ap.parse_args()
    main(a.mode, a.calib or (4 if a.mode == "mini" else 10))

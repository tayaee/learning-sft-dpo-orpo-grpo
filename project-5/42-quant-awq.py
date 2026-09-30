#!/usr/bin/env python3
"""42-quant-awq.py — Stage 4c. AWQ 4bit-g128 GEMM (zero_point).
원본: quantizaton_math.ipynb 후반 (autoawq) → llm-compressor로 교체.
주의: quantize 후 model.to('cpu') → save. tokenizer 동봉 필수.

  uv run 42-quant-awq.py --mode mini|full
"""
import argparse, os

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")


def main(mode: str, calib: int):
    # 입력: synthetic-fft-<mode>-single (전략별 산출물 중 single만 하류 사용).
    # TODO: llm-compressor AWQ 예제 흐름으로 교체 (quant_config 동일)
    # TODO: calib=text_lst(문자열 리스트) → quantize → cpu → save + tokenizer
    print(f"[p5][{mode}] calib={calib} -> {SHARED}/models/synthetic-fft-{mode}-single-awq")
    print("STUB: AWQ 본문 미구현")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument("--calib", type=int, default=None)
    a = ap.parse_args()
    main(a.mode, a.calib or (4 if a.mode == "mini" else 10))

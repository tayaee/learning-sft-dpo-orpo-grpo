#!/usr/bin/env python3
"""43-quant-fft-gptq.py — Stage 4d. FFT → GPTQ 4bit-g128 (gptqmodel 7.x).
원본: quantizaton_math.ipynb 전반 (auto-gptq) → GPTQModel.load/quantize/save.
캘리브: gsm8k-calibration-256.jsonl (05-prep-calibration.sh 생성), 학습 프롬프트 템플릿 적용.
입력: synthetic-fft-<mode>-single (전략별 산출물 중 single만 하류 사용).

  uv run 43-quant-fft-gptq.py --mode mini|full [--calib N] [--calib-file PATH]
"""
import argparse
import os

from quant_common import load_calib_texts

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")


def main(mode: str, calib: int, calib_file: str | None):
    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer

    src = f"{SHARED}/models/synthetic-fft-{mode}-single"
    out = f"{SHARED}/models/synthetic-fft-{mode}-single-gptq"
    texts = load_calib_texts(calib, calib_file)
    tok = AutoTokenizer.from_pretrained(src)
    qc = QuantizeConfig(bits=4, group_size=128)
    model = GPTQModel.load(src, qc)
    model.quantize(calibration=texts[:calib], tokenizer=tok)
    model.save(out)
    tok.save_pretrained(out)
    print(f"[{mode}] calib={calib} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument("--calib", type=int, default=None)
    ap.add_argument("--calib-file", default=None,
                    help="캘리브레이션 jsonl (기본 gsm8k-calibration-256.jsonl, "
                         "CALIB_FILE로 오버라이드 가능)")
    a = ap.parse_args()
    default_calib = int(os.environ.get("CALIB_N", "32" if a.mode == "mini" else "256"))
    main(a.mode, a.calib or default_calib, a.calib_file)

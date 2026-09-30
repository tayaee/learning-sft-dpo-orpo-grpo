#!/usr/bin/env python3
"""43-quant-fp8.py — Stage 4d (신규). FP8 w8a8 PTQ. Blackwell 네이티브.
권장: llm-compressor FP8 예제 (calib=synthetic 텍스트 리스트) → 저장.
평가는 vLLM quantization='fp8' 경로로 51-eval.sh에서, 채점은 52-score.py 공용.

  uv run 43-quant-fp8.py --mode mini|full
"""
import argparse, os

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")


def main(mode: str, calib: int):
    # 입력: synthetic-fft-<mode>-single.
    # TODO: llm-compressor FP8 (w8a8, e4m3) 양자화 → save + tokenizer 동봉
    # TODO: push_to_hub(f"tayaee/1B-math-fp8-{mode}") (선택)
    print(f"[p5][{mode}] calib={calib} -> {SHARED}/models/synthetic-fft-{mode}-single-fp8")
    print("STUB: FP8 본문 미구현 (llm-compressor 문서의 AWQ 예제와 동형)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument("--calib", type=int, default=None)
    a = ap.parse_args()
    main(a.mode, a.calib or (4 if a.mode == "mini" else 64))

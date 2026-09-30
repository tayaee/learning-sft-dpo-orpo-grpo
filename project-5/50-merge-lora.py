#!/usr/bin/env python3
"""50-merge-lora.py — Stage 5a. QLoRA 어댑터 병합 (vLLM 평가 전 필수).
원본 tried: base 로드 → 특수토큰(<mask> 등) 동일 추가 → resize_token_embeddings
→ merge_and_unload → 저장. resize 생략 시 size-mismatch 에러.

  uv run 50-merge-lora.py --mode mini|full
"""
import argparse, os

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")


def main(mode: str):
    # 입력: synthetic-qlora-<mode>-single (어댑터). single만 하류에서 사용.
    # TODO: base + adapter 로드 → 토크나이저 특수토큰 추가 → resize → merge → 저장
    print(f"[p5][{mode}] adapter={SHARED}/models/synthetic-qlora-{mode}-single"
          f" -> {SHARED}/models/synthetic-qlora-{mode}-single-merged")
    print("STUB: 병합 본문 미구현")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    main(ap.parse_args().mode)

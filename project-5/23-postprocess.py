#!/usr/bin/env python3
"""23-postprocess.py — Stage 2c. 생성 CSV 후처리 → SFT/캘리브용 JSONL.
원본: data_check.ipynb (Q/A 마커 split, 케이스별 예외, 패턴 제거+rstrip).

  uv run 23-postprocess.py --mode mini|full
출력: $P5_SHARED/datasets/synthetic-<mode>.jsonl ({"source":[q],"target":[a]})
"""
import argparse, os

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")


def main(mode: str):
    # TODO: generated-<mode>.csv → split/예외처리/패턴제거 → jsonl 저장
    print(f"[p5][{mode}] -> {SHARED}/datasets/synthetic-{mode}.jsonl")
    print("STUB: 후처리 본문 미구현 — data_check.ipynb 케이스 순서대로 이식")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    main(ap.parse_args().mode)

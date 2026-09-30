#!/usr/bin/env python3
"""21-build-prompts.py — Stage 2b. template.txt에 후보 2개씩 채워 프롬프트 생성.
원본: template.txt({dg_instruct}/{dg_output}) + notebook 후반부.

  uv run 21-build-prompts.py --mode mini|full
출력: $P5_SHARED/datasets/prompts-<mode>.parquet + HF push (tayaee/*_syntheticdatagen_prompt)
"""
import argparse, os

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")
N = {"mini": 200, "full": 10000}


def main(mode: str):
    import pandas as pd
    with open(os.path.join(os.path.dirname(__file__), "assets", "template.txt"), encoding="utf-8") as f:
        template = f.read()
    # TODO: candidates-<mode>.parquet 로드 → 행당 2개 샘플 → template.format
    # TODO: DatasetDict push_to_hub(f"tayaee/alpaca_syntheticdatagen_prompt-{mode}")
    out = f"{SHARED}/datasets/prompts-{mode}.parquet"
    print(f"[p5][{mode}] n={N[mode]} template={len(template)} chars -> {out}")
    print("STUB: 샘플링/포맷 본문 미구현")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    main(ap.parse_args().mode)

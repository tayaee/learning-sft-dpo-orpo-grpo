#!/usr/bin/env python3
"""20-select-candidates.py — Stage 2a. Alpaca에서 GSM8K 유사 후보 선별.
원본: datagen.ipynb 전반 (all-mpnet-base-v2, 500개 배치, cosine, top1000→100).

  uv run 20-select-candidates.py --mode mini|full
출력: $P5_SHARED/datasets/candidates-<mode>.parquet (dg_idx, top100 리스트)
"""
import argparse, os

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")


def main(mode: str):
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    import pandas as pd
    import numpy as np

    n_gsm = 200 if mode == "mini" else None  # mini는 후보풀 축소용
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    dg = pd.DataFrame(load_dataset("tatsu-lab/alpaca")["train"])
    dg["cri"] = dg["instruction"] + dg["output"]
    # TODO: 500개 배치 인코딩 → dg['embedding'] 저장 (원본 셀 참조)
    # TODO: GSM8K 임베딩과 cosine → 행당 top1000 중 100 샘플 → result 컬럼
    out = f"{SHARED}/datasets/candidates-{mode}.parquet"
    print(f"[p5][{mode}] rows={len(dg)} n_gsm={n_gsm} -> {out}")
    print("STUB: 인코딩/유사도 본문 미구현 — datagen.ipynb 셀 순서대로 이식")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    main(ap.parse_args().mode)

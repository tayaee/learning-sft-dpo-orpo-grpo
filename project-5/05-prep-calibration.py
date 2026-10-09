#!/usr/bin/env python3
"""05-prep-calibration.py — AWQ/GPTQ/FP8 캘리브레이션용 GSM8K 슬라이스 생성.
gsm8k-train.jsonl → seed 고정 shuffle → 앞 N개 → gsm8k-calibration-<N>.jsonl.
순서 편향을 피하려고 shuffle 필수. train에서만 자르고 test는 절대 제외.

  uv run 05-prep-calibration.py [--n 256] [--seed 42]
"""
import argparse
import json
import os
import random

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")


def main(n: int, seed: int):
    src = f"{SHARED}/datasets/gsm8k-train.jsonl"
    out = f"{SHARED}/datasets/gsm8k-calibration-{n}.jsonl"
    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(line)
    if n > len(rows):
        raise SystemExit(f"n={n} > train rows={len(rows)}")
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    with open(out, "w", encoding="utf-8") as f:
        for i in idx[:n]:
            f.write(rows[i] + "\n")
    print(f"train={len(rows)} seed={seed} n={n} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.n, a.seed)

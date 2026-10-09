#!/usr/bin/env python3
"""73-score.py — Stage 7c. '#### <숫자>' 추출 채점 (원본 get_gsm8k_res.py).
invalid율 + acc를 타깃별 표로 출력. (BLEU는 생략 — acc가 핵심 지표.)

  uv run 73-score.py --mode mini|full
입력: $P5_SHARED/outputs/eval-<mode>/*.jsonl
"""
import argparse
import glob
import json
import os
import re

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
FALLBACK_RE = re.compile(r"[Tt]he answer is:?\s*\$?\s*(\-?[0-9][0-9\.\,]*)")
INVALID = "[invalid]"


def norm(s: str):
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract(t: str) -> str:
    m = ANS_RE.search(t or "")
    if m and norm(m.group(1)) is not None:
        return m.group(1).replace(",", "")
    # 합성데이터 학습 모델은 "The answer is: X" 형식을 뱉는다 (강의의 mini 보조추출과 동등)
    m = FALLBACK_RE.search(t or "")
    if m and norm(m.group(1)) is not None:
        return m.group(1).replace(",", "")
    return INVALID


def main(mode: str):
    print(f"{'target':8} {'n':>5} {'acc':>7} {'invalid':>8}")
    for path in sorted(glob.glob(f"{SHARED}/outputs/eval-{mode}/*.jsonl")):
        name = os.path.splitext(os.path.basename(path))[0]
        correct = invalid = n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                tgt = r["target"]
                gold = extract(tgt[0] if isinstance(tgt, list) else tgt)
                preds = r.get("kd_data", [])
                ans = extract(preds[0]) if preds else INVALID
                n += 1
                if ans == INVALID:
                    invalid += 1
                elif gold != INVALID and abs(float(ans) - float(gold)) < 1e-4:
                    correct += 1
        print(f"{name:8} {n:>5} {correct / max(n, 1):>7.3f} {invalid / max(n, 1):>8.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    main(ap.parse_args().mode)

#!/usr/bin/env python3
"""52-score.py — Stage 5c. '#### <숫자>' 추출 채점 (invalid율/acc/BLEU).
원본: evaluation/get_gsm8k_res.py. base처럼 'The answer is'를 안 뱉는 모델은
CSV 덤프 후 gpt-4o-mini 재추출 (원본 방식, 선택).

  uv run 52-score.py --mode mini|full
입력: $P5_SHARED/outputs/eval-<mode>/*.jsonl
"""
import argparse, os

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")


def main(mode: str):
    # TODO: ANS_RE=r"#### (\-?[0-9\.\,]+)" 추출 → judge true/false/invalid → 표 출력
    print(f"[p5][{mode}] scoring {SHARED}/outputs/eval-{mode}/*.jsonl")
    print("STUB: 채점 본문 미구현")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    main(ap.parse_args().mode)

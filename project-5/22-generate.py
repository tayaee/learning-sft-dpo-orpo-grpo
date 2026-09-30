#!/usr/bin/env python3
"""22-generate.py — teacher(Llama-3.1-8B-Instruct)로 합성 생성 + invalid 재생성 루프.
원본: data_gen_vllm.py (apply_chat_template, batch 10000, 마커 3종 검사).
"""
import argparse, os

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")
MARKERS = ("Transformed Domain Question", "Transformed Domain Answer", "The answer is")


def main(mode: str, tp: int, maxlen: int, teacher: str, n: int):
    from vllm import LLM, SamplingParams  # noqa
    # TODO: prompts-<mode>.parquet[:n] → chat template → LLM(tp) generate
    # TODO: 마커 미포함=invalid 분리 → 재생성 루프 → generated-<mode>.csv 저장
    print(f"[p5][{mode}] teacher={teacher} tp={tp} maxlen={maxlen} n={n}"
          f" -> {SHARED}/datasets/generated-{mode}.csv")
    print("STUB: vLLM 호출부 미구현")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--maxlen", type=int, default=4096)
    ap.add_argument("--teacher", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    main(a.mode, a.tp, a.maxlen, a.teacher, a.n)

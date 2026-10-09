#!/usr/bin/env python3
"""40-quant-fft-awq.py — Stage 4a. FFT → AWQ 4bit (llmcompressor oneshot + AWQModifier).
원본: quantizaton_math.ipynb 후반 (autoawq) → llmcompressor로 교체.
입력: synthetic-fft-<mode>-single. 출력에 tokenizer 동봉 필수.
캘리브레이션: gsm8k-calibration-256.jsonl (05-prep-calibration.sh 생성,
gsm8k-train에서 seed 고정 추출). AWQ 예제 정석 256.

  uv run 40-quant-fft-awq.py --mode mini|full [--calib N] [--calib-file PATH]
"""
import argparse
import json
import os

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context.\n"
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\nGive a response as the assistant with the input conversation history\n\n"
    "### Response:\n{response}"
)


def first(v):
    return v[0] if isinstance(v, list) else v


def main(mode: str, calib: int, calib_file: str | None):
    from datasets import Dataset
    from transformers import AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier

    src = f"{SHARED}/models/synthetic-fft-{mode}-single"
    out = f"{SHARED}/models/synthetic-fft-{mode}-single-awq"
    calib_file = calib_file or os.environ.get(
        "CALIB_FILE", f"{SHARED}/datasets/gsm8k-calibration-256.jsonl")
    if not os.path.exists(calib_file):
        raise SystemExit(f"missing calibration file: {calib_file} "
                         "(run 05-prep-calibration.sh first)")
    texts = []
    with open(calib_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                texts.append(PROMPT_TEMPLATE.format(instruction=first(r["source"]),
                                                    response=first(r["target"])))
            if len(texts) >= calib:
                break
    if len(texts) < calib:
        raise SystemExit(f"calibration file has {len(texts)} rows "
                         f"< requested calib={calib}")
    print(f"[{mode}] calib_src={calib_file} n={len(texts)}")
    ds = Dataset.from_list([{"text": t} for t in texts])
    recipe = AWQModifier(ignore=["lm_head"], scheme="W4A16", targets=["Linear"])
    oneshot(model=src, dataset=ds, recipe=recipe, output_dir=out,
            max_seq_length=1024, num_calibration_samples=calib)
    AutoTokenizer.from_pretrained(src).save_pretrained(out)
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

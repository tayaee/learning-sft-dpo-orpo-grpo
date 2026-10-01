#!/usr/bin/env python3
"""42-quant-awq.py — Stage 4c. AWQ 4bit (llmcompressor oneshot + AWQModifier).
원본: quantizaton_math.ipynb 후반 (autoawq) → llmcompressor로 교체.
입력: synthetic-fft-<mode>-single. 출력에 tokenizer 동봉 필수.

  uv run 42-quant-awq.py --mode mini|full [--calib N]
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


def main(mode: str, calib: int):
    from datasets import Dataset
    from transformers import AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier

    src = f"{SHARED}/models/synthetic-fft-{mode}-single"
    out = f"{SHARED}/models/synthetic-fft-{mode}-single-awq"
    texts = []
    with open(f"{SHARED}/datasets/synthetic-{mode}.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                texts.append(PROMPT_TEMPLATE.format(instruction=first(r["source"]),
                                                    response=first(r["target"])))
            if len(texts) >= max(calib, 10):
                break
    ds = Dataset.from_list([{"text": t} for t in texts])
    recipe = AWQModifier(ignore=["lm_head"], scheme="W4A16", targets=["Linear"])
    oneshot(model=src, dataset=ds, recipe=recipe, output_dir=out,
            max_seq_length=1024, num_calibration_samples=calib)
    AutoTokenizer.from_pretrained(src).save_pretrained(out)
    print(f"[p5][{mode}] calib={calib} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument("--calib", type=int, default=None)
    a = ap.parse_args()
    default_calib = int(os.environ.get("CALIB_N", "2" if a.mode == "mini" else "10"))
    main(a.mode, a.calib or default_calib)

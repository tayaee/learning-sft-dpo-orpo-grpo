#!/usr/bin/env python3
"""43-quant-fp8.py — Stage 4d (신규). FP8 static quant (llmcompressor oneshot).
Blackwell 네이티브. 평가는 vLLM quantization='fp8' 경로 (51-eval.sh).

  uv run 43-quant-fp8.py --mode mini|full [--calib N]
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
    from llmcompressor.modifiers.quantization import QuantizationModifier

    src = f"{SHARED}/models/synthetic-fft-{mode}-single"
    out = f"{SHARED}/models/synthetic-fft-{mode}-single-fp8"
    texts = []
    with open(f"{SHARED}/datasets/synthetic-{mode}.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                texts.append(PROMPT_TEMPLATE.format(instruction=first(r["source"]),
                                                    response=first(r["target"])))
            if len(texts) >= max(calib, 64):
                break
    ds = Dataset.from_list([{"text": t} for t in texts])
    recipe = QuantizationModifier(ignore=["lm_head"], scheme="FP8",
                                  targets=["Linear"])
    oneshot(model=src, dataset=ds, recipe=recipe, output_dir=out,
            max_seq_length=1024, num_calibration_samples=calib)
    AutoTokenizer.from_pretrained(src).save_pretrained(out)
    print(f"[p5][{mode}] calib={calib} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument("--calib", type=int, default=None)
    a = ap.parse_args()
    main(a.mode, a.calib or (4 if a.mode == "mini" else 64))

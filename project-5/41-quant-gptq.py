#!/usr/bin/env python3
"""41-quant-gptq.py — Stage 4b. GPTQ 4bit-g128 (gptqmodel 7.x).
원본: quantizaton_math.ipynb 전반 (auto-gptq) → GPTQModel.load/quantize/save.
캘리브: synthetic-<mode>.jsonl에서 CALIB_N개, 학습 프롬프트 템플릿 적용.
입력: synthetic-fft-<mode>-single (전략별 산출물 중 single만 하류 사용).

  uv run 41-quant-gptq.py --mode mini|full [--calib N]
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
    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer

    src = f"{SHARED}/models/synthetic-fft-{mode}-single"
    out = f"{SHARED}/models/synthetic-fft-{mode}-single-gptq"
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
    tok = AutoTokenizer.from_pretrained(src)
    qc = QuantizeConfig(bits=4, group_size=128)
    model = GPTQModel.load(src, qc)
    model.quantize(calibration=texts[:calib], tokenizer=tok)
    model.save(out)
    tok.save_pretrained(out)
    print(f"[p5][{mode}] calib={calib} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument("--calib", type=int, default=None)
    a = ap.parse_args()
    main(a.mode, a.calib or (4 if a.mode == "mini" else 10))

#!/usr/bin/env python3
"""53-quant-qlora-gptq.py — Stage 5d. QLoRA-merged → GPTQ 4bit-g128 (gptqmodel 7.x).
FFT용 43-quant-fft-gptq.py와 동일 플로우, 입력만 merged.
캘리브: gsm8k-calibration-256.jsonl (05-prep-calibration.sh 생성), 학습 프롬프트 템플릿 적용.
입력: synthetic-qlora-<mode>-single-merged → 출력: synthetic-qlora-<mode>-single-merged-gptq

  uv run 53-quant-qlora-gptq.py --mode mini|full [--calib N] [--calib-file PATH]
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
    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer

    src = f"{SHARED}/models/synthetic-qlora-{mode}-single-merged"
    out = f"{SHARED}/models/synthetic-qlora-{mode}-single-merged-gptq"
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
    tok = AutoTokenizer.from_pretrained(src)
    qc = QuantizeConfig(bits=4, group_size=128)
    model = GPTQModel.load(src, qc)
    model.quantize(calibration=texts[:calib], tokenizer=tok)
    model.save(out)
    tok.save_pretrained(out)
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

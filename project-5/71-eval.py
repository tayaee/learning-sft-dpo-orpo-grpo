#!/usr/bin/env python3
"""71-eval.py — Stage 7b. vLLM greedy GSM8K 추론 (원본 gen_math_greedy.py).
prompt_no_input 포맷, temp 0, max 512. 타깃별 quant flag 적용.
출력: $P5_SHARED/outputs/eval-<mode>/<target>.jsonl (kd_data 포함).
GGUF 10종(fft/qlora-gguf-*)은 vLLM 미지원 → 71-eval-gguf.py(llama-cli)로 별도 평가, 여기선 SKIP.

  uv run 71-eval.py --mode mini|full --target base|fft|qlora|fft-gptq|fft-awq|fft-fp8|qlora-gptq|qlora-awq|qlora-fp8 [--n N] [--tp 1]
"""
import argparse
import json
import os

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")
BASE = os.environ.get("BASE_MODEL", "unsloth/Llama-3.2-1B")
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)
# llmcompressor 산출물(AWQ/FP8)은 compressed-tensors 포맷으로 저장된다.
QUANT = {"fft-gptq": "gptq", "qlora-gptq": "gptq",
         "fft-awq": "compressed-tensors", "qlora-awq": "compressed-tensors",
         "fft-fp8": "compressed-tensors", "qlora-fp8": "compressed-tensors",
         # 구이름 호환: gptq/awq/fp8 = fft-gptq/awq/fp8
         "gptq": "gptq", "awq": "compressed-tensors", "fp8": "compressed-tensors"}


def resolve(target: str, mode: str) -> str:
    if target == "base":
        return BASE
    if target == "fft":
        return f"{SHARED}/models/synthetic-fft-{mode}-single"
    if target == "qlora":
        return f"{SHARED}/models/synthetic-qlora-{mode}-single-merged"
    if target in ("gptq", "awq", "fp8"):  # 구이름 → fft-* 별칭
        return f"{SHARED}/models/synthetic-fft-{mode}-single-{target}"
    if target.startswith("fft-"):
        return f"{SHARED}/models/synthetic-fft-{mode}-single-{target[4:]}"
    if target.startswith("qlora-"):
        return f"{SHARED}/models/synthetic-qlora-{mode}-single-merged-{target[6:]}"
    raise SystemExit(f"unknown target: {target} (gguf는 vLLM 미지원, llama.cpp로 평가)")


def main(mode: str, target: str, n: int, tp: int):
    from vllm import LLM, SamplingParams

    rows = []
    with open(f"{SHARED}/datasets/gsm8k-test.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if n > 0:
        rows = rows[:n]
    model = resolve(target, mode)
    kw = dict(model=model, tensor_parallel_size=tp, trust_remote_code=True,
              gpu_memory_utilization=0.9, dtype="auto", enforce_eager=True)
    if target in QUANT:
        kw["quantization"] = QUANT[target]
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0, max_tokens=512)

    def q_of(r):
        s = r["source"]
        return s[0] if isinstance(s, list) else s

    prompts = [PROMPT_NO_INPUT.format(instruction=q_of(r)) for r in rows]
    outs = llm.generate(prompts, sp)
    outdir = f"{SHARED}/outputs/eval-{mode}"
    os.makedirs(outdir, exist_ok=True)
    outp = f"{outdir}/{target}.jsonl"
    with open(outp, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            r = dict(r)
            r["kd_data"] = [o.outputs[0].text]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{mode}][{target}] n={len(rows)} model={model} -> {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument("--target", required=True,
                    choices=["base", "fft", "qlora",
                             "fft-gptq", "fft-awq", "fft-fp8",
                             "qlora-gptq", "qlora-awq", "qlora-fp8",
                             "gptq", "awq", "fp8"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--tp", type=int, default=int(os.environ.get("TP", "1")))
    a = ap.parse_args()
    n = a.n if a.n is not None else int(os.environ.get("EVAL_N", "10" if a.mode == "mini" else "0"))
    main(a.mode, a.target, n, a.tp)

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "transformers==4.57.3",
#     "huggingface_hub[hf_transfer]",
#     "peft",
#     "llama-cpp-python",
# ]
# ///
"""Step 04: single-shot inference on ex02 GGUF (4-bit default).

Input: gguf-output (ex02). Next: ex05_infer_stream.
GGUF mode runs on CPU. Unset --gguf-dir for Unsloth GPU mode.

Examples:
    uv run ex04_infer_generate.py
    uv run ex04_infer_generate.py --quant 8 --prompt "Hello!"
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

GGUF_FILE = {"4": "model-Q4_K_M.gguf", "5": "model-Q5_K_M.gguf", "8": "model-Q8_0.gguf"}


def parse_args():
    p = argparse.ArgumentParser(description="Step 04: single-shot inference")
    p.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--gguf-dir", default="gguf-output")
    p.add_argument("--no-gguf", action="store_true", default=False)
    p.add_argument("--quant", default="4", choices=["4", "5", "8"])
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--prompt", default="What is 1+1?")
    p.add_argument("--system", default=None)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--seed", type=int, default=3407)
    return p.parse_args()


def run_gguf(gguf_path, prompt, system, a):
    from llama_cpp import Llama

    print(f"Load GGUF: {gguf_path}")
    llm = Llama(model_path=gguf_path, n_ctx=a.max_seq_length, verbose=False)
    text = f"{system}\n{prompt}" if system else prompt
    out = llm(text, max_tokens=a.max_new_tokens, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k)
    print(f"\nAssistant: {out['choices'][0]['text'].strip()}")


def run_unsloth(a, messages):
    import random

    import numpy as np
    import torch
    from unsloth import FastLanguageModel

    print(f"Load: {a.model}" + (f" + adapter {a.adapter}" if a.adapter else ""))
    load4 = a.quant == "4"
    load8 = a.quant == "8"
    if a.quant == "5":
        print("Note: 5-bit has no BNB load, use 16bit here (Q5_K_M for GGUF).")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=a.model,
        max_seq_length=a.max_seq_length,
        load_in_4bit=load4,
        load_in_8bit=load8,
        load_in_16bit=not (load4 or load8),
        full_finetuning=False,
    )
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter)
    FastLanguageModel.for_inference(model)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    inputs = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(
        "cuda"
    )
    do_sample = a.temperature > 0
    kw = {"max_new_tokens": a.max_new_tokens, "use_cache": True, "do_sample": do_sample}
    if do_sample:
        kw.update({"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k})
    with torch.no_grad():
        out = model.generate(input_ids=inputs, **kw)
    print(f"\nAssistant: {tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()}")


def main():
    a = parse_args()
    use_gguf = a.gguf_dir and not a.no_gguf
    print(f"Mode: {'gguf' if use_gguf else 'unsloth'} quant={a.quant} max_new={a.max_new_tokens}")
    if use_gguf:
        run_gguf(os.path.join(a.gguf_dir, GGUF_FILE[a.quant]), a.prompt, a.system, a)
        print("Done! Next: ex05_infer_stream")
        return
    import torch

    if not torch.cuda.is_available():
        print("No CUDA GPU. Unsloth mode needs GPU (or use --gguf-dir).", file=sys.stderr)
        sys.exit(1)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    msgs = []
    if a.system:
        msgs.append({"role": "system", "content": a.system})
    msgs.append({"role": "user", "content": a.prompt})
    run_unsloth(a, msgs)
    print("Done! Next: ex05_infer_stream")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Step 04: single-shot inference on ex02 GGUF.")
        print("  uv run ex04_infer_generate.py --help")
        print("  ./ex04_infer_generate.sh")
        print("CPU ok (GGUF). No-op without args.")
        sys.exit(0)
    main()

#!/usr/bin/env python3
"""query_base.py — 베이스 모델 추론 (학습 전 baseline).

사용법:
    uv run query_base.py "방정식 x^2 + 5x + 6 = 0 의 해를 구하시오."
    uv run query_base.py               # REPL 모드 (여러 질문 연속)

모델은 로컬 HF 캐시(`~/.cache/huggingface/hub/`)의 Qwen2.5-1.5B-Instruct 사용.
오프라인 환경 강제: HF_HUB_OFFLINE=1 환경변수 설정.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen2.5-1.5B-Instruct 베이스 추론")
    p.add_argument("prompt", nargs="*", help="한글 질문 (없으면 REPL)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy, >0 = sampling")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--repl", action="store_true",
                   help="인자 없이도 REPL 진입")
    return p


def load_model():
    print(f"[base] loading {BASE_MODEL} ...", flush=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
        device_map="cuda:0",
        attn_implementation="sdpa",     # flash-attn 미설치 → SDPA fallback
    )
    model.eval()
    return tok, model


def ask(tok, model, prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0.0),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    if args.temperature > 0.0:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = 0.95
    with torch.no_grad():
        out = model.generate(**ids, **gen_kwargs)
    gen = out[0, ids.input_ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


def repl_mode(tok, model):
    print("\n=== REPL 모드 ('quit' / 'exit' / Ctrl-D 종료) ===\n")
    while True:
        try:
            q = input("[you] ").strip()
        except EOFError:
            print()
            break
        if not q or q.lower() in ("quit", "exit"):
            break
        print("\n=== BASE ===")
        print(ask(tok, model, q))
        print()


def main():
    global args
    args = build_argparser().parse_args()
    tok, model = load_model()

    if args.prompt:
        print("\n=== BASE ===")
        print(ask(tok, model, " ".join(args.prompt)))
    else:
        repl_mode(tok, model)


if __name__ == "__main__":
    main()

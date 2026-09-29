#!/usr/bin/env python3
"""query_rl_orpo.py — ORPO 완료 모델 추론.

사용법:
    uv run query_rl_orpo.py --mode mini "방정식 x^2 + 5x + 6 = 0 의 해를 구하시오."
    uv run query_rl_orpo.py --mode full  "피타고라스 정리를 증명하시오."
    uv run query_rl_orpo.py --mode full              # REPL 모드

모델 경로:
    full  → ./data/orpo-full-out/merged
    mini  → ./data/orpo-mini-out/merged
사전 요구: rl-orpo-04-run-axolotl.sh + rl-orpo-05-merge.sh 완료
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATHS = {
    "mini": "./data/orpo-mini-out/merged",
    "full": "./data/orpo-full-out/merged",
}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ORPO merge 모델 추론")
    p.add_argument("prompt", nargs="*", help="한글 질문 (없으면 REPL)")
    p.add_argument("--mode", choices=["mini", "full"], default="full")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    return p


def load_model(args):
    path = MODEL_PATHS[args.mode]
    if not os.path.isdir(path):
        sys.exit(f"[orpo] ERROR: {path} 가 없습니다.\n"
                 f"  먼저 rl-orpo-04 (train) + rl-orpo-05 (merge) 를 "
                 f"--mode {args.mode} 로 실행하세요.")
    print(f"[orpo] loading {path} ...", flush=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, device_map="cuda:0", attn_implementation="sdpa")
    model.eval()
    return tok, model


def ask(tok, model, prompt: str, max_new_tokens: int, temperature: float) -> str:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95
    with torch.no_grad():
        out = model.generate(**ids, **gen_kwargs)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    args = build_argparser().parse_args()
    tok, model = load_model(args)

    if args.prompt:
        print("\n=== ORPO ===")
        print(ask(tok, model, " ".join(args.prompt),
                  args.max_new_tokens, args.temperature))
    else:
        print("\n=== REPL 모드 ('quit' / 'exit' / Ctrl-D 종료) ===\n")
        while True:
            try:
                q = input("[you] ").strip()
            except EOFError:
                print()
                break
            if not q or q.lower() in ("quit", "exit"):
                break
            print("\n=== ORPO ===")
            print(ask(tok, model, q, args.max_new_tokens, args.temperature))
            print()


if __name__ == "__main__":
    main()

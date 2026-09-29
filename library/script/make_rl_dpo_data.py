#!/usr/bin/env python3
"""make_rl_dpo_data.py — 베이스 모델 + SFT 모델로 DPO 학습용 선호 데이터 생성.

생성 원리 (rejection sampling 경량 버전):
  - 각 prompt 에 대해 K=4 개 응답을 SFT'd 모델에서 샘플링
  - 점수(score)를 매겨 가장 좋은 응답 = chosen, 가장 나쁜 응답 = rejected
  - 점수 함수:  ① CoT  블록 존재 ②  ###  헤더로 구조화된 결론  ③ 응답 길이
  - 베이스 모델 출력도 "rejected" 후보로 섞어서 데이터 다양성 확보

출력:  data/dpo-{mode}-out/train_rl-dpo.jsonl  (각 줄: {"prompt", "chosen", "rejected"})
       data/dpo-{mode}-out/sample_rl-dpo.jsonl  (디버깅용 50건)

사용:
    uv run python make_rl_dpo_data.py --sft-model ./data/sft-full-out/merged \
        --base-model Qwen/Qwen2.5-1.5B-Instruct --num-prompts 1000 \
        --out data/dpo-full-out/train_rl-dpo.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(p: str, dtype):
    print(f"[load] {p}", flush=True)
    tok = AutoTokenizer.from_pretrained(p)
    m = AutoModelForCausalLM.from_pretrained(
        p, dtype=dtype, device_map="cuda:0", attn_implementation="sdpa")
    m.eval()
    return tok, m


def chat_generate(tok, model, prompt: str, max_new_tokens: int, temperature: float) -> str:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else 1.0,
            top_p=0.95,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
    gen = out[0, ids.input_ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


def score(resp: str) -> float:
    """SFT'd 응답에 부여할 휴리스틱 점수 (높을수록 좋음)."""
    score = 0.0
    # 1.   블록 보너스
    if " segmented reasoning" in resp or "Let's" in resp:
        score += 0.5
    # 2.  ###  구조화 헤더 보너스 (SFT 데이터는 거의 항상 있음)
    if resp.count("###") >= 1:
        score += 1.0
    # 3.  결론 / 답변 단정 보너스
    if re.search(r"존재하지 않으므로|따라서|그러므로|정리하면|답은", resp):
        score += 0.5
    # 4.  적정 길이 (300–1200자) 보너스
    L = len(resp)
    if 300 <= L <= 1200:
        score += 1.0
    elif L < 100:
        score -= 1.0
    # 5.  너무 짧거나 잘린 응답 페널티
    if not resp.strip().endswith(("다.", "요.", "다).", "습니다.", ".")):
        score -= 0.5
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-model", default="./data/sft-full-out/merged")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train-jsonl", default="train.jsonl")
    ap.add_argument("--out", default="data/dpo-full-out/train_rl-dpo.jsonl",
                    help="mini 용도면 data/dpo-mini-out/ 아래로 지정할 것")
    ap.add_argument("--sample-out", default="data/dpo-full-out/sample_rl-dpo.jsonl")
    ap.add_argument("--num-prompts", type=int, default=1000)
    ap.add_argument("--samples-per-prompt", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    random.seed(args.seed)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    # ----- 입력 -----
    if not os.path.exists(args.train_jsonl) and os.path.exists(args.train_jsonl + ".gz"):
        import subprocess
        print(f"[gunzip] {args.train_jsonl}.gz -> {args.train_jsonl}", flush=True)
        subprocess.run(["gunzip", "-k", args.train_jsonl + ".gz"], check=True)

    print(f"[read] {args.train_jsonl}", flush=True)
    with open(args.train_jsonl, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    random.shuffle(rows)
    rows = rows[: args.num_prompts]
    print(f"[read] {len(rows)} prompts", flush=True)

    # ----- 모델 -----
    sft_tok, sft_model = load(args.sft_model, dtype)
    base_tok, base_model = load(args.base_model, dtype)

    # ----- 생성 -----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_pairs = 0
    accepts = []
    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            prompt = row["messages"][0]["content"]
            gold = row["messages"][1]["content"]

            # SFT 모델에서 K개 샘플
            cands = []
            for _ in range(args.samples_per_prompt):
                r = chat_generate(sft_tok, sft_model, prompt,
                                  args.max_new_tokens, temperature=0.7)
                cands.append((score(r), r))

            # base 모델에서 1개 (rejected 후보)
            base_resp = chat_generate(base_tok, base_model, prompt,
                                      args.max_new_tokens, temperature=0.0)
            base_score = score(base_resp)

            # 정렬: 가장 높은 점수 = chosen, 가장 낮은 = rejected
            cands_sorted = sorted(cands, key=lambda x: x[0], reverse=True)
            chosen_score, chosen = cands_sorted[0]
            _, rejected_sft = cands_sorted[-1]

            # base 모델 출력이 더 낮으면 그걸 rejected 로 사용
            if base_score < score(rejected_sft):
                rejected = base_resp
            else:
                rejected = rejected_sft

            # 너무 비슷하면 (점수 차이 작으면) 살짝 노이즈 주입
            if chosen.strip() == rejected.strip():
                continue

            obj = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_pairs += 1

            if len(accepts) < 50:
                accepts.append(obj)

            if (i + 1) % 10 == 0:
                print(f"[gen] {i+1}/{len(rows)}  pairs={n_pairs}", flush=True)

    # ----- sample DPO -----
    with open(args.sample_out, "w", encoding="utf-8") as f:
        for o in accepts:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"[done] {n_pairs} pairs → {args.out}")
    print(f"[done] {len(accepts)} sample → {args.sample_out}")


if __name__ == "__main__":
    main()

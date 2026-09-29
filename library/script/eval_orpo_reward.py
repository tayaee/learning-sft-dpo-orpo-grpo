#!/usr/bin/env python3
"""eval_orpo_reward.py — ORPO 효과 정량 측정 (휴리스틱 reward 기반).

SFT 모델 vs ORPO 모델이 같은 프롬프트 집합에 대해 생성한 응답을
make_rl_dpo_data.score 휴리스틱으로 채점해 평균 점수를 비교한다.
ORPO 가 선호 데이터를 잘 학습했다면 ORPO 평균 점수가 유의미하게 올라간다.

사용:
    uv run python eval_orpo_reward.py --mode full
    uv run python eval_orpo_reward.py --mode mini --num-prompts 10
"""
from __future__ import annotations

import argparse
import json
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from make_rl_dpo_data import chat_generate, load, score

PATHS = {
    "mini": "./data/orpo-mini-out/merged",
    "full": "./data/orpo-full-out/merged",
}

# 비교 대상 SFT 베이스라인도 같은 모드의 merge 모델 사용 (모드 간 섞임 방지)
SFT_PATHS = {
    "mini": "./data/sft-mini-out/merged",
    "full": "./data/sft-full-out/merged",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mini", "full"], default="full")
    ap.add_argument("--num-prompts", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None, help="결과 jsonl 저장 경로(선택)")
    args = ap.parse_args()

    if not os.path.exists("train.jsonl") and os.path.exists("train.jsonl.gz"):
        import subprocess
        print("[gunzip] train.jsonl.gz -> train.jsonl", flush=True)
        subprocess.run(["gunzip", "-k", "train.jsonl.gz"], check=True)

    with open("train.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    random.shuffle(rows)
    rows = rows[: args.num_prompts]

    sft_tok, sft_model = load(SFT_PATHS[args.mode], torch.bfloat16)
    orpo_tok, orpo_model = load(PATHS[args.mode], torch.bfloat16)

    results = []
    for i, row in enumerate(rows):
        prompt = row["messages"][0]["content"]
        r_sft = chat_generate(sft_tok, sft_model, prompt,
                              args.max_new_tokens, temperature=0.0)
        r_orpo = chat_generate(orpo_tok, orpo_model, prompt,
                               args.max_new_tokens, temperature=0.0)
        s_sft, s_orpo = score(r_sft), score(r_orpo)
        results.append({"prompt": prompt[:80], "score_sft": s_sft,
                        "score_orpo": s_orpo})
        print(f"[{i+1}/{len(rows)}] sft={s_sft:+.1f}  orpo={s_orpo:+.1f}"
              f"  {'↑' if s_orpo > s_sft else ('=' if s_orpo == s_sft else '↓')}",
              flush=True)

    avg_sft = sum(r["score_sft"] for r in results) / len(results)
    avg_orpo = sum(r["score_orpo"] for r in results) / len(results)
    wins = sum(1 for r in results if r["score_orpo"] > r["score_sft"])
    ties = sum(1 for r in results if r["score_orpo"] == r["score_sft"])

    print("\n========== 요약 ==========")
    print(f"prompts          : {len(results)}")
    print(f"SFT  평균 reward : {avg_sft:+.3f}")
    print(f"ORPO 평균 reward : {avg_orpo:+.3f}")
    print(f"개선폭           : {avg_orpo - avg_sft:+.3f}")
    print(f"win/tie/loss     : {wins}/{ties}/{len(results)-wins-ties}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()

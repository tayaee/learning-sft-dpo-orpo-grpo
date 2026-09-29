#!/usr/bin/env python3
"""make_rl_orpo_data.py — ORPO 학습용 선호 데이터 생성.

ORPO 는 DPO 와 같은 (prompt, chosen, rejected) 삼중 데이터를 사용하지만
reference model 이 없고 SFT 손실과 선호 손실을 한 번에 학습한다.

주의: axolotl ORPO 학습(TRL conversational 포맷)은 chosen/rejected 를
메시지 리스트 형식으로 요구하므로, 여기서 DPO 포맷(평문)을 변환해 저장한다:
    {"prompt": str,
     "chosen":   [{"role":"user", ...}, {"role":"assistant", ...}],
     "rejected": [{"role":"user", ...}, {"role":"assistant", ...}]}

두 가지 모드:
  1) --from-dpo : 이미 만들어 둔 data/train_rl-dpo.jsonl 을 재사용 (빠름, 권장)
  2) 직접 생성  : make_rl_dpo_data.py 의 rejection-sampling 로직을 임포트해서
                  새로 선호 쌍을 생성 (SFT 모델 필요)

출력:  --out 으로 지정한 경로 ({prompt, chosen, rejected})
       예) full  → data/orpo-full-out/train_rl-orpo.jsonl
           mini  → data/orpo-mini-out/train_rl-orpo.jsonl

사용:
    # DPO 데이터 재사용 (권장) — rl-orpo-01-make-orpo-data.sh <mini|full> 사용 권장
    uv run python make_rl_orpo_data.py --from-dpo data/dpo-full-out/train_rl-dpo.jsonl \
        --num-prompts 1000 --out data/orpo-full-out/train_rl-orpo.jsonl

    # 직접 생성 (SFT merge 모델 필요, 시간 오래 걸림)
    uv run python make_rl_orpo_data.py --generate \
        --sft-model ./data/sft-full-out/merged --num-prompts 1000
"""
from __future__ import annotations

import argparse
import json
import os
import random


def to_orpo_row(r: dict) -> dict:
    """DPO 평문 포맷 → TRL conversational 메시지 리스트 포맷."""
    return {
        "prompt": r["prompt"],
        "chosen": [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["chosen"]},
        ],
        "rejected": [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["rejected"]},
        ],
    }


def from_dpo(args):
    """DPO 선호 쌍 파일을 그대로 ORPO 형식으로 변환 (포맷이 동일하다)."""
    with open(args.from_dpo, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    random.seed(args.seed)
    random.shuffle(rows)
    rows = rows[: args.num_prompts]
    print(f"[reuse] {args.from_dpo} → {len(rows)} pairs")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(to_orpo_row(r), ensure_ascii=False) + "\n")
    print(f"[done] {len(rows)} pairs → {args.out}")
    # --mini-out 을 명시한 경우에만 mini 파일을 추가로 생성 (모드 간 덮어쓰기 방지)
    if args.mini_out:
        mini_dir = os.path.dirname(args.mini_out)
        if mini_dir:
            os.makedirs(mini_dir, exist_ok=True)
        with open(args.mini_out, "w", encoding="utf-8") as f:
            for r in rows[: args.mini_n]:
                f.write(json.dumps(to_orpo_row(r), ensure_ascii=False) + "\n")
        print(f"[done] {min(len(rows), args.mini_n)} pairs → {args.mini_out}")


def generate(args):
    """rejection sampling 으로 직접 생성 (make_rl_dpo_data 로직 재사용)."""
    import torch  # noqa: F401  (make_rl_dpo_data 가 내부에서 사용)
    from make_rl_dpo_data import chat_generate, load, score

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    if not os.path.exists(args.train_jsonl) and os.path.exists(args.train_jsonl + ".gz"):
        import subprocess
        print(f"[gunzip] {args.train_jsonl}.gz -> {args.train_jsonl}", flush=True)
        subprocess.run(["gunzip", "-k", args.train_jsonl + ".gz"], check=True)

    print(f"[read] {args.train_jsonl}", flush=True)
    with open(args.train_jsonl, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    random.seed(args.seed)
    random.shuffle(rows)
    rows = rows[: args.num_prompts]

    sft_tok, sft_model = load(args.sft_model, dtype)
    base_tok, base_model = load(args.base_model, dtype)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            prompt = row["messages"][0]["content"]
            cands = []
            for _ in range(args.samples_per_prompt):
                r = chat_generate(sft_tok, sft_model, prompt,
                                  args.max_new_tokens, temperature=0.7)
                cands.append((score(r), r))
            base_resp = chat_generate(base_tok, base_model, prompt,
                                      args.max_new_tokens, temperature=0.0)
            cands_sorted = sorted(cands, key=lambda x: x[0], reverse=True)
            _, chosen = cands_sorted[0]
            rejected = base_resp if score(base_resp) < score(cands_sorted[-1][1]) \
                else cands_sorted[-1][1]
            if chosen.strip() == rejected.strip():
                continue
            f.write(json.dumps(
                to_orpo_row({"prompt": prompt, "chosen": chosen,
                             "rejected": rejected}), ensure_ascii=False) + "\n")
            n += 1
            if (i + 1) % 10 == 0:
                print(f"[gen] {i+1}/{len(rows)}  pairs={n}", flush=True)
    print(f"[done] {n} pairs → {args.out}")

    # mini 용은 앞부분 잘라 저장
    with open(args.out, encoding="utf-8") as src, \
         open(args.mini_out, "w", encoding="utf-8") as dst:
        for j, line in enumerate(src):
            if j >= args.mini_n:
                break
            dst.write(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dpo", default=None,
                    help="재사용할 DPO 선호 쌍 jsonl (지정 시 재사용 모드)")
    ap.add_argument("--generate", action="store_true", help="직접 생성 모드")
    ap.add_argument("--sft-model", default="./data/sft-full-out/merged")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train-jsonl", default="train.jsonl")
    ap.add_argument("--out", default="data/orpo-full-out/train_rl-orpo.jsonl")
    ap.add_argument("--mini-out", default=None,
                    help="지정 시 소량 mini 파일을 추가 생성 (기본: 생성 안 함)")
    ap.add_argument("--num-prompts", type=int, default=1000)
    ap.add_argument("--samples-per-prompt", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--mini-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    if args.from_dpo:
        from_dpo(args)
    elif args.generate:
        generate(args)
    else:
        ap.error("--from-dpo 또는 --generate 중 하나는 필수입니다.")


if __name__ == "__main__":
    main()

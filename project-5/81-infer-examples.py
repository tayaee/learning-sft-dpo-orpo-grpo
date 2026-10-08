#!/usr/bin/env python3
"""81-infer-examples.py — Stage 8c. 서빙 중인 모델에 추론 예제 요청.
80-serve-vllm.sh가 띄운 OpenAI-호환 엔드포인트로 질의한다.
예제는 벤더링된 gsm8k-test.jsonl 앞 N개 + 한국어 수학 1문제로,
학습/평가와 동일한 prompt_no_input 포맷을 **completions** API로 보낸다
(서빙 토크나이저에 chat template이 없어 chat API는 400).

  # 터미널1: ./80-serve-vllm.sh mini fft
  # 터미널2:
  uv run 81-infer-examples.py --model tayaee/Llama-3.2-1B-math-fft-mini [--n 3] [--dry-run]
"""
import argparse
import json
import os

P5_ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)
KO_SAMPLE = "방정식 x^2 + 5x + 6 = 0 의 해를 구하시오."


def load_questions(n: int):
    for cand in [os.environ.get("P5_SHARED", "") + "/datasets/gsm8k-test.jsonl",
                 os.path.join(P5_ROOT, "data", "gsm8k-test.jsonl")]:
        if cand and os.path.exists(cand):
            qs = []
            with open(cand, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        q = r["source"][0] if isinstance(r["source"], list) else r["source"]
                        qs.append(q)
                    if len(qs) >= n:
                        break
            return qs
    raise SystemExit("gsm8k-test.jsonl 없음 — 00-setup.sh 먼저 실행")


def main(model: str, base_url: str, n: int, dry_run: bool):
    questions = load_questions(n) + [KO_SAMPLE]
    prompts = [PROMPT_NO_INPUT.format(instruction=q) for q in questions]
    if dry_run:
        for q, p in zip(questions, prompts):
            print(f"Q: {q}\n--- prompt ---\n{p}\n")
        return
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    for q, p in zip(questions, prompts):
        r = client.completions.create(
            model=model, prompt=p,
            temperature=0, max_tokens=512)
        print(f"Q: {q}\nA: {r.choices[0].text}\n{'=' * 60}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="서빙 모델명 (예: tayaee/Llama-3.2-1B-math-fft-mini)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(a.model, a.base_url, a.n, a.dry_run)

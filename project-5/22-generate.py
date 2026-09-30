#!/usr/bin/env python3
"""22-generate.py — teacher로 합성 생성 + invalid 재생성 루프.
원본: data_gen_vllm.py (apply_chat_template, 마커 3종 검사).
마커 미포함 행은 invalid로 분리해 재생성한다 (valid가 안정될 때까지, 최대 3회).
"""
import argparse
import os

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")
MARKERS = ("Transformed Domain Question", "Transformed Domain Answer", "The answer is")
MAX_ITERS = 3


def is_valid(t: str) -> bool:
    return all(m in (t or "") for m in MARKERS)


def main(mode: str, tp: int, maxlen: int, teacher: str, n: int):
    import pandas as pd
    from tqdm import tqdm
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    df = pd.read_parquet(f"{SHARED}/datasets/prompts-{mode}.parquet").head(n)
    df = df.reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(teacher)

    def to_chat(p):
        return tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)

    llm = LLM(model=teacher, tensor_parallel_size=tp, max_model_len=maxlen,
              trust_remote_code=True, gpu_memory_utilization=0.9,
              dtype="auto", enforce_eager=True)
    sp = SamplingParams(temperature=0.5, top_p=0.8, top_k=5,
                        repetition_penalty=1.05, max_tokens=2048)
    print("[p5] model ready", flush=True)

    df["generated"] = None
    pending = df.index.tolist()
    for it in range(1, MAX_ITERS + 1):
        chats = [to_chat(df.loc[i, "prompt"]) for i in pending]
        outs = llm.generate(chats, sp)
        nxt = []
        for i, o in zip(pending, outs):
            t = o.outputs[0].text
            df.at[i, "generated"] = t
            if not is_valid(t):
                nxt.append(i)
        print(f"[p5][{mode}] iter{it}: valid={len(pending) - len(nxt)}/{len(pending)}",
              flush=True)
        pending = nxt
        if not pending:
            break
    out = f"{SHARED}/datasets/generated-{mode}.csv"
    df.to_csv(out, index=False)
    print(f"[p5][{mode}] invalid_remain={len(pending)} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--maxlen", type=int, default=4096)
    ap.add_argument("--teacher", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    main(a.mode, a.tp, a.maxlen, a.teacher, a.n)

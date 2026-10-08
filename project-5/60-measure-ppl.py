#!/usr/bin/env python3
"""60-measure-ppl.py — Stage 6. 전 타깃 PPL 측정 + Accept/Conditional/Discard 판정.
대상 11종: base, fft, qlora(merged) + 양자화 8종(fft×4, qlora-merged×4).
GGUF 2종은 transformers로 PPL 불가 → SKIP (llama.cpp perplexity로 별도 측정).

판정 기준 (부모 FP 대비 ΔPPL):
  Δ < 0.3            → Accept (그대로 배포)
  0.3 ≤ Δ < 1.0      → Conditional (71-eval GSM8K 2차 심사 필요)
  Δ ≥ 1.0            → Discard (추론 깨짐 구간)
부모: fft-* vs fft, qlora-* vs qlora. fft/qlora 자체는 vs base 참고용으로 표시.

텍스트: $P5_SHARED/datasets/gsm8k-test.jsonl 앞 N개를 학습 프롬프트 템플릿으로
감싸서 in-domain PPL 측정 (WikiText가 아니라 수학 특화 유지 여부용).

  uv run 60-measure-ppl.py --mode mini|full [--n 32] [--targets all|base,fft,...]
  출력: stdout 표 + $P5_SHARED/outputs/ppl-<mode>.json
"""
import argparse
import json
import math
import os

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")
BASE = os.environ.get("BASE_MODEL", "unsloth/Llama-3.2-1B")
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context.\n"
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\nGive a response as the assistant with the input conversation history\n\n"
    "### Response:\n{response}"
)

# target → (kind, 경로템플릿, 부모타깃). gguf는 PPL SKIP.
TARGETS = {
    "base": ("hf", BASE, None),
    "fft": ("hf", f"{SHARED}/models/synthetic-fft-{{m}}-single", "base"),
    "qlora": ("hf", f"{SHARED}/models/synthetic-qlora-{{m}}-single-merged", "base"),
    "fft-gguf": ("gguf", f"{SHARED}/models/synthetic-fft-{{m}}-single-gguf/model-q8_0.gguf", "fft"),
    "fft-gptq": ("hf", f"{SHARED}/models/synthetic-fft-{{m}}-single-gptq", "fft"),
    "fft-awq": ("hf", f"{SHARED}/models/synthetic-fft-{{m}}-single-awq", "fft"),
    "fft-fp8": ("hf", f"{SHARED}/models/synthetic-fft-{{m}}-single-fp8", "fft"),
    "qlora-gguf": ("gguf", f"{SHARED}/models/synthetic-qlora-{{m}}-single-merged-gguf/model-q8_0.gguf", "qlora"),
    "qlora-gptq": ("hf", f"{SHARED}/models/synthetic-qlora-{{m}}-single-merged-gptq", "qlora"),
    "qlora-awq": ("hf", f"{SHARED}/models/synthetic-qlora-{{m}}-single-merged-awq", "qlora"),
    "qlora-fp8": ("hf", f"{SHARED}/models/synthetic-qlora-{{m}}-single-merged-fp8", "qlora"),
}
ORDER = ["base", "fft", "qlora",
         "fft-gguf", "fft-gptq", "fft-awq", "fft-fp8",
         "qlora-gguf", "qlora-gptq", "qlora-awq", "qlora-fp8"]


def verdict(delta):
    if delta is None:
        return "SKIP"
    if delta < 0.3:
        return "Accept"
    if delta < 1.0:
        return "Conditional"
    return "Discard"


def first(v):
    return v[0] if isinstance(v, list) else v


def load_texts(mode: str, n: int):
    texts = []
    with open(f"{SHARED}/datasets/gsm8k-test.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                texts.append(PROMPT_TEMPLATE.format(
                    instruction=first(r["source"]), response=first(r["target"])))
                if len(texts) >= n:
                    break
    return texts


def ppl_of(model_path: str, texts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()
    nll_sum, tok_count = 0.0, 0
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True,
                      max_length=1024).input_ids.cuda() \
                if torch.cuda.is_available() else \
                tok(t, return_tensors="pt", truncation=True,
                    max_length=1024).input_ids
            if torch.cuda.is_available():
                ids = ids.cuda()
                model = model.cuda() if next(model.parameters()).device.type == "cpu" else model
            labels = ids.clone()
            out = model(input_ids=ids, labels=labels)
            # loss는 평균 NLL → 토큰 수 가중 합산
            n = ids.numel()
            nll_sum += out.loss.item() * n
            tok_count += n
    return math.exp(nll_sum / max(tok_count, 1))


def main(mode: str, n: int, targets: str):
    sel = ORDER if targets == "all" else targets.split(",")
    texts = load_texts(mode, n)
    print(f"[ppl][{mode}] texts={len(texts)}")
    results = {}
    for t in sel:
        kind, tmpl, parent = TARGETS[t]
        path = tmpl.format(m=mode) if "{m}" in tmpl else tmpl
        if kind == "gguf":
            results[t] = {"ppl": None, "delta": None, "verdict": "SKIP",
                          "note": "GGUF는 transformers PPL 불가 — llama.cpp perplexity로 별도 측정"}
            print(f"{t:12} SKIP (gguf, {path})")
            continue
        if not os.path.exists(path) and t != "base":
            results[t] = {"ppl": None, "delta": None, "verdict": "SKIP",
                          "note": f"산출물 없음: {path}"}
            print(f"{t:12} SKIP (산출물 없음)")
            continue
        try:
            p = ppl_of(path, texts)
            results[t] = {"ppl": round(p, 3), "path": path}
            print(f"{t:12} ppl={p:.3f}")
        except Exception as e:  # noqa: BLE001 — 전 타깃 표를 끝까지 뽑기 위함
            results[t] = {"ppl": None, "delta": None, "verdict": "SKIP",
                          "note": f"{type(e).__name__}: {e}"}
            print(f"{t:12} SKIP ({e})")

    # Δ 및 판정 (부모가 측정됐을 때만)
    for t in sel:
        r = results[t]
        if r.get("ppl") is None:
            continue
        _, _, parent = TARGETS[t]
        if parent and results.get(parent, {}).get("ppl") is not None:
            d = r["ppl"] - results[parent]["ppl"]
            r["delta_vs_" + parent] = round(d, 3)
            r["verdict"] = verdict(d)
        elif t in ("fft", "qlora") and results.get("base", {}).get("ppl") is not None:
            d = r["ppl"] - results["base"]["ppl"]
            r["delta_vs_base"] = round(d, 3)
            r["verdict"] = "ref"  # SFT 자체의 도메인 이동이라 판정 대상 아님
        elif t == "base":
            r["verdict"] = "ref"

    print(f"\n{'target':12} {'ppl':>8} {'delta':>8} {'verdict':>11}")
    for t in sel:
        r = results[t]
        p = f"{r['ppl']:.3f}" if r.get("ppl") is not None else "-"
        dkey = next((k for k in r if k.startswith("delta")), None)
        d = f"{r[dkey]:+.3f}" if (dkey and r[dkey] is not None) else "-"
        print(f"{t:12} {p:>8} {d:>8} {r.get('verdict', '-'):>11}")

    os.makedirs(f"{SHARED}/outputs", exist_ok=True)
    outp = f"{SHARED}/outputs/ppl-{mode}.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"-> {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--targets", default="all")
    a = ap.parse_args()
    main(a.mode, a.n, a.targets)

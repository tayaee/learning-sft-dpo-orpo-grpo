# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Step 08: inference test vs ex06/ex07 vLLM server. Covers 4/5/8-bit labels.

Input: running server (ex06 or ex07). Next: ex09_cleanup.
Stdlib only. --quant all runs 3 labeled checks (4, 5, 8).

Examples:
    uv run ex08_test_vllm.py --model tayaee/my-model-gguf
    uv run ex08_test_vllm.py --model tayaee/my-model-gguf --quant 4
"""

import argparse
import json
import sys
import urllib.request

LABELS = {"4": "Q4_K_M", "5": "Q5_K_M", "8": "Q8_0"}


def parse_args():
    p = argparse.ArgumentParser(description="Step 08: test vLLM endpoint")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="tayaee/my-model-gguf")
    p.add_argument("--quant", default="all", choices=["4", "5", "8", "all"])
    p.add_argument("--prompt", default="What is 1+1?")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--timeout", type=int, default=60)
    return p.parse_args()


def chat(base_url, model, prompt, max_tokens, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    ).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        return True, data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return False, str(e)


def main():
    a = parse_args()
    quants = ["4", "5", "8"] if a.quant == "all" else [a.quant]
    print(f"Server: {a.base_url} model={a.model} quants={quants}")
    fails = 0
    for q in quants:
        ok, text = chat(a.base_url, a.model, a.prompt, a.max_tokens, a.timeout)
        tag = f"[{LABELS[q]}]"
        print(f"{'PASS' if ok else 'FAIL'} {tag} {text[:120]}")
        fails += not ok
    if fails:
        print(f"Done: {len(quants) - fails}/{len(quants)} pass. Next: ex09_cleanup")
        sys.exit(1)
    print(f"Done: all {len(quants)} pass. Next: ex09_cleanup")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Step 08: inference test vs ex06/ex07 server.")
        print("  uv run ex08_test_vllm.py --help")
        print("  ./ex08_test_vllm.sh  (needs running server)")
        print("No-op without args.")
        sys.exit(0)
    main()

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub[hf_transfer]",
# ]
# ///
"""Step 03: upload ex02 GGUF files to Hub. No GPU needed.

Input: gguf-output (ex02). Output: Hub repo files.
Next: ex04_infer_generate. Quant map: 4=Q4_K_M, 5=Q5_K_M, 8=Q8_0.

Examples:
    uv run ex03_upload_model.py --repo tayaee/my-model-gguf
    uv run ex03_upload_model.py --repo tayaee/my-model-gguf --quant 4
"""

import argparse
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

QUANT_FILE = {"4": "Q4_K_M", "5": "Q5_K_M", "8": "Q8_0"}


def parse_args():
    p = argparse.ArgumentParser(description="Step 03: upload GGUF 4/5/8-bit to Hub")
    p.add_argument("--gguf-dir", default="gguf-output")
    p.add_argument("--repo", default="tayaee/my-model-gguf")
    p.add_argument("--quant", default="all", choices=["4", "5", "8", "all"])
    return p.parse_args()


def main():
    a = parse_args()
    quants = ["4", "5", "8"] if a.quant == "all" else [a.quant]
    print(f"Dir: {a.gguf_dir} quants={quants} repo={a.repo}")

    d = Path(a.gguf_dir)
    if not d.is_dir():
        print(f"Missing dir: {a.gguf_dir} (run ex02 first).", file=sys.stderr)
        sys.exit(1)
    files = [f for f in d.glob("*.gguf") if any(q in f.name for q in [QUANT_FILE[x] for x in quants])]
    if not files:
        print(f"No GGUF files for {quants} in {a.gguf_dir}.", file=sys.stderr)
        sys.exit(1)
    print(f"Found: {[f.name for f in files]}")

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import HfApi, login

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("Warn: HF_TOKEN missing, upload may fail.")
    api = HfApi(token=token)
    api.create_repo(a.repo, exist_ok=True)
    for f in sorted(files):
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=a.repo)
        print(f"Uploaded: {f.name}")
    print(f"Repo: https://huggingface.co/{a.repo}")
    print("Done! Next: ex04_infer_generate")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Step 03: upload ex02 GGUF files to Hub. No GPU needed.")
        print("  uv run ex03_upload_model.py --help")
        print("  ./ex03_upload_model.sh")
        print("Needs gguf-output + HF_TOKEN. No-op without args.")
        sys.exit(0)
    main()

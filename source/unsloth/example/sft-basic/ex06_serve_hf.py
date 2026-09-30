# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "vllm",
# ]
# ///
"""Step 06: serve ex03 Hub repo with vLLM.

Input: Hub repo from ex03. Next: ex07_serve_local (or ex08 test).
--quant picks expected GGUF file (4=Q4_K_M, 5=Q5_K_M, 8=Q8_0).

Examples:
    uv run ex06_serve_hf.py --repo tayaee/my-model-gguf --quant 4 --run
"""

import argparse
import shutil
import subprocess
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

GGUF_FILE = {"4": "model-Q4_K_M.gguf", "5": "model-Q5_K_M.gguf", "8": "model-Q8_0.gguf"}


def parse_args():
    p = argparse.ArgumentParser(description="Step 06: serve HF repo with vLLM")
    p.add_argument("--repo", default="tayaee/my-model-gguf")
    p.add_argument("--quant", default="4", choices=["4", "5", "8"])
    p.add_argument("--filename", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu-mem", type=float, default=0.9)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--run", action="store_true", default=False)
    return p.parse_args()


def main():
    a = parse_args()
    fname = a.filename or GGUF_FILE[a.quant]
    print(f"Repo: {a.repo} quant={a.quant} file={fname}")
    print(f"Addr: {a.host}:{a.port} gpu_mem={a.gpu_mem} max_len={a.max_len}")
    cmd = [
        "vllm", "serve", a.repo,
        "--host", a.host, "--port", str(a.port),
        "--gpu-memory-utilization", str(a.gpu_mem),
        "--max-model-len", str(a.max_len),
    ]
    print("Cmd: " + " ".join(cmd))
    if not a.run:
        print("Dry run. Add --run to launch.")
        return
    if shutil.which("vllm") is None:
        print("vllm not found. Install vllm first.", file=sys.stderr)
        sys.exit(1)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Step 06: serve ex03 Hub repo with vLLM.")
        print("  uv run ex06_serve_hf.py --help")
        print("  ./ex06_serve_hf.sh")
        print("No-op without args.")
        sys.exit(0)
    main()

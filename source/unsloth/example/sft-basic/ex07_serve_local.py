# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "vllm",
# ]
# ///
"""Step 07: serve ex02 local output with vLLM.

Input: merged-output (ex02) or gguf-output + --quant. Next: ex08_test_vllm.

Examples:
    uv run ex07_serve_local.py --run
    uv run ex07_serve_local.py --gguf-dir ./gguf-output --quant 8 --run
"""

import argparse
import os
import shutil
import subprocess
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

GGUF_FILE = {"4": "model-Q4_K_M.gguf", "5": "model-Q5_K_M.gguf", "8": "model-Q8_0.gguf"}


def parse_args():
    p = argparse.ArgumentParser(description="Step 07: serve local output with vLLM")
    p.add_argument("--model-dir", default="merged-output")
    p.add_argument("--gguf-dir", default=None)
    p.add_argument("--quant", default="4", choices=["4", "5", "8"])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu-mem", type=float, default=0.9)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--run", action="store_true", default=False)
    return p.parse_args()


def main():
    a = parse_args()
    if a.gguf_dir:
        target = os.path.join(a.gguf_dir, GGUF_FILE[a.quant])
        src = f"gguf {a.gguf_dir} quant={a.quant}"
    else:
        target = a.model_dir
        src = f"merged {a.model_dir}"
    print(f"Src: {src} -> {target}")
    print(f"Addr: {a.host}:{a.port} gpu_mem={a.gpu_mem} max_len={a.max_len}")
    cmd = [
        "vllm", "serve", target,
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
    if not os.path.exists(target):
        print(f"Missing: {target} (run ex02 first).", file=sys.stderr)
        sys.exit(1)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Step 07: serve ex02 local output with vLLM.")
        print("  uv run ex07_serve_local.py --help")
        print("  ./ex07_serve_local.sh")
        print("No-op without args.")
        sys.exit(0)
    main()

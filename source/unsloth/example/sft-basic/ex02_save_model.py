# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "transformers==4.57.3",
#     "huggingface_hub[hf_transfer]",
#     "peft",
# ]
# ///
"""Step 02: save merged 16bit + GGUF 4/5/8-bit locally.

Input: ex01 adapter (unsloth-output). Output: merged-output, gguf-output.
Next: ex03_upload_model. Quant map: 4=q4_k_m, 5=q5_k_m, 8=q8_0.

Examples:
    uv run ex02_save_model.py
    uv run ex02_save_model.py --quant 4
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

QUANT_MAP = {"4": "q4_k_m", "5": "q5_k_m", "8": "q8_0"}


def parse_args():
    p = argparse.ArgumentParser(description="Step 02: save merged + GGUF 4/5/8-bit")
    p.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    p.add_argument("--adapter", default="unsloth-output")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--out-dir", default="gguf-output")
    p.add_argument("--merged-dir", default="merged-output")
    p.add_argument("--quant", default="all", choices=["4", "5", "8", "all"])
    p.add_argument("--skip-merged", action="store_true", default=False)
    return p.parse_args()


def main():
    a = parse_args()
    quants = list(QUANT_MAP.values()) if a.quant == "all" else [QUANT_MAP[a.quant]]
    print(f"Model: {a.model} + adapter {a.adapter}")
    print(f"Out: {a.out_dir} quants={quants} merged={'skip' if a.skip_merged else a.merged_dir}")

    import torch

    if not torch.cuda.is_available():
        print("No CUDA GPU. Save needs GPU.", file=sys.stderr)
        sys.exit(1)

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import login
    from unsloth import FastLanguageModel

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    print("Load model...")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=a.model,
        max_seq_length=a.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter)
        print(f"Adapter attached: {a.adapter}")

    if not a.skip_merged:
        print(f"Save merged 16bit: {a.merged_dir}")
        model.save_pretrained_merged(a.merged_dir, tokenizer=tok, save_method="merged_16bit")
        print("Merged saved.")

    print(f"Save GGUF: {a.out_dir} {quants}")
    model.save_pretrained_gguf(a.out_dir, tok, quantization_method=quants)
    print(f"Saved: {a.out_dir}/")
    print("Done! Next: ex03_upload_model")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Step 02: save merged + GGUF 4/5/8-bit from ex01 adapter.")
        print("  uv run ex02_save_model.py --help")
        print("  ./ex02_save_model.sh")
        print("Needs GPU + unsloth-output. No-op without args.")
        sys.exit(0)
    main()

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "datasets",
#     "trl==0.22.2",
#     "transformers==4.57.3",
#     "huggingface_hub[hf_transfer]",
#     "trackio",
#     "tensorboard",
# ]
# ///
"""Step 01: Unsloth SFT finetune. Output -> unsloth-output (used by ex02).

Prev: ex00_cleanup. Next: ex02_save_model.

Examples:
    uv run ex01_train_sft.py --dataset mlabonne/FineTome-100k --max-steps 60 \\
        --output-repo tayaee/my-model-test
"""

import argparse
import logging
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Step 01: Unsloth SFT finetune")
    p.add_argument("--base-model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    p.add_argument("--dataset", default="mlabonne/FineTome-100k")
    p.add_argument("--output-repo", default="tayaee/my-model-test")
    p.add_argument("--num-epochs", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--eval-split", type=float, default=0.0)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--save-local", default="unsloth-output")
    p.add_argument("--trackio-space", default=None)
    p.add_argument("--merge-model", action="store_true", default=False)
    return p.parse_args()


def main():
    a = parse_args()
    plan = f"{a.num_epochs} epoch(s)" if a.num_epochs else f"{a.max_steps} steps"
    print("Config:")
    print(f"  base: {a.base_model}")
    print(f"  dataset: {a.dataset} (samples: {a.num_samples})")
    print(f"  plan: {plan} eval_split={a.eval_split}")
    print(f"  batch: {a.batch_size}x{a.grad_accum} lr={a.lr} seq={a.max_seq_length}")
    print(f"  lora: r={a.lora_r} alpha={a.lora_alpha} seed={a.seed}")
    print(f"  out: {a.output_repo} local={a.save_local} merge={a.merge_model}")

    import torch

    if not torch.cuda.is_available():
        log.error("No CUDA GPU. Run on GPU machine or HF Jobs.")
        sys.exit(1)
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    if a.trackio_space:
        os.environ["TRACKIO_SPACE_ID"] = a.trackio_space

    from datasets import load_dataset
    from huggingface_hub import login
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import standardize_data_formats, train_on_responses_only

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        log.info("Hub login ok")
    else:
        log.warning("HF_TOKEN missing, upload may fail")

    print("Step 1/4: load model")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=a.base_model,
        max_seq_length=a.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=a.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
        lora_alpha=a.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=a.seed,
        use_rslora=False,
        loftq_config=None,
    )

    print("Step 2/4: load dataset")
    ds = load_dataset(a.dataset, split="train")
    if a.num_samples:
        ds = ds.select(range(min(a.num_samples, len(ds))))
    for col in ["messages", "conversations", "conversation"]:
        if col in ds.column_names and isinstance(ds[0][col], list):
            if col != "conversations":
                ds = ds.rename_column(col, "conversations")
            break
    ds = standardize_data_formats(ds)

    def fmt(ex):
        texts = tok.apply_chat_template(ex["conversations"], tokenize=False, add_generation_prompt=False)
        return {"text": [x.removeprefix(tok.bos_token) for x in texts]}

    ds = ds.map(fmt, batched=True)
    if a.eval_split > 0:
        s = ds.train_test_split(test_size=a.eval_split, seed=a.seed)
        train_ds, eval_ds = s["train"], s["test"]
        print(f"  train={len(train_ds)} eval={len(eval_ds)}")
    else:
        train_ds, eval_ds = ds, None
        print(f"  train={len(train_ds)} eval=none")

    print("Step 3/4: train")
    eff = a.batch_size * a.grad_accum
    steps_per_epoch = max(1, len(train_ds) // eff)
    if a.num_epochs:
        log_steps = max(1, steps_per_epoch // 10)
        save_steps = max(1, steps_per_epoch // 4)
    else:
        log_steps = max(1, a.max_steps // 20)
        save_steps = max(1, a.max_steps // 4)
    cfg = SFTConfig(
        output_dir=a.save_local,
        dataset_text_field="text",
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        warmup_steps=5,
        num_train_epochs=a.num_epochs if a.num_epochs else 1,
        max_steps=a.max_steps if a.max_steps else -1,
        learning_rate=a.lr,
        logging_steps=log_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=a.seed,
        max_length=a.max_seq_length,
        report_to=["tensorboard"] if not a.trackio_space else ["tensorboard", "trackio"],
        run_name="unsloth-sft",
        push_to_hub=True,
        hub_model_id=a.output_repo,
        save_steps=save_steps,
        save_total_limit=3,
    )
    if eval_ds is not None:
        cfg.eval_strategy = "epoch" if a.num_epochs else "steps"
        if not a.num_epochs:
            cfg.eval_steps = max(1, a.max_steps // 5)
    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=train_ds, eval_dataset=eval_ds, args=cfg)
    trainer = train_on_responses_only(
        trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n"
    )
    result = trainer.train()
    loss = result.metrics.get("train_loss")
    if loss:
        print(f"Train loss: {loss:.4f}")
    if eval_ds is not None:
        try:
            ev = trainer.evaluate()
            if ev.get("eval_loss"):
                print(f"Eval loss: {ev['eval_loss']:.4f}")
        except Exception as e:
            print(f"Eval skip: {e}")

    print("Step 4/4: save")
    if a.merge_model:
        model.push_to_hub_merged(a.output_repo, tokenizer=tok, save_method="merged_16bit")
        print(f"Pushed merged: https://huggingface.co/{a.output_repo}")
    else:
        model.save_pretrained(a.save_local)
        tok.save_pretrained(a.save_local)
        print(f"Saved local: {a.save_local}/")
        model.push_to_hub(a.output_repo, tokenizer=tok)
        print(f"Pushed adapter: https://huggingface.co/{a.output_repo}")
    print("Done! Next: ex02_save_model")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Step 01: SFT finetune. Defaults run a quick 60-step test.")
        print("  uv run ex01_train_sft.py --help")
        print("  ./ex01_train_sft.sh")
        print("Needs GPU. No-op without args.")
        sys.exit(0)
    main()

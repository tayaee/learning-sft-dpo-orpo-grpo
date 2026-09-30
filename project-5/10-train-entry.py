#!/usr/bin/env python3
"""10-train-entry.py — Stage 1/3 공용 학습 진입점 (TRL SFTTrainer).

원본: main.py + trainer436 + train_basic/FFT/QLoRA.sh → 2026 스택으로 이식.
- 데이터: {"source":[...], "target":[...]} jsonl → 학습 프롬프트 템플릿
  (원본 TokenizeProcessor/quant 노트북과 동일 문구).
- 토크나이저: "[PAD]"+ "<mask>" 추가 후 신규 임베딩 평균 초기화 (main.py resize()).
- QLoRA(--peft): bnb 4bit nf4 double_quant + LoRA r16/a32/drop0.05.
- 전략: single=플레인, ddp=torchrun DDP(자동), fsdp=full_shard auto_wrap.
- 강의 기본값: lr 1e-5, cosine, warmup 0.03, micro 2, tok/tgt 512, seed 1,
  mask_rate 0 (masking off).
- DGX 적응: gradient_checkpointing=True (원본 False), bf16, tf32 허용.

사용: torchrun ... 10-train-entry.py --model ... --train ... --out ... [--peft]
검증: python3 10-train-entry.py --train <jsonl> --out /tmp/x --dry_run
"""
import argparse
import json
import os
import random

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context.\n"
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\nGive a response as the assistant with the input conversation history\n\n"
    "### Response:\n{response}"
)
MASK_TOKEN = "<mask>"
PAD_TOKEN = "[PAD]"


def first(v):
    return v[0] if isinstance(v, list) else v


def load_texts(path):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(PROMPT_TEMPLATE.format(
                instruction=first(row["source"]), response=first(row["target"])))
    return texts


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        import numpy as np
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # dry_run 환경 (torch 없음)에서는 stdlib만


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--train", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--mode", default="mini")
    p.add_argument("--peft", action="store_true", help="QLoRA 경로 (31에서 사용)")
    p.add_argument("--strategy", default="single", choices=["single", "ddp", "fsdp"])
    p.add_argument("--accum", type=int, default=32, help="grad accum (effective batch 64 유지용)")
    p.add_argument("--micro", type=int, default=2)
    p.add_argument("--max_len", type=int, default=1024, help="tok 512 + tgt 512")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--mask_rate", type=float, default=0.0, help="원본 max_mask_rate (기본 0=off)")
    p.add_argument("--dry_run", action="store_true", help="데이터 포맷만 확인 후 종료")
    return p.parse_args()


def main():
    a = parse_args()
    set_seed(a.seed)
    texts = load_texts(a.train)
    print(f"[p5][{a.mode}][{a.strategy}] n={len(texts)} sample_chars={len(texts[0])}")
    print("---- sample ----")
    print(texts[0][:600])
    print("----------------")
    if a.dry_run:
        return

    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers import DataCollatorForLanguageModeling
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    added = []
    if tok.pad_token is None:
        added.append(PAD_TOKEN)
    added.append(MASK_TOKEN)
    n_new = tok.add_special_tokens({"additional_special_tokens": added})
    if tok.pad_token is None:
        tok.pad_token = PAD_TOKEN
    tok.padding_side = "right"

    dtype = torch.bfloat16
    if a.peft:
        if a.strategy == "fsdp":
            print("WARN: 원본 강의는 FSDP+QLoRA 불가 → DDP 동작. 학습용으로 계속.")
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_compute_dtype=dtype,
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type="nf4")
        model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb,
                                                     torch_dtype=dtype)
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        peft_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                              r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                              "gate_proj", "down_proj", "up_proj"],
                              modules_to_save=["embed_tokens", "lm_head"])
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype)

    # main.py resize(): 신규 토큰 임베딩을 기존 평균으로 초기화
    model.resize_token_embeddings(len(tok))
    if n_new > 0:
        with torch.no_grad():
            for emb in {model.get_input_embeddings(), model.get_output_embeddings()}:
                w = emb.weight.data
                w[-n_new:] = w[:-n_new].mean(dim=0, keepdim=True)

    ds = Dataset.from_list([{"text": t} for t in texts])

    if a.mask_rate > 0:
        mask_id = tok.convert_tokens_to_ids(MASK_TOKEN)
        special = set(tok.all_special_ids)

        def collator(features):
            batch = tok.pad({"input_ids": [f["input_ids"] for f in features]},
                            padding=True, return_tensors="pt")
            labels = batch["input_ids"].clone()
            prob = torch.rand(batch["input_ids"].shape)
            is_special = torch.zeros_like(prob, dtype=torch.bool)
            for s in special:
                is_special |= batch["input_ids"] == s
            do_mask = (prob < a.mask_rate) & ~is_special
            batch["input_ids"][do_mask] = mask_id
            batch["labels"] = labels
            return batch
    else:
        collator = DataCollatorForLanguageModeling(tok, mlm=False)

    fsdp = "full_shard auto_wrap" if a.strategy == "fsdp" else ""
    fsdp_cfg = {"transformer_layer_cls_to_wrap": "LlamaDecoderLayer"} if a.strategy == "fsdp" else {}
    cfg = SFTConfig(
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.micro,
        gradient_accumulation_steps=a.accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=a.warmup,
        logging_steps=200, save_strategy="epoch", save_total_limit=2,
        bf16=True, seed=a.seed, dataset_text_field="text",
        max_length=a.max_len, packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False} if a.peft else {},
        fsdp=fsdp, fsdp_config=fsdp_cfg, report_to="none",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, data_collator=collator)
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    print(f"[p5] saved → {a.out}")


if __name__ == "__main__":
    main()

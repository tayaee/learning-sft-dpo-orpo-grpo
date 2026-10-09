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

from io_common import clean_tmp, commit_dir, tmp_path

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
            texts.append(
                PROMPT_TEMPLATE.format(
                    instruction=first(row["source"]), response=first(row["target"])
                )
            )
    return texts


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # dry_run 환경 (torch 없음)에서는 stdlib만


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Llama-3.2-1B")
    p.add_argument("--train", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--mode", default="mini")
    p.add_argument("--peft", action="store_true", help="QLoRA 경로 (31에서 사용)")
    p.add_argument("--strategy", default="single", choices=["single", "ddp", "fsdp"])
    p.add_argument(
        "--accum", type=int, default=32, help="grad accum (effective batch 64 유지용)"
    )
    p.add_argument("--micro", type=int, default=2)
    p.add_argument("--max_len", type=int, default=1024, help="tok 512 + tgt 512")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--mask_rate", type=float, default=0.0, help="원본 max_mask_rate (기본 0=off)"
    )
    p.add_argument(
        "--max_rows", type=int, default=0, help="학습 행 상한 (0=전체, mini 경량화용)"
    )
    p.add_argument("--dry_run", action="store_true", help="데이터 포맷만 확인 후 종료")
    return p.parse_args()


def _save_fsdp_dcp(fsdp_model, model_cfg, tok, out_dir, dtype):
    """2노드 FSDP 전용 저장: sharded DCP + rank0 디스크 consolidation.

    full-model gather를 fabric에 던지지 않는다 (CX7 RoCE 버스트 손실로
    rank0 최종 저장이 deterministic하게 실패). fabric을 타는 것은 barrier
    (수 바이트)뿐이며, 노드 속도 차는 barrier가 흡수한다.
    shard 파일은 out_dir/shards/ 에 남긴다 (재조립·디버그용).
    """
    import torch.distributed as dist
    from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
    from torch.distributed.checkpoint import load as dcp_load
    from torch.distributed.checkpoint import save as dcp_save
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    rank = dist.get_rank()
    shard_dir = os.path.join(out_dir, "shards")
    if rank == 0:
        os.makedirs(
            shard_dir, exist_ok=True
        )  # 생성은 rank0만 (공유FS makedirs race 회피)
    dist.barrier()

    # 1) 각 rank 자기 shard만 디스크에 기록 (대형 fabric 전송 없음).
    # get_model_state_dict은 FSDP 래핑을 자동 감지해 sharded로 반환한다
    # (FSDP.state_dict_type 컨텍스트는 deprecated).
    local_sd = get_model_state_dict(fsdp_model)
    dcp_save(local_sd, storage_writer=FileSystemWriter(shard_dir))
    dist.barrier()  # 전 rank 쓰기 완료 후 rank0 읽기 시작

    # 2) rank0만: 디스크 shard → CPU full 모델 조립 (fabric 무관).
    # no_dist=True: load 플래너의 collective를 끄고 단일 프로세스로 읽음.
    if rank == 0:
        from transformers import AutoModelForCausalLM

        full = AutoModelForCausalLM.from_config(model_cfg, dtype=dtype)
        full.resize_token_embeddings(len(tok))
        full_sd = get_model_state_dict(
            full, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        dcp_load(full_sd, storage_reader=FileSystemReader(shard_dir), no_dist=True)
        missing, _ = set_model_state_dict(
            full, full_sd, options=StateDictOptions(full_state_dict=True)
        )
        assert not missing, f"DCP consolidate missing keys: {sorted(missing)[:5]}"
        tmp = clean_tmp(tmp_path(out_dir))
        full.save_pretrained(tmp, safe_serialization=True)
        tok.save_pretrained(tmp)
        commit_dir(tmp, out_dir)
    dist.barrier()


def main():
    a = parse_args()
    set_seed(a.seed)
    texts = load_texts(a.train)
    if a.max_rows > 0:
        texts = texts[: a.max_rows]
    print(f"[{a.mode}][{a.strategy}] n={len(texts)} sample_chars={len(texts[0])}")
    print("---- sample ----")
    print(texts[0][:600])
    print("----------------")
    if a.dry_run:
        return

    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
        # Compat shim: peft<=0.19는 임포트 시 BloomPreTrainedModel을 참조하지만
        # transformers 5.x에서 삭제됨. Bloom prefix-tuning 전용이라 LoRA 경로에
        # 영향 없음 (hasattr 체크에서 False → 매핑 스킵).
        try:
            from transformers import BloomPreTrainedModel  # noqa
        except ImportError:
            import transformers as _tf

            class _BloomStub:
                pass

            _tf.BloomPreTrainedModel = _BloomStub
        if a.strategy == "fsdp":
            print("WARN: FSDP+QLoRA unsupported, DDP fallback (training only).")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            a.model, quantization_config=bnb, torch_dtype=dtype
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype)

    # main.py resize_at_begin: PEFT 래핑보다 먼저 수행 (래핑 후 resize 불가).
    # 신규 토큰 임베딩을 기존 평균으로 초기화.
    model.resize_token_embeddings(len(tok))
    if n_new > 0:
        with torch.no_grad():
            for emb in {model.get_input_embeddings(), model.get_output_embeddings()}:
                w = emb.weight.data
                w[-n_new:] = w[:-n_new].mean(dim=0, keepdim=True)

    if a.peft:
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ],
            modules_to_save=["embed_tokens", "lm_head"],
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    ds = Dataset.from_list([{"text": t} for t in texts])

    # trl은 map 배치를 먼저 패딩한 텐서(input_ids/attention_mask/labels 포함)로
    # collator를 호출하므로, 배치는 여기서 최종 max에 맞춰 다시 패딩한다.
    # mask_rate>0일 때만 <mask> 치환을 적용한다 (기본 0=강의와 동일하게 off).
    mask_id = tok.convert_tokens_to_ids(MASK_TOKEN)
    special = set(tok.all_special_ids)

    def collator(features):
        ids = [torch.as_tensor(f["input_ids"]) for f in features]
        pad = torch.nn.utils.rnn.pad_sequence
        input_ids = pad(ids, batch_first=True, padding_value=tok.pad_token_id)
        if "attention_mask" in features[0]:
            ams = [torch.as_tensor(f["attention_mask"]) for f in features]
            attention_mask = pad(ams, batch_first=True, padding_value=0)
        else:
            attention_mask = (input_ids != tok.pad_token_id).long()
        if "labels" in features[0]:
            lbs = [torch.as_tensor(f["labels"]) for f in features]
            labels = pad(lbs, batch_first=True, padding_value=-100)
        else:
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
        if a.mask_rate > 0:
            prob = torch.rand(input_ids.shape)
            is_special = torch.zeros_like(prob, dtype=torch.bool)
            for s in special:
                is_special |= input_ids == s
            do_mask = (prob < a.mask_rate) & ~is_special & (labels != -100)
            input_ids[do_mask] = mask_id
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    fsdp = "full_shard auto_wrap" if a.strategy == "fsdp" else ""
    fsdp_cfg = (
        {"transformer_layer_cls_to_wrap": "LlamaDecoderLayer"}
        if a.strategy == "fsdp"
        else {}
    )
    # 2노드 FSDP는 중간 체크포인트(full gather)를 건너뛰고 최종 DCP 저장만 수행.
    # _dcp_2node에서도 P5_DCP_SAVE=0이면 기존 경로(trainer.save_model) 사용.
    _world = int(os.environ.get("WORLD_SIZE", "1"))
    _dcp_2node = (
        a.strategy == "fsdp"
        and _world > 1
        and os.environ.get("P5_DCP_SAVE", "1") == "1"
    )
    cfg = SFTConfig(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.micro,
        gradient_accumulation_steps=a.accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=a.warmup,
        logging_steps=200,
        save_strategy="no" if _dcp_2node else "epoch",
        save_total_limit=2,
        bf16=True,
        seed=a.seed,
        dataset_text_field="text",
        max_length=a.max_len,
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False} if a.peft else {},
        fsdp=fsdp,
        fsdp_config=fsdp_cfg,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        data_collator=collator,
    )
    # TRL _save_checkpoint은 매번 create_model_card → trackio import를 타는데,
    # huggingface_hub>=1(CommitOperationAdd 삭제)과 충돌해 ImportError로 죽는다.
    # 모델 카드는 파이프라인 산출물이 아니므로 no-op (버전 조합과 무관하게 동작).
    trainer.create_model_card = lambda *a, **k: {}
    trainer.train()
    if _dcp_2node:
        _save_fsdp_dcp(trainer.model, model.config, tok, a.out, dtype)
    else:
        tmp = clean_tmp(tmp_path(a.out))
        trainer.save_model(tmp)
        tok.save_pretrained(tmp)
        commit_dir(tmp, a.out)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()

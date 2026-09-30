# Unsloth SFT basic

Ordered pipeline `ex00`–`ex09`. Each step uses the prev step output.
`unsloth_sft_example.py` kept as-is (origin). HF user: `tayaee`.
Quant: 4=Q4_K_M, 5=Q5_K_M, 8=Q8_0.

All `.py` print help with no args (exit 0, VSCode safe).
All `.sh` carry chained defaults, no required env.

```bash
./ex00_cleanup.sh
./ex01_train_sft.sh        # GPU, quick 60-step default
./ex02_save_model.sh       # GPU, uses unsloth-output
./ex03_upload_model.sh     # needs HF_TOKEN, uses gguf-output
./ex04_infer_generate.sh   # CPU, uses gguf-output Q4
./ex05_infer_stream.sh     # CPU, uses gguf-output Q5
./ex06_serve_hf.sh &       # or ./ex07_serve_local.sh &
./ex08_test_vllm.sh        # needs running server
./ex09_cleanup.sh
```

Env overrides: `MODEL`, `ADAPTER`, `REPO`, `QUANT` (4/5/8/all),
`GGUF_DIR`, `OUT_DIR`, `MERGED_DIR`, `MODEL_DIR`, `PROMPT`, `PORT`.

## ex00_cleanup — clean before run

- Input: local dirs (`unsloth-output`, `merged-output`, `gguf-output`).
- Process: delete each dir if present, else skip.
- Output: clean work dir. Next: `ex01_train_sft`.

## ex01_train_sft — finetune

- Input: base `LiquidAI/LFM2.5-1.2B-Instruct`, dataset `mlabonne/FineTome-100k`
  (1000 samples, 60 steps default). Needs GPU.
- Process: LoRA SFT via Unsloth + `SFTTrainer`, responses-only mask.
- Output: adapter in `unsloth-output/` + Hub `tayaee/my-model-test`.
  Next: `ex02_save_model`.

## ex02_save_model — save merged + GGUF

- Input: base model + adapter `unsloth-output/` (ex01). Needs GPU.
- Process: merge LoRA to 16bit, convert to GGUF per `--quant`.
- Output: `merged-output/` (16bit) + `gguf-output/` (Q4_K_M/Q5_K_M/Q8_0).
  Next: `ex03_upload_model`.

## ex03_upload_model — upload GGUF

- Input: `gguf-output/*.gguf` (ex02), `--quant` filter. No GPU.
  Needs `HF_TOKEN`.
- Process: create Hub repo if missing, upload matching files.
- Output: Hub `tayaee/my-model-gguf` with 4/5/8-bit files.
  Next: `ex04_infer_generate`.

## ex04_infer_generate — single-shot inference

- Input: `gguf-output/model-Q4_K_M.gguf` (ex02) + prompt. CPU ok.
  Empty `GGUF_DIR` switches to Unsloth GPU mode (`--no-gguf`).
- Process: one `model.generate()` call (or llama-cpp for GGUF).
- Output: printed assistant reply. Next: `ex05_infer_stream`.

## ex05_infer_stream — streaming inference

- Input: `gguf-output/model-Q5_K_M.gguf` (ex02) + prompt. CPU ok.
  Empty `GGUF_DIR` switches to Unsloth GPU mode.
- Process: `model.generate()` with `TextStreamer` (tokens as made).
- Output: streamed assistant reply. Next: `ex06_serve_hf`.

## ex06_serve_hf — vLLM serve from Hub

- Input: Hub repo `tayaee/my-model-gguf` (ex03). `--quant` picks
  the expected GGUF filename for info.
- Process: launch `vllm serve <repo>` on `0.0.0.0:8000`.
- Output: running OpenAI-compatible server. Next: `ex08_test_vllm`
  (or use `ex07` instead).

## ex07_serve_local — vLLM serve local

- Input: `merged-output/` (ex02), or `gguf-output` + `--quant` file.
- Process: launch `vllm serve <local path>` on `0.0.0.0:8000`.
- Output: running OpenAI-compatible server. Next: `ex08_test_vllm`.

## ex08_test_vllm — inference test

- Input: running server from ex06/ex07 (`http://localhost:8000/v1`),
  model `tayaee/my-model-gguf`. Stdlib only.
- Process: POST `/chat/completions`, one check per `--quant`
  (`all` = 4, 5, 8 labels).
- Output: `PASS`/`FAIL` per quant, exit 1 on any fail.
  Next: `ex09_cleanup`.

## ex09_cleanup — clean after run

- Input: local dirs (`unsloth-output`, `merged-output`, `gguf-output`).
- Process: delete each dir if present, else skip.
- Output: clean work dir. Pipeline done.

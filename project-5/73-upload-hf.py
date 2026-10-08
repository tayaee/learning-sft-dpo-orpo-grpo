#!/usr/bin/env python3
"""73-upload-hf.py — Stage 7a. 로컬 산출물을 HF Hub(tayaee/*)에 업로드.
HF 산출물은 Stage 8b 서빙의 입력이 된다 (local 경로로도 서빙 가능).

  uv run 73-upload-hf.py --mode mini|full --targets all|fft,qlora,fft-gptq,...,qlora-gguf [--dry-run]
전제: hf auth login (쓰기 권한 토큰).
"""
import argparse
import os

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")
HF_ID = os.environ.get("HF_ID", "tayaee")

# target → (로컬 디렉토리 템플릿, HF repo 템플릿, 업로드 방식)
# 양자화 총 8종: fft×4 + qlora-merged×4. 구이름(gptq/awq/fp8/gguf)은 fft-* 별칭.
TARGETS = {
    "fft": ("synthetic-fft-{m}-single", "p5-1B-math-fft-{m}", "folder"),
    "qlora": ("synthetic-qlora-{m}-single-merged", "p5-1B-math-qlora-{m}", "folder"),
    "fft-gptq": ("synthetic-fft-{m}-single-gptq", "p5-1B-math-fft-gptq-{m}", "folder"),
    "fft-awq": ("synthetic-fft-{m}-single-awq", "p5-1B-math-fft-awq-{m}", "folder"),
    "fft-fp8": ("synthetic-fft-{m}-single-fp8", "p5-1B-math-fft-fp8-{m}", "folder"),
    "fft-gguf": ("synthetic-fft-{m}-single-gguf/model-q8_0.gguf",
             "p5-1B-math-fft-gguf-{m}", "gguf"),
    "qlora-gptq": ("synthetic-qlora-{m}-single-merged-gptq", "p5-1B-math-qlora-gptq-{m}", "folder"),
    "qlora-awq": ("synthetic-qlora-{m}-single-merged-awq", "p5-1B-math-qlora-awq-{m}", "folder"),
    "qlora-fp8": ("synthetic-qlora-{m}-single-merged-fp8", "p5-1B-math-qlora-fp8-{m}", "folder"),
    "qlora-gguf": ("synthetic-qlora-{m}-single-merged-gguf/model-q8_0.gguf",
             "p5-1B-math-qlora-gguf-{m}", "gguf"),
    "gptq": ("synthetic-fft-{m}-single-gptq", "p5-1B-math-fft-gptq-{m}", "folder"),
    "awq": ("synthetic-fft-{m}-single-awq", "p5-1B-math-fft-awq-{m}", "folder"),
    "fp8": ("synthetic-fft-{m}-single-fp8", "p5-1B-math-fft-fp8-{m}", "folder"),
    "gguf": ("synthetic-fft-{m}-single-gguf/model-q8_0.gguf",
             "p5-1B-math-fft-gguf-{m}", "gguf"),
}


def main(mode: str, targets: str, dry_run: bool):
    sel = list(TARGETS) if targets == "all" else targets.split(",")
    plan = []
    for t in sel:
        local_t, repo_t, kind = TARGETS[t]
        plan.append((t, f"{SHARED}/models/" + local_t.format(m=mode),
                     f"{HF_ID}/" + repo_t.format(m=mode), kind))
    for t, local, repo, kind in plan:
        print(f"[{t}] {local} -> {repo} ({kind})")
        if dry_run:
            continue
        if not os.path.exists(local):
            print(f"  SKIP: 로컬 산출물 없음 (먼저 해당 stage 실행)")
            continue
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo, exist_ok=True)
        if kind == "folder":
            api.upload_folder(repo_id=repo, folder_path=local)
        else:  # gguf 단일 파일 (tokenizer는 폴더 타깃 repo에 동봉 전제)
            api.upload_file(repo_id=repo, path_or_fileobj=local,
                            path_in_repo=os.path.basename(local))
        print(f"  OK: https://huggingface.co/{repo}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument("--targets", default="all")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(a.mode, a.targets, a.dry_run)

#!/usr/bin/env python3
"""74-upload-hf.py — Stage 7a. 로컬 산출물을 HF Hub(tayaee/*)에 업로드.
HF 산출물은 Stage 8b 서빙의 입력이 된다 (local 경로로도 서빙 가능).

  uv run 74-upload-hf.py --mode mini|full --targets all|fft,qlora,fft-gptq,...,fft-gguf-q4_k_m [--dry-run]
전제: hf auth login (쓰기 권한 토큰).
"""

import argparse
import os

SHARED = os.environ.get("P5_SHARED", "/rosenas/data/AIML/project-5-shared")
HF_ID = os.environ.get("HF_ID", "tayaee")
# repo명에 베이스 모델명을 포함한다. BASE_MODEL 뒷부분만 사용하므로
# 미러(unsloth/…)와 원본(meta-llama/…)이 같은 repo명으로 수렴한다.
# 예: unsloth/Llama-3.2-1B → tayaee/p5-Llama-3.2-1B-math-fft-mini
SLUG = os.environ.get("BASE_MODEL", "unsloth/Llama-3.2-1B").split("/")[-1]

# target → (로컬 디렉토리 템플릿, HF repo 템플릿, 업로드 방식)
# 양자화 총 16종: HF 폴더 6종(gptq/awq/fp8 × fft/qlora) + GGUF 10종(fft/qlora × 5).
# 구이름(gptq/awq/fp8/gguf)은 fft-* 별칭. fft-gguf/qlora-gguf/gguf 키는 q8_0 호환용으로 유지.
TARGETS = {
    "fft": ("synthetic-fft-{m}-single", "{slug}-math-fft-{m}", "folder"),
    "qlora": ("synthetic-qlora-{m}-single-merged", "{slug}-math-qlora-{m}", "folder"),
    "fft-gptq": ("synthetic-fft-{m}-single-gptq", "{slug}-math-fft-gptq-{m}", "folder"),
    "fft-awq": ("synthetic-fft-{m}-single-awq", "{slug}-math-fft-awq-{m}", "folder"),
    "fft-fp8": ("synthetic-fft-{m}-single-fp8", "{slug}-math-fft-fp8-{m}", "folder"),
    "fft-gguf": (
        "synthetic-fft-{m}-single-gguf/model-q8_0.gguf",
        "{slug}-math-fft-gguf-{m}",
        "gguf",
    ),
    "qlora-gptq": (
        "synthetic-qlora-{m}-single-merged-gptq",
        "{slug}-math-qlora-gptq-{m}",
        "folder",
    ),
    "qlora-awq": (
        "synthetic-qlora-{m}-single-merged-awq",
        "{slug}-math-qlora-awq-{m}",
        "folder",
    ),
    "qlora-fp8": (
        "synthetic-qlora-{m}-single-merged-fp8",
        "{slug}-math-qlora-fp8-{m}",
        "folder",
    ),
    "qlora-gguf": (
        "synthetic-qlora-{m}-single-merged-gguf/model-q8_0.gguf",
        "{slug}-math-qlora-gguf-{m}",
        "gguf",
    ),
    "gptq": ("synthetic-fft-{m}-single-gptq", "{slug}-math-fft-gptq-{m}", "folder"),
    "awq": ("synthetic-fft-{m}-single-awq", "{slug}-math-fft-awq-{m}", "folder"),
    "fp8": ("synthetic-fft-{m}-single-fp8", "{slug}-math-fft-fp8-{m}", "folder"),
    "gguf": (
        "synthetic-fft-{m}-single-gguf/model-q8_0.gguf",
        "{slug}-math-fft-gguf-{m}",
        "gguf",
    ),
}

# GGUF 5종 변종 (42/52 산출물): fft/qlora × {q8_0,q6_k,q5_k_m,q4_k_m,q3_k_m}.
# gguf 단일파일 업로드 분기 그대로 사용 (path_in_repo=파일명).
GQUANTS = ["q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m"]
for _base, _local in (
    ("fft", "synthetic-fft-{m}-single"),
    ("qlora", "synthetic-qlora-{m}-single-merged"),
):
    for _q in GQUANTS:
        TARGETS[f"{_base}-gguf-{_q}"] = (
            f"{_local}-gguf/model-{_q}.gguf",
            "{slug}-math-" + f"{_base}-gguf-{_q}" + "-{m}",
            "gguf",
        )


def main(mode: str, targets: str, dry_run: bool):
    sel = list(TARGETS) if targets == "all" else targets.split(",")
    plan = []
    for t in sel:
        local_t, repo_t, kind = TARGETS[t]
        plan.append(
            (
                t,
                f"{SHARED}/models/" + local_t.format(m=mode),
                f"{HF_ID}/" + repo_t.format(m=mode, slug=SLUG),
                kind,
            )
        )
    for t, local, repo, kind in plan:
        print(f"[{t}] {local} -> {repo} ({kind})")
        if dry_run:
            continue
        if not os.path.exists(local):
            print(f"  SKIP: no local output (run that stage first)")
            continue
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo, exist_ok=True)
        if kind == "folder":
            api.upload_folder(repo_id=repo, folder_path=local)
        else:  # gguf 단일 파일 (tokenizer는 폴더 타깃 repo에 동봉 전제)
            api.upload_file(
                repo_id=repo,
                path_or_fileobj=local,
                path_in_repo=os.path.basename(local),
            )
        print(f"  OK: https://huggingface.co/{repo}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mini", choices=["mini", "full"])
    ap.add_argument("--targets", default="all")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(a.mode, a.targets, a.dry_run)

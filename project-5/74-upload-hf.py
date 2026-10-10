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

# target → (로컬 디렉토리 템플릿, 업로드 방식). {m}=mode, {s}=strat.
# 양자화 총 16종: HF 폴더 6종(gptq/awq/fp8 × fft/qlora) + GGUF 10종(fft/qlora × 5).
# 구이름(gptq/awq/fp8/gguf)은 fft-* 별칭. fft-gguf/qlora-gguf 키는 q8_0 호환용으로 유지.
TARGETS = {
    "fft": ("synthetic-fft-{m}-{s}", "folder"),
    "qlora": ("synthetic-qlora-{m}-{s}-merged", "folder"),
    "fft-gptq": ("synthetic-fft-{m}-{s}-gptq", "folder"),
    "fft-awq": ("synthetic-fft-{m}-{s}-awq", "folder"),
    "fft-fp8": ("synthetic-fft-{m}-{s}-fp8", "folder"),
    "fft-gguf": (
        "synthetic-fft-{m}-{s}-gguf/model-q8_0.gguf",
        "gguf",
    ),
    "qlora-gptq": (
        "synthetic-qlora-{m}-{s}-merged-gptq",
        "folder",
    ),
    "qlora-awq": (
        "synthetic-qlora-{m}-{s}-merged-awq",
        "folder",
    ),
    "qlora-fp8": (
        "synthetic-qlora-{m}-{s}-merged-fp8",
        "folder",
    ),
    "qlora-gguf": (
        "synthetic-qlora-{m}-{s}-merged-gguf/model-q8_0.gguf",
        "gguf",
    ),
}


# HF repo명 규칙 (models-tree.txt 설계 결정 3):
# single 동결 ({slug}-math-{target}-{m}, full이면 -{m} 생략),
# ddp/fsdp는 -{s} 삽입, full은 무표기.
def repo_name(target: str, strat: str, mode: str) -> str:
    r = f"{SLUG}-math-{target}"
    if strat != "single":
        r += f"-{strat}"
    if mode != "full":
        r += f"-{mode}"
    return f"{HF_ID}/{r}"


# 구이름 별칭 → 정식 키 (같은 repo에 업로드, 기존 동작 유지).
ALIAS = {"gptq": "fft-gptq", "awq": "fft-awq", "fp8": "fft-fp8", "gguf": "fft-gguf"}

# GGUF 5종 변종 (42/52 산출물): fft/qlora × {q8_0,q6_k,q5_k_m,q4_k_m,q3_k_m}.
# gguf 단일파일 업로드 분기 그대로 사용 (path_in_repo=파일명).
GQUANTS = ["q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m"]
for _base, _local in (
    ("fft", "synthetic-fft-{m}-{s}"),
    ("qlora", "synthetic-qlora-{m}-{s}-merged"),
):
    for _q in GQUANTS:
        TARGETS[f"{_base}-gguf-{_q}"] = (
            f"{_local}-gguf/model-{_q}.gguf",
            "gguf",
        )


def main(mode: str, strat: str, targets: str, dry_run: bool):
    sel = list(TARGETS) if targets == "all" else targets.split(",")
    plan = []
    for t in sel:
        key = ALIAS.get(t, t)
        local_t, kind = TARGETS[key]
        plan.append(
            (
                t,
                f"{SHARED}/models/" + local_t.format(m=mode, s=strat),
                repo_name(key, strat, mode),
                kind,
            )
        )
    for t, local, repo, kind in plan:
        print(f"[{t}] {local} -> {repo} ({kind})")
        if dry_run:
            continue
        if not os.path.exists(local):
            print("  SKIP: no local output (run that stage first)")
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
    ap.add_argument(
        "--strat",
        default=os.environ.get("STRAT", "single"),
        choices=["single", "ddp", "fsdp"],
    )
    ap.add_argument("--targets", default="all")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(a.mode, a.strat, a.targets, a.dry_run)

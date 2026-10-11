#!/usr/bin/env python3
"""check_report_to.py — train job에 report_to를 주기 전 미리 체크.

10-train-entry.py와 동일한 CSV 파싱을 적용하고, 각 연동 백엔드 패키지의
실제 설치 여부(find_spec, import 없이 감지)를 기준으로 Trainer가 인식하게
될 report_to, 활성/비활성, 미설치 시 필요한 패키지, 보고서 저장 위치를 보여준다.
(transformers 본체는 import하지 않는다 — Trainer와 동일한 is_*_available()
기준과 동등하면서도 무거운 의존성 체인을 타지 않기 위함.)

사용 (project-5 디렉토리에서):
    uv run --no-sync check_report_to.py --report_to all
    uv run --no-sync check_report_to.py --report_to tensorboard,mlflow,trackio,dvclive,wandb
    uv run --no-sync check_report_to.py --report_to none
    uv run --no-sync check_report_to.py --report_to all --out /path/to/model-out

종료 코드: 0=학습 시 로그 남음(요청 전체 활성), 1=로그 안 남거나 문제 있음.
"""

import argparse
import importlib.metadata
import importlib.util
import os
import sys


# transformers INTEGRATION_TO_CALLBACK 키(5.x)와 1:1 대응.
# pip: 설치 명령용. import: is_*_available()가 감지하는 모듈명.
KNOWN = {
    "tensorboard": {
        "pip": "tensorboard",
        "import": "tensorboard",
        "env": ["TENSORBOARD_LOGGING_DIR"],
        "out": lambda out: os.getenv("TENSORBOARD_LOGGING_DIR")
        or (os.path.join(out, "runs") if out else "$OUT/runs"),
        "note": "로컬 이벤트 파일. rank0만 기록.",
    },
    "mlflow": {
        "pip": "mlflow",
        "import": "mlflow",
        "env": ["MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME"],
        "out": lambda out: os.getenv("MLFLOW_TRACKING_URI") or "./mlruns",
        "note": "미지정 시 실행 cwd의 ./mlruns. 비교용은 절대경로 권장.",
    },
    "trackio": {
        "pip": "trackio>=0.21.1",
        "import": "trackio",
        "env": [],
        "out": lambda out: "./trackio/ (project=huggingface)",
        "note": "로그인 불필요(로컬). rank0만 기록.",
    },
    "dvclive": {
        "pip": "dvclive",
        "import": "dvclive",
        "env": ["HF_DVCLIVE_LOG_MODEL"],
        "out": lambda out: "./dvclive/ (metrics.tsv, params.yaml)",
        "note": "cwd 기준. rank0만 기록.",
    },
    "wandb": {
        "pip": "wandb",
        "import": "wandb",
        "env": ["WANDB_PROJECT", "WANDB_MODE", "WANDB_API_KEY"],
        "out": lambda out: f"cloud wandb.ai/{os.getenv('WANDB_PROJECT', 'huggingface')} + ./wandb/",
        "note": "유일하게 인증 필요(wandb login). offline이면 WANDB_MODE=offline.",
    },
    "comet_ml": {
        "pip": "comet-ml",
        "import": "comet_ml",
        "env": ["COMET_API_KEY", "COMET_PROJECT_NAME"],
        "out": lambda out: "comet.com 클라우드",
        "note": "SaaS. API 키 필요.",
    },
    "swanlab": {
        "pip": "swanlab",
        "import": "swanlab",
        "env": ["SWANLAB_API_KEY"],
        "out": lambda out: "swanlab 클라우드 + ./swanlog/",
        "note": "SaaS.",
    },
    "clearml": {
        "pip": "clearml",
        "import": "clearml",
        "env": [],
        "out": lambda out: "ClearML 서버(셀프호스팅/클라우드)",
        "note": "MLOps 확장형. 무거움.",
    },
    "dagshub": {
        "pip": "dagshub",
        "import": "dagshub",
        "env": ["MLFLOW_TRACKING_URI"],
        "out": lambda out: "DagsHub 원격 (MLFLOW_TRACKING_URI로 지정)",
        "note": "DagsHub 계정 필요.",
    },
    "neptune": {
        "pip": "neptune",
        "import": "neptune",
        "env": ["NEPTUNE_API_TOKEN", "NEPTUNE_PROJECT"],
        "out": lambda out: "neptune.ai 클라우드",
        "note": "SaaS. API 토큰 필요.",
    },
    "codecarbon": {
        "pip": "codecarbon",
        "import": "codecarbon",
        "env": [],
        "out": lambda out: "./codecarbon / emissions.csv",
        "note": "실험추적 아님(탄소 측정). loss용 아님.",
    },
    "azure_ml": {
        "pip": "azureml-sdk",
        "import": "azureml.core",
        "env": [],
        "out": lambda out: "Azure ML workspace run history",
        "note": "Azure 환경 종속.",
    },
    "flyte": {
        "pip": "flytekit",
        "import": "flytekit",
        "env": [],
        "out": lambda out: "Flyte deck / K8s 파이프라인",
        "note": "단일 학습 로그용 아님.",
    },
    "kubeflow": {
        "pip": "kfp",
        "import": "kfp",
        "env": [],
        "out": lambda out: "Kubeflow Pipelines UI (자동 추가)",
        "note": "설치돼 있으면 transformers가 자동 추가.",
    },
}


def parse_report_to(s):
    """10-train-entry.py와 동일한 파싱. SFTConfig에 넘어가는 값을 반환."""
    items = [x.strip() for x in (s or "").split(",") if x.strip()]
    if not items or items == ["none"]:
        return "none"
    if len(items) == 1:
        return items[0]
    return [x for x in items if x != "none"]


def is_available(name):
    meta = KNOWN[name]
    if name == "mlflow" and os.getenv("DISABLE_MLFLOW_INTEGRATION", "FALSE").upper() == "TRUE":
        return False
    try:
        return importlib.util.find_spec(meta["import"]) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report_to", required=True, help='예: "all", "none", "tensorboard,wandb"')
    p.add_argument("--out", default=None, help="모델 출력 디렉토리 (tensorboard 경로 표시에 사용)")
    a = p.parse_args()

    try:
        tf_ver = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        tf_ver = "(미설치)"
    known_names = sorted(KNOWN.keys())
    available = {n for n in known_names if is_available(n)}

    effective = parse_report_to(a.report_to)
    print(f"input: {a.report_to!r}  ->  Trainer report_to: {effective!r} (transformers {tf_ver})")
    print(f"인지: {len(known_names)}종, 설치됨: {len(available)}종")
    print(f"installed: {', '.join(sorted(available)) if available else '(없음 — all이어도 로그 안 남음)'}")

    if effective == "none":
        print("-> 로깅 없음. Trainer 콜백 추가 안 됨.")
        return 0

    problems = 0
    if a.report_to.strip() == "all":
        targets = sorted(available)
        print(f"-> all 해석: {targets if targets else '(설치된 연동 없음)'}")
        missing = sorted(set(known_names) - available)
        if missing:
            print("not installed (all에 포함 안 됨):")
            for n in missing:
                print(f"  [----] {n}: pip install {KNOWN[n]['pip']}")
        if not targets:
            print("RESULT: FAIL — all이지만 설치된 연동이 없어 로그가 남지 않음.")
            return 1
    else:
        req = effective if isinstance(effective, list) else [effective]
        print(f"-> 요청: {req}")
        targets = req

    for n in targets:
        if n not in KNOWN:
            print(f"  [UNKNOWN] {n}: 오타? 유효값: {', '.join(known_names)}, all, none")
            problems += 1
            continue
        meta = KNOWN[n]
        if n in available:
            try:
                out = meta["out"](a.out)
            except Exception:
                out = "(경로 확인 실패)"
            env_show = ""
            if meta.get("env"):
                env_show = " | " + ", ".join(
                    f"{k}={os.getenv(k) or '(미설정)'}" for k in meta["env"]
                )
            print(f"  [OK] {n}: 출력 -> {out}{env_show}")
            if meta.get("note"):
                print(f"       {meta['note']}")
            if n == "wandb" and not os.getenv("WANDB_API_KEY"):
                print("       HINT: WANDB_API_KEY 미설정 — 'wandb login' 상태 확인 필요 (실패 시 잡 중단 가능).")
        else:
            print(f"  [MISSING] {n}: pip install {meta['pip']}")
            problems += 1

    if problems:
        print(f"RESULT: FAIL — {problems}건 문제 (위 MISSING/UNKNOWN 해결 후 재실행).")
        return 1
    print("RESULT: OK — 요청 전체 활성. 학습 시 위 위치에 보고서 생성.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

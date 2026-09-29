#!/usr/bin/env python3
"""eval_compare_table.py — lm-eval 결과 2개(이전 모델 vs 새 모델) 비교표 생성.

eval-compare-* 스크립트가 만든 결과 디렉터리 구조:
    <root>/
      prev-general/results_*.json     ← lm_eval --tasks tinyArc,...  출력
      prev-korean/results_*.json      ← lm_eval --tasks kobest_*      출력
      new-general/results_*.json
      new-korean/results_*.json

사용:
    uv run python eval_compare_table.py outputs/lm_eval_results/sft-mini \
        --labels "prev=Base,new=SFT"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

SUITES = {"general": "평가 ① General 지능 (TinyBenchmarks)",
          "korean": "평가 ② Korean (KO-BEST)"}


def find_results(d: str) -> list[str]:
    """디렉터리 안의 results_*.json 경로들 (최신순)."""
    return sorted(glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True, include_hidden=True))


def load_metrics(d: str) -> dict[str, tuple[float, float]] | None:
    """{task: (score, stderr)} — acc_norm 우선, 없으면 acc."""
    files = find_results(d)
    if not files:
        return None
    data = json.load(open(files[-1], encoding="utf-8"))
    out: dict[str, tuple[float, float]] = {}
    for task, m in data.get("results", {}).items():
        if not isinstance(m, dict):
            continue
        if "acc_norm,none" in m:
            score, stderr, metric = m["acc_norm,none"], m.get("acc_norm_stderr,none"), "acc_norm"
        elif "acc,none" in m:
            score, stderr, metric = m["acc,none"], m.get("acc_stderr,none"), "acc"
        else:
            continue
        try:
            stderr_val = float(stderr) if (stderr is not None and stderr != "N/A") else float("nan")
        except (ValueError, TypeError):
            stderr_val = float("nan")
        try:
            score_val = float(score) if (score is not None and score != "N/A") else float("nan")
        except (ValueError, TypeError):
            score_val = float("nan")
        out[task] = (score_val, stderr_val, metric)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="lm-eval 결과 비교표 생성")
    ap.add_argument("root", help="eval-compare 스크립트의 결과 루트 디렉터리")
    ap.add_argument("--labels", default="prev=이전 모델,new=새 모델",
                    help='표 머리글 표시명, 예) "prev=Base,new=SFT"')
    args = ap.parse_args()

    label_map: dict[str, str] = {}
    for pair in args.labels.split(","):
        k, _, v = pair.partition("=")
        label_map[k.strip()] = v.strip()
    keys = list(label_map)

    if not os.path.isdir(args.root):
        sys.exit(f"[eval-table] ERROR: {args.root} 가 없습니다.")

    found_any = False
    for suite, title in SUITES.items():
        cols: dict[str, dict] = {}
        metrics_seen = "acc_norm"
        for k in keys:
            d = os.path.join(args.root, f"{k}-{suite}")
            m = load_metrics(d) if os.path.isdir(d) else None
            cols[k] = m or {}
            if m:
                found_any = True
                if any(v[2] == "acc" for v in m.values()):
                    metrics_seen = "acc / acc_norm"
        if not any(cols.values()):
            continue

        tasks: list[str] = []
        for k in keys:
            for t in cols[k]:
                if t not in tasks:
                    tasks.append(t)
        tasks.sort()

        print(f"\n### {title}\n")
        header = "| task | metric | " + " | ".join(label_map[k] for k in keys) + " | Δ(%p) |"
        print(header)
        print("|---|---|" + "---|" * len(keys) + "---|")
        for t in tasks:
            vals = []
            for k in keys:
                v = cols[k].get(t)
                vals.append(f"{v[0]*100:.1f}" if v else "-")
            nums = [cols[k][t][0] for k in keys if t in cols[k]]
            delta = f"{(nums[-1]-nums[0])*100:+.1f}" if len(nums) == len(keys) else "-"
            print(f"| {t} | {metrics_seen} | " + " | ".join(vals) + f" | {delta} |")

    if not found_any:
        sys.exit("[eval-table] ERROR: results_*.json 을 찾지 못했습니다. "
                 "먼저 eval-compare 스크립트를 실행하세요.")


if __name__ == "__main__":
    main()

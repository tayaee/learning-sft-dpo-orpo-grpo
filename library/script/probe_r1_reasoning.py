#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai",
# ]
# ///
"""probe_r1_reasoning.py — OpenRouter reasoning/content 분리 관측 프로브.

CoT/reasoning 구조를 눈으로 보기 위한 읽기 전용 도구. 학습·데이터 생성 없음.
나중에 teacher trace 수집(distill/RLVR)으로 확장할 때 --out JSONL을 재사용.

사용법:
    export OPENROUTER_API_KEY=...
    uv run --script library/script/probe_r1_reasoning.py "질문..."
    uv run --script library/script/probe_r1_reasoning.py --model deepseek/deepseek-r1 --out outputs/reasoning-probe/x.jsonl "질문..."

출력: stdout에 [Thinking] / [Final Answer] 구분 스트리밍 + --out 지정 시 JSONL 1줄 저장
    {"model": ..., "prompt": ..., "reasoning": ..., "content": ...}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenRouter reasoning/content 분리 프로브")
    p.add_argument("prompt", nargs="*", help="질문 (없으면 기본 예제)")
    p.add_argument("--model", default="deepseek/deepseek-r1")
    p.add_argument("--out", default=None,
                   help="저장 경로 (없으면 outputs/reasoning-probe/<utc시각>.jsonl)")
    p.add_argument("--no-save", action="store_true", help="파일 저장 안 함")
    return p


DEFAULT_PROMPT = "한 유리잔은 5달러인데 두 번째 잔마다 40%를 할인해 준다고 한다. 16잔을 사려면 총 얼마가 필요한가?"


def main() -> None:
    args = build_argparser().parse_args()
    prompt = " ".join(args.prompt).strip() or DEFAULT_PROMPT

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("[오류] OPENROUTER_API_KEY 환경 변수가 설정되지 않았습니다.\n"
                 "  export OPENROUTER_API_KEY=... 후 다시 실행하세요.")

    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    print(f"model: {args.model}")
    print(f"질문: {prompt}\n")
    print("-" * 50)
    print("[모델의 사고 과정 (Thinking Process)]")

    try:
        stream = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
    except Exception as e:
        sys.exit(f"[오류] API 호출 실패: {e}")

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    is_thinking = True

    try:
        for chunk in stream:
            delta = chunk.choices[0].delta
            reasoning_chunk = getattr(delta, "reasoning", None)
            content_chunk = getattr(delta, "content", None)

            if reasoning_chunk:
                reasoning_parts.append(reasoning_chunk)
                sys.stdout.write(reasoning_chunk)
                sys.stdout.flush()

            if content_chunk:
                if is_thinking:
                    print("\n" + "-" * 50)
                    print("[최종 답변 (Final Answer)]")
                    is_thinking = False
                cleaned = content_chunk.replace("<think>", "").replace("</think>", "")
                content_parts.append(cleaned)
                sys.stdout.write(cleaned)
                sys.stdout.flush()
    except Exception as e:
        print(f"\n[오류] 스트리밍 중단: {e}", file=sys.stderr)

    print("\n" + "=" * 50)

    if args.no_save:
        return

    out = args.out or os.path.join(
        "outputs", "reasoning-probe",
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".jsonl",
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    record = {
        "model": args.model,
        "prompt": prompt,
        "reasoning": "".join(reasoning_parts),
        "content": "".join(content_parts),
    }
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

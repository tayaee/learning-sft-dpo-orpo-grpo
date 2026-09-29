#!/bin/bash
# mini pipeline driver — runs each stage's mini scripts and captures file changes.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ ! -f "$ROOT_DIR/pyproject.toml" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR/library/function:$ROOT_DIR/library/script${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/library/script/scripts_common.sh"  # noqa: cross-stage driver stays at root

OUT=files-changed.txt
: > "$OUT"
echo "$(date) — Mini pipeline test (sft → dpo → orpo → grpo)" >> "$OUT"
echo "" >> "$OUT"
echo "Capture: find . -newer /tmp/mark -type f -ls 2>/dev/null" >> "$OUT"
echo "Symlinks shown separately with find ... -type l -ls 2>/dev/null" >> "$OUT"
echo "" >> "$OUT"

stage_header() {
  local stage="$1" mode="$2"
  echo "" >> "$OUT"
  echo "================================================================================" >> "$OUT"
  echo "STAGE: ${stage} (${mode})" >> "$OUT"
  echo "================================================================================" >> "$OUT"
}

stage_in_config() {
  local stage="$1" mode="$2"
  echo "" >> "$OUT"
  echo "### Stage start: ${stage} ${mode} — in/ and config/ ###" >> "$OUT"
  for kind in in config; do
    echo "" >> "$OUT"
    echo "## data/${stage}-${mode}-${kind}/" >> "$OUT"
    find "data/${stage}-${mode}-${kind}" \( -type f -o -type l \) -ls 2>/dev/null >> "$OUT"
    echo "(end)" >> "$OUT"
  done
}

stage_out() {
  local stage="$1" mode="$2"
  echo "" >> "$OUT"
  echo "### Stage end: ${stage} ${mode} — out/ ###" >> "$OUT"
  echo "## data/${stage}-${mode}-out/" >> "$OUT"
  find "data/${stage}-${mode}-out" \( -type f -o -type l \) -ls 2>/dev/null >> "$OUT"
  echo "(end)" >> "$OUT"
}

run_one() {
  local script="$1"
  echo "" >> "$OUT"
  echo "--- Script: ${script} ---" >> "$OUT"
  touch /tmp/mark
  bash "$script" >> "$OUT" 2>&1
  local rc=$?
  echo "" >> "$OUT"
  echo "Exit code: ${rc}" >> "$OUT"
  echo "Files newer than /tmp/mark:" >> "$OUT"
  find . -newer /tmp/mark \( -type f -o -type l \) -ls 2>/dev/null >> "$OUT"
  echo "(end of changes for ${script})" >> "$OUT"
  return $rc
}

# === STAGE 1: SFT ===
stage_header sft mini
stage_in_config sft mini
for s in phase/sft/algorithm/lora/tool/axolotl/sft-mini-05-review-sft-config-mini.sh phase/sft/algorithm/lora/tool/axolotl/sft-mini-06-run-axolotl-mini.sh phase/sft/algorithm/lora/tool/axolotl/sft-mini-07-merge-mini.sh phase/sft/algorithm/lora/tool/axolotl/sft-mini-11-lm-eval-sft-mini.sh phase/sft/algorithm/lora/tool/axolotl/sft-mini-12-eval-compare-mini.sh; do
  run_one "$s" || { echo "FAILED: $s" >> "$OUT"; break; }
done
stage_out sft mini

# === STAGE 2: DPO ===
stage_header dpo mini
stage_in_config dpo mini
for s in phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-01-make-rl-dpo-data.sh phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-02-sanity-check-rl-dpo-data-mini.sh phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-03-review-rl-dpo-config-mini.sh phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-04-run-axolotl-mini.sh phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-05-test-models-mini.sh phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-06-merge-mini.sh phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-09-lm-eval-rl-dpo-mini.sh phase/pref/algorithm/dpo/tool/axolotl/rl-dpo-mini-10-eval-compare-mini.sh; do
  run_one "$s" || { echo "FAILED: $s" >> "$OUT"; break; }
done
stage_out dpo mini

# === STAGE 3: ORPO ===
stage_header orpo mini
stage_in_config orpo mini
for s in phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-01-make-orpo-data-mini.sh phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-02-sanity-check-orpo-data-mini.sh phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-03-review-orpo-config-mini.sh phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-04-run-axolotl-mini.sh phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-05-merge-mini.sh phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-06-test-model-mini.sh phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-08-reward-eval-mini.sh phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-09-lm-eval-rl-orpo-mini.sh phase/pref/algorithm/orpo/tool/axolotl/rl-orpo-mini-10-eval-compare-mini.sh; do
  run_one "$s" || { echo "FAILED: $s" >> "$OUT"; break; }
done
stage_out orpo mini

# === STAGE 4: GRPO ===
stage_header grpo mini
stage_in_config grpo mini
for s in phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-mini-01-review-data-mini.sh phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-mini-02-review-reward-fn.sh phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-mini-03-review-rl-grpo-config-mini.sh phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-mini-04-run-axolotl-mini.sh phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-mini-05-test-model-mini.sh phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-mini-06-merge-mini.sh phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-mini-09-lm-eval-rl-grpo-mini.sh phase/rl/algorithm/grpo/tool/axolotl/rl-grpo-mini-10-eval-compare-mini.sh; do
  run_one "$s" || { echo "FAILED: $s" >> "$OUT"; break; }
done
stage_out grpo mini

echo "" >> "$OUT"
echo "================================================================================" >> "$OUT"
echo "PIPELINE COMPLETE — $(date)" >> "$OUT"
echo "================================================================================" >> "$OUT"
#!/usr/bin/env bash
# check_report_to.sh — 5종 리포팅 연동 INSTALL + CONFIG 상태 green/red 표시.
#   ./check_report_to.sh                  # 1회 확인 (기본 5종)
#   ./check_report_to.sh --watch [초]      # 모두 green 될 때까지 반복 (기본 30초)
# 종료 코드: 0=5종 INSTALL·CONFIG 모두 READY, 1=미준비 있음.
set -uo pipefail
cd "$(dirname "$0")"

FIVE="tensorboard,mlflow,trackio,dvclive,wandb"
WATCH=0; INTERVAL=30; REPORT_TO="$FIVE"; OUT_ARG=(); OUT_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --watch) WATCH=1; [ "${2:-}" != "" ] && [[ "$2" =~ ^[0-9]+$ ]] && { INTERVAL="$2"; shift; }; shift ;;
    --report-to) REPORT_TO="$2"; shift 2 ;;
    --out) OUT_ARG=(--out "$2"); OUT_DIR="$2"; shift 2 ;;
    *) echo "usage: $0 [--watch [secs]] [--report-to csv] [--out dir]" >&2; exit 2 ;;
  esac
done
IFS=',' read -ra NAMES <<< "$REPORT_TO"

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then GREEN=$'\e[32m'; RED=$'\e[31m'; RST=$'\e[0m'; else GREEN=''; RED=''; RST=''; fi
ok() { printf '%b● READY%b' "$GREEN" "$RST"; }
bad() { printf '%b● NOT READY%b' "$RED" "$RST"; }

# config 점검: 전역 $raw (체커 출력) 사용. echo "이유"; return 0=READY.
config_check() {
  case "$1" in
    tensorboard|trackio|dvclive) echo "추가 설정 불필요"; return 0 ;;  # 추가 설정 불필요
    mlflow)
      local uri="${MLFLOW_TRACKING_URI:-./mlruns}" path
      case "$uri" in
        http://*|https://*) echo "원격 $uri (연결 미확인)"; return 0 ;;
        file:*) path="${uri#file:}" ;;
        *) path="$uri" ;;
      esac
      if [ -e "$path" ] && [ ! -w "$path" ]; then echo "저장소 쓰기 불가: $path"; return 1; fi
      if [ ! -e "$path" ] && [ ! -w "$(dirname "$path")" ]; then echo "부모디렉토리 쓰기 불가: $(dirname "$path")"; return 1; fi
      echo "저장소: $uri"; return 0 ;;
    wandb)
      if [ -n "${WANDB_API_KEY:-}" ]; then echo "API키: env 설정됨"; return 0; fi
      if [ -f ~/.netrc ] && grep -q "machine api.wandb.ai" ~/.netrc; then echo "API키: ~/.netrc 로그인됨"; return 0; fi
      echo "'wandb login' 필요 (WANDB_API_KEY도 미설정)"; return 1 ;;
  esac
  return 0
}

# 보고서 저장 위치 (정보 표시용 — 체커는 아무 파일도 쓰지 않음).
output_path() {
  case "$1" in
    tensorboard)
      if [ -n "${TENSORBOARD_LOGGING_DIR:-}" ]; then echo "$TENSORBOARD_LOGGING_DIR";
      elif [ -n "$OUT_DIR" ]; then echo "$OUT_DIR/runs";
      else echo '$OUT/runs'; fi ;;
    mlflow) echo "${MLFLOW_TRACKING_URI:-./mlruns}" ;;
    trackio) echo "./trackio/ (project=huggingface)" ;;
    dvclive) echo "./dvclive/" ;;
    wandb) echo "cloud wandb.ai/${WANDB_PROJECT:-huggingface} + ./wandb/" ;;
    *) echo "(벤더 문서 참조)" ;;
  esac
}

one_pass() {
  local raw rc n inst conf reason overall=0
  raw=$(uv run --no-sync check_report_to.py --report_to "$REPORT_TO" "${OUT_ARG[@]}" 2>&1)
  rc=$?
  echo "$raw" | grep -v "pytree"
  echo "---- install ----"
  declare -A INST_OK=()
  for n in "${NAMES[@]}"; do
    if echo "$raw" | grep -q "^\s*\[OK\] $n:"; then inst="$(ok)"; INST_OK[$n]=1;
    else inst="$(bad)"; INST_OK[$n]=0; overall=1; fi
    printf '%-12s install : %b\n' "$n" "$inst"
  done
  echo "---- config ----"
  for n in "${NAMES[@]}"; do
    if [ "${INST_OK[$n]}" = "1" ] && reason=$(config_check "$n"); then conf="$(ok)";
    else conf="$(bad)"; overall=1; [ "${INST_OK[$n]}" = "1" ] || reason="패키지 미설치"; fi
    printf '%-12s config  : %b  (%s)\n' "$n" "$conf" "$reason"
  done
  echo "P5_REPORT_TO=${P5_REPORT_TO:-(미설정, 기본 none)}"
  echo "---- output ----"
  for n in "${NAMES[@]}"; do
    printf '%-12s output  : %s\n' "$n" "$(output_path "$n")"
  done
  [ "$overall" = "0" ] && echo "ALL GREEN."
  return $overall
}

if [ "$WATCH" = "1" ]; then
  while true; do
    echo "== $(date -u +%FT%TZ) =="
    if one_pass; then exit 0; fi
    echo "retry in ${INTERVAL}s ... (Ctrl-C to stop)"
    sleep "$INTERVAL"
  done
else
  one_pass
fi

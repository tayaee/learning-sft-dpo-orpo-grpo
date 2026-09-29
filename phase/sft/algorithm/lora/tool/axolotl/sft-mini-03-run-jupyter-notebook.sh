#!/usr/bin/env bash
# run-jupyter-notebook.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ ! -f "$ROOT_DIR/pyproject.toml" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="$ROOT_DIR/library/function:$ROOT_DIR/library/script${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT_DIR/library/script/scripts_common.sh"
parse_flags "$@"
set -euo pipefail

PORT="${1:-8888}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' not installed. See https://docs.astral.sh/uv/"
  exit 1
fi

REQUIRED_PKGS=("jupyter" "notebook" "ipykernel")
MISSING_PKGS=()
for pkg in "${REQUIRED_PKGS[@]}"; do
  if ! uv pip show "$pkg" >/dev/null 2>&1; then
    MISSING_PKGS+=("$pkg")
  fi
done

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
  echo "Adding missing packages: ${MISSING_PKGS[*]}"
  uv add "${MISSING_PKGS[@]}"
fi

KERNEL_DISPLAY_NAME="Python (learning-sft-and-rl)"
KERNEL_NAME="learning-sft-and-rl"
if ! uv run jupyter kernelspec list 2>/dev/null | grep -q "$KERNEL_NAME"; then
  echo "Registering kernel: $KERNEL_NAME"
  uv run ipython kernel install --user --name="$KERNEL_NAME" --display-name="$KERNEL_DISPLAY_NAME" >/dev/null 2>&1 || true
fi

HOSTNAME="${JUPYTER_HOSTNAME:-}"
if [ -z "$HOSTNAME" ] && command -v hostname >/dev/null 2>&1; then
  HOSTNAME="$(hostname -f 2>/dev/null || hostname --fqdn 2>/dev/null || true)"
fi
if { [ -z "$HOSTNAME" ] || [[ "$HOSTNAME" != *.* ]]; } && command -v hostname >/dev/null 2>&1; then
  ALT="$(hostname -A 2>/dev/null | awk '{print $1}' || true)"
  if [ -n "$ALT" ] && [[ "$ALT" == *.* ]]; then
    HOSTNAME="$ALT"
  fi
fi
if [ -z "$HOSTNAME" ] && command -v hostname >/dev/null 2>&1; then
  HOSTNAME="$(hostname 2>/dev/null || true)"
fi

if [ -z "$HOSTNAME" ]; then
  echo "Error: Cannot determine hostname." >&2
  exit 1
fi

RESOLVED_IP=""
if command -v avahi-resolve >/dev/null 2>&1; then
  RESOLVED_IP="$(avahi-resolve -n "$HOSTNAME" 2>/dev/null | awk '{print $2}' | head -n1 || true)"
fi
if [ -z "$RESOLVED_IP" ] && command -v getent >/dev/null 2>&1; then
  RESOLVED_IP="$(getent hosts "$HOSTNAME" 2>/dev/null | awk '{print $1}' | head -n1 || true)"
fi

if [ -z "$RESOLVED_IP" ]; then
  echo "Error: Cannot resolve '${HOSTNAME}'." >&2
  exit 1
fi

echo "input: pyproject.toml, uv.lock"
echo "output: Jupyter Server (http://${HOSTNAME}:${PORT})"

echo "Host FQDN: ${HOSTNAME} -> ${RESOLVED_IP}"
echo "Starting Jupyter Notebook at http://${HOSTNAME}:${PORT}"

set -x
uv run jupyter notebook --port="$PORT" --no-browser --ip=0.0.0.0
set +x

echo "input: pyproject.toml, uv.lock"
echo "output: Jupyter Server (http://${HOSTNAME}:${PORT})"

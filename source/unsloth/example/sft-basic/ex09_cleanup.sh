#!/usr/bin/env bash
# ex09_cleanup.sh — step 09: clean local artifacts after the run.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

exec uv run ./ex09_cleanup.py --clean "$@"

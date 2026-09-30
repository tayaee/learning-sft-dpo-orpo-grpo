#!/usr/bin/env bash
# ex00_cleanup.sh — step 00: clean local artifacts.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

exec uv run ./ex00_cleanup.py --clean "$@"

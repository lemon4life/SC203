#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

exec "$PYTHON_BIN" "$REPO_ROOT/src/run_dossr.py" \
  --config "$REPO_ROOT/configs/baseline_realsr.yaml" \
  "$@"

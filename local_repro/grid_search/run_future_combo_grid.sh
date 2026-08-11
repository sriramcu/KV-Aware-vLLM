#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$HOME/KV-Aware-vLLM}"
MAX_QUESTIONS="${MAX_QUESTIONS:-250}"

exec python "$REPO/local_repro/grid_search/run_future_combo_grid.py" \
  --max-questions "$MAX_QUESTIONS" \
  "$@"

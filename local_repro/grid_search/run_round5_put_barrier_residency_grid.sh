#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$HOME/KV-Aware-vLLM}"
MAX_QUESTIONS="${MAX_QUESTIONS:-1}"

exec python "$REPO/local_repro/grid_search/run_round5_put_barrier_residency_grid.py" \
  --max-questions "$MAX_QUESTIONS" \
  "$@"

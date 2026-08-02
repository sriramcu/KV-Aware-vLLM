#!/bin/bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$REPO/local_repro/grid_search/run_pressure_grid.py" --max-questions 1 "$@"

#!/bin/bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULL_PATCH="$BUNDLE_DIR/sc_pressure_grid.patch"
UPGRADE_PATCH="$BUNDLE_DIR/sc_pressure_grid_v1_to_v2.patch"
REPO="${1:-$PWD}"

if [[ ! -d "$REPO/.git" ]]; then
  echo "Repository root not found at: $REPO" >&2
  echo "Run this from KV-Aware-vLLM or pass its path as the first argument." >&2
  exit 2
fi
for patch in "$FULL_PATCH" "$UPGRADE_PATCH"; do
  if [[ ! -f "$patch" ]]; then
    echo "Missing patch: $patch" >&2
    exit 2
  fi
done

validate_tree() {
  local count
  for path in \
    local_repro/cpu_offload_lmcache_sriram.py \
    local_repro/run_driver.sh \
    local_repro/grid_search/run_pressure_grid.py \
    local_repro/grid_search/run_smoke_grid.sh \
    local_repro/grid_search/submit_pressure_grid.sbatch \
    local_repro/grid_search/README.md; do
    [[ -f "$REPO/$path" ]] || {
      echo "Validation failed: missing $path" >&2
      return 1
    }
  done

  python - "$REPO" <<'PY'
from pathlib import Path
import sys
repo = Path(sys.argv[1])
for relative in (
    "local_repro/cpu_offload_lmcache_sriram.py",
    "local_repro/grid_search/run_pressure_grid.py",
):
    path = repo / relative
    compile(path.read_text(encoding="utf-8"), str(path), "exec")

text = (repo / "local_repro/cpu_offload_lmcache_sriram.py").read_text(
    encoding="utf-8"
)
required = (
    "[SC_DRIVER_REQUEST_METRIC]",
    "[SC_DRIVER_PHASE_SUMMARY]",
    "[SC_DRIVER_OUTPUT_CONSISTENCY]",
    "[SC_DRIVER_METRICS_DONE]",
)
for marker in required:
    assert marker in text, marker
assert "cold_request_metrics = None\n        if _sc_io_trace_enabled():" in text
assert "if _sc_io_trace_enabled():\n            warm_request_metrics" in text

readme = (repo / "local_repro/grid_search/README.md").read_text(
    encoding="utf-8"
)
assert "SC_LMCACHE_IO_TRACE_ENABLE=1" in readme
assert "disables all" in readme
print("Python compilation and metric-switch validation: PASS")
PY

  bash -n "$REPO/local_repro/run_driver.sh"
  bash -n "$REPO/local_repro/grid_search/run_smoke_grid.sh"
  bash -n "$REPO/local_repro/grid_search/submit_pressure_grid.sbatch"
  echo "Shell syntax: PASS"

  count="$(
    cd "$REPO"
    python local_repro/grid_search/run_pressure_grid.py --max-questions 1 --print-grid \
      | grep -cE '^[0-9]{2}\. '
  )"
  [[ "$count" == "16" ]] || {
    echo "Grid validation failed: expected 16 configurations, found $count" >&2
    return 1
  }
  echo "Grid definition: PASS (16 configurations)"

  grep -q 'default=1' "$REPO/local_repro/grid_search/run_pressure_grid.py"
  grep -q '=== DONE ===' "$REPO/local_repro/run_driver.sh"
  grep -q '"SC_LMCACHE_IO_TRACE_ENABLE": "1"' \
    "$REPO/local_repro/grid_search/run_pressure_grid.py"
  echo "Smoke defaults and completion markers: PASS"
}

make_backup() {
  local mode="$1"
  local stamp backup metadata
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  backup="${HOME}/sc_pressure_grid_backup_${stamp}.tar.gz"
  metadata="${backup}.metadata.txt"

  if [[ "$mode" == "full" ]]; then
    tar -czf "$backup" -C "$REPO" \
      local_repro/cpu_offload_lmcache_sriram.py \
      local_repro/run_driver.sh
  else
    tar -czf "$backup" -C "$REPO" \
      local_repro/cpu_offload_lmcache_sriram.py \
      local_repro/run_driver.sh \
      local_repro/grid_search/README.md
  fi

  {
    echo "created_utc=$stamp"
    echo "repo=$REPO"
    echo "apply_mode=$mode"
    echo "The archive contains every pre-existing file touched by this apply mode."
    if [[ "$mode" == "full" ]]; then
      echo "New files created by the full patch (remove these when manually restoring):"
      printf '  %s\n' \
        local_repro/grid_search/README.md \
        local_repro/grid_search/run_pressure_grid.py \
        local_repro/grid_search/run_smoke_grid.sh \
        local_repro/grid_search/submit_pressure_grid.sbatch
    fi
  } > "$metadata"

  echo "Backup created: $backup"
  echo "Backup metadata: $metadata"
}

cd "$REPO"

if git apply --reverse --check "$FULL_PATCH" >/dev/null 2>&1; then
  echo "The switch-enabled SC pressure-grid patch appears to be already applied."
  validate_tree
  exit 0
fi

if git apply --check "$FULL_PATCH" >/dev/null 2>&1; then
  make_backup full
  git apply "$FULL_PATCH"
elif git apply --check "$UPGRADE_PATCH" >/dev/null 2>&1; then
  echo "Detected the earlier pressure-grid implementation; applying switch-only upgrade."
  make_backup upgrade
  git apply "$UPGRADE_PATCH"
else
  echo "Patch check failed." >&2
  echo "The tree matches neither the post-refactor source nor the earlier pressure-grid bundle." >&2
  echo "No files were changed." >&2
  exit 1
fi

chmod 755 \
  "$REPO/local_repro/run_driver.sh" \
  "$REPO/local_repro/grid_search/run_pressure_grid.py" \
  "$REPO/local_repro/grid_search/run_smoke_grid.sh" \
  "$REPO/local_repro/grid_search/submit_pressure_grid.sbatch"

validate_tree

echo "Switch-enabled SC pressure-grid smoke implementation applied successfully."
echo "The lightweight SC_DRIVER_* metric records follow SC_LMCACHE_IO_TRACE_ENABLE:"
echo "  1 = enabled"
echo "  0 = disabled"
echo "The grid sets it to 1; the vanilla profile sets it to 0."
echo "Start the MAX_QUESTIONS=1 grid with:"
echo "  cd $REPO"
echo "  ./local_repro/grid_search/run_smoke_grid.sh"

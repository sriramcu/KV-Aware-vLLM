#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$BUNDLE_DIR/p0_first_half.patch"
REPO="${1:-/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM}"
EXPECTED_HEAD="3f9342de59adb5fe649f6bc3823a153fd3ed507d"

TARGETS=(
  third_party/LMCache/lmcache/v1/storage_backend/storage_manager.py
  third_party/LMCache/lmcache/v1/storage_backend/local_disk_backend.py
)

declare -A EXPECTED_SHA256=(
  [third_party/LMCache/lmcache/v1/storage_backend/storage_manager.py]="7c2d899da8be4f1678b1b2da44c87b55ced4cc2814495e24e8cce3c424d2f226"
  [third_party/LMCache/lmcache/v1/storage_backend/local_disk_backend.py]="c00d481bc35dd49d001d6c1a5e6f519ddb818c3390b67f8db263998bb10968cf"
)

cd "$REPO"
ROOT="$(git rev-parse --show-toplevel)"
if [[ "$ROOT" != "$REPO" ]]; then
  echo "Resolved Git root: $ROOT"
  cd "$ROOT"
fi

actual_head="$(git rev-parse HEAD)"
if [[ "$actual_head" != "$EXPECTED_HEAD" ]]; then
  echo "WARNING: expected parent commit $EXPECTED_HEAD, found $actual_head"
  echo "The exact target-file hashes will still be checked before applying."
fi

for target in "${TARGETS[@]}"; do
  [[ -f "$target" ]] || { echo "Missing target: $target" >&2; exit 1; }

done

if ! git diff --quiet -- "${TARGETS[@]}" || \
   ! git diff --cached --quiet -- "${TARGETS[@]}"; then
  echo "Refusing to patch: one of the two target files already has tracked changes." >&2
  git status --short -- "${TARGETS[@]}" >&2
  exit 1
fi

for target in "${TARGETS[@]}"; do
  actual="$(sha256sum "$target" | awk '{print $1}')"
  expected="${EXPECTED_SHA256[$target]}"
  if [[ "$actual" != "$expected" ]]; then
    echo "Refusing to patch: baseline hash mismatch for $target" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    exit 1
  fi
  echo "Baseline verified: $target"
done

git apply --check "$PATCH_FILE"
git apply "$PATCH_FILE"

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared/gpfs/home/sriramc2/venvs/kvaware/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python
fi

"$PYTHON_BIN" - "${TARGETS[@]}" <<'PYSYNTAX'
from pathlib import Path
import sys
for raw in sys.argv[1:]:
    path = Path(raw)
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print(f"Syntax verified: {path}")
PYSYNTAX
git diff --check -- "${TARGETS[@]}"

echo
echo "P0 first-half patch applied successfully."
echo "No pip reinstall is required because LMCache is installed editable."
echo
git diff --stat -- "${TARGETS[@]}"
echo
echo "Configured knobs for the test job:"
echo "  LMCACHE_P0_LOOKUP_MAX_INFLIGHT=1"
echo "  LMCACHE_P0_DISK_PUT_MAX_PENDING=8"
echo
echo "Submit the included Slurm job with:"
echo "  sbatch $BUNDLE_DIR/03_h100_p0_first_half.sbatch"

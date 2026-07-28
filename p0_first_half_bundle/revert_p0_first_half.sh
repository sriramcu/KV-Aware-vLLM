#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$BUNDLE_DIR/p0_first_half.patch"
REPO="${1:-/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM}"

cd "$REPO"
git apply --reverse --check "$PATCH_FILE"
git apply --reverse "$PATCH_FILE"

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared/gpfs/home/sriramc2/venvs/kvaware/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python
fi
"$PYTHON_BIN" - <<'PYSYNTAX'
from pathlib import Path
for raw in (
    "third_party/LMCache/lmcache/v1/storage_backend/storage_manager.py",
    "third_party/LMCache/lmcache/v1/storage_backend/local_disk_backend.py",
):
    path = Path(raw)
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print(f"Syntax verified: {path}")
PYSYNTAX

echo "P0 first-half patch reverted."

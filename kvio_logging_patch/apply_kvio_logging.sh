#!/bin/bash
set -euo pipefail

REPO="${REPO:-/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM}"
SITE_PACKAGES="${SITE_PACKAGES:-/mnt/shared/gpfs/home/sriramc2/venvs/kvaware/lib/python3.12/site-packages}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_ROOT="${BACKUP_ROOT:-$REPO/runs/kvio_logging_backup_$STAMP}"

mkdir -p "$BACKUP_ROOT/repo/local_repro"
mkdir -p "$BACKUP_ROOT/site/lmcache/v1/lookup_client"
mkdir -p "$BACKUP_ROOT/site/lmcache/v1/storage_backend"

cp -a "$REPO/local_repro/run_driver.sh" \
      "$BACKUP_ROOT/repo/local_repro/run_driver.sh"
cp -a "$REPO/local_repro/cpu_offload_lmcache_sriram.py" \
      "$BACKUP_ROOT/repo/local_repro/cpu_offload_lmcache_sriram.py"
cp -a "$SITE_PACKAGES/lmcache/v1/lookup_client/lmcache_async_lookup_client.py" \
      "$BACKUP_ROOT/site/lmcache/v1/lookup_client/lmcache_async_lookup_client.py"
cp -a "$SITE_PACKAGES/lmcache/v1/cache_engine.py" \
      "$BACKUP_ROOT/site/lmcache/v1/cache_engine.py"
cp -a "$SITE_PACKAGES/lmcache/v1/storage_backend/storage_manager.py" \
      "$BACKUP_ROOT/site/lmcache/v1/storage_backend/storage_manager.py"
cp -a "$SITE_PACKAGES/lmcache/v1/storage_backend/local_disk_backend.py" \
      "$BACKUP_ROOT/site/lmcache/v1/storage_backend/local_disk_backend.py"

echo "Backups written to: $BACKUP_ROOT"

(
  cd "$REPO"
  patch --dry-run -p1 < "$PATCH_DIR/kvio_repo_logging.patch"
)
(
  cd "$SITE_PACKAGES"
  patch --dry-run -p1 < "$PATCH_DIR/kvio_lmcache_logging.patch"
)

(
  cd "$REPO"
  patch -p1 < "$PATCH_DIR/kvio_repo_logging.patch"
)
(
  cd "$SITE_PACKAGES"
  patch -p1 < "$PATCH_DIR/kvio_lmcache_logging.patch"
)

bash -n "$REPO/local_repro/run_driver.sh"
python -m py_compile \
  "$REPO/local_repro/cpu_offload_lmcache_sriram.py" \
  "$SITE_PACKAGES/lmcache/v1/lookup_client/lmcache_async_lookup_client.py" \
  "$SITE_PACKAGES/lmcache/v1/cache_engine.py" \
  "$SITE_PACKAGES/lmcache/v1/storage_backend/storage_manager.py" \
  "$SITE_PACKAGES/lmcache/v1/storage_backend/local_disk_backend.py"

echo "Logging-only instrumentation applied successfully."
echo "Set SRIRAM_KV_IO_TRACE=1 to enable it; run_driver.sh enables it by default."

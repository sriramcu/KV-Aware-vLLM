#!/bin/bash
#SBATCH --job-name=kvaware_p0ab
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -euxo pipefail

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
export SRIRAM_STACK_DUMP_INTERVAL=120
export SRIRAM_KV_IO_TRACE=1
export SRIRAM_KV_MEM_DEBUG=1
export SRIRAM_KV_MEM_DEBUG_INTERVAL_S=5

# P0 first-half controls. These limits are per LMCache worker/rank.
export LMCACHE_P0_LOOKUP_MAX_INFLIGHT="${LMCACHE_P0_LOOKUP_MAX_INFLIGHT:-4}"
export LMCACHE_P0_DISK_PUT_MAX_PENDING="${LMCACHE_P0_DISK_PUT_MAX_PENDING:-8}"
export LMCACHE_P0_CLIENT_LOOKUP_ADMISSION_TIMEOUT_MS="${LMCACHE_P0_CLIENT_LOOKUP_ADMISSION_TIMEOUT_MS:-0}"

export LMCACHE_P0_CLIENT_LOOKUP_MAX_INFLIGHT=0
export LMCACHE_P0_LOOKUP_MAX_INFLIGHT=0
export LMCACHE_P0_DISK_PUT_MAX_PENDING=0

# Irrelevant when Gate C is disabled, but remove it for clarity.
unset LMCACHE_P0_CLIENT_LOOKUP_ADMISSION_TIMEOUT_MS

# Reproduce the existing workload shape unless overridden at submission time.
export MAX_QUESTIONS="${MAX_QUESTIONS:-250}"
export SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-25}"
export SUBMISSION_BATCH_SIZE="$MAX_QUESTIONS"

# Preserve other jobs' cache directories. run_driver still creates a fresh,
# job-specific directory for this run.
export CLEAN_OLD_LMCACHE=0

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs
source "$VENV/bin/activate"
cd "$REPO"

echo "=== P0 FIRST HALF PREFLIGHT ==="
date
hostname
nvidia-smi
git rev-parse HEAD
git status --short -- \
  third_party/LMCache/lmcache/v1/storage_backend/storage_manager.py \
  third_party/LMCache/lmcache/v1/storage_backend/local_disk_backend.py

echo "LMCACHE_P0_LOOKUP_MAX_INFLIGHT=$LMCACHE_P0_LOOKUP_MAX_INFLIGHT"
echo "LMCACHE_P0_DISK_PUT_MAX_PENDING=$LMCACHE_P0_DISK_PUT_MAX_PENDING"
echo "MAX_QUESTIONS=$MAX_QUESTIONS"
echo "SUBMISSION_BATCH_SIZE=$SUBMISSION_BATCH_SIZE"

python - <<'PYSYNTAX'
from pathlib import Path
for raw in (
    "third_party/LMCache/lmcache/v1/storage_backend/storage_manager.py",
    "third_party/LMCache/lmcache/v1/storage_backend/local_disk_backend.py",
):
    path = Path(raw)
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print(f"Syntax verified: {path}")
PYSYNTAX

python - <<'PY'
from pathlib import Path
import inspect
import lmcache
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskWorker
from lmcache.v1.storage_backend.storage_manager import StorageManager

expected = Path(
    "/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/third_party/LMCache/lmcache"
).resolve()
actual = Path(lmcache.__file__).resolve().parent
print("lmcache_path:", actual)
print("storage_manager_source:", inspect.getsourcefile(StorageManager))
print("local_disk_worker_source:", inspect.getsourcefile(LocalDiskWorker))
assert actual == expected
assert hasattr(StorageManager, "_p0_acquire_lookup_admission")
assert hasattr(LocalDiskWorker, "acquire_put_admission")
print("P0 first-half source preflight: OK")
PY

echo "=== P0 FIRST HALF WORKLOAD START ==="
"$REPO/local_repro/run_driver.sh" wo_gnn 2>&1

echo "=== P0 FIRST HALF JOB PASSED ==="
date

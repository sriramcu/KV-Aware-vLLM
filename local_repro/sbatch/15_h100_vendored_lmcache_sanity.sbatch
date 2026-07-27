#!/bin/bash
#SBATCH --job-name=vendored_lmcache_sanity
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -euxo pipefail

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware
EXPECTED="$REPO/third_party/LMCache/lmcache"

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs

source "$VENV/bin/activate"
cd "$REPO"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export DATASET_NAME=hotpotqa
export MAX_QUESTIONS=4
export SUBMISSION_BATCH_SIZE=2

# Critical: do not let the driver delete caches from previous jobs.
export CLEAN_OLD_LMCACHE=0

echo "=== SOURCE AND ENVIRONMENT CHECK ==="
date
hostname
nvidia-smi
git rev-parse HEAD

python - <<'PY'
from importlib import metadata
from pathlib import Path
import lmcache
import torch

expected = Path(
    "/mnt/shared/gpfs/home/sriramc2/"
    "KV-Aware-vLLM/third_party/LMCache/lmcache"
).resolve()
actual = Path(lmcache.__file__).resolve().parent

print("lmcache_version:", metadata.version("lmcache"))
print("lmcache_path:", actual)
print("torch_version:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())

assert actual == expected
assert torch.cuda.is_available()

import lmcache.c_ops
print("lmcache.c_ops: OK")
PY

echo "=== START TINY END-TO-END WORKLOAD ==="
bash "$REPO/local_repro/run_driver.sh" wo_gnn 2>&1

echo "=== END-TO-END SANITY PASSED ==="
date

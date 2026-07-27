#!/bin/bash
#SBATCH --job-name=verify_lmcache_src
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -euxo pipefail

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware
EXPECTED="$REPO/third_party/LMCache/lmcache"

source "$VENV/bin/activate"
cd "$REPO"

echo "=== ENVIRONMENT ==="
hostname
nvidia-smi
git rev-parse HEAD

python - <<'PY'
from importlib import metadata
from pathlib import Path
import importlib
import inspect
import torch
import lmcache

expected = Path(
    "/mnt/shared/gpfs/home/sriramc2/"
    "KV-Aware-vLLM/third_party/LMCache/lmcache"
).resolve()

print("lmcache_version:", metadata.version("lmcache"))
print("lmcache_file:", lmcache.__file__)
print("lmcache_loader:", lmcache.__loader__)
print("torch_version:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())

assert torch.cuda.is_available()
assert lmcache.__file__ is not None
assert Path(lmcache.__file__).resolve().parent == expected

modules = [
    "lmcache.integration.vllm.vllm_v1_adapter",
    "lmcache.v1.cache_engine",
    "lmcache.v1.gpu_connector.gpu_connectors",
    "lmcache.v1.lookup_client.lmcache_async_lookup_client",
    "lmcache.v1.memory_management",
    "lmcache.v1.storage_backend.local_cpu_backend",
    "lmcache.v1.storage_backend.local_disk_backend",
    "lmcache.v1.storage_backend.storage_manager",
]

for name in modules:
    module = importlib.import_module(name)
    path = Path(inspect.getsourcefile(module)).resolve()
    print(name, "->", path)
    assert expected in path.parents

print("Python source imports: OK")
PY

echo "=== C_OPS ==="
python - <<'PY'
import lmcache.c_ops
print("lmcache.c_ops file:", lmcache.c_ops.__file__)
print("lmcache.c_ops import: OK")
PY

echo "=== METADATA ==="
python -m pip show lmcache
python -m pip list --editable
python -m pip check || true

echo "=== VERIFICATION PASSED ==="

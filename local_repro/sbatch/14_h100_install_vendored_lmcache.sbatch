#!/bin/bash
#SBATCH --job-name=install_lmcache_src
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -euxo pipefail

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
SRC="$REPO/third_party/LMCache"
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs

# Prevent two installation jobs from modifying the same shared venv.
exec 9>/mnt/shared/gpfs/home/sriramc2/venvs/kvaware.lmcache-install.lock
flock -n 9 || {
    echo "Another LMCache installation is already running."
    exit 1
}

source "$VENV/bin/activate"
cd "$REPO"

echo "=== NODE AND GPU ==="
date
hostname
nvidia-smi

echo "=== SOURCE REVISION ==="
git rev-parse HEAD
git status --short -- third_party/LMCache

# Refuse to build from uncommitted LMCache source.
if [[ -n "$(git status --porcelain -- third_party/LMCache)" ]]; then
    echo "Vendored LMCache has uncommitted changes."
    exit 1
fi

echo "=== CURRENT ENVIRONMENT ==="
which python
python --version
python -m pip show lmcache
python - <<'PY'
from importlib import metadata
from pathlib import Path
import lmcache
import torch

print("before_lmcache_version:", metadata.version("lmcache"))
print("before_lmcache_path:", Path(lmcache.__file__).resolve())
print("torch_version:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
assert torch.cuda.is_available()
PY

echo "=== PYTHON SYNTAX CHECK ==="
python -m compileall -q "$SRC/lmcache"

echo "=== EDITABLE SOURCE INSTALL ==="

# LMCache is vendored inside another Git repository, so provide the known
# upstream version explicitly rather than letting setuptools-scm inspect
# the parent KV-Aware-vLLM repository.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.4.6

# Build against the installed CUDA-12 Torch environment.
export LMCACHE_CUDA_MAJOR=12

# Avoid excessive compilation parallelism.
export MAX_JOBS=8

python -m pip install \
    --editable "$SRC" \
    --no-build-isolation \
    --no-deps

echo "=== VERIFY EDITABLE INSTALL ==="
python - <<'PY'
from importlib import metadata
from pathlib import Path
import importlib
import inspect
import json
import torch
import lmcache

expected = Path(
    "/mnt/shared/gpfs/home/sriramc2/"
    "KV-Aware-vLLM/third_party/LMCache/lmcache"
).resolve()

actual = Path(lmcache.__file__).resolve().parent

print("lmcache_version:", metadata.version("lmcache"))
print("lmcache_actual_path:", actual)
print("lmcache_expected_path:", expected)
print("torch_version:", torch.__version__)
print("torch_cuda:", torch.version.cuda)

if actual != expected:
    raise RuntimeError(
        f"LMCache did not resolve to vendored source: {actual}"
    )

direct_url = metadata.distribution("lmcache").read_text("direct_url.json")
print("direct_url_json:", direct_url)

if direct_url:
    data = json.loads(direct_url)
    if not data.get("dir_info", {}).get("editable", False):
        raise RuntimeError("LMCache metadata does not report editable=true")

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
    source_file = Path(inspect.getsourcefile(module)).resolve()
    print(name, "->", source_file)
    if expected not in source_file.parents:
        raise RuntimeError(
            f"{name} was not imported from vendored source: {source_file}"
        )

print("All modified LMCache modules resolve to vendored source.")
PY

echo "=== VERIFY COMPILED EXTENSION ==="
python -c \
    'import lmcache.c_ops; print("lmcache.c_ops import: OK")'

echo "=== PIP METADATA ==="
python -m pip show lmcache
python -m pip list --editable

echo "=== PIP CHECK ==="
python -m pip check || true

echo "=== INSTALL SANITY PASSED ==="
date

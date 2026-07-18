#!/bin/bash
#SBATCH --job-name=pip_vllm_editable
#SBATCH --partition=gpu-a30-dchen
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euxo pipefail

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs

source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware/bin/activate

cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM

echo "=== NODE INFO ==="
date
hostname
whoami
pwd

echo "=== PYTHON INFO ==="
which python
python --version
python -m pip --version

echo "=== GPU INFO ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

echo "=== TORCH/CUDA INFO ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

echo "=== CUDA TOOLKIT INFO ==="
which nvcc || true
nvcc --version || true
echo "CUDA_HOME=${CUDA_HOME:-unset}"
readlink -f /usr/local/cuda || true
ls -ld /usr/local/cuda* 2>/dev/null || true

echo "=== START PIP INSTALL EDITABLE ==="
MAX_JOBS=4 python -m pip install -e . --no-build-isolation -v

echo "=== VERIFY VLLM INSTALL ==="
python - <<'PY'
import pathlib
import vllm

import torch
print("torch:", torch.__version__, torch.version.cuda)

print("vllm:", vllm.__file__)
print("vllm package dir:", pathlib.Path(vllm.__file__).parent)

import vllm._C
print("vllm._C OK")

from vllm import LLM, SamplingParams
print("LLM import OK")
print("SamplingParams import OK")

from vllm.platforms import current_platform
print("platform:", current_platform)
print("device_type:", repr(current_platform.device_type))
PY

echo "=== DONE ==="
date
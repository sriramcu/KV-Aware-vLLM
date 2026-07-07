#!/bin/bash
#SBATCH --job-name=kvaware_a30_sanity
#SBATCH --partition=gpu-a30-dchen
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euxo pipefail

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs

echo "=== BASIC INFO ==="
date
hostname
whoami
pwd
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware/bin/activate
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM

echo "=== PYTHON/VENV ==="
which python
python --version

echo "=== TORCH ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

echo "=== VLLM LOCATION ==="
python - <<'PY'
import vllm
print(vllm.__file__)
PY

echo "=== PATCH CHECKS ==="
test -f local_repro/cpu_offload_lmcache_sriram.py
grep -n "HKV_ROOT" local_repro/cpu_offload_lmcache_sriram.py
grep -n "VLLM_KV_IMPORTANCE_ENABLE" local_repro/cpu_offload_lmcache_sriram.py
grep -R "def _get_request_block_tiers" -n . | head
grep -R "target_tiers=target_tiers" -n . | head
grep -R "\[SY\] req_id" -n . | head || true

echo "=== DONE ==="

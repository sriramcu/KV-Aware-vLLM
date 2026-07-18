#!/bin/bash
#SBATCH --job-name=vllm_smoke
#SBATCH --partition=gpu-h100-dchen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH -o /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH -e /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -euxo pipefail

cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM

source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware/bin/activate

export PYTHONPATH=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM:${PYTHONPATH:-}
export HF_HOME=/mnt/shared/gpfs/home/sriramc2/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "===== GPU ====="
nvidia-smi

echo "===== ENV ====="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY

echo "===== VLLM CONFIG SMOKE ====="
python local_repro/smoke_vllm_config.py

echo "===== DONE ====="
date
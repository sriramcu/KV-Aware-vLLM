#!/bin/bash
#SBATCH --job-name=vllm_lmcache_smoke
#SBATCH --partition=gpu-h100-dchen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH -o /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH -e /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -euxo pipefail

cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware/bin/activate

export PYTHONPATH=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM:${PYTHONPATH:-}
export HF_HOME=/mnt/shared/gpfs/home/sriramc2/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export CUDA_DEVICE_ORDER=PCI_BUS_ID

nvidia-smi
python local_repro/smoke_vllm_lmcache.py
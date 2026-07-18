#!/bin/bash
#SBATCH --job-name=h100_all_ds_250
#SBATCH --partition=gpu-h100-dchen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH -o /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH -e /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -euo pipefail

cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware/bin/activate

export PYTHONPATH=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM:${PYTHONPATH:-}
export HF_HOME=/mnt/shared/gpfs/home/sriramc2/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/mnt/shared/gpfs/home/sriramc2/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/mnt/shared/gpfs/home/sriramc2/.cache/huggingface/transformers
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MAX_QUESTIONS=250

for DS in hotpotqa 2wikimultihop musique medical; do
  echo "============================================================"
  echo "RUNNING DATASET=$DS WO_GNN"
  echo "============================================================"
  export DATASET_NAME="$DS"
  bash local_repro/run_driver.sh wo_gnn 2>&1

  echo "============================================================"
  echo "RUNNING DATASET=$DS WITH_GNN"
  echo "============================================================"
  export DATASET_NAME="$DS"
  bash local_repro/run_driver.sh with_gnn 2>&1
done
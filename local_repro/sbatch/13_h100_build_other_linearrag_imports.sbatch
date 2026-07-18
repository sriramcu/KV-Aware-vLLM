#!/bin/bash
#SBATCH --job-name=build_lr_imports
#SBATCH --partition=gpu-h100-dchen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH -o /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH -e /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -euo pipefail

cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/Hierarchical_KV/LinearRAG
source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware/bin/activate

export PYTHONPATH=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM:/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/Hierarchical_KV/LinearRAG:${PYTHONPATH:-}
export HF_HOME=/mnt/shared/gpfs/home/sriramc2/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/mnt/shared/gpfs/home/sriramc2/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/mnt/shared/gpfs/home/sriramc2/.cache/huggingface/transformers
export CUDA_DEVICE_ORDER=PCI_BUS_ID

for DS in 2wikimultihop musique medical; do
  echo "============================================================"
  echo "BUILDING LinearRAG IMPORT FOR $DS"
  echo "============================================================"

  python build_import_only.py \
    --dataset_name "$DS" \
    --embedding_model_path model/all-mpnet-base-v2 \
    2>&1

  echo "CHECKING $DS"
  ls -lh import/$DS/
done
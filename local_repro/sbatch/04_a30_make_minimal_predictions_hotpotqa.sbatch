#!/bin/bash
#SBATCH --job-name=make_preds_hotpotqa
#SBATCH --partition=gpu-a30-dchen
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euxo pipefail

source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware/bin/activate

cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/Hierarchical_KV/LinearRAG

python make_minimal_predictions.py

ls -lh results/hotpotqa/self_generated/predictions.json
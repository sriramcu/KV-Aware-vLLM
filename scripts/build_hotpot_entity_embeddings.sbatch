#!/usr/bin/env bash
#SBATCH --job-name=kvq650-10batch
#SBATCH --partition=gpu-h100-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.err
set -Eeuo pipefail
REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029/bin/activate
cd "$REPO"
python "$HOME/repair_hotpot_entity_embeddings.py" --repo "$REPO" --dataset hotpotqa --device cuda --batch-size 128
python scripts/h100_next8/preflight_dataset.py --repo "$REPO" --dataset hotpotqa --questions 650 --policy bayes

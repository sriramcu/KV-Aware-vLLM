#!/usr/bin/env bash
#SBATCH --job-name=gnn-h100-smoke
#SBATCH --partition=gpu-h100-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err
set -Eeuo pipefail
export KV_GNN_PROFILE=h100_smoke
exec bash /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/scripts/run_shortq_gnn_exclusive_common.sh

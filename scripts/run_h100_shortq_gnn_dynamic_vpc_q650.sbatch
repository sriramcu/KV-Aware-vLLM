#!/usr/bin/env bash
#SBATCH --job-name=kvaware-gnn-vpc-q650
#SBATCH --partition=gpu-h100-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -Eeuo pipefail
export KV_GNN_DYNAMIC_PROFILE=h100_q650
exec bash /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/scripts/run_shortq_gnn_dynamic_vpc_common.sh

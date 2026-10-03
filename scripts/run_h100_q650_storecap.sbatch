#!/usr/bin/env bash
#SBATCH --job-name=kvaware-storecap-q650
#SBATCH --partition=gpu-h100-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.err

set -Eeuo pipefail
REPO="${REPO:-/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM}"
exec bash "$REPO/scripts/run_h100_q650_storecap.sh"

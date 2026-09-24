#!/usr/bin/env bash
#SBATCH --job-name=l40-gnn-q200-thresh
#SBATCH --partition=gpu-l40-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -Eeuo pipefail

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro

mkdir -p "${RUN_ROOT}/logs"
cd "$REPO"

export KV_GNN_PROFILE=l40_q200

# L40 batch sweep showed BS=1 was fastest.
export GNN_INFERENCE_BATCH_SIZE=1

# Never modify the learned/threshold distribution for this experiment.
export SMOKE_FORCE_MIN_UNIQUE_PER_TIER=0

echo "========================================"
echo "L40 Short-Q threshold policy Q200"
echo "Policy: GPU >= 6 -> L0"
echo "        CPU >= 5 -> L1"
echo "        otherwise -> L2"
echo "GNN inference batch: ${GNN_INFERENCE_BATCH_SIZE}"
echo "Job: ${SLURM_JOB_ID}"
echo "========================================"

bash scripts/run_shortq_gnn_exclusive_common.sh
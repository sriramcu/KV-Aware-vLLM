#!/usr/bin/env bash
#SBATCH --job-name=l40-gnn-q200
#SBATCH --partition=gpu-l40-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -Eeuo pipefail

GNN_BS="${1:-1}"

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro

mkdir -p "${RUN_ROOT}/logs"
cd "$REPO"

# Same L40 GNN configuration as the validated smoke,
# but increase the workload from Q96 -> Q200.
export KV_GNN_PROFILE=l40_smoke

export NUM_QUESTIONS=200
export SUBMISSION_BATCH_SIZE=200

# Keep same generation settings as your Q200 no-GNN comparison.
export MIN_TOKENS=32
export MAX_TOKENS=128

# L40 sweep showed BS=1 is best; can still override from CLI.
export GNN_INFERENCE_BATCH_SIZE="$GNN_BS"

# Do not artificially force tier coverage in the performance run.
export SMOKE_FORCE_MIN_UNIQUE_PER_TIER=0

# Same diagnostics/safeguards as comparison job.
export LMCACHE_MP_STAGE1_FRESHNESS_GUARD_S=270
export LMCACHE_MP_CONGESTION_DEBUG=1
export LMCACHE_MP_CONGESTION_LOG_EVERY=25
export LMCACHE_MP_CONGESTION_SLOW_S=5
export LMCACHE_MP_CHTHM_DEBUG=1

echo "========================================"
echo "L40 Short-Q GNN Q200"
echo "questions:              ${NUM_QUESTIONS}"
echo "submission batch:       ${SUBMISSION_BATCH_SIZE}"
echo "GNN inference batch:    ${GNN_INFERENCE_BATCH_SIZE}"
echo "Slurm job:              ${SLURM_JOB_ID}"
echo "========================================"

bash scripts/run_shortq_gnn_exclusive_common.sh
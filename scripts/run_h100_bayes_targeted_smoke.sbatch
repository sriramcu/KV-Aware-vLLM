#!/usr/bin/env bash
#SBATCH --job-name=kvbayes-smoke3
#SBATCH --partition=gpu-h100-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.err
set -Eeuo pipefail
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
export KV_BAYES_PRIOR_FILE="${KV_BAYES_PRIOR_FILE:-/mnt/shared/gpfs/home/sriramc2/runs/kvaware/rank_h100_calibrated.json}"
[[ -s "$KV_BAYES_PRIOR_FILE" ]] || { echo "ERROR: calibrate priors first" >&2; exit 2; }
export KV_BAYES_DIRECT_COST_FEEDBACK=1 KV_BAYES_USE_COST_PROXIES=0
export KV_BAYES_READMISSION=1 KV_BAYES_POSTERIOR_MODE=fixed
export KV_BAYES_PRIOR_WEIGHT=1 KV_BAYES_GPU_GB=10
export LMCACHE_L2_SIZE_GB="${LMCACHE_L2_SIZE_GB:-100}"
export KV_BAYES_DISK_GB="$LMCACHE_L2_SIZE_GB"
export NUM_QUESTIONS="${NUM_QUESTIONS:-80}" SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-20}"
export MIN_TOKENS="${MIN_TOKENS:-16}" MAX_TOKENS="${MAX_TOKENS:-16}"
export LMCACHE_L1_SIZE_GB=225 SMOKE_FORCE_MIN_UNIQUE_PER_PLACEMENT=0
export EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-bayes_three_cost_q${NUM_QUESTIONS}_b${SUBMISSION_BATCH_SIZE}}"
bash scripts/run_h100_bayes_canonical_smoke.sh

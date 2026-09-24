#!/usr/bin/env bash
#SBATCH --job-name=l40-gnn-q200-top2
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

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro

mkdir -p "${RUN_ROOT}/logs"

cd "$REPO"

# Reuse the validated L40 configuration.
export KV_GNN_PROFILE=l40_smoke

# Q200 experiment.
export NUM_QUESTIONS=200
export SUBMISSION_BATCH_SIZE=200
export MIN_TOKENS=32
export MAX_TOKENS=128

# L40 sweep showed request batching did not help.
export GNN_INFERENCE_BATCH_SIZE=1

# Never alter the natural policy distribution.
export SMOKE_FORCE_MIN_UNIQUE_PER_TIER=0

# Same diagnostics / safety settings as the earlier Q200 runs.
export LMCACHE_MP_STAGE1_FRESHNESS_GUARD_S=270

export LMCACHE_MP_CONGESTION_DEBUG=1
export LMCACHE_MP_CONGESTION_LOG_EVERY=25
export LMCACHE_MP_CONGESTION_SLOW_S=5

export LMCACHE_MP_CHTHM_DEBUG=1

echo "============================================================"
echo "L40 Short-Q Q200"
echo "Policy is selected exclusively by chunk_voting.py"
echo
python - <<'PY'
from Hierarchical_KV.shortq_placement.chunk_voting import (
    selected_vote_policy_name,
)
print("chunk vote policy:", selected_vote_policy_name())
PY
echo "============================================================"

bash scripts/run_shortq_gnn_exclusive_common.sh
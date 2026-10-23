#!/usr/bin/env bash
#SBATCH --job-name=kvbayes-cell
#SBATCH --partition=gpu-h100-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.err
set -Eeuo pipefail
REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
cd "$REPO"
: "${SWEEP_CELL:?supply SWEEP_CELL}"
case "$SWEEP_CELL" in
  nognn_original|nognn_canonical|gnn_original|gnn_canonical|bayes_original|bayes_canonical|bayes_random_canonical) ;;
  *) [[ "$SWEEP_CELL" == b_* || "$SWEEP_CELL" == c_* ]] || { echo "Invalid cell: $SWEEP_CELL" >&2; exit 2; } ;;
esac
export NUM_QUESTIONS=650 SUBMISSION_BATCH_SIZE=100
export MIN_TOKENS=128 MAX_TOKENS=128 MAX_NUM_SEQS=16
export LMCACHE_L1_SIZE_GB="${SWEEP_CPU_GB:-225}"
export LMCACHE_L2_SIZE_GB="${SWEEP_DISK_GB:-100}"
export LMCACHE_MP_MAX_INFLIGHT_STORE_GB="${SWEEP_STORE_CAP_GB:-0.25}"
export KV_PASSAGE_ORDER_POLICY="${SWEEP_ORDER:-canonical}"
export KV_REQUEST_ORDER=seeded_shuffle KV_REQUEST_ORDER_SEED=0
export REQUEST_ORDER=seeded_shuffle REQUEST_ORDER_SEED=0
export WARM_ORDER=random WARM_ORDER_SEED=1
export ENABLE_VPC=1 KV_GNN_AWARE_VPC=1
export KV_MUTUAL_PREFIX=1 KV_VPC_SUFFICIENT_BYPASS=1
export KV_COLD_WARM_STORE_BARRIER=1 LMCACHE_MP_PREFIX_DIAGNOSTICS=1
export VLLM_PHASE_TRACE=1 VLLM_PHASE_TRACE_DEVICE_TIMING=0
export LMCACHE_L2_LOAD_MAX_IN_FLIGHT=2 LMCACHE_L2_LOOKUP_MAX_IN_FLIGHT=16
export EXPERIMENT_LABEL="${SWEEP_CELL}"
export GRID_LABEL="${SWEEP_CELL}_q650_b100"
export KV_BAYES_ENABLE=0
if [[ "$SWEEP_CELL" == nognn_* || "${SWEEP_POLICY:-}" == "nognn" ]]; then
  export KV_GNN_AWARE_VPC=0
  # This baseline has no Bayesian disk/GPU *policy budgets*. CPU allocation
  # and disk store cap are real runtime controls; other budgets are not.
  bash scripts/h100_10batch/run_h100_nognn_q650_safe.sbatch
  exit
fi
export KV_GNN_DYNAMIC_PROFILE=h100_q650
export KV_GNN_L1_BACKING=1 KV_GNN_L2_BACKING=0
export GNN_CHUNK_VOTE_POLICY=threshold_with_drop_plurality
export KV_BAYES_GPU_GB="${SWEEP_GPU_GB:-10}"
export KV_BAYES_DISK_GB="$LMCACHE_L2_SIZE_GB"
if [[ "$SWEEP_CELL" == bayes_* || "${SWEEP_POLICY:-}" == "bayes" ]]; then
  export KV_BAYES_ENABLE=1
  export KV_BAYES_PRIOR_FILE="${SWEEP_PRIOR_FILE:-/mnt/shared/gpfs/home/sriramc2/runs/kvaware/rank_h100_calibrated.json}"
  [[ -s "$KV_BAYES_PRIOR_FILE" ]] || { echo "Missing calibrated prior: $KV_BAYES_PRIOR_FILE" >&2; exit 2; }
  export KV_BAYES_PRIOR_WEIGHT="${SWEEP_PRIOR_WEIGHT:-1}"
  export KV_BAYES_DIRECT_COST_FEEDBACK=1 KV_BAYES_USE_COST_PROXIES=0
  export KV_BAYES_READMISSION=1 KV_BAYES_POSTERIOR_MODE="${SWEEP_POSTERIOR_MODE:-fixed}"
  export KV_BAYES_SCORE_PRIOR_COUNT="${SWEEP_SCORE_PRIOR_COUNT:-10}"
  export KV_BAYES_COST_PRIOR_COUNT="${SWEEP_COST_PRIOR_COUNT:-5}"
  if [[ "$SWEEP_CELL" == bayes_random_canonical ]]; then
    export KV_BAYES_RANDOM_SCORE_SEED=42
  fi
fi
export KV_BAYES_SCORE_FORMAT=expected_ordinal_0_100
bash scripts/run_shortq_gnn_dynamic_vpc_common.sh

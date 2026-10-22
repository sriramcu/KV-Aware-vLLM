#!/usr/bin/env bash
# Run INSIDE an allocated 2x H100 `srun`; do not submit another Slurm job.
set -Eeuo pipefail
export KV_GNN_DYNAMIC_PROFILE=h100_smoke
export KV_BAYES_ENABLE=1
export KV_PASSAGE_ORDER_POLICY=canonical
export KV_REQUEST_ORDER=seeded_shuffle KV_REQUEST_ORDER_SEED=0
export WARM_ORDER=random WARM_ORDER_SEED=1
export NUM_QUESTIONS="${NUM_QUESTIONS:-12}"
export SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-4}"
export MIN_TOKENS="${MIN_TOKENS:-16}" MAX_TOKENS="${MAX_TOKENS:-16}"
export MAX_NUM_SEQS=16
export LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-225}"
export LMCACHE_MP_MAX_INFLIGHT_STORE_GB=0.25
export LMCACHE_L2_LOAD_MAX_IN_FLIGHT=2 LMCACHE_L2_LOOKUP_MAX_IN_FLIGHT=16
export KV_MUTUAL_PREFIX=1 KV_VPC_SUFFICIENT_BYPASS=1
export KV_COLD_WARM_STORE_BARRIER=1 LMCACHE_MP_PREFIX_DIAGNOSTICS=1
export KV_GNN_AWARE_VPC=1 KV_GNN_L1_BACKING=1 KV_GNN_L2_BACKING=0
export GNN_CHUNK_VOTE_POLICY=threshold_with_drop_plurality
export VLLM_PHASE_TRACE=1 VLLM_PHASE_TRACE_DEVICE_TIMING=0
export EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-bayes_canonical_smoke}"
export KV_BAYES_GPU_GB="${KV_BAYES_GPU_GB:-10}"
export LMCACHE_L2_SIZE_GB="${LMCACHE_L2_SIZE_GB:-100}"
export KV_BAYES_DISK_GB="${KV_BAYES_DISK_GB:-$LMCACHE_L2_SIZE_GB}"
export KV_BAYES_PRIOR_WEIGHT="${KV_BAYES_PRIOR_WEIGHT:-1}"
# This narrowly binned prior is diagnostic only: derived from the initial
# Q12 smoke score range, not an independently calibrated production prior.
SMOKE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KV_BAYES_PRIOR_FILE="${KV_BAYES_PRIOR_FILE:-${SMOKE_REPO}/Hierarchical_KV/shortq_placement/priors/rank_smoke_diagnostic.json}"
# Default is prior-only: LMCache diagnostic log intervals are not exclusive GET costs.
export KV_BAYES_USE_COST_PROXIES="${KV_BAYES_USE_COST_PROXIES:-0}"
exec bash /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/scripts/run_shortq_gnn_dynamic_vpc_common.sh

#!/usr/bin/env bash
#SBATCH --job-name=kvaware-canon-gnn-q650
#SBATCH --partition=gpu-h100-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.err

set -Eeuo pipefail
# Reproduce main job 29265 settings EXCEPT canonical passage order + trace.
# The two independent request shuffle seeds are intentionally different.
export KV_GNN_DYNAMIC_PROFILE=h100_q650
export KV_PASSAGE_ORDER_POLICY="${KV_PASSAGE_ORDER_POLICY:-canonical}"
export KV_REQUEST_ORDER=seeded_shuffle
export KV_REQUEST_ORDER_SEED=0
export WARM_ORDER=random
export WARM_ORDER_SEED=1
export SUBMISSION_BATCH_SIZE=200
export MIN_TOKENS=128 MAX_TOKENS=128
export MAX_NUM_SEQS=16
export LMCACHE_L1_SIZE_GB=225
export LMCACHE_MP_MAX_INFLIGHT_STORE_GB=0.25
export LMCACHE_L2_LOAD_MAX_IN_FLIGHT=2
export LMCACHE_L2_LOOKUP_MAX_IN_FLIGHT=16
export KV_MUTUAL_PREFIX=1
export KV_VPC_SUFFICIENT_BYPASS=1
export KV_COLD_WARM_STORE_BARRIER=1
export LMCACHE_MP_PREFIX_DIAGNOSTICS=1
export VLLM_GNN_AWARE_VPC_WINDOW=256
export KV_GNN_AWARE_VPC=1 KV_GNN_L1_BACKING=1 KV_GNN_L2_BACKING=0
export GNN_CHUNK_VOTE_POLICY=threshold_with_drop_plurality
export GNN_CHUNK_GPU_MIN=8 GNN_CHUNK_CPU_MIN=2
export EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-canonical_trace_q650_b200_seed0_seed1}"
export VLLM_PHASE_TRACE="${VLLM_PHASE_TRACE:-1}"
# Optional GPU event timing is UNVALIDATED on H100 capture streams; off by default.
export VLLM_PHASE_TRACE_DEVICE_TIMING="${VLLM_PHASE_TRACE_DEVICE_TIMING:-0}"
exec bash /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/scripts/run_shortq_gnn_dynamic_vpc_common.sh

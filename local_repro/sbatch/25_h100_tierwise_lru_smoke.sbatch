#!/bin/bash
#SBATCH --job-name=tier_lru
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -euxo pipefail

HOME_DIR=/mnt/shared/gpfs/home/sriramc2
REPO="$HOME_DIR/KV-Aware-vLLM"
RUN_ROOT="$HOME_DIR/runs/kvaware_repro"
SCRATCH2_ROOT=/scratch2/sriramc2

mkdir -p "$RUN_ROOT/logs"
mkdir -p "$SCRATCH2_ROOT"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1

# ------------------------------------------------------------
# Workload
# ------------------------------------------------------------
export DATASET_NAME=medical
# change for larger runs
export MAX_QUESTIONS=8
export SUBMISSION_BATCH_SIZE=8
export VLLM_MAX_NUM_SEQS=4

# ------------------------------------------------------------
# GNN placement / importance
# ------------------------------------------------------------
export VLLM_KV_IMPORTANCE_ENABLE=1
export VLLM_KV_RANDOM_PLACEMENT_ENABLE=0
export VLLM_KV_CPU_ONLY=0
export GNN_KV_BLOCK_SIZE=16

# ------------------------------------------------------------
# Full-sacrifice vLLM tierwise LRU
# ------------------------------------------------------------
export VLLM_KV_TIERWISE_LRU_ENABLE=1
export VLLM_KV_TIERWISE_LRU_TIERS=gpu,cpu,disk
export VLLM_KV_TIERWISE_LRU_CHUNK_SIZE=512

# Detailed per-block trace.
# Keep =1 for this post-fix 8-question validation run.
# Set =0 for larger runs.
export VLLM_KV_TIERWISE_LRU_TRACE_ENABLE=1

# Aggregate statistics.
# Keep enabled even for large runs.
export VLLM_KV_TIERWISE_LRU_SUMMARY_ENABLE=1
export VLLM_KV_TIERWISE_LRU_SUMMARY_INTERVAL=1000

# ------------------------------------------------------------
# LMCache
# ------------------------------------------------------------
export SC_LMCACHE_PROFILE=vanilla
export SC_LMCACHE_MAX_LOCAL_CPU_SIZE=20
export SC_LMCACHE_LOCAL_DISK_ENABLE=1
export SC_LMCACHE_MAX_LOCAL_DISK_SIZE=100
export SC_LMCACHE_DATA_DIR="$SCRATCH2_ROOT/kvaware_tierlru_${SLURM_JOB_ID}"

export LMCACHE_LOOKUP_TIMEOUT_MS=3000
export CLEAN_OLD_LMCACHE=0

# Keep LMCache tier trace ON for this small validation run so we can
# compare its gpu/cpu/disk chunk votes against vLLM assignments.
# Set =0 for larger runs.
export SC_LMCACHE_TIER_TRACE_ENABLE=1

export SC_LMCACHE_IO_TRACE_ENABLE=0
export SC_LMCACHE_LOAD_TRACE_ENABLE=0
export SC_LMCACHE_MEMORY_TRACE_ENABLE=0
export SC_LMCACHE_LIFECYCLE_TRACE_ENABLE=0

# Fill in remaining defaults without overriding explicit values above.
source "$REPO/local_repro/sc_lmcache_knobs.sh"

echo "============================================================"
echo "TIERWISE LRU TEST"
echo "job                     = ${SLURM_JOB_ID}"
echo "host                    = $(hostname)"
echo "questions               = $MAX_QUESTIONS"
echo "submission batch        = $SUBMISSION_BATCH_SIZE"
echo "max_num_seqs            = $VLLM_MAX_NUM_SEQS"
echo "tierwise LRU            = $VLLM_KV_TIERWISE_LRU_ENABLE"
echo "tiers                   = $VLLM_KV_TIERWISE_LRU_TIERS"
echo "tier chunk size         = $VLLM_KV_TIERWISE_LRU_CHUNK_SIZE"
echo "tierwise trace          = $VLLM_KV_TIERWISE_LRU_TRACE_ENABLE"
echo "tierwise summary        = $VLLM_KV_TIERWISE_LRU_SUMMARY_ENABLE"
echo "summary interval        = $VLLM_KV_TIERWISE_LRU_SUMMARY_INTERVAL"
echo "LMCache tier trace      = $SC_LMCACHE_TIER_TRACE_ENABLE"
echo "LMCache data dir        = $SC_LMCACHE_DATA_DIR"
echo "============================================================"

nvidia-smi

"$REPO/local_repro/run_driver.sh" with_gnn 2>&1

echo "=== TIERWISE LRU TEST DONE ==="
date
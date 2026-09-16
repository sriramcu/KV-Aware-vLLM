#!/bin/bash
#SBATCH --job-name=tier_lru_q650
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
# Workload — match historical Q650 configuration
# ------------------------------------------------------------
export DATASET_NAME=medical
export MAX_QUESTIONS=650
export SUBMISSION_BATCH_SIZE=650
export VLLM_MAX_NUM_SEQS=16

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

# Detailed per-block tracing is too verbose for Q650.
export VLLM_KV_TIERWISE_LRU_TRACE_ENABLE=0

# Keep aggregate eviction statistics enabled.
export VLLM_KV_TIERWISE_LRU_SUMMARY_ENABLE=1
export VLLM_KV_TIERWISE_LRU_SUMMARY_INTERVAL=1000

# ------------------------------------------------------------
# LMCache — vanilla Q650-style configuration
# ------------------------------------------------------------
export SC_LMCACHE_PROFILE=vanilla
export SC_LMCACHE_MAX_LOCAL_CPU_SIZE=20
export SC_LMCACHE_LOCAL_DISK_ENABLE=1
export SC_LMCACHE_MAX_LOCAL_DISK_SIZE=450

export SC_LMCACHE_DATA_DIR="$SCRATCH2_ROOT/kvaware_tierlru_q650_${SLURM_JOB_ID}"

export LMCACHE_LOOKUP_TIMEOUT_MS=3000
export CLEAN_OLD_LMCACHE=1

# Disable verbose traces for the timed run.
export SC_LMCACHE_TIER_TRACE_ENABLE=0
export SC_LMCACHE_IO_TRACE_ENABLE=0
export SC_LMCACHE_LOAD_TRACE_ENABLE=0
export SC_LMCACHE_MEMORY_TRACE_ENABLE=0
export SC_LMCACHE_LIFECYCLE_TRACE_ENABLE=0

# Fill in remaining LMCache defaults without overriding values above.
source "$REPO/local_repro/sc_lmcache_knobs.sh"

echo "============================================================"
echo "Q650 TIERWISE LRU RUN"
echo "job                     = ${SLURM_JOB_ID}"
echo "host                    = $(hostname)"
echo "dataset                 = $DATASET_NAME"
echo "questions               = $MAX_QUESTIONS"
echo "submission batch        = $SUBMISSION_BATCH_SIZE"
echo "max_num_seqs            = $VLLM_MAX_NUM_SEQS"
echo "tierwise LRU            = $VLLM_KV_TIERWISE_LRU_ENABLE"
echo "tiers                   = $VLLM_KV_TIERWISE_LRU_TIERS"
echo "tier chunk size         = $VLLM_KV_TIERWISE_LRU_CHUNK_SIZE"
echo "tierwise trace          = $VLLM_KV_TIERWISE_LRU_TRACE_ENABLE"
echo "tierwise summary        = $VLLM_KV_TIERWISE_LRU_SUMMARY_ENABLE"
echo "summary interval        = $VLLM_KV_TIERWISE_LRU_SUMMARY_INTERVAL"
echo "CPU LMCache/rank        = $SC_LMCACHE_MAX_LOCAL_CPU_SIZE GiB"
echo "disk limit              = $SC_LMCACHE_MAX_LOCAL_DISK_SIZE GiB"
echo "LMCache data dir        = $SC_LMCACHE_DATA_DIR"
echo "============================================================"

echo "=== SCRATCH2 FILESYSTEM ==="
df -h "$SCRATCH2_ROOT"

nvidia-smi

"$REPO/local_repro/run_driver.sh" with_gnn 2>&1

echo "=== Q650 TIERWISE LRU RUN DONE ==="
date
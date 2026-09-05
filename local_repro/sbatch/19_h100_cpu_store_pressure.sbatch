#!/bin/bash
#SBATCH --job-name=lmcache_store_pressure
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -euxo pipefail

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1

# ------------------------------------------------------------
# Purpose: deliberately trigger LMCache's graceful
# "CPU memory pressure -> stop storing" path.
# ------------------------------------------------------------

export SC_LMCACHE_PROFILE=custom

# Small CPU staging/cache pool PER TP RANK.
# This is deliberately much smaller than a normal experiment.
export SC_LMCACHE_MAX_LOCAL_CPU_SIZE=4

# Enough simultaneous distinct requests to put pressure on that pool.
export DATASET_NAME=medical
export MAX_QUESTIONS=20
export SUBMISSION_BATCH_SIZE=20
export VLLM_MAX_NUM_SEQS=20

export LMCACHE_LOOKUP_TIMEOUT_MS=3000

# We NEED disk enabled.
# Slow async disk PUTs retain references to CPU MemoryObjs.
export SC_LMCACHE_LOCAL_DISK_ENABLE=1

# Do NOT bound disk PUTs for this test.
# We specifically want queued PUTs retaining CPU objects.
export SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE=0

# Keep this close to vanilla behavior.
export SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE=0
export SC_LMCACHE_WORKER_LOOKUP_ADMISSION_ENABLE=0

export SC_LMCACHE_DISK_RESIDENT_PUT_DEDUP_ENABLE=0
export SC_LMCACHE_COLD_WARM_PUT_BARRIER_ENABLE=0

export SC_LMCACHE_PVTSC_ENABLE=0
export SC_LMCACHE_PVTSC_WITHIN_FILE_ENABLE=0

# Diagnostics.
export SC_LMCACHE_MEMORY_TRACE_ENABLE=1
export SC_LMCACHE_MEMORY_TRACE_INTERVAL_S=1
export SC_LMCACHE_IO_TRACE_ENABLE=1
export SC_LMCACHE_LOAD_TRACE_ENABLE=1

# Optional, very verbose:
export SC_LMCACHE_LIFECYCLE_TRACE_ENABLE=0

export CLEAN_OLD_LMCACHE=1

# Use the normal job-specific shared-NFS path.
unset SC_LMCACHE_DATA_DIR

source "$REPO/local_repro/sc_lmcache_knobs.sh"

echo "============================================================"
echo "LMCache graceful store-pressure reproduction"
echo "CPU pool/rank      = $SC_LMCACHE_MAX_LOCAL_CPU_SIZE GiB"
echo "questions          = $MAX_QUESTIONS"
echo "submission batch   = $SUBMISSION_BATCH_SIZE"
echo "max_num_seqs       = $VLLM_MAX_NUM_SEQS"
echo "disk PUT admission = $SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE"
echo "============================================================"

"$REPO/local_repro/run_driver.sh" wo_gnn 2>&1
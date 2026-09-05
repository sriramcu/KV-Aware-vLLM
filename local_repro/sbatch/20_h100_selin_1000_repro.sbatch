#!/bin/bash
#SBATCH --job-name=1000_repro
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=06:00:00
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

# ============================================================
# SELIN 1000-Q BEHAVIOR REPRODUCTION
# ============================================================

export SC_LMCACHE_PROFILE=custom

# Same workload shape as Selin.
export DATASET_NAME=hotpotqa
export MAX_QUESTIONS=1000
export SUBMISSION_BATCH_SIZE=1000
export VLLM_MAX_NUM_SEQS=20

# Same CPU pool.
export SC_LMCACHE_MAX_LOCAL_CPU_SIZE=100
export SC_LMCACHE_MAX_LOCAL_DISK_SIZE=500

# Same timeout.
export LMCACHE_LOOKUP_TIMEOUT_MS=3000

# Disk ON, node-local scratch like Selin.
export SC_LMCACHE_LOCAL_DISK_ENABLE=1
export SC_LMCACHE_DATA_DIR="/scratch/sriramc2/selin_1000_${SLURM_JOB_ID}"

# Vanilla pressure behavior -- no newer protection mechanisms.
export SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE=0
export SC_LMCACHE_WORKER_LOOKUP_ADMISSION_ENABLE=0

export SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE=0
export SC_LMCACHE_DISK_RESIDENT_PUT_DEDUP_ENABLE=0
export SC_LMCACHE_COLD_WARM_PUT_BARRIER_ENABLE=0

export SC_LMCACHE_SERIALIZER_FAIRNESS_ENABLE=0

export SC_LMCACHE_PVTSC_ENABLE=0
export SC_LMCACHE_PVTSC_WITHIN_FILE_ENABLE=0

# Keep the useful CPU-pressure diagnostics.
export SC_LMCACHE_MEMORY_TRACE_ENABLE=1
export SC_LMCACHE_MEMORY_TRACE_INTERVAL_S=5

# Avoid gigantic I/O trace logs for a 1000-Q run.
export SC_LMCACHE_IO_TRACE_ENABLE=0
export SC_LMCACHE_LOAD_TRACE_ENABLE=0
export SC_LMCACHE_LIFECYCLE_TRACE_ENABLE=0

export CLEAN_OLD_LMCACHE=1

source "$REPO/local_repro/sc_lmcache_knobs.sh"

echo "============================================================"
echo "SELIN 1000-Q REPRO"
echo "dataset          = $DATASET_NAME"
echo "questions        = $MAX_QUESTIONS"
echo "submission batch = $SUBMISSION_BATCH_SIZE"
echo "max_num_seqs     = $VLLM_MAX_NUM_SEQS"
echo "CPU/rank         = $SC_LMCACHE_MAX_LOCAL_CPU_SIZE GiB"
echo "disk             = $SC_LMCACHE_DATA_DIR"
echo "============================================================"

echo "=== SCRATCH PREFLIGHT ==="
df -h /scratch/sriramc2
df -B1 /scratch/sriramc2

"$REPO/local_repro/run_driver.sh" wo_gnn 2>&1
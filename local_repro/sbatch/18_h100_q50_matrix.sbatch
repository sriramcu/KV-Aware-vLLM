#!/bin/bash
#SBATCH --job-name=kvaware_q50_matrix
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=codenimbus-003-1
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/q50_matrix_logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/q50_matrix_logs/%x_%j.err

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/q50_matrix_logs

set -euxo pipefail

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1

: "${MATRIX_MODE:?MATRIX_MODE is required: wo_gnn|with_gnn}"
: "${DATASET_NAME:?DATASET_NAME is required: medical|hotpotqa}"
: "${MATRIX_DISK:?MATRIX_DISK is required: 0|1}"
: "${MATRIX_STORAGE:?MATRIX_STORAGE is required: none|nfs|scratch}"
: "${MATRIX_ALL_KNOBS:?MATRIX_ALL_KNOBS is required: 0|1}"
: "${MATRIX_MAX_INFLIGHT:?MATRIX_MAX_INFLIGHT is required}"

case "$MATRIX_MODE" in wo_gnn|with_gnn) ;; *) exit 2 ;; esac
case "$DATASET_NAME" in medical|hotpotqa) ;; *) exit 2 ;; esac
case "$MATRIX_DISK" in 0|1) ;; *) exit 2 ;; esac
case "$MATRIX_ALL_KNOBS" in 0|1) ;; *) exit 2 ;; esac

export SC_LMCACHE_PROFILE=custom
export MAX_QUESTIONS=50
export SUBMISSION_BATCH_SIZE=50
export VLLM_MAX_NUM_SEQS=16
export SC_LMCACHE_MAX_LOCAL_CPU_SIZE=20
export LMCACHE_LOOKUP_TIMEOUT_MS=3000
export SC_LMCACHE_MAX_LOCAL_CPU_SIZE=20
export CLEAN_OLD_LMCACHE=1

# Fixed vanilla mechanisms for every matrix cell.
export SC_LMCACHE_WORKER_LOOKUP_ADMISSION_ENABLE=0
export SC_LMCACHE_WORKER_LOOKUP_MAX_INFLIGHT=1
export SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE=0
export SC_LMCACHE_DISK_PUT_MAX_PENDING=8
export SC_LMCACHE_COLD_WARM_PUT_BARRIER_ENABLE=0

# Instrumentation stays on; these are not policy knobs.
export SC_LMCACHE_IO_TRACE_ENABLE=1
export SC_LMCACHE_LOAD_TRACE_ENABLE=1
export SC_LMCACHE_MEMORY_TRACE_ENABLE=1
export SC_LMCACHE_MEMORY_SNAPSHOT_ENABLE=0
export SC_LMCACHE_LOOKUP_TRACE_ENABLE=0
export SC_LMCACHE_REQUEST_TRACE_ENABLE=0
export SC_LMCACHE_TIER_TRACE_ENABLE=0
export SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE=0
export SC_LMCACHE_DISK_PUT_RESIDENCY_TRACE_ENABLE=1

if [[ "$MATRIX_ALL_KNOBS" == "1" ]]; then
  export SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE=1
  export SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT="$MATRIX_MAX_INFLIGHT"
  export SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_WAIT_MS=0
  export SC_LMCACHE_SERIALIZER_FAIRNESS_ENABLE=1
  export SC_LMCACHE_SERIALIZER_CPU_BURST_RATIO=16
  export SC_LMCACHE_DISK_RESIDENT_PUT_DEDUP_ENABLE=1
  export SC_LMCACHE_PVTSC_ENABLE=1
  export SC_LMCACHE_PVTSC_WITHIN_FILE_ENABLE=1
  export SC_LMCACHE_PVTSC_READ_CHUNK_MIB=4
else
  export SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE=0
  export SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT="$MATRIX_MAX_INFLIGHT"
  export SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_WAIT_MS=0
  export SC_LMCACHE_SERIALIZER_FAIRNESS_ENABLE=0
  export SC_LMCACHE_SERIALIZER_CPU_BURST_RATIO=16
  export SC_LMCACHE_DISK_RESIDENT_PUT_DEDUP_ENABLE=0
  export SC_LMCACHE_PVTSC_ENABLE=0
  export SC_LMCACHE_PVTSC_WITHIN_FILE_ENABLE=0
  export SC_LMCACHE_PVTSC_READ_CHUNK_MIB=4
fi

export SC_LMCACHE_LOCAL_DISK_ENABLE="$MATRIX_DISK"
if [[ "$MATRIX_DISK" == "0" ]]; then
  [[ "$MATRIX_STORAGE" == "none" ]] || exit 2
  unset SC_LMCACHE_DATA_DIR
elif [[ "$MATRIX_STORAGE" == "nfs" ]]; then
  unset SC_LMCACHE_DATA_DIR
elif [[ "$MATRIX_STORAGE" == "scratch" ]]; then
  export SC_LMCACHE_DATA_DIR="/scratch/sriramc2/kvaware_q50_${SLURM_JOB_ID}"
else
  exit 2
fi

source "$REPO/local_repro/sc_lmcache_knobs.sh"

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs
"$REPO/local_repro/run_driver.sh" "$MATRIX_MODE" 2>&1
ls /scratch/sriramc2
echo "ls command above"

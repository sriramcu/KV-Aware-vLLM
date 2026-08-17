#!/bin/bash
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16

set -euxo pipefail

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

if [[ "${SC_LMCACHE_SOURCE_DEFAULT_KNOBS:-1}" == "1" ]]; then
  source "$REPO/local_repro/sc_lmcache_knobs.sh"
fi

: "${GRID_RUN_ID:?GRID_RUN_ID is required}"
: "${GRID_RUN_LABEL:?GRID_RUN_LABEL is required}"
: "${GRID_CONFIG_SHA256:?GRID_CONFIG_SHA256 is required}"
: "${MAX_QUESTIONS:?MAX_QUESTIONS is required}"
: "${SUBMISSION_BATCH_SIZE:?SUBMISSION_BATCH_SIZE is required}"
: "${DATASET_NAME:?DATASET_NAME is required}"
: "${VLLM_MAX_NUM_SEQS:?VLLM_MAX_NUM_SEQS is required}"
: "${SC_LMCACHE_MAX_LOCAL_CPU_SIZE:?SC_LMCACHE_MAX_LOCAL_CPU_SIZE is required}"
: "${ROUND5_STORAGE:?ROUND5_STORAGE is required}"
: "${SC_LMCACHE_COLD_WARM_PUT_BARRIER_ENABLE:?barrier enable is required}"
: "${SC_LMCACHE_DISK_RESIDENT_PUT_DEDUP_ENABLE:?resident PUT dedup enable is required}"
: "${SC_LMCACHE_DISK_PUT_RESIDENCY_TRACE_ENABLE:?PUT residency trace enable is required}"

# Critical Round-5 hygiene: NFS cells must remain on the driver's shared NFS default
# even if the launching shell still has a stale scratch override exported.
case "$ROUND5_STORAGE" in
  shared_nfs)
    unset SC_LMCACHE_DATA_DIR
    ;;
  node_local_ext4)
    : "${SC_LMCACHE_DATA_DIR:?SC_LMCACHE_DATA_DIR is required for node-local cells}"
    case "$SC_LMCACHE_DATA_DIR" in
      /scratch/*/*) ;;
      *)
        echo "Unsafe/non-scratch SC_LMCACHE_DATA_DIR=$SC_LMCACHE_DATA_DIR" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "Unknown ROUND5_STORAGE=$ROUND5_STORAGE" >&2
    exit 2
    ;;
esac

export PYTHONFAULTHANDLER=1

printf '%s\n' \
  "=== SC ROUND5 PUT BARRIER x RESIDENT DEDUP GRID JOB ===" \
  "GRID_RUN_ID=$GRID_RUN_ID" \
  "GRID_RUN_LABEL=$GRID_RUN_LABEL" \
  "GRID_CONFIG_SHA256=$GRID_CONFIG_SHA256" \
  "ROUND5_STORAGE=$ROUND5_STORAGE" \
  "SC_ROUND5_NODELIST=${SC_ROUND5_NODELIST:-unknown}" \
  "MAX_QUESTIONS=$MAX_QUESTIONS" \
  "DATASET_NAME=$DATASET_NAME" \
  "SUBMISSION_BATCH_SIZE=$SUBMISSION_BATCH_SIZE" \
  "VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS" \
  "SC_LMCACHE_MAX_LOCAL_CPU_SIZE=$SC_LMCACHE_MAX_LOCAL_CPU_SIZE" \
  "SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT=$SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT" \
  "SC_LMCACHE_COLD_WARM_PUT_BARRIER_ENABLE=$SC_LMCACHE_COLD_WARM_PUT_BARRIER_ENABLE" \
  "SC_LMCACHE_DISK_RESIDENT_PUT_DEDUP_ENABLE=$SC_LMCACHE_DISK_RESIDENT_PUT_DEDUP_ENABLE" \
  "SC_LMCACHE_DISK_PUT_RESIDENCY_TRACE_ENABLE=$SC_LMCACHE_DISK_PUT_RESIDENCY_TRACE_ENABLE" \
  "SC_LMCACHE_DATA_DIR=${SC_LMCACHE_DATA_DIR:-driver-default-shared-NFS}"

"$REPO/local_repro/run_driver.sh" wo_gnn 2>&1

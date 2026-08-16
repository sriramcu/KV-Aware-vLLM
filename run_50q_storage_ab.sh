#!/usr/bin/env bash
# Submit a matched 50-question storage A/B for the frozen no-GNN configuration.
# A: shared NFS/default LMCache data dir
# B: node-local ext4 scratch LMCache data dir
set -euo pipefail

REPO="${REPO:-/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM}"
SBATCH_SCRIPT="$REPO/local_repro/sbatch/02_h100_wo_gnn.sbatch"
NODELIST="${NODELIST:-codenimbus-003-1}"
SCRATCH_LMCACHE="${SCRATCH_LMCACHE:-/scratch/sriramc2/lmcache_med_s4_seq16_q50}"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "Missing sbatch script: $SBATCH_SCRIPT" >&2
  exit 1
fi

NODE_ARGS=()
if [[ -n "$NODELIST" ]]; then
  NODE_ARGS+=(--nodelist="$NODELIST")
fi

COMMON_EXPORT="MAX_QUESTIONS=50"
COMMON_EXPORT+=",DATASET_NAME=medical"
COMMON_EXPORT+=",SUBMISSION_BATCH_SIZE=250"
COMMON_EXPORT+=",VLLM_MAX_NUM_SEQS=16"
COMMON_EXPORT+=",VLLM_KV_IMPORTANCE_ENABLE=0"
COMMON_EXPORT+=",SC_LMCACHE_SOURCE_DEFAULT_KNOBS=1"
COMMON_EXPORT+=",SC_LMCACHE_PROFILE=custom"
COMMON_EXPORT+=",SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE=1"
COMMON_EXPORT+=",SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT=4"
COMMON_EXPORT+=",SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_WAIT_MS=0"
COMMON_EXPORT+=",SC_LMCACHE_WORKER_LOOKUP_ADMISSION_ENABLE=0"
COMMON_EXPORT+=",SC_LMCACHE_WORKER_LOOKUP_MAX_INFLIGHT=4"
COMMON_EXPORT+=",SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE=0"
COMMON_EXPORT+=",SC_LMCACHE_DISK_PUT_MAX_PENDING=8"
COMMON_EXPORT+=",SC_LMCACHE_IO_TRACE_ENABLE=1"
COMMON_EXPORT+=",SC_LMCACHE_LOAD_TRACE_ENABLE=1"
COMMON_EXPORT+=",SC_LMCACHE_MEMORY_TRACE_ENABLE=1"
COMMON_EXPORT+=",SC_LMCACHE_MEMORY_SNAPSHOT_ENABLE=0"
COMMON_EXPORT+=",SC_LMCACHE_LOOKUP_TRACE_ENABLE=0"
COMMON_EXPORT+=",SC_LMCACHE_REQUEST_TRACE_ENABLE=0"
COMMON_EXPORT+=",SC_LMCACHE_TIER_TRACE_ENABLE=0"
COMMON_EXPORT+=",SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE=0"
COMMON_EXPORT+=",SC_LMCACHE_LIFECYCLE_TRACE_ENABLE=0"
COMMON_EXPORT+=",SC_LMCACHE_MEMORY_TRACE_INTERVAL_S=5"
COMMON_EXPORT+=",SC_LMCACHE_LONG_PIN_THRESHOLD_S=10"
COMMON_EXPORT+=",SC_LMCACHE_LONG_REF_THRESHOLD_S=10"
COMMON_EXPORT+=",SC_DRIVER_RESOURCE_MONITOR_ENABLE=1"
COMMON_EXPORT+=",SC_DRIVER_RESOURCE_MONITOR_INTERVAL_S=30"
COMMON_EXPORT+=",CLEAN_OLD_LMCACHE=1"

cd "$REPO"

echo "Submitting NFS/default-path Q=50 job..."
# Drop SC_LMCACHE_DATA_DIR from sbatch's environment so run_driver.sh uses its default shared NFS path.
JID_NFS=$(env -u SC_LMCACHE_DATA_DIR sbatch --parsable \
  --job-name=kvaware_nfs_q50_med_s4_seq16 \
  "${NODE_ARGS[@]}" \
  --export=ALL,"$COMMON_EXPORT" \
  "$SBATCH_SCRIPT" | cut -d';' -f1)
echo "NFS job: $JID_NFS"

echo "Submitting scratch Q=50 job dependent on NFS completion..."
JID_SCRATCH=$(sbatch --parsable \
  --dependency=afterany:"$JID_NFS" \
  --job-name=kvaware_scratch_q50_med_s4_seq16 \
  "${NODE_ARGS[@]}" \
  --export=ALL,"$COMMON_EXPORT",SC_LMCACHE_DATA_DIR="$SCRATCH_LMCACHE" \
  "$SBATCH_SCRIPT" | cut -d';' -f1)
echo "Scratch job: $JID_SCRATCH"

echo "\nWatch with:"
echo "  squeue -u $USER"
echo "\nExpected logs under:"
echo "  /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/kvaware_nfs_q50_med_s4_seq16_${JID_NFS}.out"
echo "  /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/kvaware_scratch_q50_med_s4_seq16_${JID_SCRATCH}.out"
echo "\nScratch cache path used by second job: $SCRATCH_LMCACHE"

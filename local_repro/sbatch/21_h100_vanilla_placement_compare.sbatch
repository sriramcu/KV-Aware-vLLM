#!/bin/bash
#SBATCH --job-name=kvaware_vanilla_placement
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

: "${BASELINE_MODE:?BASELINE_MODE is required: wo_gnn|with_gnn}"
: "${BASELINE_STORAGE:?BASELINE_STORAGE is required: scratch|nfs}"
: "${BASELINE_MAX_QUESTIONS:?BASELINE_MAX_QUESTIONS is required}"
: "${BASELINE_CPU_GIB:?BASELINE_CPU_GIB is required}"

case "$BASELINE_MODE" in
  wo_gnn|with_gnn) ;;
  *) echo "Invalid BASELINE_MODE=$BASELINE_MODE" >&2; exit 2 ;;
esac

case "$BASELINE_STORAGE" in
  scratch|nfs) ;;
  *) echo "Invalid BASELINE_STORAGE=$BASELINE_STORAGE" >&2; exit 2 ;;
esac

# Prevent stale experiment settings inherited from the submitting shell from
# contaminating this vanilla comparison. The BASELINE_* variables above are
# intentionally separate from SC_LMCACHE_* for this reason.
while IFS='=' read -r name _; do
  if [[ "$name" == SC_LMCACHE_* ]]; then
    unset "$name"
  fi
done < <(env)

export SC_LMCACHE_PROFILE=vanilla
export SC_LMCACHE_MAX_LOCAL_CPU_SIZE="$BASELINE_CPU_GIB"
export SC_LMCACHE_LOCAL_DISK_ENABLE=1

export DATASET_NAME=medical
export MAX_QUESTIONS="$BASELINE_MAX_QUESTIONS"
export SUBMISSION_BATCH_SIZE="$BASELINE_MAX_QUESTIONS"
export VLLM_MAX_NUM_SEQS=16
export LMCACHE_LOOKUP_TIMEOUT_MS=3000
export CLEAN_OLD_LMCACHE=1

if [[ "$BASELINE_STORAGE" == "scratch" ]]; then
  export SC_LMCACHE_DATA_DIR="/scratch/sriramc2/kvaware_vanilla_${BASELINE_MODE}_${SLURM_JOB_ID}"
else
  # Empty/unset lets run_driver.sh create its normal job-specific shared-NFS path.
  unset SC_LMCACHE_DATA_DIR
fi

# Vanilla profile fills all experimental functional controls as disabled.
source "$REPO/local_repro/sc_lmcache_knobs.sh"

export SC_LMCACHE_IO_TRACE_ENABLE=1
export SC_LMCACHE_LOAD_TRACE_ENABLE=1
export SC_LMCACHE_TIER_TRACE_ENABLE=1
export SC_LMCACHE_DISK_PUT_RESIDENCY_TRACE_ENABLE=1
export SC_LMCACHE_MEMORY_TRACE_ENABLE=1

echo "============================================================"
echo "VANILLA PLACEMENT COMPARISON"
echo "mode              = $BASELINE_MODE"
echo "storage           = $BASELINE_STORAGE"
echo "dataset           = $DATASET_NAME"
echo "questions         = $MAX_QUESTIONS"
echo "submission batch  = $SUBMISSION_BATCH_SIZE"
echo "max_num_seqs      = $VLLM_MAX_NUM_SEQS"
echo "CPU/rank          = $SC_LMCACHE_MAX_LOCAL_CPU_SIZE GiB"
echo "LMCache data dir  = ${SC_LMCACHE_DATA_DIR:-driver-default shared NFS}"
echo "============================================================"
sc_lmcache_print_knobs

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs
"$REPO/local_repro/run_driver.sh" "$BASELINE_MODE" 2>&1

#!/bin/bash
#SBATCH --job-name=kvaware_q300_policy
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
SCRATCH2_ROOT=/scratch2/sriramc2

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1

# ============================================================
# Required experiment selectors
# ============================================================
#
# BASELINE_PLACEMENT:
#   duplicate = original/default LMCache CPU+disk duplication
#   random    = exclusive random CPU/disk placement
#   gnn       = exclusive GNN CPU/disk placement
#
# BASELINE_TUNED:
#   0 = vanilla LMCache knobs
#   1 = R16 + cold/warm PUT barrier + scheduler lookup K4
#
: "${BASELINE_PLACEMENT:?BASELINE_PLACEMENT is required: duplicate|random|gnn}"
: "${BASELINE_TUNED:?BASELINE_TUNED is required: 0|1}"

case "$BASELINE_PLACEMENT" in
  duplicate|random|gnn)
    ;;
  *)
    echo "Invalid BASELINE_PLACEMENT=$BASELINE_PLACEMENT" >&2
    exit 2
    ;;
esac

case "$BASELINE_TUNED" in
  0|1)
    ;;
  *)
    echo "Invalid BASELINE_TUNED=$BASELINE_TUNED" >&2
    exit 2
    ;;
esac


# ============================================================
# Fixed dimensions: match jobs 20092-20094
# ============================================================

export DATASET_NAME=medical
export MAX_QUESTIONS=300
export SUBMISSION_BATCH_SIZE=300
export VLLM_MAX_NUM_SEQS=16

export LMCACHE_LOOKUP_TIMEOUT_MS=3000

# Existing run_driver.sh cleanup mechanism.
export CLEAN_OLD_LMCACHE=1


# ============================================================
# Remove inherited LMCache experiment knobs
# ============================================================

# Prevent an interactive shell / previous experiment from contaminating
# this run. We restore the desired values below.
while IFS='=' read -r name _; do
  if [[ "$name" == SC_LMCACHE_* ]]; then
    unset "$name"
  fi
done < <(env)


# ============================================================
# Start from vanilla LMCache
# ============================================================

export SC_LMCACHE_PROFILE=vanilla

# 100 GiB CPU LMCache per TP rank.
export SC_LMCACHE_MAX_LOCAL_CPU_SIZE=100

# All four new experiments use /scratch2 disk.
export SC_LMCACHE_LOCAL_DISK_ENABLE=1

# Fill in vanilla defaults.
source "$REPO/local_repro/sc_lmcache_knobs.sh"


# ============================================================
# Placement policy
# ============================================================

# These experiments are never CPU-only.
export VLLM_KV_CPU_ONLY=0

# Keep random placement deterministic across comparisons.
export VLLM_KV_RANDOM_SEED="${BASELINE_RANDOM_SEED:-0}"

case "$BASELINE_PLACEMENT" in

  duplicate)
    # No GNN.
    # No random target tiers.
    #
    # Adapter returns target_tiers=None, restoring ordinary LMCache
    # behavior: enabled backends receive the normal duplicated placement.
    MODE=wo_gnn
    export VLLM_KV_RANDOM_PLACEMENT_ENABLE=0
    ;;

  random)
    # Same exclusive random placement policy used by job 20093.
    MODE=wo_gnn
    export VLLM_KV_RANDOM_PLACEMENT_ENABLE=1
    ;;

  gnn)
    # Same full-coverage GNN placement policy used by job 20094.
    MODE=with_gnn
    export VLLM_KV_RANDOM_PLACEMENT_ENABLE=0
    ;;

esac


# ============================================================
# Optional tuning bundle
# ============================================================
#
# BASELINE_TUNED=1 changes ONLY:
#
#   K4:
#     scheduler lookup admission enabled
#     max 4 outstanding scheduler lookups
#
#   R16:
#     CPU-preferred single-serializer fairness
#     up to 16 CPU selections per forced disk selection
#
#   PUT barrier:
#     drain cold LocalDiskBackend PUTs before warm starts
#
# It does NOT additionally enable:
#   - PVTSC
#   - within-file PVTSC
#   - worker admission
#   - disk-PUT admission
#   - resident PUT dedup
#

if [[ "$BASELINE_TUNED" == "1" ]]; then

  # K4
  export SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE=1
  export SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT=4
  export SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_WAIT_MS=0

  # R16
  export SC_LMCACHE_SERIALIZER_FAIRNESS_ENABLE=1
  export SC_LMCACHE_SERIALIZER_CPU_BURST_RATIO=16

  # Cold -> warm PUT drain barrier
  export SC_LMCACHE_COLD_WARM_PUT_BARRIER_ENABLE=1

fi


# ============================================================
# Full logging: same instrumentation family as 20092-20094
# ============================================================

export SC_LMCACHE_IO_TRACE_ENABLE=1
export SC_LMCACHE_LOAD_TRACE_ENABLE=1
export SC_LMCACHE_TIER_TRACE_ENABLE=1
export SC_LMCACHE_DISK_PUT_RESIDENCY_TRACE_ENABLE=1
export SC_LMCACHE_MEMORY_TRACE_ENABLE=1


# ============================================================
# /scratch2 configuration
# ============================================================

[[ -d "$SCRATCH2_ROOT" ]] || {
  echo "Missing $SCRATCH2_ROOT on $(hostname)" >&2
  exit 2
}

[[ -w "$SCRATCH2_ROOT" ]] || {
  echo "$SCRATCH2_ROOT is not writable on $(hostname)" >&2
  exit 2
}

TUNE_TAG=vanilla
if [[ "$BASELINE_TUNED" == "1" ]]; then
  TUNE_TAG=r16_barrier_k4
fi

export SC_LMCACHE_DATA_DIR="$SCRATCH2_ROOT/kvaware_q300_${BASELINE_PLACEMENT}_${TUNE_TAG}_${SLURM_JOB_ID}"


# ============================================================
# Configuration printout
# ============================================================

echo "============================================================"
echo "SCRATCH2 Q300 POLICY/TUNING COMPARISON"
echo "host                    = $(hostname)"
echo "placement               = $BASELINE_PLACEMENT"
echo "mode                    = $MODE"
echo "tuned bundle            = $BASELINE_TUNED"
echo "random placement        = $VLLM_KV_RANDOM_PLACEMENT_ENABLE"
echo "random seed             = $VLLM_KV_RANDOM_SEED"
echo "CPU-only routing        = $VLLM_KV_CPU_ONLY"
echo "disk enabled            = $SC_LMCACHE_LOCAL_DISK_ENABLE"
echo "questions               = $MAX_QUESTIONS"
echo "submission batch        = $SUBMISSION_BATCH_SIZE"
echo "max_num_seqs            = $VLLM_MAX_NUM_SEQS"
echo "CPU/rank                = $SC_LMCACHE_MAX_LOCAL_CPU_SIZE GiB"
echo "lookup timeout          = $LMCACHE_LOOKUP_TIMEOUT_MS ms"
echo "LMCache data dir        = $SC_LMCACHE_DATA_DIR"
echo "CLEAN_OLD_LMCACHE       = $CLEAN_OLD_LMCACHE"
echo "============================================================"

echo "=== SCRATCH2 FILESYSTEM ==="
findmnt -no SOURCE,FSTYPE,OPTIONS "$SCRATCH2_ROOT" || true
df -hT "$SCRATCH2_ROOT"

echo "=== EFFECTIVE LMCache KNOBS ==="
sc_lmcache_print_knobs


# ============================================================
# Run
# ============================================================

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs

"$REPO/local_repro/run_driver.sh" "$MODE" 2>&1

echo "=== DONE ==="
date
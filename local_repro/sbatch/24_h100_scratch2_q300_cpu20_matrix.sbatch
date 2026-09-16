#!/bin/bash
#SBATCH --job-name=kvaware_q300_cpu20
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
# Experiment selectors
# ============================================================
#
# BASELINE_PLACEMENT:
#   cpuonly   = all LMCache KV -> CPU, disk disabled
#   duplicate = normal LMCache CPU+disk duplication
#   random    = exclusive random CPU/disk placement
#   gnn       = exclusive GNN CPU/disk placement
#
# BASELINE_TUNED:
#   0 = vanilla functional knobs
#   1 = K4 + R16 + PUT barrier + PVTSC + within-file PVTSC
#

: "${BASELINE_PLACEMENT:?required: cpuonly|duplicate|random|gnn}"
: "${BASELINE_TUNED:?required: 0|1}"

case "$BASELINE_PLACEMENT" in
  cpuonly|duplicate|random|gnn) ;;
  *)
    echo "Invalid BASELINE_PLACEMENT=$BASELINE_PLACEMENT" >&2
    exit 2
    ;;
esac

case "$BASELINE_TUNED" in
  0|1) ;;
  *)
    echo "Invalid BASELINE_TUNED=$BASELINE_TUNED" >&2
    exit 2
    ;;
esac


# ============================================================
# Fixed experimental dimensions
# ============================================================

export DATASET_NAME=medical
export MAX_QUESTIONS="${BASELINE_MAX_QUESTIONS:-300}"
export SUBMISSION_BATCH_SIZE="$MAX_QUESTIONS"
export VLLM_MAX_NUM_SEQS=16

# 20 GiB CPU LMCache per TP rank.
export BASELINE_CPU_GIB=20

export LMCACHE_LOOKUP_TIMEOUT_MS=3000
export CLEAN_OLD_LMCACHE=1


# ============================================================
# Remove stale SC_* knobs inherited from submitting shell
# ============================================================

while IFS='=' read -r name _; do
  if [[ "$name" == SC_LMCACHE_* ]]; then
    unset "$name"
  fi
done < <(env)


# ============================================================
# Base LMCache configuration
# ============================================================

export SC_LMCACHE_PROFILE=vanilla
export SC_LMCACHE_MAX_LOCAL_CPU_SIZE="$BASELINE_CPU_GIB"

# Same diagnostic logging family as the previous Q300 runs.
export SC_LMCACHE_IO_TRACE_ENABLE=1
export SC_LMCACHE_LOAD_TRACE_ENABLE=1
export SC_LMCACHE_TIER_TRACE_ENABLE=1
export SC_LMCACHE_DISK_PUT_RESIDENCY_TRACE_ENABLE=1
export SC_LMCACHE_MEMORY_TRACE_ENABLE=1


# ============================================================
# Placement policy
# ============================================================

export VLLM_KV_RANDOM_SEED="${BASELINE_RANDOM_SEED:-0}"

case "$BASELINE_PLACEMENT" in

  cpuonly)
    MODE=wo_gnn

    export SC_LMCACHE_LOCAL_DISK_ENABLE=0
    export VLLM_KV_CPU_ONLY=1
    export VLLM_KV_RANDOM_PLACEMENT_ENABLE=0

    export SC_LMCACHE_DATA_DIR=""
    ;;

  duplicate)
    MODE=wo_gnn

    export SC_LMCACHE_LOCAL_DISK_ENABLE=1
    export VLLM_KV_CPU_ONLY=0
    export VLLM_KV_RANDOM_PLACEMENT_ENABLE=0
    ;;

  random)
    MODE=wo_gnn

    export SC_LMCACHE_LOCAL_DISK_ENABLE=1
    export VLLM_KV_CPU_ONLY=0
    export VLLM_KV_RANDOM_PLACEMENT_ENABLE=1
    ;;

  gnn)
    MODE=with_gnn

    export SC_LMCACHE_LOCAL_DISK_ENABLE=1
    export VLLM_KV_CPU_ONLY=0
    export VLLM_KV_RANDOM_PLACEMENT_ENABLE=0
    ;;

esac


# ============================================================
# Tuned bundle
# ============================================================
#
# Tuned =
#   K4 scheduler admission
#   R16 CPU-preferred serializer fairness
#   cold -> warm PUT drain barrier
#   PVTSC
#   within-file PVTSC, 4 MiB subreads
#
# Everything else remains vanilla.
#

if [[ "$BASELINE_TUNED" == "1" ]]; then

  # K4
  export SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE=1
  export SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT=4
  export SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_WAIT_MS=0

  # R16
  export SC_LMCACHE_SERIALIZER_FAIRNESS_ENABLE=1
  export SC_LMCACHE_SERIALIZER_CPU_BURST_RATIO=16

  # Cold -> warm disk PUT drain.
  export SC_LMCACHE_COLD_WARM_PUT_BARRIER_ENABLE=1

  # Post-vLLM-timeout synthetic completion / cancellation.
  export SC_LMCACHE_PVTSC_ENABLE=1
  export SC_LMCACHE_PVTSC_WITHIN_FILE_ENABLE=1
  export SC_LMCACHE_PVTSC_READ_CHUNK_MIB=4

  # Keep these at vanilla behavior.
  export SC_LMCACHE_WORKER_LOOKUP_ADMISSION_ENABLE=0
  export SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE=0
  export SC_LMCACHE_DISK_RESIDENT_PUT_DEDUP_ENABLE=0

fi


# Source AFTER overrides so it validates our values and fills defaults
# for everything we did not explicitly change.
source "$REPO/local_repro/sc_lmcache_knobs.sh"


# ============================================================
# /scratch2 setup for disk-enabled cells
# ============================================================

if [[ "$SC_LMCACHE_LOCAL_DISK_ENABLE" == "1" ]]; then

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
    TUNE_TAG=r16_barrier_k4_pvtsc
  fi

  export SC_LMCACHE_DATA_DIR="$SCRATCH2_ROOT/kvaware_q300_cpu20_${BASELINE_PLACEMENT}_${TUNE_TAG}_${SLURM_JOB_ID}"

else
  export SC_LMCACHE_DATA_DIR=""
fi


# ============================================================
# Configuration printout
# ============================================================

echo "============================================================"
echo "Q300 / CPU20 PLACEMENT MATRIX"
echo "host                    = $(hostname)"
echo "placement               = $BASELINE_PLACEMENT"
echo "mode                    = $MODE"
echo "tuned                   = $BASELINE_TUNED"
echo "CPU/rank                = $SC_LMCACHE_MAX_LOCAL_CPU_SIZE GiB"
echo "disk enabled            = $SC_LMCACHE_LOCAL_DISK_ENABLE"
echo "CPU-only routing        = $VLLM_KV_CPU_ONLY"
echo "random placement        = $VLLM_KV_RANDOM_PLACEMENT_ENABLE"
echo "random seed             = $VLLM_KV_RANDOM_SEED"
echo "questions               = $MAX_QUESTIONS"
echo "submission batch        = $SUBMISSION_BATCH_SIZE"
echo "max_num_seqs            = $VLLM_MAX_NUM_SEQS"
echo "lookup timeout          = $LMCACHE_LOOKUP_TIMEOUT_MS ms"
echo "PVTSC                   = $SC_LMCACHE_PVTSC_ENABLE"
echo "within-file PVTSC       = $SC_LMCACHE_PVTSC_WITHIN_FILE_ENABLE"
echo "PVTSC chunk MiB         = $SC_LMCACHE_PVTSC_READ_CHUNK_MIB"
echo "LMCache data dir        = ${SC_LMCACHE_DATA_DIR:-CPU-only/no disk}"
echo "============================================================"

if [[ "$SC_LMCACHE_LOCAL_DISK_ENABLE" == "1" ]]; then
  echo "=== SCRATCH2 FILESYSTEM ==="
  findmnt -no SOURCE,FSTYPE,OPTIONS "$SCRATCH2_ROOT" || true
  df -hT "$SCRATCH2_ROOT"
fi

echo "=== EFFECTIVE LMCACHE KNOBS ==="
sc_lmcache_print_knobs

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs

"$REPO/local_repro/run_driver.sh" "$MODE" 2>&1

echo "=== DONE ==="
date
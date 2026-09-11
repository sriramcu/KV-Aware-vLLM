#!/bin/bash
#SBATCH --job-name=kvaware_scratch2_q300
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

: "${BASELINE_MODE:?BASELINE_MODE is required: wo_gnn|with_gnn}"
: "${BASELINE_DISK:?BASELINE_DISK is required: 0|1}"
: "${BASELINE_MAX_QUESTIONS:?BASELINE_MAX_QUESTIONS is required}"
: "${BASELINE_CPU_GIB:?BASELINE_CPU_GIB is required}"

case "$BASELINE_MODE" in
  wo_gnn|with_gnn) ;;
  *) echo "Invalid BASELINE_MODE=$BASELINE_MODE" >&2; exit 2 ;;
esac

case "$BASELINE_DISK" in
  0|1) ;;
  *) echo "Invalid BASELINE_DISK=$BASELINE_DISK" >&2; exit 2 ;;
esac

# Prevent stale experiment settings inherited from the submitting shell from
# contaminating the comparison. BASELINE_* variables intentionally survive.
while IFS='=' read -r name _; do
  if [[ "$name" == SC_LMCACHE_* ]]; then
    unset "$name"
  fi
done < <(env)

export SC_LMCACHE_PROFILE=vanilla
export SC_LMCACHE_MAX_LOCAL_CPU_SIZE="$BASELINE_CPU_GIB"
export SC_LMCACHE_LOCAL_DISK_ENABLE="$BASELINE_DISK"

export DATASET_NAME=medical
export MAX_QUESTIONS="$BASELINE_MAX_QUESTIONS"
export SUBMISSION_BATCH_SIZE="$BASELINE_MAX_QUESTIONS"
export VLLM_MAX_NUM_SEQS=16
export LMCACHE_LOOKUP_TIMEOUT_MS=3000
export CLEAN_OLD_LMCACHE=1

if [[ "$BASELINE_DISK" == "1" ]]; then
  export VLLM_KV_CPU_ONLY=0

  [[ -d "$SCRATCH2_ROOT" ]] || {
    echo "Missing $SCRATCH2_ROOT on $(hostname)" >&2
    exit 2
  }

  [[ -w "$SCRATCH2_ROOT" ]] || {
    echo "$SCRATCH2_ROOT is not writable on $(hostname)" >&2
    exit 2
  }

  export SC_LMCACHE_DATA_DIR="$SCRATCH2_ROOT/kvaware_vanilla_${BASELINE_MODE}_${SLURM_JOB_ID}"
else
  export VLLM_KV_CPU_ONLY=1
  unset SC_LMCACHE_DATA_DIR
fi

echo "CPU-only routing  = $VLLM_KV_CPU_ONLY"

# Vanilla profile keeps functional experimental mechanisms disabled.
source "$REPO/local_repro/sc_lmcache_knobs.sh"

# Full diagnostic logging, matching the 19590-style run.
export SC_LMCACHE_IO_TRACE_ENABLE=1
export SC_LMCACHE_LOAD_TRACE_ENABLE=1
export SC_LMCACHE_TIER_TRACE_ENABLE=1
export SC_LMCACHE_DISK_PUT_RESIDENCY_TRACE_ENABLE=1
export SC_LMCACHE_MEMORY_TRACE_ENABLE=1

echo "============================================================"
echo "SCRATCH2 Q300 PLACEMENT COMPARISON"
echo "host              = $(hostname)"
echo "mode              = $BASELINE_MODE"
echo "disk enabled      = $BASELINE_DISK"
echo "dataset           = $DATASET_NAME"
echo "questions         = $MAX_QUESTIONS"
echo "submission batch  = $SUBMISSION_BATCH_SIZE"
echo "max_num_seqs      = $VLLM_MAX_NUM_SEQS"
echo "CPU/rank          = $SC_LMCACHE_MAX_LOCAL_CPU_SIZE GiB"
echo "CLEAN_OLD_LMCACHE = $CLEAN_OLD_LMCACHE"
echo "LMCache data dir  = ${SC_LMCACHE_DATA_DIR:-none (CPU-only)}"
echo "============================================================"

if [[ "$BASELINE_DISK" == "1" ]]; then
  findmnt -no SOURCE,FSTYPE,OPTIONS "$SCRATCH2_ROOT" || true
  df -hT "$SCRATCH2_ROOT"
fi

sc_lmcache_print_knobs

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs
"$REPO/local_repro/run_driver.sh" "$BASELINE_MODE" 2>&1

if [[ "$BASELINE_DISK" == "1" ]]; then
  echo "=== CACHE SIZE ==="
  du -sh "$SC_LMCACHE_DATA_DIR" || true
fi

echo "=== DONE ==="

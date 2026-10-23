#!/usr/bin/env bash
#SBATCH --job-name=kvq650-10batch
#SBATCH --partition=gpu-h100-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.err
set -Eeuo pipefail

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
cd "$REPO"
: "${CASE_ID:?Must supply CASE_ID=P1,P2,P3,P4,P5,P6,P7,P8,P11 or P12}"
case "$CASE_ID" in
  P1)  POLICY=nognn; GPU=0;  CPU=225; DISK=500; CAP=0.25; MODE=fixed;        WEIGHT=1;   SCORE_COUNT=10;   COST_COUNT=5 ;;
  P2)  POLICY=bayes; GPU=25; CPU=225; DISK=500; CAP=0.25; MODE=fixed;        WEIGHT=0.5; SCORE_COUNT=10;   COST_COUNT=5 ;;
  P3)  POLICY=bayes; GPU=30; CPU=225; DISK=300; CAP=0.25; MODE=pseudo_count; WEIGHT=1;   SCORE_COUNT=1000; COST_COUNT=100 ;;
  P4)  POLICY=nognn; GPU=0;  CPU=225; DISK=700; CAP=0.25; MODE=fixed;        WEIGHT=1;   SCORE_COUNT=10;   COST_COUNT=5 ;;
  P5)  POLICY=nognn; GPU=0;  CPU=225; DISK=300; CAP=0.25; MODE=fixed;        WEIGHT=1;   SCORE_COUNT=10;   COST_COUNT=5 ;;
  P6)  POLICY=bayes; GPU=25; CPU=225; DISK=500; CAP=0.25; MODE=fixed;        WEIGHT=0.2; SCORE_COUNT=10;   COST_COUNT=5 ;;
  P7)  POLICY=bayes; GPU=30; CPU=225; DISK=300; CAP=0.25; MODE=fixed;        WEIGHT=0.2; SCORE_COUNT=10;   COST_COUNT=5 ;;
  P8)  POLICY=nognn; GPU=0;  CPU=100; DISK=100; CAP=0;    MODE=fixed;        WEIGHT=1;   SCORE_COUNT=10;   COST_COUNT=5 ;;
  P11) POLICY=bayes; GPU=25; CPU=225; DISK=500; CAP=0;    MODE=fixed;        WEIGHT=1;   SCORE_COUNT=10;   COST_COUNT=5 ;;
  P12) POLICY=nognn; GPU=0;  CPU=225; DISK=500; CAP=0;    MODE=fixed;        WEIGHT=1;   SCORE_COUNT=10;   COST_COUNT=5 ;;
  *) echo "Invalid CASE_ID: ${CASE_ID}" >&2; exit 2 ;;
esac

# All launch parameters are explicit, keeping runs comparable to job 29920.
export SWEEP_CELL="c_${CASE_ID}_${POLICY}"
export SWEEP_POLICY="$POLICY"
export SWEEP_ORDER=original
export SWEEP_GPU_GB="$GPU"
export SWEEP_CPU_GB="$CPU"
export SWEEP_DISK_GB="$DISK"
export SWEEP_STORE_CAP_GB="$CAP"
export SWEEP_PRIOR_WEIGHT="$WEIGHT"
export SWEEP_POSTERIOR_MODE="$MODE"
export SWEEP_SCORE_PRIOR_COUNT="$SCORE_COUNT"
export SWEEP_COST_PRIOR_COUNT="$COST_COUNT"
export SWEEP_PRIOR_FILE="/mnt/shared/gpfs/home/sriramc2/runs/kvaware/h100_expected_ordinal_exploratory.json"
export GRID_LABEL="c_${CASE_ID}_${POLICY}_q650_b100"

# Default: reclaim the entire /scratch2/sriramc2 allocation BEFORE EACH run,
# while preserving that directory itself. Both GNN and no-GNN launchers honor it.
# Never overlap with other jobs that use the same scratch root.
export CLEAN_OLD_LMCACHE=1
export VLLM_PHASE_TRACE=1 VLLM_PHASE_TRACE_DEVICE_TIMING=0
export KV_COLD_WARM_STORE_BARRIER=1
export KV_MUTUAL_PREFIX=1 KV_VPC_SUFFICIENT_BYPASS=1
export LMCACHE_MP_PREFIX_DIAGNOSTICS=1

# Do not silently use a rank_head prior for the ordinal experiment.
if [[ "$POLICY" == bayes ]]; then
  [[ -s "$SWEEP_PRIOR_FILE" ]] || { echo "Missing ordinal prior: $SWEEP_PRIOR_FILE" >&2; exit 2; }
fi
[[ -s scripts/h100_10batch/run_h100_bayes_sweep_cell_safe.sbatch ]] || {
  echo "Install generated launchers: python scripts/h100_10batch/install.py" >&2
  exit 2
}
[[ -s scripts/h100_10batch/run_h100_nognn_q650_safe.sbatch ]] || {
  echo "Install generated launchers: python scripts/h100_10batch/install.py" >&2
  exit 2
}

echo "[10BATCH] case=$CASE_ID policy=$POLICY GPU_policy=$GPU CPU_L1=$CPU DISK_physical_and_policy=$DISK PUT_cap=$CAP posterior=$MODE w=$WEIGHT kA=$SCORE_COUNT kB=$COST_COUNT"
echo "[10BATCH] job=${SLURM_JOB_ID:-unset} node=$(hostname) repo=$REPO"

# The generated Bayesian sweep launcher delegates no-GNN runs to the generated
# safe baseline, and otherwise invokes the installed live GNN/LMCache code.
bash scripts/h100_10batch/run_h100_bayes_sweep_cell_safe.sbatch

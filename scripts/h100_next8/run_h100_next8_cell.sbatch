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
: "${CASE_ID:?Supply CASE_ID from submit_8_sbatch_only.sh}"

# Keep every pair matched. Only the stress dimension changes between pairs.
# All Bayes runs use the successful 29954 configuration except CPU scarcity.
case "$CASE_ID" in
  B200_NG) POLICY=nognn; DATASET=medical; Q=650; B=200; GPU=0;  CPU=225; DISK=500 ;;
  B200_BY) POLICY=bayes; DATASET=medical; Q=650; B=200; GPU=25; CPU=225; DISK=500 ;;
  CPU100_NG) POLICY=nognn; DATASET=medical; Q=650; B=100; GPU=0;  CPU=100; DISK=400 ;;
  CPU100_BY) POLICY=bayes; DATASET=medical; Q=650; B=100; GPU=25; CPU=100; DISK=400 ;;
  HOTPOT_NG) POLICY=nognn; DATASET=hotpotqa; Q=650; B=100; GPU=0;  CPU=225; DISK=500 ;;
  HOTPOT_BY) POLICY=bayes; DATASET=hotpotqa; Q=650; B=100; GPU=25; CPU=225; DISK=500 ;;
  Q1000_NG) POLICY=nognn; DATASET=medical; Q=1000; B=100; GPU=0;  CPU=225; DISK=500 ;;
  Q1000_BY) POLICY=bayes; DATASET=medical; Q=1000; B=100; GPU=25; CPU=225; DISK=500 ;;
  *) echo "ERROR: invalid CASE_ID=$CASE_ID" >&2; exit 2 ;;
esac

# Before touching any scratch, verify that this dataset can actually be read.
# Hotpot needs all three graph assets for the GNN precompute.
export NEXT8_DATASET="$DATASET" NEXT8_Q="$Q" NEXT8_B="$B"
export NEXT8_CASE="$CASE_ID" NEXT8_POLICY="$POLICY"
python3 scripts/h100_next8/preflight_dataset.py \
  --repo "$REPO" --dataset "$DATASET" --questions "$Q" --policy "$POLICY"

# Match 29944/29954 except for the intended experimental dimension.
export SWEEP_CELL="c_next8_${CASE_ID}_${POLICY}"
export SWEEP_POLICY="$POLICY" SWEEP_ORDER=original
export SWEEP_GPU_GB="$GPU" SWEEP_CPU_GB="$CPU" SWEEP_DISK_GB="$DISK"
export SWEEP_STORE_CAP_GB=0.25
export SWEEP_PRIOR_WEIGHT=0.5 SWEEP_POSTERIOR_MODE=fixed
export SWEEP_SCORE_PRIOR_COUNT=10 SWEEP_COST_PRIOR_COUNT=5
export SWEEP_PRIOR_FILE=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/h100_expected_ordinal_exploratory.json
export GRID_LABEL="next8_${CASE_ID}_${DATASET}_q${Q}_b${B}"

# IMPORTANT: user preference, reclaim whole scratch root before EVERY job.
# This is destructive; singleton must protect jobs sharing this root.
export CLEAN_OLD_LMCACHE=1
export VLLM_PHASE_TRACE=1 VLLM_PHASE_TRACE_DEVICE_TIMING=0
export KV_COLD_WARM_STORE_BARRIER=1
export KV_MUTUAL_PREFIX=1 KV_VPC_SUFFICIENT_BYPASS=1
export LMCACHE_MP_PREFIX_DIAGNOSTICS=1

if [[ "$POLICY" == bayes ]]; then
  [[ -s "$SWEEP_PRIOR_FILE" ]] || { echo "Missing expected ordinal prior: $SWEEP_PRIOR_FILE" >&2; exit 2; }
  python3 - "$SWEEP_PRIOR_FILE" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); assert p.get('score_format') == 'expected_ordinal_0_100', (
    'Wrong histogram format in prior:', p.get('score_format'))
PY
fi
for f in scripts/h100_next8/run_h100_bayes_sweep_variable.sbatch \
         scripts/h100_next8/run_h100_nognn_variable.sbatch \
         scripts/h100_next8/run_shortq_gnn_dynamic_vpc_variable.sh; do
  [[ -s "$f" ]] || { echo "Not installed: $f; run python scripts/h100_next8/install.py" >&2; exit 2; }
done

echo "[NEXT8] case=$CASE_ID policy=$POLICY dataset=$DATASET Q=$Q B=$B GPU_policy=$GPU CPU=$CPU Disk=$DISK cap=0.25 posterior=fixed w=0.5 score=expected_ordinal_0_100 cleanup=1"
echo "[NEXT8] slurm=${SLURM_JOB_ID:-unset} node=$(hostname)"
bash scripts/h100_next8/run_h100_bayes_sweep_variable.sbatch

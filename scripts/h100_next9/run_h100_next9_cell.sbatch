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
: "${CASE_ID:?Supply CASE_ID (see submit_9_sbatch_only.sh)}"

# 29954's calibrated score histogram and original cost calibration (for SCAR_BY).
CAL_PRIOR=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/h100_expected_ordinal_exploratory.json
PRIOR_DIR="$REPO/scripts/h100_next9/priors"
UNIFORM_PRIOR="$PRIOR_DIR/uniform_scores_equal_cost_10ms.json"
HYBRID_PRIOR="$PRIOR_DIR/calibrated_scores_equal_cost_10ms.json"

# Common invariant configuration: Medical Q650 B100, original passage order,
# capped 0.25GiB PUT, max_num_seqs16, H100 TP2, GPU policy 25GiB.
POLICY=bayes GPU=25 CPU=225 DISK=500 PRIOR="$UNIFORM_PRIOR"
POSTERIOR=fixed W=0.5 KA=10 KB=5 ADAPT_COST=1
case "$CASE_ID" in
  P01_CAL_SCORE_COSTOFF)
    PRIOR="$HYBRID_PRIOR"; ADAPT_COST=0 ;;
  P02_UNIF_BOTH_FIXED)
    ADAPT_COST=1 ;;
  P03_UNIF_SCORE_FIXED)
    ADAPT_COST=0 ;;
  P04_UNIF_BOTH_HI)
    POSTERIOR=pseudo_count; KA=1000; KB=100; ADAPT_COST=1 ;;
  P05_UNIF_BOTH_LO)
    POSTERIOR=pseudo_count; KA=10; KB=1; ADAPT_COST=1 ;;
  P06_UNIF_SCORE_HI)
    POSTERIOR=pseudo_count; KA=1000; KB=100; ADAPT_COST=0 ;;
  P07_UNIF_SCORE_LO)
    POSTERIOR=pseudo_count; KA=10; KB=1; ADAPT_COST=0 ;;
  SCAR_NG)
    POLICY=nognn; GPU=0; CPU=100; DISK=222; PRIOR="$CAL_PRIOR"; ADAPT_COST=1 ;;
  SCAR_BY)
    CPU=100; DISK=222; PRIOR="$CAL_PRIOR"; ADAPT_COST=1 ;;
  *) echo "ERROR: unknown CASE_ID=$CASE_ID" >&2; exit 2 ;;
esac

# Previously the launcher generated next8_* names, which were rejected by
# the historical dispatcher. The c_* prefix is intentionally mandatory.
export SWEEP_CELL="c_next9_${CASE_ID}_${POLICY}"
export SWEEP_POLICY="$POLICY" SWEEP_ORDER=original
export SWEEP_GPU_GB="$GPU" SWEEP_CPU_GB="$CPU" SWEEP_DISK_GB="$DISK"
export SWEEP_STORE_CAP_GB=0.25
export SWEEP_PRIOR_WEIGHT="$W" SWEEP_POSTERIOR_MODE="$POSTERIOR"
export SWEEP_SCORE_PRIOR_COUNT="$KA" SWEEP_COST_PRIOR_COUNT="$KB"
export SWEEP_PRIOR_FILE="$PRIOR"
export KV_BAYES_ADAPT_COST="$ADAPT_COST"
export NEXT8_DATASET=medical NEXT8_Q=650 NEXT8_B=100
export GRID_LABEL="next9_${CASE_ID}_medical_q650_b100"

# Keep the established cache-cleaning and diagnostic semantics.
# WARNING: This deletes EVERY item under /scratch2/sriramc2 before each run.
export CLEAN_OLD_LMCACHE=1
export VLLM_PHASE_TRACE=1 VLLM_PHASE_TRACE_DEVICE_TIMING=0
export KV_COLD_WARM_STORE_BARRIER=1
export KV_MUTUAL_PREFIX=1 KV_VPC_SUFFICIENT_BYPASS=1
export LMCACHE_MP_PREFIX_DIAGNOSTICS=1

printf '[NEXT9] case=%s policy=%s Q=650 B=100 gpu=%s cpu=%s disk=%s posterior=%s w=%s kA=%s kB=%s score_prior=%s adapt_cost=%s put_cap=0.25 clean_all=1\n' \
  "$CASE_ID" "$POLICY" "$GPU" "$CPU" "$DISK" "$POSTERIOR" "$W" "$KA" "$KB" "$PRIOR" "$ADAPT_COST"
echo "[NEXT9] SWEEP_CELL=$SWEEP_CELL"

for f in "$PRIOR" \
         scripts/h100_next8/run_h100_bayes_sweep_variable.sbatch \
         scripts/h100_next8/run_h100_nognn_variable.sbatch \
         scripts/h100_next8/run_shortq_gnn_dynamic_vpc_variable.sh; do
  [[ -s "$f" ]] || { echo "ERROR: missing file: $f" >&2; exit 2; }
done
python3 scripts/h100_next9/verify.py --repo "$REPO" --case "$CASE_ID"

# DRY_RUN=1 permits a login-node case audit: no GPUs and no scratch cleanup.
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo '[NEXT9] DRY RUN SUCCESS: no workload launched or scratch deleted'
  exit 0
fi

# Existing next8 preflight runs before any destructive scratch cleanup.
python3 scripts/h100_next8/preflight_dataset.py --repo "$REPO" \
  --dataset medical --questions 650 --policy "$POLICY"

# The next8 dispatcher is a known working copy with physical L2 enforcement,
# original ordering and variable Q/B/dataset; its cost-timing flag remains ON
# in Bayes regardless of KV_BAYES_ADAPT_COST. The latter only gates updating
# the mapper's cost posterior, not collecting device-completion measurements.
bash scripts/h100_next8/run_h100_bayes_sweep_variable.sbatch

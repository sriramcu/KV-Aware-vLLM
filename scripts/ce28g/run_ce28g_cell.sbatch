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
: "${CASE_ID:?Specify CASE_ID from scripts/ce28g/submit_8_sbatch_only.sh}"
CAL_PRIOR=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/h100_expected_ordinal_exploratory.json
UNIFORM_PRIOR="$REPO/scripts/h100_next9/priors/uniform_scores_equal_cost_10ms.json"
POSTERIOR=fixed W=0.5 KA=10 KB=5 PRIOR="$CAL_PRIOR" ADAPT_COST=1
L1_EVICT=0.20 L2_EVICT=0.10 WM_MAP=0
case "$CASE_ID" in
  CAL_ON_BASE)  ADAPT_COST=1 ;;
  CAL_OFF_BASE) ADAPT_COST=0 ;;
  CAL_ON_WM5)  ADAPT_COST=1; L1_EVICT=0.05; L2_EVICT=0.05; WM_MAP=1 ;;
  CAL_OFF_WM5) ADAPT_COST=0; L1_EVICT=0.05; L2_EVICT=0.05; WM_MAP=1 ;;
  P04_HI_ON)   PRIOR="$UNIFORM_PRIOR"; POSTERIOR=pseudo_count; KA=1000; KB=100; ADAPT_COST=1; L1_EVICT=0.05; L2_EVICT=0.05; WM_MAP=1 ;;
  P05_LO_ON)   PRIOR="$UNIFORM_PRIOR"; POSTERIOR=pseudo_count; KA=10; KB=1; ADAPT_COST=1; L1_EVICT=0.05; L2_EVICT=0.05; WM_MAP=1 ;;
  P06_HI_OFF)  PRIOR="$UNIFORM_PRIOR"; POSTERIOR=pseudo_count; KA=1000; KB=100; ADAPT_COST=0; L1_EVICT=0.05; L2_EVICT=0.05; WM_MAP=1 ;;
  P07_LO_OFF)  PRIOR="$UNIFORM_PRIOR"; POSTERIOR=pseudo_count; KA=10; KB=1; ADAPT_COST=0; L1_EVICT=0.05; L2_EVICT=0.05; WM_MAP=1 ;;
  *) echo "ERROR: unknown CE28G CASE_ID=$CASE_ID" >&2; exit 2 ;;
esac
export SWEEP_CELL="c_ce28g_${CASE_ID}_bayes" SWEEP_POLICY=bayes SWEEP_ORDER=original
export SWEEP_GPU_GB=25 SWEEP_CPU_GB=225 SWEEP_DISK_GB=500 SWEEP_STORE_CAP_GB=0.25
export SWEEP_PRIOR_FILE="$PRIOR" SWEEP_PRIOR_WEIGHT="$W"
export SWEEP_POSTERIOR_MODE="$POSTERIOR" SWEEP_SCORE_PRIOR_COUNT="$KA"
export SWEEP_COST_PRIOR_COUNT="$KB" KV_BAYES_ADAPT_COST="$ADAPT_COST"
export KV_BAYES_USE_WATERMARK_CAPACITY="$WM_MAP"
export LMCACHE_L1_EVICTION_TRIGGER=0.80 LMCACHE_L1_EVICTION_RATIO="$L1_EVICT"
export LMCACHE_L2_EVICTION_TRIGGER=0.90 LMCACHE_L2_EVICTION_RATIO="$L2_EVICT"
# Cost V2 uses CuPy stream-event intervals; old completion remains diagnostic.
# Native phase instrumentation is retained to audit both CPU and Disk H2D.
export LMCACHE_BAYES_H2D_DIAG=1 KV_BAYES_DIRECT_COST_FEEDBACK=1 KV_BAYES_USE_COST_PROXIES=0
export NEXT8_DATASET=medical NEXT8_Q=650 NEXT8_B=100
export CLEAN_OLD_LMCACHE=1
export VLLM_PHASE_TRACE=1 VLLM_PHASE_TRACE_DEVICE_TIMING=0
export KV_COLD_WARM_STORE_BARRIER=1 KV_MUTUAL_PREFIX=1 KV_VPC_SUFFICIENT_BYPASS=1
export LMCACHE_MP_PREFIX_DIAGNOSTICS=1
export GRID_LABEL="CE28G_${CASE_ID}_medical_q650_b100"
printf '[CE28G_CONFIG] case=%s prior=%s posterior=%s w=%s kA=%s kB=%s adapt_cost=%s l1_ratio=%s l2_ratio=%s wm_mapper=%s estimator=gpu_transfer_region\n' \
  "$CASE_ID" "$PRIOR" "$POSTERIOR" "$W" "$KA" "$KB" "$ADAPT_COST" "$L1_EVICT" "$L2_EVICT" "$WM_MAP"
python3 scripts/ce28g/verify_ce28g.py --repo "$REPO" --prior "$PRIOR"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "[CE28G] DRY_RUN successful. No scratch cleanup, GPU or job launch."
  exit 0
fi
python3 scripts/h100_next8/preflight_dataset.py --repo "$REPO" \
  --dataset medical --questions 650 --policy bayes
# THIS WILL DELETE EVERY CHILD OF /scratch2/sriramc2; KEEP SINGLETON SERIES.
bash scripts/h100_next8/run_h100_bayes_sweep_variable.sbatch

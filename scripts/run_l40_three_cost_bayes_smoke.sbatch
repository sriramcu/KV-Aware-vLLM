#!/usr/bin/env bash
#SBATCH --job-name=kvbayes-l40-smoke
#SBATCH --partition=gpu-l40-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.err
# Job-local adaptation of patched H100 common runner to L40/8B TP1.
# Does NOT alter source files, overwrite the H100 prior, or delete /scratch2.
set -Eeuo pipefail
REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware
: "${L40_PRIOR_PATH:?Set L40_PRIOR_PATH to the preflight job rank_l40_calibrated.json}"
[[ -s "$L40_PRIOR_PATH" ]] || { echo "L40 prior missing: $L40_PRIOR_PATH" >&2; exit 2; }
[[ -f "$REPO/experiments/calibrate_bayesian_recompute.py" ]] || { echo 'Three-cost patch is not installed' >&2; exit 2; }
cd "$REPO"
source "$VENV/bin/activate"

# All cache writes are in a distinct NFS directory. Never wipe /scratch2,
# the shared L40 NFS cache root, or another job's data.
export CLEAN_OLD_LMCACHE=0
export KV_GNN_DYNAMIC_PROFILE=l40_smoke
export KV_BAYES_ENABLE=1 KV_BAYES_PRIOR_FILE="$L40_PRIOR_PATH"
export KV_BAYES_DIRECT_COST_FEEDBACK=1 KV_BAYES_USE_COST_PROXIES=0
export KV_BAYES_READMISSION=1 KV_BAYES_POSTERIOR_MODE=fixed KV_BAYES_PRIOR_WEIGHT=1
export LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-32}"
export LMCACHE_L2_SIZE_GB="${LMCACHE_L2_SIZE_GB:-4}"
export KV_BAYES_GPU_GB="${KV_BAYES_GPU_GB:-2}"
export KV_BAYES_DISK_GB="$LMCACHE_L2_SIZE_GB"
export LMCACHE_MP_MAX_INFLIGHT_STORE_GB="${LMCACHE_MP_MAX_INFLIGHT_STORE_GB:-0.25}"
export KV_PASSAGE_ORDER_POLICY=canonical
export KV_REQUEST_ORDER=seeded_shuffle KV_REQUEST_ORDER_SEED=0
export WARM_ORDER=random WARM_ORDER_SEED=1
export NUM_QUESTIONS="${NUM_QUESTIONS:-40}"
export SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-10}"
export MIN_TOKENS=16 MAX_TOKENS=16 MAX_NUM_SEQS=8
export KV_MUTUAL_PREFIX=1 KV_VPC_SUFFICIENT_BYPASS=1
export KV_COLD_WARM_STORE_BARRIER=1 LMCACHE_MP_PREFIX_DIAGNOSTICS=1
export KV_STAGE1_STARVATION_FALLBACK=1
export KV_GNN_AWARE_VPC=1 KV_GNN_L1_BACKING=1 KV_GNN_L2_BACKING=0
export GNN_CHUNK_VOTE_POLICY=threshold_with_drop_plurality
export VLLM_PHASE_TRACE=1 VLLM_PHASE_TRACE_DEVICE_TIMING=0
export EXPERIMENT_LABEL="l40_three_cost_q${NUM_QUESTIONS}_b${SUBMISSION_BATCH_SIZE}"

for candidate in "${CUDA_HOME:-}" /usr/local/cuda-13.1 /usr/local/cuda-13.0 /usr/local/cuda; do
  if [[ -n "$candidate" && -x "$candidate/bin/nvcc" ]]; then
    export L40_CUDA_HOME="$candidate"
    break
  fi
done
: "${L40_CUDA_HOME:?No installed CUDA toolkit with nvcc}"
mkdir -p "$RUN_ROOT/_l40_three_cost_joblocal" "$RUN_ROOT/lmcache_mp_l2_l40"
BASE="$REPO/scripts/run_shortq_gnn_dynamic_vpc_common.sh"
COPY="$RUN_ROOT/_l40_three_cost_joblocal/common_${SLURM_JOB_ID}.sh"
python - "$BASE" "$COPY" <<'PY'
from pathlib import Path
import sys
src, dst = map(Path, sys.argv[1:])
s = src.read_text()
old = 'case "$PROFILE" in\n  h100_smoke)'
profile = '''case "$PROFILE" in
  l40_smoke)
    CUDA_HOME="${L40_CUDA_HOME:?}"
    MODEL=meta-llama/Llama-3.1-8B-Instruct
    TP=1; QUANTIZATION=; GPU_UTIL=0.55; MAX_SEQS="${MAX_NUM_SEQS:-8}"
    L1_GB="$LMCACHE_L1_SIZE_GB"
    NUM_QUESTIONS="${NUM_QUESTIONS:-40}"
    SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-10}"
    MIN_TOKENS="${MIN_TOKENS:-16}"; MAX_TOKENS="${MAX_TOKENS:-16}"
    RUN_LABEL=l40_bayesian_dynamic_vpc_smoke
    L2_DIR="${RUN_ROOT}/lmcache_mp_l2_l40/bayes_${SLURM_JOB_ID}"
    SMOKE_FORCE_MIN_UNIQUE_PER_PLACEMENT=0
    ;;
  h100_smoke)'''
assert s.count(old) == 1, 'common runner changed; do not rewrite blindly'
s = s.replace(old,profile,1)
dst.write_text(s)
print('L40 job-local common runner:',dst)
PY
bash -n "$COPY"
sha256sum "$BASE" "$COPY"
echo "[L40_BAYES_SMOKE] job=$SLURM_JOB_ID model=8B TP1 L1=$LMCACHE_L1_SIZE_GB L2=$LMCACHE_L2_SIZE_GB prior=$L40_PRIOR_PATH"
echo "[L40_BAYES_SMOKE] no shared root cleanup; job-local NFS L2=$RUN_ROOT/lmcache_mp_l2_l40/bayes_${SLURM_JOB_ID}"
# The common runner will invoke the already-patched LMCache adapter JSON, GNN
# score precompute, both cold/warm waves, and cost observations at wave boundaries.
bash "$COPY"
RUN_DIR="$RUN_ROOT/l40_bayesian_dynamic_vpc_smoke_${EXPERIMENT_LABEL}_${SLURM_JOB_ID}"
[[ -s "$RUN_DIR/results/summary.json" ]] || { echo "Missing summary under $RUN_DIR" >&2; exit 80; }
python - "$RUN_DIR" "$NUM_QUESTIONS" <<'PY'
import json, pathlib, sys
r=pathlib.Path(sys.argv[1]); q=int(sys.argv[2]); s=json.loads((r/'results/summary.json').read_text())
assert s['cold']['successful']==q and s['warm']['successful']==q, s
h=r/'results/bayes_wave_history.json'
assert h.is_file() and h.stat().st_size>10, h
assert list((r/'logs/phase_trace').glob('phase_engine.pid*.jsonl'))
config=json.loads((r/'run_config.json').read_text())
assert float(config['lmcache_l2_size_gb'])==4.0 or float(config['lmcache_l2_size_gb'])>0
print('L40_BAYES_SMOKE_PASS',r)
print('cost observation diagnostics in:',r/'logs/driver.log')
PY

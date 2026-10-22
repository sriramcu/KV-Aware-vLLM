#!/usr/bin/env bash
#SBATCH --job-name=kvbayes-l40-q650
#SBATCH --partition=gpu-l40-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.err

# A SINGLE Q650 cold+warm L40/8B/TP1 Bayesian run. The H100 common launcher
# is adapted in a job-local copy only. No edits to tracked source files.
# Historical L40 reference: jobs 29425/29428 (Q650, B200, MNS10, 128 tokens,
# original ordering, selective VPC, 64-GiB L1, U0.25, NFS L2).
# DIFFERENCES: Bayesian mapping, updated three-cost feedback, physical 100-GiB
# L2 target + LRU, contemporary instrumentation. Not an exact matched A/B.
set -Eeuo pipefail
: "${SLURM_JOB_ID:?Run with sbatch}"
: "${L40_PRIOR_PATH:?Provide calibrated L40/8B/TP1 prior via --export}"

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware
L40_L2_ROOT="$RUN_ROOT/lmcache_mp_l2_l40"
BASE="$REPO/scripts/run_shortq_gnn_dynamic_vpc_common.sh"
[[ -s "$L40_PRIOR_PATH" ]] || { echo "Missing L40 prior: $L40_PRIOR_PATH" >&2; exit 2; }
[[ -f "$REPO/experiments/calibrate_bayesian_recompute.py" ]] || { echo 'Three-cost patch absent' >&2; exit 2; }
[[ -f "$BASE" ]] || { echo "Missing common launcher: $BASE" >&2; exit 2; }
[[ -x "$VENV/bin/python" ]] || { echo "Missing Python venv: $VENV" >&2; exit 2; }
cd "$REPO"
source "$VENV/bin/activate"

# Verify provenance: never silently use an H100/70B cost prior on L40/8B.
python - "$L40_PRIOR_PATH" <<'PY'
import json, math, sys
p = json.load(open(sys.argv[1]))
c = p.get('cost_prior_ms', {})
assert all(math.isfinite(float(c[x])) and float(c[x]) > 0 for x in ('cpu', 'disk', 'recompute')), c
m = p.get('cost_calibration', {})
assert m.get('recompute_prior_measured') is True and int(m.get('recompute_samples',0)) >= 3, m
print('[L40_Q650_PRIOR]', sys.argv[1], c, flush=True)
PY

# Match the earlier L40 Q650 workload/serving controls as closely as possible.
export KV_GNN_DYNAMIC_PROFILE=l40_bayes_q650
export VLLM_VPC_POLICY=selective
export KV_BAYES_ENABLE=1 KV_BAYES_PRIOR_FILE="$L40_PRIOR_PATH"
export KV_BAYES_DIRECT_COST_FEEDBACK=1 KV_BAYES_USE_COST_PROXIES=0
export KV_BAYES_READMISSION=1 KV_BAYES_POSTERIOR_MODE=fixed KV_BAYES_PRIOR_WEIGHT=1
export KV_GNN_AWARE_VPC=1 KV_GNN_L1_BACKING=1 KV_GNN_L2_BACKING=0
export LMCACHE_L1_SIZE_GB=64
export LMCACHE_L2_SIZE_GB="${LMCACHE_L2_SIZE_GB:-100}"
export KV_BAYES_DISK_GB="$LMCACHE_L2_SIZE_GB"
export KV_BAYES_GPU_GB="${KV_BAYES_GPU_GB:-2}"
export LMCACHE_L2_EVICTION_TRIGGER=0.90 LMCACHE_L2_EVICTION_RATIO=0.10
export LMCACHE_MP_MAX_INFLIGHT_STORE_GB=0.25
export LMCACHE_L2_LOOKUP_MAX_IN_FLIGHT=16 LMCACHE_L2_LOAD_MAX_IN_FLIGHT=2
export LMCACHE_MAX_WORKERS=8
export KV_FS_SHARED_WORKERS=5 KV_FS_LOOKUP_WORKERS=1 KV_FS_RETRIEVE_WORKERS=2
export KV_FS_STORE_WORKERS=0 KV_FS_DELETE_WORKERS=0
export KV_PASSAGE_ORDER_POLICY=original
export KV_REQUEST_ORDER=seeded_shuffle KV_REQUEST_ORDER_SEED=0
export WARM_ORDER=random WARM_ORDER_SEED=1
export NUM_QUESTIONS=650 SUBMISSION_BATCH_SIZE=200
export MIN_TOKENS=128 MAX_TOKENS=128 MAX_NUM_SEQS=10
export L40_GPU_MEMORY_UTILIZATION=0.55
export KV_MUTUAL_PREFIX=1 KV_VPC_SUFFICIENT_BYPASS=1
export KV_STAGE1_STARVATION_FALLBACK=1 KV_COLD_WARM_STORE_BARRIER=1
export LMCACHE_MP_PREFIX_DIAGNOSTICS=1
export VLLM_VPC_DIAGNOSTICS=1 VLLM_VPC_DIAG_INTERVAL_S=5
export VLLM_VPC_STATS_INTERVAL_S=5
export VLLM_GNN_AWARE_VPC_WINDOW=256
export GNN_CHUNK_VOTE_POLICY=threshold_with_drop_plurality
export GNN_CHUNK_CPU_MIN=2 GNN_CHUNK_GPU_MIN=8
export VLLM_PHASE_TRACE=1 VLLM_PHASE_TRACE_DEVICE_TIMING=0
export CLEAN_OLD_LMCACHE=1
export EXPERIMENT_LABEL=l40_bayes_three_cost_original_b200_mns10

# Use the installed CUDA version on the allocated L40 machine.
for candidate in "${CUDA_HOME:-}" /usr/local/cuda-13.1 /usr/local/cuda-13.0 /usr/local/cuda; do
  if [[ -n "$candidate" && -x "$candidate/bin/nvcc" ]]; then
    export L40_CUDA_HOME="$candidate"
    break
  fi
done
: "${L40_CUDA_HOME:?No CUDA toolkit with nvcc found}"

# With CLEAN_OLD_LMCACHE=1, the job-local launcher removes ONLY the children
# of this exact NFS L40 cache directory, NEVER /scratch2, and never run logs.
# Do not execute concurrently with another user job using this L40 L2 root.
[[ "$L40_L2_ROOT" == "$RUN_ROOT/lmcache_mp_l2_l40" ]] || exit 10
[[ ! -L "$L40_L2_ROOT" ]] || { echo "Unsafe symlinked L40 NFS cache root" >&2; exit 10; }
mkdir -p "$L40_L2_ROOT" "$RUN_ROOT/_l40_bayes_q650_joblocal"
COPY="$RUN_ROOT/_l40_bayes_q650_joblocal/common_${SLURM_JOB_ID}.sh"

python - "$BASE" "$COPY" <<'PY'
from pathlib import Path
import sys
src, dest = map(Path, sys.argv[1:])
s = src.read_text()
def replace_once(old, new, label):
    global s
    occurrences = s.count(old)
    if occurrences != 1:
        raise SystemExit(f'Unsupported live common runner: {label}, found {occurrences}; refusing unsafe rewrite')
    s = s.replace(old, new, 1)

profile = '''  l40_bayes_q650)
    CUDA_HOME="${L40_CUDA_HOME:?}"
    MODEL=meta-llama/Llama-3.1-8B-Instruct
    TP=1; QUANTIZATION=; GPU_UTIL="${L40_GPU_MEMORY_UTILIZATION:-0.55}"
    MAX_SEQS="${MAX_NUM_SEQS:-10}"
    L1_GB="$LMCACHE_L1_SIZE_GB"
    NUM_QUESTIONS=650; SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-200}"
    MIN_TOKENS="${MIN_TOKENS:-128}"; MAX_TOKENS="${MAX_TOKENS:-128}"
    RUN_LABEL=l40_bayesian_dynamic_vpc_q650
    L2_DIR="${RUN_ROOT}/lmcache_mp_l2_l40/bayes_q650_${SLURM_JOB_ID}"
    SMOKE_FORCE_MIN_UNIQUE_PER_PLACEMENT=0
    ;;
'''
replace_once('case "$PROFILE" in\n  h100_smoke)',
             'case "$PROFILE" in\n' + profile + '  h100_smoke)', 'L40 Q650 profile')
replace_once('  local root=/scratch2/sriramc2\n  [[ "$root" == "/scratch2/sriramc2" ]] || exit 12',
'''  local root
  if [[ "$PROFILE" == "l40_bayes_q650" ]]; then
    root="${RUN_ROOT}/lmcache_mp_l2_l40"
    [[ "$root" == "$RUN_ROOT/lmcache_mp_l2_l40" && ! -L "$root" ]] || {
      echo "Unsafe L40 NFS cleanup root: $root" >&2; exit 12;
    }
  else
    root=/scratch2/sriramc2
    [[ "$root" == "/scratch2/sriramc2" ]] || exit 12
  fi''', 'NFS-only cleanup guard')
replace_once('LOG_DIR="${RUN_DIR}/logs"\n',
'''LOG_DIR="${RUN_DIR}/logs"
export VLLM_VPC_STATS_JSONL="${LOG_DIR}/vpc_periodic.jsonl"
export VLLM_VPC_STATS_INTERVAL_S="${VLLM_VPC_STATS_INTERVAL_S:-5}"
''', 'periodic VPC instrumentation')
replace_once('  "vpc": true,\n',
'''  "vpc": true,
  "vpc_policy": "$VLLM_VPC_POLICY",
''', 'VPC mode in config')
replace_once(
    "if aware:\n"
    "    assert '[GNN_AWARE_VPC_INIT]' in vl, 'GNN-aware VPC was requested but not initialized'\n"
    "else:\n"
    "    assert '[GNN_AWARE_VPC_INIT]' not in vl, 'GNN-aware VPC initialized while gate is off'\n",
    "mode = __import__('os').environ['VLLM_VPC_POLICY']\n"
    "assert f'[VPC_POLICY_INIT] mode={mode} ' in vl, (mode, 'VPC mode did not initialize')\n"
    "if mode == 'window':\n"
    "    assert '[GNN_AWARE_VPC_INIT]' in vl, 'window compatibility marker missing'\n"
    "else:\n"
    "    assert '[GNN_AWARE_VPC_INIT]' not in vl, 'unexpected window marker in non-window mode'\n",
    'selective-mode marker validation')
dest.write_text(s)
print('[L40_Q650_JOBLOCAL_RUNNER]', dest)
PY
bash -n "$COPY"
sha256sum "$BASE" "$COPY"

echo "[L40_Q650] job=$SLURM_JOB_ID model=Llama-3.1-8B TP1 prior=$L40_PRIOR_PATH"
echo "[L40_Q650] original-order Q650 B200 MNS10 128-token, VPC=selective, L1=64GiB, L2=${LMCACHE_L2_SIZE_GB}GiB (LRU), GPU-policy=${KV_BAYES_GPU_GB}GiB, U=0.25GiB"
echo "[L40_Q650] CLEAN_OLD_LMCACHE=1 cleans *only* L40 NFS root $L40_L2_ROOT (not H100 scratch)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true
bash "$COPY"

RUN_DIR="$RUN_ROOT/l40_bayesian_dynamic_vpc_q650_${EXPERIMENT_LABEL}_${SLURM_JOB_ID}"
python - "$RUN_DIR" "$LMCACHE_L2_SIZE_GB" <<'PY'
import json, pathlib, sys
r = pathlib.Path(sys.argv[1]); l2 = float(sys.argv[2])
s = json.loads((r/'results/summary.json').read_text())
c = json.loads((r/'run_config.json').read_text())
h = json.loads((r/'results/bayes_wave_history.json').read_text())
assert all(s[p]['successful'] == 650 and s[p]['requests'] == 650 for p in ('cold','warm')), s
assert c['model'] == 'meta-llama/Llama-3.1-8B-Instruct' and c['tp'] == 1, c
assert c['max_num_seqs'] == 10 and c['l1_gb'] == 64, c
assert c['passage_order_policy'] == 'original' and c['vpc_policy'] == 'selective', c
assert float(c['lmcache_l2_size_gb']) == l2 and float(c['bayes_disk_gb']) == l2, c
assert len(h) >= 2 and h[-1]['cost_sample_counts']['recompute'] > 0, h[-1]
assert list((r/'logs/phase_trace').glob('phase_engine.pid*.jsonl')), r
print('[L40_Q650_PASS]', r)
print('[L40_Q650_E2E] cold=',s['cold']['elapsed_seconds'],'warm=',s['warm']['elapsed_seconds'])
print('[L40_Q650_FINAL_COSTS]', h[-1]['cost_estimate_ms'])
PY

# Success-only cleanup: keep all serving logs/metrics/placement records, remove
# only THIS run's NFS L2 files to avoid leaving ~100 GiB of disk objects.
L40_THIS_L2="$L40_L2_ROOT/bayes_q650_${SLURM_JOB_ID}"
if [[ "${L40_CLEAN_AFTER:-1}" == 1 && "$L40_THIS_L2" == "$RUN_ROOT/lmcache_mp_l2_l40/bayes_q650_${SLURM_JOB_ID}" && ! -L "$L40_THIS_L2" ]]; then
  echo "[L40_Q650_POST_CLEAN] Removing only this run's disk objects: $L40_THIS_L2"
  rm -rf -- "$L40_THIS_L2"
fi

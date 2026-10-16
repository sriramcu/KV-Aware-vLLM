#!/usr/bin/env bash
#SBATCH --job-name=kvaware-l40-gnn-mns
#SBATCH --partition=gpu-l40-dchen
#SBATCH --nodelist=codenimbus-002-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.err

# L40/8B/TP1 Q650 replication: GNN threshold 8 at MNS10/7/4,
# plus threshold 22 at MNS10. Clean dedicated L40 LMCache cache root on NFS.
set -Eeuo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch, not bash}"
: "${MAX_NUM_SEQS:?Supply MAX_NUM_SEQS=10|7|4}"
: "${GNN_CHUNK_GPU_MIN:?Supply GNN_CHUNK_GPU_MIN=8|22}"
case "$MAX_NUM_SEQS:$GNN_CHUNK_GPU_MIN" in
  10:8|7:8|4:8|10:22) ;;
  *) echo "ERROR: unexpected configuration: MNS=$MAX_NUM_SEQS GPU_MIN=$GNN_CHUNK_GPU_MIN" >&2; exit 2 ;;
esac

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware
BASE_COMMON="$REPO/scripts/run_shortq_gnn_dynamic_vpc_common.sh"
[[ -f "$BASE_COMMON" ]] || { echo "Missing common runner: $BASE_COMMON" >&2; exit 2; }
[[ -x "$VENV/bin/python" ]] || { echo "Missing Python: $VENV" >&2; exit 2; }
[[ -f "$REPO/vllm/v1/core/vpc_policies/stats.py" ]] || { echo "Missing four-mode VPC refactor" >&2; exit 2; }

# Existing L40 smokes used CUDA 13.0/13.1; use installed nvcc instead of guessing.
for cuda_candidate in "${CUDA_HOME:-}" /usr/local/cuda-13.1 /usr/local/cuda-13.0 /usr/local/cuda; do
  if [[ -n "$cuda_candidate" && -x "$cuda_candidate/bin/nvcc" ]]; then
    export L40_CUDA_HOME="$cuda_candidate"
    break
  fi
done
: "${L40_CUDA_HOME:?No CUDA toolkit with nvcc found}"

# H100 Q650 controls adapted to the previously validated 1xL40/8B profile.
export KV_GNN_DYNAMIC_PROFILE=l40_q650
export VLLM_VPC_POLICY=selective
export KV_GNN_AWARE_VPC=1
export VLLM_GNN_AWARE_VPC_WINDOW=256
export KV_CHUNK_PLACEMENT_MODE=gnn
export GNN_CHUNK_VOTE_POLICY=threshold_with_drop_plurality
export GNN_CHUNK_CPU_MIN=2
export GNN_CHUNK_GPU_MIN
unset GNN_RANDOM_MATCH_DISTRIBUTION || true
export GNN_RANDOM_CHUNK_SEED=0
export LMCACHE_L1_SIZE_GB=64
export LMCACHE_MP_MAX_INFLIGHT_STORE_GB=0.25
export LMCACHE_L2_LOOKUP_MAX_IN_FLIGHT=16
export LMCACHE_L2_LOAD_MAX_IN_FLIGHT=2
export LMCACHE_MAX_WORKERS=8
export KV_GNN_L1_BACKING=1
export KV_GNN_L2_BACKING=0
export KV_REQUEST_ORDER=seeded_shuffle
export KV_REQUEST_ORDER_SEED=0
export WARM_ORDER=random
export WARM_ORDER_SEED=1
export SUBMISSION_BATCH_SIZE=200
export MIN_TOKENS=128
export MAX_TOKENS=128
export KV_MUTUAL_PREFIX=1
export KV_VPC_SUFFICIENT_BYPASS=1
export KV_STAGE1_STARVATION_FALLBACK=1
export KV_COLD_WARM_STORE_BARRIER=1
export LMCACHE_MP_PREFIX_DIAGNOSTICS=1
export KV_FS_SHARED_WORKERS=5
export KV_FS_LOOKUP_WORKERS=1
export KV_FS_RETRIEVE_WORKERS=2
export KV_FS_STORE_WORKERS=0
export KV_FS_DELETE_WORKERS=0
export VLLM_VPC_DIAGNOSTICS=1
export VLLM_VPC_DIAG_INTERVAL_S=5
export VLLM_VPC_STATS_INTERVAL_S=5
export L40_GPU_MEMORY_UTILIZATION=0.55
export L40_L2_ROOT="$RUN_ROOT/lmcache_mp_l2_l40"
export CLEAN_OLD_LMCACHE=1
export EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-l40_mns${MAX_NUM_SEQS}_gnn_t${GNN_CHUNK_GPU_MIN}}"
[[ "$EXPERIMENT_LABEL" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid EXPERIMENT_LABEL' >&2; exit 2; }
mkdir -p "$RUN_ROOT" "$L40_L2_ROOT"

# Job-local immutable copy of the LIVE common runner, like the verified H100
# four-mode sweep. No edits to tracked repo files or other run directories.
COMMON_DIR="$RUN_ROOT/_l40_q650_gnn_mns_common_copies"
mkdir -p "$COMMON_DIR"
COPIED_COMMON="$COMMON_DIR/common_${SLURM_JOB_ID}.sh"
"$VENV/bin/python" - "$BASE_COMMON" "$COPIED_COMMON" <<'PY'
from pathlib import Path
import sys
src, dest = map(Path, sys.argv[1:])
s = src.read_text()

def replace_once(old, new, label):
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"Unsupported live common runner: {label} matched {n} times; refusing to launch")
    s = s.replace(old, new, 1)

profile = '''  l40_q650)
    CUDA_HOME="${L40_CUDA_HOME:?}"
    MODEL=meta-llama/Llama-3.1-8B-Instruct
    TP=1; QUANTIZATION=; GPU_UTIL="${L40_GPU_MEMORY_UTILIZATION:-0.55}"
    MAX_SEQS="${MAX_NUM_SEQS:?}"
    L1_GB="$LMCACHE_L1_SIZE_GB"
    NUM_QUESTIONS=650; SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-200}"
    MIN_TOKENS="${MIN_TOKENS:-128}"; MAX_TOKENS="${MAX_TOKENS:-128}"
    RUN_LABEL=l40_shortq_gnn_dynamic_vpc_q650
    L2_DIR="${L40_L2_ROOT:?}/gnn_q650_${SLURM_JOB_ID}"
    SMOKE_FORCE_MIN_UNIQUE_PER_PLACEMENT=0
    ;;
'''
replace_once('case "$PROFILE" in\n  h100_smoke)',
             'case "$PROFILE" in\n' + profile + '  h100_smoke)',
             'L40 profile')

# Patch L40 cache cleaning to target the actual NFS L2 cache root.
# The original common runner's clean_old_lmcache() is H100-only and would
# otherwise delete /scratch2/sriramc2. For L40, CLEAN_OLD_LMCACHE=1
# must instead purge ONLY the known L40 LMCache L2 root on NFS.
replace_once('  local root=/scratch2/sriramc2\n  [[ "$root" == "/scratch2/sriramc2" ]] || exit 12',
'''  local root
  if [[ "$PROFILE" == "l40_q650" ]]; then
    [[ "$CLEAN_OLD_LMCACHE" == "1" ]] || { echo 'L40 requires CLEAN_OLD_LMCACHE=1' >&2; exit 12; }
    root="${L40_L2_ROOT:?L40_L2_ROOT must point to the L40 NFS LMCache cache root}"
    [[ "$root" == "$RUN_ROOT/lmcache_mp_l2_l40" ]] || {
      echo "Unsafe L40 L2 cleanup root: $root" >&2; exit 12;
    }
    [[ ! -L "$root" ]] || { echo "Refusing symlinked L40 L2 root: $root" >&2; exit 12; }
  else
    root=/scratch2/sriramc2
    [[ "$root" == "/scratch2/sriramc2" ]] || exit 12
  fi''', 'L40 NFS-only cleanup root guard')

# Retain the original clean_old_lmcache invocation, now patched for L40.
replace_once('clean_old_lmcache\nmkdir -p "$LOG_DIR" "$RESULT_DIR" "$PLACEMENT_DIR" "$L2_DIR"',
'''clean_old_lmcache
[[ "$PROFILE" != "l40_q650" || "$CLEAN_OLD_LMCACHE" == "1" ]] || exit 12
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$PLACEMENT_DIR" "$L2_DIR"''', 'L40 NFS cleanup invocation')

replace_once('LOG_DIR="${RUN_DIR}/logs"\n',
'''LOG_DIR="${RUN_DIR}/logs"
export VLLM_VPC_STATS_JSONL="${LOG_DIR}/vpc_periodic.jsonl"
export VLLM_VPC_STATS_INTERVAL_S="${VLLM_VPC_STATS_INTERVAL_S:-5}"
''', 'periodic stats path')

replace_once('  "vpc": true,\n',
'''  "vpc": true,
  "vpc_policy": "$VLLM_VPC_POLICY",
  "vpc_diagnostics": "$VLLM_VPC_DIAGNOSTICS",
  "gnn_chunk_gpu_min": ${GNN_CHUNK_GPU_MIN},
  "gnn_chunk_cpu_min": ${GNN_CHUNK_CPU_MIN},
''', 'manifest fields')

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
    "    assert '[GNN_AWARE_VPC_INIT]' not in vl, 'window initialized in non-window mode'\n",
    'four-mode VPC policy validation')

dest.write_text(s)
print(f"Job-local common script: {dest} (derived from {src})")
PY
bash -n "$COPIED_COMMON"
sha256sum "$BASE_COMMON" "$COPIED_COMMON"
echo "[L40_REPLICATION] job=$SLURM_JOB_ID mns=$MAX_NUM_SEQS threshold=$GNN_CHUNK_GPU_MIN policy=$VLLM_VPC_POLICY model=8B TP=1" 
echo "L40 NFS L2 root: $L40_L2_ROOT (CLEAN_OLD_LMCACHE=$CLEAN_OLD_LMCACHE; purge prior contents before each run)"
echo "L2 path: $L40_L2_ROOT/gnn_q650_${SLURM_JOB_ID}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true
# On job exit, remove this run's NFS L2 cache too; the result/diagnostic
# files are saved separately under RUN_ROOT. Never remove unrelated paths.
L40_THIS_L2="$L40_L2_ROOT/gnn_q650_${SLURM_JOB_ID}"
cleanup_this_l40_l2() {
  local status=$?
  trap - EXIT
  if [[ "${CLEAN_OLD_LMCACHE:-}" == "1" &&         "$L40_L2_ROOT" == "$RUN_ROOT/lmcache_mp_l2_l40" &&         "$L40_THIS_L2" == "$RUN_ROOT/lmcache_mp_l2_l40/gnn_q650_${SLURM_JOB_ID}" &&         ! -L "$L40_L2_ROOT" && ! -L "$L40_THIS_L2" ]]; then
    echo "[L40_NFS_CLEANUP] Removing this job's LMCache L2 objects: $L40_THIS_L2"
    rm -rf -- "$L40_THIS_L2" || true
  fi
  exit "$status"
}
trap cleanup_this_l40_l2 EXIT
bash "$COPIED_COMMON"

RUN_DIR="$RUN_ROOT/l40_shortq_gnn_dynamic_vpc_q650_${EXPERIMENT_LABEL}_${SLURM_JOB_ID}"
[[ -f "$RUN_DIR/results/summary.json" ]] || { echo "No Q650 result: $RUN_DIR" >&2; exit 60; }
"$VENV/bin/python" "$REPO/scripts/analyze_mp_congestion_chthm.py" \
  --lmcache-log "$RUN_DIR/logs/lmcache.log" --vllm-log "$RUN_DIR/logs/vllm.log" \
  --results-dir "$RUN_DIR/results" --strict-phase \
  --output "$RUN_DIR/mp_diag_summary.json" > "$RUN_DIR/mp_diag_summary.txt"

"$VENV/bin/python" - "$RUN_DIR" "$MAX_NUM_SEQS" "$GNN_CHUNK_GPU_MIN" <<'PY'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1]); mns=int(sys.argv[2]); threshold=int(sys.argv[3])
config=json.loads((run/'run_config.json').read_text())
summary=json.loads((run/'results/summary.json').read_text())
diag=json.loads((run/'mp_diag_summary.json').read_text())
placement=json.loads((run/'placement/gnn_placement_summary.json').read_text())
assert config['max_num_seqs'] == mns and config['gnn_chunk_gpu_min'] == threshold
assert config['vpc_policy'] == 'selective'
assert config['chunk_vote']['policy'] == 'threshold_with_drop_plurality'
assert config['chunk_vote']['config']['gpu_min'] == threshold
assert config['chunk_vote']['config']['cpu_min'] == 2
assert summary['cold']['successful'] == summary['cold']['requests'] == 650
assert summary['warm']['successful'] == summary['warm']['requests'] == 650
attrib=diag['phase_attribution']
assert attrib['unknown_scheduler_admit_count'] == 0, attrib
assert attrib['scheduler_admit_by_phase'] == {'cold':650, 'warm':650}, attrib
for phase in ('cold', 'warm'):
    assert diag['raw_chthm'][phase]['requests'] == 650
files=sorted((run/'logs').glob('vpc_periodic.pid*.jsonl'))
assert files, f'No VPC periodic telemetry in {run}/logs'
reports=[]
for f in files:
    reports.extend(json.loads(line) for line in f.read_text().splitlines() if line.strip())
assert any(r.get('policy') == 'selective' for r in reports), 'No selective VPC snapshots'
counts=placement['unique_runtime_placement_counts']
print('L40_GNN_MNS_Q650_PASS', 'mns=', mns, 'gpu_threshold=', threshold,
      'labels=', counts, 'periodic_snapshots=', len(reports))
if counts.get('gpu', 0) == 0:
    print('WARNING: ZERO GPU-classified chunks; threshold comparison cannot demonstrate selective GPU retention')
for phase in ('cold', 'warm'):
    d=diag['raw_chthm'][phase]
    print('CHTHM',phase,'raw_observation_coverage_pct=',d['raw_external_observation_coverage_pct'],
          'rates_exact=',d['rates_exact'])
print('run_dir=',run)
PY

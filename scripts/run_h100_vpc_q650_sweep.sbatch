#!/usr/bin/env bash
#SBATCH --job-name=kvaware-vpc-q650
#SBATCH --partition=gpu-h100-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.err

# Four-mode, actual H100 TP2 Q650 policy sweep. Uses the existing project
# GNN + LMCache MP runner, but validates explicit VPC policy selection.
# Never deletes shared /scratch2/sriramc2 contents.
set -Eeuo pipefail

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware




: "${SLURM_JOB_ID:?Run this with sbatch}"
: "${VLLM_VPC_POLICY:?Set VLLM_VPC_POLICY=vanilla|window|tierwise|selective}"
case "$VLLM_VPC_POLICY" in
  vanilla|window|tierwise|selective) ;;
  *) echo "Invalid VLLM_VPC_POLICY=$VLLM_VPC_POLICY" >&2; exit 2 ;;
esac

mkdir -p "$RUN_ROOT"
[[ -x "$VENV/bin/python" ]] || { echo "Missing venv: $VENV" >&2; exit 2; }
[[ -f "$REPO/vllm/v1/core/vpc_policies/stats.py" ]] || {
  echo "VPC policy refactor not found in live repo" >&2; exit 2;
}

# Identical workload, placement, and lower-tier controls in every arm.
export KV_GNN_DYNAMIC_PROFILE=h100_q650
export VLLM_VPC_POLICY
export VLLM_GNN_AWARE_VPC_WINDOW=256
export KV_GNN_AWARE_VPC=1               # legacy knob; explicit mode overrides it
export KV_GNN_L1_BACKING=1
export KV_GNN_L2_BACKING=0
export LMCACHE_L1_SIZE_GB=225
export LMCACHE_MP_MAX_INFLIGHT_STORE_GB=0.25
export LMCACHE_L2_LOOKUP_MAX_IN_FLIGHT=16
export LMCACHE_L2_LOAD_MAX_IN_FLIGHT=2
export MAX_NUM_SEQS=16
export SUBMISSION_BATCH_SIZE=200
export KV_REQUEST_ORDER=seeded_shuffle
export KV_REQUEST_ORDER_SEED=0
export WARM_ORDER=random
export WARM_ORDER_SEED=1
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
export GNN_CHUNK_VOTE_POLICY=threshold_with_drop_plurality
export GNN_RANDOM_CHUNK_SEED=0
unset GNN_RANDOM_MATCH_DISTRIBUTION || true

export CLEAN_OLD_LMCACHE="${CLEAN_OLD_LMCACHE:-1}"
export VLLM_VPC_STATS_INTERVAL_S=5

EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-vpc4_gnn_drop_u025_${VLLM_VPC_POLICY}_${SLURM_JOB_ID}}"
[[ "$EXPERIMENT_LABEL" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "Invalid EXPERIMENT_LABEL=$EXPERIMENT_LABEL" >&2; exit 2;
}
export EXPERIMENT_LABEL

echo "========== SCRATCH BEFORE RUN =========="
echo "Job: ${SLURM_JOB_ID}"
echo "Policy: ${VLLM_VPC_POLICY}"
echo "CLEAN_OLD_LMCACHE=${CLEAN_OLD_LMCACHE}"

du -sh /scratch2/sriramc2 2>&1 || true
df -h /scratch2 2>&1 || true
echo "========================================" 

# The uploaded common Q650 runner still checks a W256-only initialization
# marker. Make an immutable job-specific COPY of the LIVE runner and update
# only that assertion, the run manifest, and the periodic stats path.
# The production common script is NOT overwritten.
BASE_COMMON="$REPO/scripts/run_shortq_gnn_dynamic_vpc_common.sh"
SWEEP_COMMON_DIR="$RUN_ROOT/_h100_vpc_sweep_common_copies"
mkdir -p "$SWEEP_COMMON_DIR"
SWEEP_COMMON="$SWEEP_COMMON_DIR/common_${SLURM_JOB_ID}.sh"

"$VENV/bin/python" - "$BASE_COMMON" "$SWEEP_COMMON" <<'PY'
from pathlib import Path
import sys
src, dest = map(Path, sys.argv[1:])
s = src.read_text()

def replace_once(old, new, what):
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(
            f"Unexpected LIVE common script ({what}: {count} matches). "
            "No changes made; review live launcher before submitting sweep."
        )
    s = s.replace(old, new, 1)

replace_once(
    'LOG_DIR="${RUN_DIR}/logs"\n',
    'LOG_DIR="${RUN_DIR}/logs"\n'
    '# Sweep-only periodic VPC counters in the same run log directory.\n'
    'export VLLM_VPC_STATS_JSONL="${LOG_DIR}/vpc_periodic.jsonl"\n'
    'export VLLM_VPC_STATS_INTERVAL_S="${VLLM_VPC_STATS_INTERVAL_S:-5}"\n',
    'periodic statistics path',
)
replace_once(
    '  "vpc": true,\n',
    '  "vpc": true,\n'
    '  "vpc_policy": "$VLLM_VPC_POLICY",\n',
    'run configuration policy field',
)
replace_once(
    "if aware:\n"
    "    assert '[GNN_AWARE_VPC_INIT]' in vl, 'GNN-aware VPC was requested but not initialized'\n"
    "else:\n"
    "    assert '[GNN_AWARE_VPC_INIT]' not in vl, 'GNN-aware VPC initialized while gate is off'\n",
    "mode = __import__('os').environ['VLLM_VPC_POLICY']\n"
    "assert f'[VPC_POLICY_INIT] mode={mode} ' in vl, (mode, 'VPC mode did not initialize')\n"
    "if mode == 'window':\n"
    "    assert '[GNN_AWARE_VPC_INIT]' in vl, 'W256 compatibility marker missing'\n"
    "else:\n"
    "    assert '[GNN_AWARE_VPC_INIT]' not in vl, 'window initialized in non-window mode'\n",
    'legacy VPC validation replacement',
)
dest.write_text(s)
print(f"Copied and adapted LIVE runner: {src} -> {dest}")
PY

sha256sum "$BASE_COMMON" "$SWEEP_COMMON"
echo "[H100_VPC_SWEEP] job=$SLURM_JOB_ID mode=$VLLM_VPC_POLICY label=$EXPERIMENT_LABEL"
# All original GNN precompute, LMCache MP, vLLM TP2 and Q650 analysis steps.
bash "$SWEEP_COMMON"

echo "========== SCRATCH AFTER RUN =========="
du -sh /scratch2/sriramc2 2>&1 || true
df -h /scratch2 2>&1 || true
echo "======================================="

# Fail closed if the expected per-run periodic statistics are absent or the
# policy did not execute. The common runner already checks 650/650 success.
RUN_DIR="$RUN_ROOT/h100_shortq_gnn_dynamic_vpc_q650_${EXPERIMENT_LABEL}_${SLURM_JOB_ID}"
"$VENV/bin/python" - "$RUN_DIR" "$VLLM_VPC_POLICY" <<'PY'
import json, pathlib, sys
run_dir, policy = pathlib.Path(sys.argv[1]), sys.argv[2]
assert run_dir.is_dir(), f"Run directory missing: {run_dir}"
config = json.loads((run_dir / 'run_config.json').read_text())
assert config['vpc_policy'] == policy, config.get('vpc_policy')
summary = json.loads((run_dir / 'results' / 'summary.json').read_text())
assert summary['cold']['successful'] == 650, summary['cold']
assert summary['warm']['successful'] == 650, summary['warm']
files = sorted((run_dir / 'logs').glob('vpc_periodic.pid*.jsonl'))
assert files, 'No periodic VPC stats; check scheduler subprocess environment'
reports = []
for f in files:
    for line in f.read_text().splitlines():
        if line.strip():
            reports.append(json.loads(line))
reports = [r for r in reports if r.get('policy') == policy]
assert reports, f"No VPC stats for mode={policy}"
last = max(reports, key=lambda r: r.get('cached_reclaims', 0))
if policy == 'selective':
    assert any(sum(r.get('admission_reject_counts', {}).values()) > 0
               for r in reports), 'Selective admission never rejected a block'
    assert all(v == 0 for r in reports for k, v in r.get('victim_counts', {}).items()
               if k != 'gpu'), 'Selective reclaimed a non-GPU cached block'
else:
    assert any(r.get('cached_reclaims', 0) > 0 for r in reports), (
        policy, 'No cached VPC eviction observed in full Q650 run')
print('H100_VPC_POLICY_SWEEP_PASS', 'policy=', policy,
      'max_cached_reclaims=', last['cached_reclaims'],
      'select_calls=', last['select_calls'],
      'admission_rejects=', max((sum(r.get('admission_reject_counts', {}).values())
                                   for r in reports), default=0))
print('run_dir=', run_dir)
PY

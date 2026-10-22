#!/usr/bin/env bash
#SBATCH --job-name=kvbayes-l40-preflight
#SBATCH --partition=gpu-l40-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.err
# L40-only functional preflight; does not replace H100 TP2 calibration/smokes.
# Does NOT clear /scratch2 or any existing NFS LMCache directory.
set -Eeuo pipefail
REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
VENV=/mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029
ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware
RUN_DIR="${ROOT}/l40_three_cost_preflight_${SLURM_JOB_ID:?run via sbatch}"
mkdir -p "$RUN_DIR"
cd "$REPO"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/shared/gpfs/home/sriramc2/.cache/huggingface}"
export CUDA_MPS_PIPE_DIRECTORY="/tmp/kvaware-no-mps-${USER}-${SLURM_JOB_ID}-DO-NOT-CREATE"
for candidate in "${CUDA_HOME:-}" /usr/local/cuda-13.1 /usr/local/cuda-13.0 /usr/local/cuda; do
  if [[ -n "$candidate" && -x "$candidate/bin/nvcc" ]]; then
    export CUDA_HOME="$candidate"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    break
  fi
done
[[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]] || { echo 'No CUDA toolkit found' >&2; exit 2; }
[[ -f experiments/calibrate_bayesian_recompute.py ]] || { echo 'Incremental patch not installed' >&2; exit 2; }
grep -q 'class PhaseRecomputeCostReader' Hierarchical_KV/shortq_placement/bayesian_mapper.py || { echo 'Missing recompute reader' >&2; exit 2; }
{
  echo "Job=$SLURM_JOB_ID node=$(hostname) CPUs=${SLURM_CPUS_PER_TASK:-unknown}"
  git rev-parse HEAD
  git status --short
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} | tee "$RUN_DIR/provenance.txt"
echo '=== Existing CPU regression tests ==='
python -m unittest tests.test_bayesian_mapper_cpu -v > "$RUN_DIR/cpu_tests.log" 2>&1 || { tail -90 "$RUN_DIR/cpu_tests.log"; exit 10; }

# Temporary job-local copies: leave the shared/installed H100 calibration scripts untouched.
python - "$REPO" "$RUN_DIR" <<'PY'
from pathlib import Path
import sys
repo, run = map(Path, sys.argv[1:])
t = (repo/'experiments/calibrate_bayesian_tier_costs.py').read_text()
a = 'if torch.cuda.device_count() != 2:'
b = 'if torch.cuda.device_count() != 1:'
assert t.count(a) == 1, 'transfer calibrator shape changed; refuse edit'
t = t.replace(a,b,1)
(run/'l40_transfer_calibrator.py').write_text(t)
r = (repo/'experiments/calibrate_bayesian_recompute.py').read_text()
a = "llm = LLM(model=args.model, tensor_parallel_size=2, quantization='fp8',"
b = "llm = LLM(model=args.model, tensor_parallel_size=1, quantization=None,"
assert r.count(a) == 1, 'recompute calibrator shape changed; refuse edit'
r = r.replace(a,b,1)
(run/'l40_recompute_calibrator.py').write_text(r)
print('Wrote job-local L40 TP1 calibration adapters (original source untouched)')
PY

PRIOR_IN="$REPO/Hierarchical_KV/shortq_placement/priors/rank_smoke_diagnostic.json"
TRANSFER_PRIOR="$RUN_DIR/l40_transfer_only.json"
L40_PRIOR="$RUN_DIR/rank_l40_calibrated.json"
mkdir -p "$RUN_DIR/calibration_scratch" "$RUN_DIR/prefill_trace"
echo '=== L40 TP1 8B transfer prior (NFS, 64MiB/chunk) ==='
python "$RUN_DIR/l40_transfer_calibrator.py" \
  --prior-in "$PRIOR_IN" --prior-out "$TRANSFER_PRIOR" \
  --mib-per-rank 64 --repeats 5 --scratch "$RUN_DIR/calibration_scratch" \
  > "$RUN_DIR/transfer_calibration.log" 2>&1 || { tail -100 "$RUN_DIR/transfer_calibration.log"; exit 20; }

echo '=== L40 TP1 Llama 3.1 8B recompute prior ==='
python "$RUN_DIR/l40_recompute_calibrator.py" \
  --prior-in "$TRANSFER_PRIOR" --prior-out "$L40_PRIOR" \
  --trace-dir "$RUN_DIR/prefill_trace" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu-utilization 0.55 --rounds 3 --min-samples 3 \
  > "$RUN_DIR/recompute_calibration.log" 2>&1 || { tail -150 "$RUN_DIR/recompute_calibration.log"; exit 30; }

python - "$L40_PRIOR" <<'PY'
import json, math, sys
p = json.load(open(sys.argv[1]))
print('L40 prior costs, ms/512:', p['cost_prior_ms'])
m = p['cost_calibration']
assert m['recompute_prior_measured'] is True
assert m['recompute_samples'] >= 3
assert all(math.isfinite(float(v)) and float(v) > 0 for v in p['cost_prior_ms'].values())
print('recompute samples:', m['recompute_samples'])
print('NOTE: L40 TP1 8B prior must NOT be fed to H100 TP2 70B runs.')
PY

# Verify patched LMCache fs_native L2 capacity and eviction JSON is accepted at startup.
# This is a configuration/startup check, NOT a physical LRU-eviction-under-load test.
L2_DIR="$RUN_DIR/lmcache_l2_smoke"
mkdir -p "$L2_DIR"
PORT=$((24000 + SLURM_JOB_ID % 12000))
L2_JSON=$(python - "$L2_DIR" <<'PY'
import json,sys
print(json.dumps({'type':'fs_native','base_path':sys.argv[1],
  'num_workers':4,'use_odirect':False,'max_capacity_gb':1.0,
  'eviction':{'eviction_policy':'LRU','trigger_watermark':0.90,'eviction_ratio':0.10}}))
PY
)
echo "$L2_JSON" > "$RUN_DIR/l2_adapter_config.json"
echo '=== LMCache L2 adapter startup smoke (1 GiB, NFS) ==='
LM_PID=''
cleanup() { if [[ -n "$LM_PID" ]]; then kill -TERM "$LM_PID" 2>/dev/null || true; wait "$LM_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT
lmcache server --host localhost --port "$PORT" --chunk-size 512 \
  --l1-size-gb 8 --l2-load-max-in-flight 2 --l2-lookup-max-in-flight 16 \
  --l2-store-max-inflight-gb 0.25 --l2-store-policy gnn_dynamic \
  --eviction-policy LRU --max-workers 4 --l2-adapter "$L2_JSON" \
  > "$RUN_DIR/lmcache_l2_startup.log" 2>&1 &
LM_PID=$!
READY=0
for i in $(seq 1 60); do
  kill -0 "$LM_PID" 2>/dev/null || break
  if python - "$PORT" <<'PY' >/dev/null 2>&1
import socket,sys
s=socket.socket();s.settimeout(0.25)
try: ok = s.connect_ex(('127.0.0.1',int(sys.argv[1]))) == 0
finally: s.close()
raise SystemExit(0 if ok else 1)
PY
  then READY=1; break; fi
  sleep 1
done
if [[ "$READY" != 1 ]]; then
  echo 'LMCache L2 startup did not complete; see startup log' >&2
  tail -100 "$RUN_DIR/lmcache_l2_startup.log" >&2
  exit 40
fi
sleep 3
kill -0 "$LM_PID" 2>/dev/null || { echo 'LMCache exited after opening port' >&2; exit 41; }
echo "L40_THREE_COST_PREFLIGHT_PASS job=$SLURM_JOB_ID output=$RUN_DIR"
echo 'LIMITATIONS: no live Bayesian Q12/Q80 waves, no L2 eviction pressure, no TP2, no H100 prior validation.'

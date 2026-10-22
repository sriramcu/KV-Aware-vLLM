#!/usr/bin/env bash
#SBATCH --job-name=kvbayes-recompute
#SBATCH --partition=gpu-h100-dchen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware/%x_%j.err
set -Eeuo pipefail
REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware
source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029/bin/activate
REPO_ROOT=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
INPUT="$RUN_ROOT/rank_h100_calibrated.json"
JOB_DIR="$RUN_ROOT/h100_recompute_only_${SLURM_JOB_ID}"
OUTPUT="$JOB_DIR/rank_h100_three_cost.json"
mkdir -p "$JOB_DIR"
test -f "$INPUT"
test -f experiments/calibrate_bayesian_recompute.py
cp "$INPUT" "$JOB_DIR/input_h100_cost_prior.json"
echo "Slurm job $SLURM_JOB_ID; CPUs=${SLURM_CPUS_PER_TASK:-unknown}"
scontrol show job "$SLURM_JOB_ID" | grep -E 'NumCPUs=|ReqTRES=|AllocTRES=' || true
export CUDA_MPS_PIPE_DIRECTORY="/tmp/kvaware-no-mps-${USER}-${SLURM_JOB_ID}-DO-NOT-CREATE"
python experiments/calibrate_bayesian_recompute.py \
  --prior-in "$JOB_DIR/input_h100_cost_prior.json" \
  --prior-out "$OUTPUT" \
  --trace-dir "$JOB_DIR/prefill_trace"
python - "$OUTPUT" <<'PY'
import json, math, sys
p = json.load(open(sys.argv[1]))
c = p['cost_prior_ms']
m = p['cost_calibration']
assert m.get('recompute_prior_measured') is True, m
assert m.get('recompute_samples', 0) >= 3, m
assert all(math.isfinite(float(v)) and float(v)>0 for v in c.values()), c
print('VERIFIED calibrated three-cost JSON:', sys.argv[1])
print('Costs (ms):', c)
print('Recompute samples:', m['recompute_samples'])
PY

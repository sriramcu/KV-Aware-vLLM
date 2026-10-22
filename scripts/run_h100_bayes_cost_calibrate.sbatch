#!/usr/bin/env bash
#SBATCH --job-name=kvbayes-calibrate
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
source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029/bin/activate
cd "$REPO"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} CPUs=${SLURM_CPUS_PER_TASK:-unset}"; nproc
scontrol show job "$SLURM_JOB_ID" | grep -E 'NumCPUs=|ReqTRES=|AllocTRES=' || true
export CUDA_MPS_PIPE_DIRECTORY="/tmp/kvaware-no-mps-${USER}-${SLURM_JOB_ID}-DO-NOT-CREATE"
CALIB_DIR="/mnt/shared/gpfs/home/sriramc2/runs/kvaware/calib_${SLURM_JOB_ID}"
mkdir -p "$CALIB_DIR"
python experiments/calibrate_bayesian_tier_costs.py \
  --prior-in "$REPO/Hierarchical_KV/shortq_placement/priors/rank_smoke_diagnostic.json" \
  --prior-out "$CALIB_DIR/transfer_only.json" \
  --scratch /scratch2/sriramc2
python experiments/calibrate_bayesian_recompute.py \
  --prior-in "$CALIB_DIR/transfer_only.json" \
  --prior-out /mnt/shared/gpfs/home/sriramc2/runs/kvaware/rank_h100_calibrated.json \
  --trace-dir "$CALIB_DIR/prefill_trace"

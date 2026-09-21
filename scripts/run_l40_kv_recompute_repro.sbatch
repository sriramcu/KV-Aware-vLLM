#!/usr/bin/env bash
#SBATCH --job-name=kv-recompute-l40
#SBATCH --partition=gpu-l40-dchen
#SBATCH --nodelist=codenimbus-002-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err
set -Eeuo pipefail
REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029/bin/activate
# Existing L40 project scripts use this toolkit path.
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_MPS_PIPE_DIRECTORY="/tmp/kvaware-no-mps-${USER}-${SLURM_JOB_ID}-DO-NOT-CREATE"
export REPO
export WORK="/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/repro_l40_${SLURM_JOB_ID}"
cd "$REPO"
./scripts/run_kv_recompute_repro_local.sh

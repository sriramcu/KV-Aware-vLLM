#!/bin/bash
#SBATCH --job-name=kvaware_h100_wo_gnn
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -euxo pipefail
REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
if [[ "${SC_LMCACHE_SOURCE_DEFAULT_KNOBS:-1}" == "1" ]]; then
  # Existing environment values win over defaults in this file.
  source "$REPO/local_repro/sc_lmcache_knobs.sh"
fi
export PYTHONFAULTHANDLER=1
export MAX_QUESTIONS="${MAX_QUESTIONS:-250}"
export SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-50}"
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0
export DATASET_NAME="medical"

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs
"$REPO/local_repro/run_driver.sh" wo_gnn 2>&1

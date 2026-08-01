#!/bin/bash
#SBATCH --job-name=kvaware_h100_with_gnn
#SBATCH --partition=gpu-h100-dchen
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16

set -euxo pipefail
export PYTHONPATH=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM:${PYTHONPATH:-}
REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
if [[ "${SC_LMCACHE_SOURCE_DEFAULT_KNOBS:-1}" == "1" ]]; then
  # Existing environment values win over defaults in this file.
  source "$REPO/local_repro/sc_lmcache_knobs.sh"
fi
# export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs
/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/run_driver.sh with_gnn 2>&1



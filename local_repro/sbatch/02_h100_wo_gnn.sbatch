#!/bin/bash
#SBATCH --job-name=kvaware_h100_wo_gnn
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=260G
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -euxo pipefail
export PYTHONPATH=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM:${PYTHONPATH:-}
export PYTHONFAULTHANDLER=1
export SRIRAM_STACK_DUMP_INTERVAL=120
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs
/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/run_driver.sh wo_gnn 2>&1

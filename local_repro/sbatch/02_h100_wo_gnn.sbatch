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
export PYTHONPATH=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM:${PYTHONPATH:-}
export PYTHONFAULTHANDLER=1
export SRIRAM_STACK_DUMP_INTERVAL=120
export SRIRAM_KV_IO_TRACE=0
export SRIRAM_KV_MEM_DEBUG=0

export LMCACHE_P0_CLIENT_LOOKUP_ADMISSION_TIMEOUT_MS=0 # default
export LMCACHE_P0_CLIENT_LOOKUP_MAX_INFLIGHT=0 
export LMCACHE_P0_LOOKUP_MAX_INFLIGHT=0  # default 4
export LMCACHE_P0_DISK_PUT_MAX_PENDING=0 # default 8
export MAX_QUESTIONS=250
export SUBMISSION_BATCH_SIZE="$MAX_QUESTIONS"
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0


DATASET_NAME="medical"

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs
/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/run_driver.sh wo_gnn 2>&1

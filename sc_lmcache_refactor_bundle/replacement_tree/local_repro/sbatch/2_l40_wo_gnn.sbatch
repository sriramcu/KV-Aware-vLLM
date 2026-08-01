#!/bin/bash
#SBATCH --job-name=kvaware_l40_wo_gnn
#SBATCH --partition=gpu-l40-dchen
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

echo "=== MEMORY BEFORE START ==="
free -h
grep -E \
'^(MemTotal|MemFree|MemAvailable|Cached|Dirty|Writeback):' \
/proc/meminfo

set -euxo pipefail

export PYTHONPATH=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM:${PYTHONPATH:-}
REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
if [[ "${SC_LMCACHE_SOURCE_DEFAULT_KNOBS:-1}" == "1" ]]; then
  # Existing environment values win over defaults in this file.
  source "$REPO/local_repro/sc_lmcache_knobs.sh"
fi
export PYTHONFAULTHANDLER=1

export MAX_QUESTIONS=2
export SUBMISSION_BATCH_SIZE=2
export CUDA_LAUNCH_BLOCKING=1

# H100 default is 0.65. L40 needs more of its 48 GB to fit 70B FP8.
export VLLM_GPU_MEMORY_UTILIZATION=0.90
export VLLM_TENSOR_PARALLEL_SIZE=2
export VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs

echo "=== L40 NODE CHECK ==="
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "VLLM_DISABLED_KERNELS=$VLLM_DISABLED_KERNELS"

nvidia-smi

/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/run_driver.sh \
    wo_gnn 2>&1


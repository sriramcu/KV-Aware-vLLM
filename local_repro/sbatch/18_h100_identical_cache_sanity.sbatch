#!/bin/bash
#SBATCH --job-name=identical_cache_sanity
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -euxo pipefail

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM

source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware/bin/activate

export PYTHONPATH="$REPO:$REPO/third_party/LMCache:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs

cd "$REPO"

echo
echo "================================================================"
echo "TEST 1: IDENTICAL REQUESTS -> GPU PREFIX CACHE"
echo "================================================================"

python local_repro/sanity_identical_gpu.py \
    --num-requests 250

echo
echo "================================================================"
echo "TEST 2: IDENTICAL REQUESTS -> CPU-ONLY LMCACHE"
echo "================================================================"

python local_repro/sanity_identical_cpu.py \
    --num-requests 250

echo
echo "================================================================"
echo "BOTH SANITY TESTS COMPLETED"
echo "================================================================"
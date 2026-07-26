#!/bin/bash
#SBATCH --job-name=actual_lmcache_replay
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/actual_lmcache_replay_probe/slurm_%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/actual_lmcache_replay_probe/slurm_%x_%j.err

set -euo pipefail

TOOLS=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/actual_lmcache_replay_probe
OUTPUT_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/actual_lmcache_replay_probe
SCRATCH_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/actual_lmcache_replay_scratch
MANIFEST=${OUTPUT_ROOT}/manifest.json

mkdir -p "$OUTPUT_ROOT" "$SCRATCH_ROOT"

echo "=== ACTUAL LMCACHE FILE REPLAY ==="
date
hostname
id
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
free -h
df -hT "$OUTPUT_ROOT" "$SCRATCH_ROOT"
mount | grep -E ' /mnt/shared ' || true
nfsstat -m || true
nfsstat -c || true
ip -s link || true
nvidia-smi || true

python3 "$TOOLS/actual_lmcache_replay_probe.py" run \
  --manifest "$MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --scratch-root "$SCRATCH_ROOT" \
  --pairs-per-phase 400 \
  --partition round_robin \
  --phase-timeout-s 240 \
  --writers 6 \
  --writer-headstart-s 10 \
  --writer-max-scratch-gib 160 \
  --sample-interval-s 1 \
  --stability-check-s 5

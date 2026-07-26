#!/bin/bash
#SBATCH --job-name=nfs_lmcache_topology
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/nfs_lmcache_topology_probe/slurm_%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/nfs_lmcache_topology_probe/slurm_%x_%j.err

set -euo pipefail

ROOT=/mnt/shared/gpfs/home/sriramc2/runs/nfs_lmcache_topology_probe
TOOLS=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/nfs_lmcache_topology_probe

mkdir -p "$ROOT"

echo "=== NFS / LMCACHE TOPOLOGY PROBE ==="
date
hostname
id
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
free -h
df -hT "$ROOT"
mount | grep -E ' /mnt/shared ' || true
nfsstat -m || true
nfsstat -c || true
ip -s link || true
nvidia-smi || true

python3 "$TOOLS/nfs_lmcache_topology_probe.py" run \
  --root "$ROOT" \
  --file-mib 80 \
  --readers 2 \
  --stress-duration-s 120 \
  --baseline-duration-s 30 \
  --writer-headstart-s 10 \
  --direct-chunk-mib 8 \
  --sample-interval-s 1

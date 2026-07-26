#!/bin/bash
#SBATCH --job-name=gpfs_mix_probe_h100
#SBATCH --partition=gpu-h100-dchen
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/gpfs_io_probe/slurm_%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/gpfs_io_probe/slurm_%x_%j.err
set -euo pipefail
ROOT=/mnt/shared/gpfs/home/sriramc2/runs/gpfs_io_probe
TOOLS=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/gpfs_probe
mkdir -p "$ROOT"
echo "=== GPFS MIXED I/O PROBE ==="
date; hostname; id
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
free -h
grep -E '^(MemTotal|MemFree|MemAvailable|Cached|Dirty|Writeback):' /proc/meminfo
ulimit -a
df -h "$ROOT"
mount | grep -E 'gpfs|/mnt/shared' || true
nvidia-smi || true
python3 "$TOOLS/gpfs_mixed_io_probe.py" \
  --root "$ROOT" \
  --phases 0,1,2,3,0 \
  --reader-files-per-phase 25 \
  --reader-chunk-mib 8 \
  --writer-file-mib 80 \
  --writer-max-files 128 \
  --writer-headstart-s 3 \
  --sample-interval-s 1
LATEST_RUN="$(find "$ROOT/runs" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
echo "=== COMPACT RESULT ==="
python3 "$TOOLS/summarize_gpfs_probe.py" "$LATEST_RUN/summary.json"
echo "=== OUTPUT LOCATION ==="
echo "$LATEST_RUN"

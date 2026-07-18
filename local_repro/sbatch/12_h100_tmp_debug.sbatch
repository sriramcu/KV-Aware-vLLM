#!/bin/bash
#SBATCH --job-name=h100_tmp_debug
#SBATCH --partition=gpu-h100-dchen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH -o /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH -e /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err

set -euxo pipefail

echo "HOST=$(hostname)"
echo "JOB=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "=== df ==="
df -h /
df -h /tmp
df -ih /
df -ih /tmp

echo "=== /tmp top-level sizes ==="
sudo -n true 2>/dev/null && SUDO=sudo || SUDO=""
$SUDO du -xhd1 /tmp 2>/dev/null | sort -h | tail -50 || true

echo "=== /tmp largest files/dirs visible to me ==="
find /tmp -xdev -user "$USER" -maxdepth 3 -printf '%s %p\n' 2>/dev/null | sort -n | tail -50 || true

echo "=== my /tmp dirs ==="
ls -lah /tmp | grep -E "sriram|sr_|kv|lmcache|pip|tmp" || true

echo "=== cleanup my obvious tmp files ==="
rm -rf /tmp/sr_${USER}_* /tmp/sr_${SLURM_JOB_ID} /tmp/sriramc2_lmcache_vllm /tmp/lmcache_hit_hook /tmp/sriramc2_prometheus_vllm
rm -f /tmp/kv_importance_tiers_sriramc2_*.json
rm -rf /tmp/pip-* /tmp/tmp*

echo "=== after cleanup ==="
df -h /tmp
df -ih /tmp
du -xhd1 /tmp 2>/dev/null | sort -h | tail -30 || true
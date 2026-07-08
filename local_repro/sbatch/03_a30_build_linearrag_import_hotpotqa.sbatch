#!/bin/bash
#SBATCH --job-name=build_import_hotpotqa
#SBATCH --partition=gpu-a30-dchen
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euxo pipefail

mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs

source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware/bin/activate

cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/Hierarchical_KV/LinearRAG

echo "=== BASIC INFO ==="
date
hostname
whoami
pwd
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

echo "=== START BUILD IMPORT ==="
python build_import_only.py

echo "=== CHECK OUTPUTS ==="
find import/hotpotqa -maxdepth 1 -type f -exec ls -lh {} \;

echo "=== DONE ==="
date
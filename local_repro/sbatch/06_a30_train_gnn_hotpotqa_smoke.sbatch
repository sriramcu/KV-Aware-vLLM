#!/bin/bash
#SBATCH --job-name=train_gnn_hotpotqa
#SBATCH --partition=gpu-a30-dchen
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euxo pipefail

source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware/bin/activate

cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/Hierarchical_KV

mkdir -p hierarchical-kv-gnn-3tier-compression
mkdir -p outputs

python rag_query_gnn_predictor.py \
  --data-path data/all_multigraph_gnn_train.pt \
  --device cuda \
  --importance-levels 3 \
  --balanced-sampling \
  --lr-scheduler plateau \
  --early-stop-patience 2 \
  --epochs 2 \
  --batch-size 16 \
  --output-model hierarchical-kv-gnn-3tier-compression/model.pt \
  --output-metrics hierarchical-kv-gnn-3tier-compression/metrics.json

ls -lh hierarchical-kv-gnn-3tier-compression/model.pt
ls -lh hierarchical-kv-gnn-3tier-compression/metrics.json

python - <<'PY'
import torch
p = "hierarchical-kv-gnn-3tier-compression/model.pt"
ckpt = torch.load(p, map_location="cpu")
print("checkpoint keys:", ckpt.keys())
print("config:", ckpt.get("config"))
PY
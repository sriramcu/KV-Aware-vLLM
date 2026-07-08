#!/bin/bash
#SBATCH --job-name=build_gnn_data_hotpotqa
#SBATCH --partition=gpu-a30-dchen
#SBATCH --output=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.out
#SBATCH --error=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euxo pipefail

source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware/bin/activate

cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/Hierarchical_KV

mkdir -p data

python build_multigraph_gnn_data_from_linearrag.py \
  --datasets hotpotqa \
  --results-root LinearRAG/results \
  --linearrag-import-dir LinearRAG/import \
  --tokenizer LinearRAG/model/all-mpnet-base-v2 \
  --max-seq-len 64 \
  --max-retrieved 5 \
  --skip-missing \
  --output data/all_multigraph_gnn_train.pt

ls -lh data/all_multigraph_gnn_train.pt

python - <<'PY'
import torch
p = "data/all_multigraph_gnn_train.pt"
d = torch.load(p, map_location="cpu")
print("keys:", sorted(d.keys()))
print("token_ids:", d["token_ids"].shape)
print("retrieved_node_ids:", d["retrieved_node_ids"].shape)
print("graph_names:", d["graph_names"])
print("vllm_block_size:", d["vllm_block_size"])
PY
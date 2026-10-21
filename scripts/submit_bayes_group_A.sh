#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware
# Sequential dependency prevents concurrent jobs from clearing the same scratch.
prev=""
for cell in nognn_original nognn_canonical gnn_original gnn_canonical bayes_original bayes_canonical bayes_random_canonical; do
  case "$cell" in *_original) order=original;; *) order=canonical;; esac
  dep=(); [[ -z "$prev" ]] || dep=("--dependency=afterok:$prev")
  id=$(sbatch --parsable "${dep[@]}" --export="ALL,SWEEP_CELL=$cell,SWEEP_ORDER=$order" scripts/run_h100_bayes_sweep_cell.sbatch)
  echo "$cell -> $id"; prev="$id"
done

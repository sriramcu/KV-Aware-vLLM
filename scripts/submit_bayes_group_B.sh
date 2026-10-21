#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware
# Four hardware-policy budgets x (no-GNN, Bayesian).
# GPU/disk budgets are mapper parameters, NOT enforced physical capacity limits.
# No-GNN sees CPU capacity and store cap but ignores the two mapper budgets.
prev=""
while read -r label gpu cpu disk cap; do
  for policy in nognn bayes; do
    cell="b_${label}_${policy}"
    dep=(); [[ -z "$prev" ]] || dep=("--dependency=afterok:$prev")
    id=$(sbatch --parsable "${dep[@]}" --export="ALL,SWEEP_CELL=$cell,SWEEP_POLICY=$policy,SWEEP_ORDER=canonical,SWEEP_GPU_GB=$gpu,SWEEP_CPU_GB=$cpu,SWEEP_DISK_GB=$disk,SWEEP_STORE_CAP_GB=$cap" scripts/run_h100_bayes_sweep_cell.sbatch)
    echo "$cell -> $id"; prev="$id"
  done
done <<'ROWS'
ref 10 225 100 0.25
cpu100 10 100 100 0.25
disk300 10 225 300 0.25
uncapped 5 100 100 0
ROWS

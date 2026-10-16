#!/usr/bin/env bash
# Run from the repository root, or any directory; four Slurm jobs serialized.
# afterany serializes the four runs even if an earlier startup fails.
set -Eeuo pipefail
REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
JOB_SCRIPT="$REPO/scripts/run_l40_gnn_mns_replication_q650_nfs_clean.sbatch"
[[ -f "$JOB_SCRIPT" ]] || { echo "Missing $JOB_SCRIPT" >&2; exit 2; }
mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware

j1=$(sbatch --parsable --job-name=kvaware-l40-mns-grid --export=ALL,MAX_NUM_SEQS=10,GNN_CHUNK_GPU_MIN=8,EXPERIMENT_LABEL=mns10_t8 "$JOB_SCRIPT")
echo "MNS10 threshold8: $j1"
j2=$(sbatch --parsable --dependency="afterany:$j1" --job-name=kvaware-l40-mns-grid --export=ALL,MAX_NUM_SEQS=7,GNN_CHUNK_GPU_MIN=8,EXPERIMENT_LABEL=mns7_t8 "$JOB_SCRIPT")
echo "MNS7 threshold8: $j2"
j3=$(sbatch --parsable --dependency="afterany:$j2" --job-name=kvaware-l40-mns-grid --export=ALL,MAX_NUM_SEQS=4,GNN_CHUNK_GPU_MIN=8,EXPERIMENT_LABEL=mns4_t8 "$JOB_SCRIPT")
echo "MNS4 threshold8: $j3"
j4=$(sbatch --parsable --dependency="afterany:$j3" --job-name=kvaware-l40-mns-grid --export=ALL,MAX_NUM_SEQS=10,GNN_CHUNK_GPU_MIN=22,EXPERIMENT_LABEL=mns10_t22 "$JOB_SCRIPT")
echo "MNS10 threshold22: $j4"
printf '\nSubmitted %s -> %s -> %s -> %s (afterany chain)\n' "$j1" "$j2" "$j3" "$j4"
printf 'Monitor with: squeue -j %s,%s,%s,%s\n' "$j1" "$j2" "$j3" "$j4"

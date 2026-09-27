#!/usr/bin/env bash
set -Eeuo pipefail

cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM

SBATCH_SCRIPT="scripts/run_h100_q650_storecap.sbatch"

# Fixed across all 8 experiments.
COMMON="LOOKUP_PF=16,PF=2,MQ_TIMEOUT=120,HEARTBEAT_INTERVAL=40"

prev=""

submit_job() {
    local vars="$1"
    local name="$2"
    local jid

    if [[ -z "$prev" ]]; then
        jid=$(sbatch --parsable \
            --job-name="$name" \
            --export="ALL,${COMMON},${vars}" \
            "$SBATCH_SCRIPT")
    else
        jid=$(sbatch --parsable \
            --dependency="afterany:${prev}" \
            --job-name="$name" \
            --export="ALL,${COMMON},${vars}" \
            "$SBATCH_SCRIPT")
    fi

    # Some Slurm installs return JOBID;CLUSTER with --parsable.
    jid="${jid%%;*}"

    echo "$name -> $jid"
    prev="$jid"
}

# 1. GNN: C=300, U=60, B=650, reverse
submit_job \
  "ARM=gnn,L1_GB=300,STORE_CAP_GB=60,SUBMISSION_BATCH_SIZE=650,WARM_ORDER=reverse,EXPERIMENT_LABEL=gnn_c300_u60_b650_reverse" \
  "gnn-c300-u60-r"

# 2. no-GNN: C=210, U=60, B=650, reverse
submit_job \
  "ARM=nognn,L1_GB=210,STORE_CAP_GB=60,SUBMISSION_BATCH_SIZE=650,WARM_ORDER=reverse,GRID_LABEL=nognn_c210_u60_b650_reverse" \
  "nognn-c210-u60-r"

# 3. GNN: C=300, U=10, B=650, reverse
submit_job \
  "ARM=gnn,L1_GB=300,STORE_CAP_GB=10,SUBMISSION_BATCH_SIZE=650,WARM_ORDER=reverse,EXPERIMENT_LABEL=gnn_c300_u10_b650_reverse" \
  "gnn-c300-u10-r"

# 4. no-GNN: C=300, U=10, B=650, reverse
submit_job \
  "ARM=nognn,L1_GB=300,STORE_CAP_GB=10,SUBMISSION_BATCH_SIZE=650,WARM_ORDER=reverse,GRID_LABEL=nognn_c300_u10_b650_reverse" \
  "nognn-c300-u10-r"

# 5. GNN: C=300, U=10, B=50, reverse
submit_job \
  "ARM=gnn,L1_GB=300,STORE_CAP_GB=10,SUBMISSION_BATCH_SIZE=50,WARM_ORDER=reverse,EXPERIMENT_LABEL=gnn_c300_u10_b50_reverse" \
  "gnn-c300-u10-b50-r"

# 6. no-GNN: C=300, U=10, B=50, reverse
submit_job \
  "ARM=nognn,L1_GB=300,STORE_CAP_GB=10,SUBMISSION_BATCH_SIZE=50,WARM_ORDER=reverse,GRID_LABEL=nognn_c300_u10_b50_reverse" \
  "nognn-c300-u10-b50-r"

# 7. GNN: C=300, U=10, B=50, same
submit_job \
  "ARM=gnn,L1_GB=300,STORE_CAP_GB=10,SUBMISSION_BATCH_SIZE=50,WARM_ORDER=same,EXPERIMENT_LABEL=gnn_c300_u10_b50_same" \
  "gnn-c300-u10-b50-s"

# 8. no-GNN: C=300, U=10, B=50, same
submit_job \
  "ARM=nognn,L1_GB=300,STORE_CAP_GB=10,SUBMISSION_BATCH_SIZE=50,WARM_ORDER=same,GRID_LABEL=nognn_c300_u10_b50_same" \
  "nognn-c300-u10-b50-s"

echo
echo "All 8 submitted in a serialized afterany chain."
echo "First job: runs immediately when resources are available."
echo "Each subsequent job starts after the preceding job terminates."
squeue -u "$USER"

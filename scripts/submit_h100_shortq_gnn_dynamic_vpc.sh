#!/usr/bin/env bash
set -Eeuo pipefail

# Submit the functional smoke first; Q650 starts only if the smoke exits 0.
# The two feature gates are inherited by both jobs and can be independently set:
#   KV_GNN_AWARE_VPC=0/1
#   KV_GNN_L1_BACKING=0/1
SMOKE_JOB=$(sbatch --parsable scripts/run_h100_shortq_gnn_dynamic_vpc_smoke.sbatch)
Q650_JOB=$(sbatch --parsable --dependency="afterok:${SMOKE_JOB}" scripts/run_h100_shortq_gnn_dynamic_vpc_q650.sbatch)
echo "smoke_job=${SMOKE_JOB}"
echo "q650_job=${Q650_JOB} dependency=afterok:${SMOKE_JOB}"

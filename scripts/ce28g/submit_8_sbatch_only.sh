#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
for case_id in CAL_ON_BASE CAL_OFF_BASE CAL_ON_WM5 CAL_OFF_WM5 \
               P04_HI_ON P05_LO_ON P06_HI_OFF P07_LO_OFF; do
  sbatch --dependency=singleton --job-name=kvq650-10batch \
    --export=ALL,CASE_ID="$case_id" scripts/ce28g/run_ce28g_cell.sbatch
done

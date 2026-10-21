#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware
# Choose winner AFTER groups A and B. Environment overrides determine winner
# capacity/order/cap settings. Four fixed-prior weights incl reference 1.
: "${GROUP_C_GPU_GB:=10}" "${GROUP_C_CPU_GB:=225}"
: "${GROUP_C_DISK_GB:=100}" "${GROUP_C_STORE_CAP_GB:=0.25}"
: "${GROUP_C_ORDER:=canonical}"
prev=""
for entry in 'w0:0' 'w025:0.25' 'w1:1' 'w4:4'; do
  label=${entry%%:*}; weight=${entry#*:}
  dep=(); [[ -z "$prev" ]] || dep=("--dependency=afterok:$prev")
  id=$(sbatch --parsable "${dep[@]}" --export="ALL,SWEEP_CELL=c_$label,SWEEP_POLICY=bayes,SWEEP_ORDER=$GROUP_C_ORDER,SWEEP_GPU_GB=$GROUP_C_GPU_GB,SWEEP_CPU_GB=$GROUP_C_CPU_GB,SWEEP_DISK_GB=$GROUP_C_DISK_GB,SWEEP_STORE_CAP_GB=$GROUP_C_STORE_CAP_GB,SWEEP_PRIOR_WEIGHT=$weight" scripts/run_h100_bayes_sweep_cell.sbatch)
  echo "c_$label -> $id"; prev="$id"
done

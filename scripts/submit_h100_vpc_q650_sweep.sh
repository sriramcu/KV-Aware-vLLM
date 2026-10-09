#!/usr/bin/env bash
# Submit the four GNN-placement / VPC-policy Q650 arms in ONE Slurm dependency chain.
# Usage: bash scripts/submit_h100_vpc_q650_sweep.sh
# Default order leaves strict tierwise last because MNS12 on L40 livelocked.
set -Eeuo pipefail
REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
SCRIPT="$REPO/scripts/run_h100_vpc_q650_sweep.sbatch"
[[ -f "$SCRIPT" ]] || { echo "Missing $SCRIPT" >&2; exit 2; }
mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware
TAG="${VPC_SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"
[[ "$TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Bad VPC_SWEEP_TAG=$TAG" >&2; exit 2; }
PREV="${VPC_SWEEP_AFTER_JOB:-}"
# afterany ensures a failed tier-independent arm does not conceal subsequent results.
DEP_TYPE="${VPC_SWEEP_DEPENDENCY:-afterany}"
[[ "$DEP_TYPE" == "afterany" || "$DEP_TYPE" == "afterok" ]] || exit 2
for mode in vanilla window selective tierwise; do
    label="vpc4_gnn_drop_u025_${mode}_${TAG}"
    opts=(--parsable --export="ALL,VLLM_VPC_POLICY=${mode},EXPERIMENT_LABEL=${label}")
    if [[ -n "$PREV" ]]; then
        opts+=(--dependency="${DEP_TYPE}:${PREV}")
    fi
    jid=$(sbatch "${opts[@]}" "$SCRIPT")
    jid="${jid%%;*}"
    [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Bad sbatch response: $jid" >&2; exit 3; }
    echo "SUBMITTED mode=${mode} job=${jid} label=${label}${PREV:+ ${DEP_TYPE}=${PREV}}"
    PREV="$jid"
done
echo "Last job: $PREV"
echo "Jobs execute sequentially; close your shell safely."

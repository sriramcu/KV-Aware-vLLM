#!/usr/bin/env bash
# Submit the four L40 VPC modes serially, to avoid port/GPU/L2 contention.
# Usage: bash scripts/submit_l40_vpc_smokes.sh quick|pressure|disk [mode ...]
set -Eeuo pipefail

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
SCRIPT="${REPO}/scripts/run_l40_vpc_policy_smoke.sbatch"
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware
[[ -f "$SCRIPT" ]] || { echo "Missing $SCRIPT; copy the downloaded sbatch script first" >&2; exit 2; }
mkdir -p "$RUN_ROOT"

phase="${1:-quick}"
if (($# > 0)); then shift; fi
if (($# > 0)); then
  modes=("$@")
else
  modes=(vanilla window tierwise selective)
fi

case "$phase" in
  quick)
    q=24; b=8; mns=4; util=0.60; tokens=32; u=0.05; require_evictions=0
    ;;
  pressure)
    # More churn, cached evictions expected for vanilla/window/tierwise.
    q=96; b=24; mns=12; util=0.52; tokens=64; u=0.05; require_evictions=1
    ;;
  disk)
    # Small disk-enabled integration smoke, not a disk-throughput benchmark.
    q=32; b=16; mns=8; util=0.55; tokens=64; u=0.25; require_evictions=0
    ;;
  *)
    echo "Usage: $0 quick|pressure|disk [vanilla|window|tierwise|selective ...]" >&2
    exit 2
    ;;
esac

stamp="${VPC_SMOKE_TAG:-$(date +%Y%m%d_%H%M%S)}"
prev="${VPC_SMOKE_AFTER_JOB:-}"
for mode in "${modes[@]}"; do
  case "$mode" in vanilla|window|tierwise|selective) ;; *) echo "Unknown VPC mode: $mode" >&2; exit 2;; esac
  label="${phase}_${mode}_${stamp}"
  args=(--parsable --export="ALL,VLLM_VPC_POLICY=${mode},EXPERIMENT_LABEL=${label},NUM_QUESTIONS=${q},SUBMISSION_BATCH_SIZE=${b},MAX_NUM_SEQS=${mns},GPU_MEMORY_UTILIZATION=${util},MIN_TOKENS=${tokens},MAX_TOKENS=${tokens},LMCACHE_MP_MAX_INFLIGHT_STORE_GB=${u},REQUIRE_VPC_EVICTIONS=${require_evictions},GNN_SMOKE_MIN_UNIQUE_PER_PLACEMENT=1")
  if [[ -n "$prev" ]]; then
    args+=("--dependency=afterany:${prev}")
  fi
  jid=$(sbatch "${args[@]}" "$SCRIPT")
  jid="${jid%%;*}"
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch reply: $jid" >&2; exit 3; }
  echo "SUBMITTED mode=${mode} phase=${phase} job=${jid} label=${label}${prev:+ afterany=$prev}"
  prev="$jid"
done
printf '\nResults will appear under: %s/l40_vpc_smoke_%s_<jobid>\n' "$RUN_ROOT" "${phase}_<mode>_${stamp}"

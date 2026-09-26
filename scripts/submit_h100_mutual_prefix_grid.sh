#!/usr/bin/env bash
set -Eeuo pipefail

# Six-run mutual-prefix grid. Both arms use the same scheduler/storage semantics:
#   - mutual prefix ON
#   - VPC-sufficient bypass ON
#   - legacy Stage-1 starvation fallback ON
#   - occupancy fallback OFF
#   - prefix diagnostics ON
# Only the learned GNN placement/VPC-retention policy differs between arms.
#
# Runs are chained with afterany because both experiment scripts perform an
# aggressive /scratch2/sriramc2 cleanup. Serializing the jobs also makes the
# filesystem-pressure comparisons easier to interpret.

GNN_SCRIPT="scripts/run_h100_shortq_gnn_dynamic_vpc_q650.sbatch"
NOGNN_SCRIPT="scripts/run_h100_l0off_q650_l1_prefetch_grid_cleanall.sbatch"
DEPENDENCY_TYPE="${DEPENDENCY_TYPE:-afterany}"
DRY_RUN="${DRY_RUN:-0}"

common_exports=(
  KV_FS_PER_OP_WORKERS=1
  KV_FS_LOOKUP_WORKERS=1
  KV_FS_STORE_WORKERS=0
  KV_FS_DELETE_WORKERS=0
  KV_VPC_SUFFICIENT_BYPASS=1
  KV_MUTUAL_PREFIX=1
  KV_STAGE1_STARVATION_FALLBACK=1
  KV_STAGE1_OCCUPANCY_FALLBACK=0
  LMCACHE_MP_PREFIX_DIAGNOSTICS=1
  LMCACHE_MP_CONGESTION_DEBUG=1
  LMCACHE_MP_CHTHM_DEBUG=1
  LMCACHE_MP_STAGE1_FRESHNESS_GUARD_S=270
)

# label PF shared retrieve
configs=(
  "pf2_s5_r2 2 5 2"
  "pf8_s5_r2 8 5 2"
  "pf8_s10_r5 8 10 5"
)

join_exports() {
  local IFS=,
  echo "$*"
}

submit_one() {
  local arm="$1" label="$2" pf="$3" shared="$4" retrieve="$5" dep="${6:-}"
  local script run_label
  local -a exports=("${common_exports[@]}")
  exports+=(
    "LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT=${pf}"
    "KV_FS_SHARED_WORKERS=${shared}"
    "KV_FS_RETRIEVE_WORKERS=${retrieve}"
  )

  if [[ "$arm" == "gnn" ]]; then
    script="$GNN_SCRIPT"
    run_label="gnn_mutual_${label}"
    exports+=(
      "EXPERIMENT_LABEL=${run_label}"
      "KV_GNN_AWARE_VPC=1"
      "KV_GNN_L1_BACKING=1"
    )
  else
    script="$NOGNN_SCRIPT"
    run_label="nognn_mutual_${label}"
    exports+=(
      "GRID_LABEL=${run_label}"
      "LMCACHE_L1_SIZE_GB=200"
      "GPU_MEMORY_UTILIZATION=0.65"
      "MAX_NUM_SEQS=16"
      "ENABLE_VPC=1"
    )
  fi

  local export_arg
  export_arg="ALL,$(join_exports "${exports[@]}")"
  local -a cmd=(sbatch --parsable --export="$export_arg")
  if [[ -n "$dep" ]]; then
    cmd+=(--dependency="${DEPENDENCY_TYPE}:${dep}")
  fi
  cmd+=("$script")

  { printf '%q ' "${cmd[@]}"; echo; } >&2
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "${cmd[@]}"
}

prev_job=""
for row in "${configs[@]}"; do
  read -r label pf shared retrieve <<<"$row"
  for arm in gnn nognn; do
    if [[ "$DRY_RUN" == "1" ]]; then
      submit_one "$arm" "$label" "$pf" "$shared" "$retrieve" "$prev_job"
      continue
    fi
    raw_job_id="$(submit_one "$arm" "$label" "$pf" "$shared" "$retrieve" "$prev_job" | tail -n1)"
    job_id="${raw_job_id%%;*}"
    echo "submitted arm=${arm} config=${label} job=${job_id}${prev_job:+ dependency=${DEPENDENCY_TYPE}:${prev_job}}" >&2
    prev_job="$job_id"
  done
done

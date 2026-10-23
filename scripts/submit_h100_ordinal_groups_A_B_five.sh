#!/usr/bin/env bash
# Five selected H100 Bayesian expected-ordinal reruns from Groups A/B.
# Serial afterok chain; no no-GNN, random control, or duplicate Group B reference.
# Group B uses ORIGINAL passage ordering, as requested.
set -Eeuo pipefail

REPO=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware
PRIOR="$RUN_ROOT/h100_expected_ordinal_exploratory.json"
LAUNCHER=scripts/run_h100_bayes_sweep_cell.sbatch

cd "$REPO"
[[ -s "$PRIOR" ]] || { echo "ERROR: missing prior: $PRIOR" >&2; exit 1; }
[[ -f "$LAUNCHER" ]] || { echo "ERROR: missing launcher: $LAUNCHER" >&2; exit 1; }
bash -n "$LAUNCHER"
grep -q 'LMCACHE_L2_SIZE_GB' "$LAUNCHER" || {
  echo 'ERROR: H100 launcher does not appear to enforce physical L2 capacity.' >&2
  exit 1
}
grep -q 'KV_BAYES_SCORE_FORMAT' scripts/run_shortq_gnn_dynamic_vpc_common.sh || {
  echo 'ERROR: expected-ordinal score format not wired into shared launcher.' >&2
  exit 1
}

python - "$PRIOR" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p['score_format'] == 'expected_ordinal_0_100', p.get('score_format')
assert p['cost_calibration']['recompute_prior_measured'] is True
assert p['cost_calibration']['recompute_samples'] >= 3
assert abs(p['cost_prior_ms']['recompute'] - 80.83623) < 0.001
assert len(p['score_bin_edges']) == len(p['score_bin_probabilities']) + 1
print('Verified ordinal-score prior and measured H100 recomputation cost.')
PY

# Each sweep cell uses the Slurm job name "kvbayes-cell".  Avoid accidental
# overlap if a previous seven-job chain was already submitted.
if command -v squeue >/dev/null 2>&1; then
  existing="$(squeue -h -u "$USER" -n kvbayes-cell -o '%i %t %j' 2>/dev/null || true)"
  if [[ -n "$existing" ]]; then
    echo 'ERROR: Existing H100 sweep jobs are queued/running:' >&2
    echo "$existing" >&2
    echo 'Cancel or finish the previous chain before submitting this one.' >&2
    exit 1
  fi
fi

# WARNING: CLEAN_OLD_LMCACHE=1 in the existing shared launcher is a BROAD
# cleanup: it removes ALL immediate children of /scratch2/sriramc2, not just
# LMCache L2 directories.  Run only if that entire scratch tree is disposable
# and no other job/user process relies on it.  It runs at the start of EACH job.
echo 'WARNING: CLEAN_OLD_LMCACHE=1 wipes EVERYTHING under /scratch2/sriramc2'
echo '         at the start of each job; it does not only delete old L2 data.'
if [[ -d /scratch2/sriramc2 ]]; then
  echo 'Current scratch entries (top level):'
  find /scratch2/sriramc2 -mindepth 1 -maxdepth 1 -printf '%f\n' | sort | head -80
  df -h /scratch2/sriramc2 || true
fi

mkdir -p "$RUN_ROOT"
RECEIPT="$RUN_ROOT/h100_ordinal_AB_five_submitted_$(date +%Y%m%d_%H%M%S).tsv"
printf 'group\thistorical_job\tcell\tordering\tgpu_gb\tcpu_gb\tdisk_gb\tstore_cap_gb\tnew_job\n' > "$RECEIPT"

prev=''
submit_cell() {
  local group="$1" historical="$2" cell="$3" ordering="$4"
  local gpu="$5" cpu="$6" disk="$7" cap="$8"
  local -a dep=()
  if [[ -n "$prev" ]]; then
    dep=("--dependency=afterok:$prev")
  fi
  local exports
  exports="ALL,SWEEP_CELL=$cell,SWEEP_POLICY=bayes,SWEEP_ORDER=$ordering,SWEEP_GPU_GB=$gpu,SWEEP_CPU_GB=$cpu,SWEEP_DISK_GB=$disk,SWEEP_STORE_CAP_GB=$cap,SWEEP_PRIOR_FILE=$PRIOR,KV_BAYES_SCORE_FORMAT=expected_ordinal_0_100,KV_BAYES_RANDOM_SCORE_SEED=,CLEAN_OLD_LMCACHE=1"
  local jid
  jid="$(sbatch --parsable --cpus-per-task=16 "${dep[@]}" --export="$exports" "$LAUNCHER")"
  jid="${jid%%;*}"
  [[ "$jid" =~ ^[0-9]+$ ]] || {
    echo "ERROR: unexpected sbatch response: $jid" >&2
    exit 1
  }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$group" "$historical" "$cell" "$ordering" "$gpu" "$cpu" "$disk" "$cap" "$jid" | tee -a "$RECEIPT"
  prev="$jid"
}

# Group A: the two non-random Bayesian GNN conditions (original + canonical).
submit_cell A 29765 bayes_original            original  10 225 100 0.25
submit_cell A 29766 bayes_canonical           canonical 10 225 100 0.25

# Group B: remove the duplicate reference and use ORIGINAL ordering throughout.
submit_cell B 29771 b_cpu100_ordinal_original original  10 100 100 0.25
submit_cell B 29773 b_disk300_ordinal_original original  10 225 300 0.25
submit_cell B 29775 b_uncapped_ordinal_original original   5 100 100 0

echo "Submission manifest: $RECEIPT"
echo 'Five jobs submitted sequentially (afterok); any failed job blocks successors.'
echo 'No --mem is specified; each H100 job requests 16 CPUs.'

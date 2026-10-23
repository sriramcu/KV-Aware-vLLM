#!/usr/bin/env python3
"""Generate SAFE copies of the installed H100 launchers (no live-file overwrite).

Source: current repository's scripts/run_h100_bayes_sweep_cell.sbatch and
scripts/run_h100_mp_nognn_q650_current_cleanall.sbatch.  Do not use an old
archived launcher: the current repository includes L2-capacity hotfixes.
"""
import argparse
import re
import shlex
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--repo', default='/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM')
p.add_argument('--prior', default='/mnt/shared/gpfs/home/sriramc2/runs/kvaware/h100_expected_ordinal_exploratory.json')
a = p.parse_args()
repo = Path(a.repo).resolve()
out = repo / 'scripts/h100_10batch'
sweep_in = repo / 'scripts/run_h100_bayes_sweep_cell.sbatch'
nognn_in = repo / 'scripts/run_h100_mp_nognn_q650_current_cleanall.sbatch'
common = repo / 'scripts/run_shortq_gnn_dynamic_vpc_common.sh'
prior = Path(a.prior)
if not prior.is_file() or prior.stat().st_size == 0:
    raise RuntimeError(f'Historical expected-ordinal prior missing: {prior}')

# Fail closed rather than silently creating scientifically unmatched runs.
for path in [sweep_in, nognn_in, common]:
    if not path.is_file():
        raise RuntimeError(f'Missing current source: {path}')
b = sweep_in.read_text()
n = nognn_in.read_text()
c = common.read_text()
for name, source in [('bayes sweep', b), ('no-GNN baseline', n)]:
    for marker in ['LMCACHE_L2_SIZE_GB', 'LMCACHE_MP_MAX_INFLIGHT_STORE_GB']:
        if marker not in source:
            raise RuntimeError(f'{name} missing {marker}; apply the physical L2 capacity patch before installing')
for marker in ['max_capacity_gb', 'CLEAN_OLD_LMCACHE', 'bayes_score_prior_count', 'bayes_cost_prior_count']:
    if marker not in c:
        raise RuntimeError(f'GNN common launcher missing {marker}; repository may not have necessary patches')
if 'max_capacity_gb' not in n:
    raise RuntimeError('no-GNN baseline does not enforce physical L2 capacity')

# Point the standard dispatcher at our safe no-GNN copy.
old = 'bash scripts/run_h100_mp_nognn_q650_current_cleanall.sbatch'
if b.count(old) != 1:
    raise RuntimeError('Cannot safely locate no-GNN delegation in Bayesian sweep')
b = b.replace(old, 'bash scripts/h100_10batch/run_h100_nognn_q650_safe.sbatch')

# The source explicitly overwrites posterior mode with fixed. Honor SWEEP_*.
pattern = r'export KV_BAYES_READMISSION=1 KV_BAYES_POSTERIOR_MODE=fixed'
if len(re.findall(pattern,b)) != 1:
    raise RuntimeError('Cannot locate one legacy forced-fixed posterior assignment')
b = re.sub(pattern, '''export KV_BAYES_READMISSION=1 KV_BAYES_POSTERIOR_MODE="${SWEEP_POSTERIOR_MODE:-fixed}"
  export KV_BAYES_SCORE_PRIOR_COUNT="${SWEEP_SCORE_PRIOR_COUNT:-10}"
  export KV_BAYES_COST_PRIOR_COUNT="${SWEEP_COST_PRIOR_COUNT:-5}"''', b)
if not re.search(r'export\s+LMCACHE_L2_SIZE_GB=', b):
    raise RuntimeError('Sweep script does not explicitly export the physical L2 size from SWEEP_DISK_GB')
if 'export KV_BAYES_DISK_GB="$LMCACHE_L2_SIZE_GB"' not in b:
    raise RuntimeError('Mapper Disk size is not tied to physical L2 capacity')

# Give no-GNN the SAME guarded, parameter-controlled full-root cleanup
# semantics as the Bayesian common launcher. Keep per-job paths for isolation
# when cleanup is deliberately disabled, and never overwrite original files.
pathline = r'^LMCACHE_L2_DIR="\$\{LMCACHE_L2_ROOT\}/medical_q650_dup_vanilla_mp"\s*$'
n, count = re.subn(pathline,
    'LMCACHE_L2_DIR="${LMCACHE_L2_ROOT}/kvaware_10batch_nognn_${SLURM_JOB_ID}"',n, flags=re.M)
if count != 1:
    raise RuntimeError('Cannot locate the expected fixed no-GNN L2 path')
func = r'^safe_clean_scratch_user_root\(\) \{.*?^\}'
safe = '''safe_clean_scratch_user_root() {
    local expected="/scratch2/sriramc2/lmcache_mp_l2/kvaware_10batch_nognn_${SLURM_JOB_ID}"
    [[ "$SCRATCH_USER_ROOT" == "/scratch2/sriramc2" && "$LMCACHE_L2_DIR" == "$expected" ]] || {
        echo "ERROR: unexpected scratch root or L2 path" >&2
        exit 12
    }
    mkdir -p "$SCRATCH_USER_ROOT"
    if [[ "${CLEAN_OLD_LMCACHE:-1}" == "1" ]]; then
        echo "FULL CLEAN of $SCRATCH_USER_ROOT (preserve root directory)"
        find "$SCRATCH_USER_ROOT" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
    elif [[ "${CLEAN_OLD_LMCACHE:-1}" == "0" ]]; then
        [[ ! -e "$LMCACHE_L2_DIR" ]] || {
            echo "ERROR: refusing to reuse old L2 dir: $LMCACHE_L2_DIR" >&2
            exit 12
        }
        echo "Cleanup disabled; using a fresh unique L2 path"
    else
        echo "ERROR: CLEAN_OLD_LMCACHE must be 0 or 1" >&2
        exit 12
    fi
    mkdir -p "$LMCACHE_L2_DIR"
}'''
n, count = re.subn(func, lambda match: safe, n, flags=re.M|re.S)
if count != 1 or n.count('safe_clean_scratch_user_root') != 2:
    raise RuntimeError('Failed to replace no-GNN scratch cleanup function')
# The only allowed recursive deletion is the exact top-level find guarded
# by the hardcoded scratch-root and job-specific path checks above.
if n.count('find "$SCRATCH_USER_ROOT" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +') != 1:
    raise RuntimeError('no-GNN scratch cleaner is not the expected guarded implementation')
for name, body in [('run_h100_bayes_sweep_cell_safe.sbatch',b), ('run_h100_nognn_q650_safe.sbatch',n)]:
    target = out/name
    target.write_text(body)
    target.chmod(0o750)
    subprocess.run(['bash','-n',str(target)],check=True)
    print('Installed:',target)
wrapper=out/'run_h100_10batch_cell.sbatch'
submit=out/'submit_10_sbatch_only.sh'
for target in [wrapper,submit]:
    if not target.is_file(): raise RuntimeError(f'Missing packaged file {target}')
    subprocess.run(['bash','-n',str(target)],check=True)
lines=submit.read_text().splitlines()
if len(lines)!=10 or any(not line.startswith('sbatch ') for line in lines):
    raise RuntimeError('Expected exactly ten sbatch commands, no other lines')
ids=[re.search(r'CASE_ID=(P\d+)',line).group(1) for line in lines]
if ids != ['P1','P2','P3','P4','P5','P6','P7','P8','P11','P12']:
    raise RuntimeError(f'Unexpected job identifiers: {ids}')
if any('--mem' in line for line in lines):
    raise RuntimeError('No --mem allowed on CodeNimbus')
print('PASS: 10 sbatch lines, syntax, L2 enforcement, guarded full-root CLEAN_OLD_LMCACHE, posterior switches.')
print('READY: from repository root run: bash scripts/h100_10batch/submit_10_sbatch_only.sh')

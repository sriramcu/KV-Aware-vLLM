#!/usr/bin/env python3
"""Create variable-dataset/Q/B launcher copies from the proven 10-job safe copies.

This does NOT modify prior working launchers. It refuses to build against
unpatched (non-physically-capped) cache launchers.
"""
import argparse
import re
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--repo', default='/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM')
a=p.parse_args()
repo=Path(a.repo).resolve()
old=repo/'scripts/h100_10batch'
out=repo/'scripts/h100_next8'
sources={
 'bayes': old/'run_h100_bayes_sweep_cell_safe.sbatch',
 'nognn': old/'run_h100_nognn_q650_safe.sbatch',
 'common':repo/'scripts/run_shortq_gnn_dynamic_vpc_common.sh',
}
for tag,f in sources.items():
 if not f.is_file():
  raise RuntimeError(f'Missing {tag}: {f}; original 10-batch installer must have run')
b=sources['bayes'].read_text()
n=sources['nognn'].read_text()
c=sources['common'].read_text()

def replace_one(text, old, new, context):
 ct=text.count(old)
 if ct!=1:
  raise RuntimeError(f'{context}: wanted exactly one instance of {old!r}, found {ct}')
 return text.replace(old,new)

# Strict version gates: no silent reversion to old unbounded L2.
for tag,text in [('bayes dispatcher',b),('no-GNN',n)]:
 for marker in ('LMCACHE_L2_SIZE_GB','LMCACHE_MP_MAX_INFLIGHT_STORE_GB'):
  if marker not in text:
   raise RuntimeError(f'{tag} lacks {marker}; physical L2 enforcement not installed')
for marker in ('max_capacity_gb','CLEAN_OLD_LMCACHE','bayes_score_prior_count'):
 if marker not in c:
  raise RuntimeError(f'GNN common missing {marker}; check installed repository patches')
if 'CLEAN_OLD_LMCACHE' not in n or 'find "$SCRATCH_USER_ROOT" -mindepth 1 -maxdepth 1' not in n:
 raise RuntimeError('Expected guarded scratch cleanup in previous safe no-GNN launcher')
if 'KV_BAYES_POSTERIOR_MODE="${SWEEP_POSTERIOR_MODE:-fixed}"' not in b:
 raise RuntimeError('Previous safe dispatcher missing configurable posterior mode')

# Fix historical forced-Q650/B100 and old q650-b100 run labels.
b=replace_one(b,'export NUM_QUESTIONS=650 SUBMISSION_BATCH_SIZE=100',
              'export NUM_QUESTIONS="${NEXT8_Q:?}" SUBMISSION_BATCH_SIZE="${NEXT8_B:?}"',
              'Bayes Q/B control')
b=replace_one(b,'export GRID_LABEL="${SWEEP_CELL}_q650_b100"',
              'export GRID_LABEL="${SWEEP_CELL}_${NEXT8_DATASET}_q${NEXT8_Q}_b${NEXT8_B}"',
              'Bayes label')
# nognn source has independent default; dataset must arrive from wrapper.
b=replace_one(b,'export KV_BAYES_ENABLE=0',
              'export DATASET_NAME="${NEXT8_DATASET:?}"\nexport KV_BAYES_ENABLE=0',
              'dataset passthrough')
b=replace_one(b,'bash scripts/h100_10batch/run_h100_nognn_q650_safe.sbatch',
              'bash scripts/h100_next8/run_h100_nognn_variable.sbatch',
              'no-GNN delegation')
# The selected ordinal prior MUST be compatible (regardless of sed patch state).
old_call='bash scripts/run_shortq_gnn_dynamic_vpc_common.sh'
new_call=('bash scripts/h100_next8/run_shortq_gnn_dynamic_vpc_variable.sh'
          if 'export KV_BAYES_SCORE_FORMAT=expected_ordinal_0_100' in b else
          'export KV_BAYES_SCORE_FORMAT=expected_ordinal_0_100\n'
          'bash scripts/h100_next8/run_shortq_gnn_dynamic_vpc_variable.sh')
b=replace_one(b,old_call,new_call,'GNN common invocation')

# Common launcher bypasses incoming DATASET_NAME and Q today; parameterize those
# two ONLY. Keep TP2, FP8, max_num_seqs16, 512 chunk, store/barrier, etc.
c=replace_one(c,'DATASET_NAME=medical','DATASET_NAME="${NEXT8_DATASET:-medical}"',
              'common dataset name')
c=replace_one(c,'NUM_QUESTIONS=650; SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-650}"',
              'NUM_QUESTIONS="${NEXT8_Q:-650}"; SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-650}"',
              'common question count')
# Keep source RUN_LABEL q650 as a historical launcher family name; SWEEP_CELL
# suffix and run_config provide actual dataset,Q,B and make paths unambiguous.

# The old safe no-GNN copy already honors DATASET_NAME/Q/B and enforces L2 cap.
# Give it a distinct job-specific L2 path, respecting the same guarded cleanup.
if n.count('kvaware_10batch_nognn_')!=2:
 raise RuntimeError(f'Expected two paired 10batch L2 safeguards, found {n.count("kvaware_10batch_nognn_")}')
n=n.replace('kvaware_10batch_nognn_', 'kvaware_next8_nognn_')

outputs={
 'run_h100_bayes_sweep_variable.sbatch':b,
 'run_h100_nognn_variable.sbatch':n,
 'run_shortq_gnn_dynamic_vpc_variable.sh':c,
}
for name,body in outputs.items():
 target=out/name
 target.write_text(body)
 target.chmod(0o750)
 subprocess.run(['bash','-n',str(target)],check=True)
 print('Created:',target)
for name in ('run_h100_next8_cell.sbatch','submit_8_sbatch_only.sh'):
 target=out/name
 subprocess.run(['bash','-n',str(target)],check=True)
lines=(out/'submit_8_sbatch_only.sh').read_text().splitlines()
expected=['B200_NG','B200_BY','CPU100_NG','CPU100_BY',
          'HOTPOT_NG','HOTPOT_BY','Q1000_NG','Q1000_BY']
if len(lines)!=8 or not all(x.startswith('sbatch ') for x in lines):
 raise RuntimeError('Submit-only script must contain EXACTLY eight sbatch command lines')
actual=[re.search(r'CASE_ID=([A-Z0-9_]+)',x).group(1) for x in lines]
if actual!=expected:
 raise RuntimeError(f'Submit order mismatch: {actual}')
assert all('--dependency=singleton' in x and '--job-name=kvq650-10batch' in x for x in lines)
assert all('--mem' not in x for x in lines)
assert 'export KV_BAYES_SCORE_FORMAT=expected_ordinal_0_100' in b
assert 'CLEAN_OLD_LMCACHE=1' in (out/'run_h100_next8_cell.sbatch').read_text()
print('PASS: syntax, 8 case IDs, same singleton family, CPU16, ordinal prior, variable Q/B/dataset, physical L2 guard, full scratch cleanup.')
print('READY: bash scripts/h100_next8/submit_8_sbatch_only.sh')

#!/usr/bin/env python3
"""Install next9 prior assets and a narrow cost-observation admission switch (v3).

Runs on a CodeNimbus login node without Slurm allocation or GPU. Reads prior
from runs/kvaware and modifies ONLY the pure-Python Bayesian mapper. Existing
next8 launchers are reused unchanged; there is no scratch cleanup here.
"""
import argparse
import ast
import re
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--repo', default='.', help='KV-Aware-vLLM repository root')
p.add_argument('--calibrated-prior', default='/mnt/shared/gpfs/home/sriramc2/runs/kvaware/h100_expected_ordinal_exploratory.json')
p.add_argument('--check-only', action='store_true', help='Validate installed assets without modifying')
a = p.parse_args()
repo = Path(a.repo).resolve()
pack = repo / 'scripts/h100_next9'
mapper_path = repo / 'Hierarchical_KV/shortq_placement/bayesian_mapper.py'
cal_path = Path(a.calibrated_prior)
uniform = pack / 'priors/uniform_scores_equal_cost_10ms.json'
hybrid = pack / 'priors/calibrated_scores_equal_cost_10ms.json'
next8 = repo / 'scripts/h100_next8'
required = [mapper_path, uniform, cal_path,
            next8 / 'run_h100_bayes_sweep_variable.sbatch',
            next8 / 'run_h100_nognn_variable.sbatch',
            next8 / 'run_shortq_gnn_dynamic_vpc_variable.sh',
            next8 / 'preflight_dataset.py']
for f in required:
    if not f.is_file() or not f.stat().st_size:
        raise SystemExit(f'ERROR: missing required file {f}. Next8 install must already have completed.')

def validate_prior(path, *, is_uniform=False):
    j = json.loads(Path(path).read_text())
    assert j.get('score_format') == 'expected_ordinal_0_100', f'Wrong score format in {path}'
    e = j['score_bin_edges']; q = j['score_bin_probabilities']; costs = j['cost_prior_ms']
    assert len(e) == len(q)+1 and len(q)>=2 and all(b>a for a,b in zip(e,e[1:])), path
    assert all(x>=0 for x in q) and abs(sum(q)-1)<1e-6, path
    assert all(isinstance(costs[k],(float,int)) and costs[k]>0 for k in ('cpu','disk','recompute')), path
    if is_uniform:
        assert len(q)==256 and e[0]==0 and e[-1]==100, path
        assert all(abs(x-1/256)<1e-10 for x in q), path
        assert all(abs(x-(100*i/256))<1e-10 for i,x in enumerate(e)), path
        assert costs == {'cpu':10.0,'disk':10.0,'recompute':10.0}, path
    return j

uniform_data = validate_prior(uniform,is_uniform=True)
calibrated = validate_prior(cal_path)
want_hybrid = dict(calibrated)
want_hybrid['cost_prior_ms'] = dict(uniform_data['cost_prior_ms'])
want_hybrid['next9_prior_origin'] = {
    'score_histogram': str(cal_path),
    'costs': 'equal nominal 10 ms copied from next9 uniform prior',
    'purpose': 'score calibration ablation P01 vs P03; cost prior identical',
}
if a.check_only:
    if not hybrid.is_file(): raise SystemExit('Missing hybrid prior; first run install.py without --check-only')
    actual_hybrid=validate_prior(hybrid)
    assert actual_hybrid==want_hybrid, 'Hybrid prior differs from expected calibrated-scores+equal-cost construction'
else:
    if hybrid.is_file():
        if json.loads(hybrid.read_text()) != want_hybrid:
            raise SystemExit(f'ERROR: {hybrid} already exists but differs from expected; will not overwrite')
    else:
        hybrid.write_text(json.dumps(want_hybrid,indent=2,sort_keys=True)+'\n')
    print('Hybrid prior:',hybrid)

text = mapper_path.read_text()
marker = 'self.adapt_cost = cost_switch == "1"'
if marker not in text:
    if a.check_only: raise SystemExit('ERROR: mapper lacks next9 KV_BAYES_ADAPT_COST guard')
    old1 = '        self.prior_ms = {t: float(self.prior["cost_prior_ms"][t]) for t in ("cpu", "disk", "recompute")}\n'
    new1 = old1 + '''        cost_switch = os.environ.get("KV_BAYES_ADAPT_COST", "1")
        if cost_switch not in ("0", "1"):
            raise ValueError("KV_BAYES_ADAPT_COST must be 0 or 1")
        self.adapt_cost = cost_switch == "1"
'''
    # Prior/diagnostic fields are stable, but the cost-update code has small
    # variations across CodeNimbus checkouts. Do not match a literal three-line
    # block; find the actual writes in observe_cost() using Python's syntax tree.
    old3 = '            "cost_sample_counts": dict(self.cost_counts),\n'
    new3 = '            "cost_adaptation_enabled": self.adapt_cost,\n' + old3
    for old, new, title in [(old1,new1,'cost flag'), (old3,new3,'profile flag')]:
        if text.count(old)!=1:
            raise SystemExit(f'ERROR: expected exactly one {title} anchor (found {text.count(old)}); no mapper changes made.')
        text=text.replace(old,new)

    tree=ast.parse(text,filename=str(mapper_path))
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WaveBayesianMapper']
    assert len(classes)==1, 'ERROR: cannot uniquely locate WaveBayesianMapper'
    methods=[n for n in classes[0].body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='observe_cost']
    assert len(methods)==1, 'ERROR: cannot uniquely locate observe_cost()'
    fn=methods[0]
    def writes_cost_counter(node):
        if not isinstance(node,(ast.Assign,ast.AugAssign,ast.AnnAssign)):
            return set()
        targets = ([node.target] if isinstance(node,(ast.AugAssign,ast.AnnAssign))
                   else node.targets)
        found=set()
        for target in targets:
            for item in ast.walk(target):
                if (isinstance(item,ast.Subscript) and isinstance(item.ctx,ast.Store)
                    and isinstance(item.value,ast.Attribute)
                    and isinstance(item.value.value,ast.Name)
                    and item.value.value.id=='self'
                    and item.value.attr in {'cost_sum_ms','cost_counts'}):
                    found.add(item.value.attr)
        return found
    groups=[]
    writes=[]
    for index,stmt in enumerate(fn.body):
        group=set()
        for node in ast.walk(stmt):
            group.update(writes_cost_counter(node))
        if group:
            groups.append((index,stmt,group))
            writes.extend(group)
    if set(writes)!={'cost_sum_ms','cost_counts'} or writes.count('cost_sum_ms')!=1 or writes.count('cost_counts')!=1:
        raise SystemExit('ERROR: observe_cost() cost counter writes could not be uniquely identified; '
                         'no mapper changes made. Please provide that method to adapt the patch.')
    first_idx, first_stmt, _=groups[0]
    # Retain existing tier/measurement validation. The guard runs before the
    # first cost mutation but does not suppress validation or collection.
    if not any(any(isinstance(n,ast.Raise) for n in ast.walk(stmt)) for stmt in fn.body[:first_idx]):
        raise SystemExit('ERROR: cannot identify cost observation validation before mutation; no mapper changes made.')
    lines=text.splitlines(keepends=True)
    line=lines[first_stmt.lineno-1]
    indent=re.match(r'^[ \t]*',line).group(0)
    if len(indent)<=4:
        raise SystemExit('ERROR: cost-counter writes at unexpected indentation; no mapper changes made.')
    lines.insert(first_stmt.lineno-1,indent+'if not self.adapt_cost:\n'
                    +indent+'    return  # Instrumentation remains ON; freeze CPU/Disk posterior.\n')
    text=''.join(lines)
    compile(text,str(mapper_path),'exec')
    back=mapper_path.with_name(mapper_path.name+'.next9_before_cost_guard.bak')
    if back.exists() and back.read_text()!=mapper_path.read_text():
        raise SystemExit(f'ERROR: existing backup {back} differs; inspect before patching')
    if not back.exists():
        shutil.copy2(mapper_path,back)
    mapper_path.write_text(text)
    print('Patched mapper:',mapper_path)
    print('Unmodified backup:',back)
else:
    assert 'if not self.adapt_cost:' in text and '"cost_adaptation_enabled": self.adapt_cost' in text, 'Partial mapper patch'
    print('Mapper switch already installed:',mapper_path)

# Behavior test: same prior and source rows, both score posterior and timing
# reader work, but cost posterior is frozen only with KV_BAYES_ADAPT_COST=0.
# Import the lightweight standalone mapper directly; no torch/GPU/vLLM.
spec = importlib.util.spec_from_file_location('next9_test_mapper',mapper_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with tempfile.TemporaryDirectory(prefix='next9_test_') as d:
    tmp=Path(d)
    (tmp/'scores.jsonl').write_text(json.dumps({'source_index':0,'chunks':[{'chunk_hash':'a'*32,'score':73.0}]})+'\n')
    prior=str(uniform)
    previous_cost_switch=os.environ.get('KV_BAYES_ADAPT_COST')
    try:
        for switch in ('0','1'):
            os.environ['KV_BAYES_ADAPT_COST']=switch
            m=mod.WaveBayesianMapper(prior,0.5,25,225,500,
                tmp/'scores.jsonl',tmp/'meta.json',tmp/'vpc.json')
            if m.adapt_cost != (switch=='1'):
                raise SystemExit(f'ERROR: KV_BAYES_ADAPT_COST={switch}, but mapper.adapt_cost={m.adapt_cost!r}')
            initial_counts=dict(m.cost_counts)
            initial_sums=dict(m.cost_sum_ms)
            initial_cpu=float(m.estimated_cost('cpu'))
            initial_disk=float(m.estimated_cost('disk'))
            m.observe_scores([0])
            m.observe_cost('cpu',40,count=12)
            m.observe_cost('disk',90,count=3)
            final_counts=dict(m.cost_counts)
            final_sums=dict(m.cost_sum_ms)
            if len(m.seen_scores)!=1:
                raise SystemExit(f'ERROR: score posterior test failed: seen_scores={len(m.seen_scores)}')
            if switch=='0':
                # Do NOT assume the mapper only has two counters: some local
                # variants also record additional tiers/telemetry. Cost OFF
                # means every cost posterior counter and sum is unchanged.
                if (final_counts!=initial_counts or final_sums!=initial_sums
                    or float(m.estimated_cost('cpu'))!=initial_cpu
                    or float(m.estimated_cost('disk'))!=initial_disk):
                    raise SystemExit('ERROR: cost OFF did not freeze mapper posterior.\n'
                                     f'counts before={initial_counts!r}, after={final_counts!r}\n'
                                     f'sums before={initial_sums!r}, after={final_sums!r}\n'
                                     f'estimates before=({initial_cpu},{initial_disk}), '
                                     f'after=({m.estimated_cost("cpu")},{m.estimated_cost("disk")})')
            else:
                # Use relative deltas, robust to extra cost counters.
                expected_changes={'cpu':12, 'disk':3}
                for tier, delta in expected_changes.items():
                    actual_delta=final_counts.get(tier,0)-initial_counts.get(tier,0)
                    if actual_delta!=delta:
                        raise SystemExit(f'ERROR: cost ON tier {tier!r} incremented by '
                                         f'{actual_delta} (expected {delta}). '
                                         f'counts before={initial_counts!r}, after={final_counts!r}')
                if not (m.estimated_cost('cpu')>initial_cpu and m.estimated_cost('disk')>initial_disk):
                    raise SystemExit('ERROR: cost ON failed to update CPU and Disk estimates')
            m.publish_wave([0],'cold')
            if m.profiles[-1].get('cost_adaptation_enabled')!=(switch=='1'):
                raise SystemExit(f'ERROR: profile flag disagrees with switch={switch}')
            print(f'[NEXT9 COST TEST OK] adapt_cost={switch}, '
                  f'counts_before={initial_counts}, counts_after={final_counts}, '
                  f'cpu_ms={m.estimated_cost("cpu"):.5g}, disk_ms={m.estimated_cost("disk"):.5g}')
    finally:
        if previous_cost_switch is None:
            os.environ.pop('KV_BAYES_ADAPT_COST',None)
        else:
            os.environ['KV_BAYES_ADAPT_COST']=previous_cost_switch

# Explicitly verify that the already-patched next8 dispatcher leaves direct
# cost logging ON and still selects the expected ordinal score format.
bayes=(next8/'run_h100_bayes_sweep_variable.sbatch').read_text()
nognn=(next8/'run_h100_nognn_variable.sbatch').read_text()
common=(next8/'run_shortq_gnn_dynamic_vpc_variable.sh').read_text()
assert '"$SWEEP_CELL" == b_* || "$SWEEP_CELL" == c_*' in bayes, 'Dispatcher does not accept c_next9_* cell names'
assert 'export KV_BAYES_DIRECT_COST_FEEDBACK=1 KV_BAYES_USE_COST_PROXIES=0' in bayes, 'Cost instrumentation unexpectedly changed'
assert 'export KV_BAYES_SCORE_FORMAT=expected_ordinal_0_100' in bayes, 'Missing expected-ordinal format override'
assert 'KV_BAYES_POSTERIOR_MODE="${SWEEP_POSTERIOR_MODE:-fixed}"' in bayes, 'Posterior mode not configurable'
assert 'SWEEP_SCORE_PRIOR_COUNT' in bayes and 'SWEEP_COST_PRIOR_COUNT' in bayes, 'Pseudo-count config missing'
assert 'LMCACHE_L2_SIZE_GB' in bayes and 'LMCACHE_L2_SIZE_GB' in nognn, 'Physical L2-capacity enforcement absent'
assert 'CLEAN_OLD_LMCACHE' in common and 'CLEAN_OLD_LMCACHE' in nognn, 'Scratch cleanup guard missing'
assert 'find "$SCRATCH_USER_ROOT" -mindepth 1 -maxdepth 1' in nognn, 'NoGNN scratch root cleanup is not the expected safe version'
for file in [pack/'run_h100_next9_cell.sbatch',pack/'submit_9_sbatch_only.sh']:
    subprocess.run(['bash','-n',str(file)],check=True)
lines=(pack/'submit_9_sbatch_only.sh').read_text().splitlines()
assert len(lines)==9 and all(line.startswith('sbatch --dependency=singleton --job-name=kvq650-10batch ') for line in lines)
assert all('--mem' not in line for line in lines)
print('PASS: prior schemas, hybrid consistency, mapper cost-ON/OFF behavior, next8 dispatcher, ordinal scores, L2 physical limits, 9 serial submissions, syntax.')

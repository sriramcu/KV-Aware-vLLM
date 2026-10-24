#!/usr/bin/env python3
"""CE28G static safety preflight. Does not run GPU or touch scratch."""
import argparse
import ast
import json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--repo',required=True)
p.add_argument('--prior',required=True)
a=p.parse_args()
r=Path(a.repo)
for rel in ('Hierarchical_KV/shortq_placement/bayesian_mapper.py',
            'third_party/LMCache/lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py',
            'scripts/h100_next8/run_shortq_gnn_dynamic_vpc_variable.sh'):
    f=r/rel
    assert f.is_file(),f
    if f.suffix=='.py': ast.parse(f.read_text(),filename=str(f))
mapper=(r/'Hierarchical_KV/shortq_placement/bayesian_mapper.py').read_text()
transfer=(r/'third_party/LMCache/lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py').read_text()
launcher=(r/'scripts/h100_next8/run_shortq_gnn_dynamic_vpc_variable.sh').read_text()
assert r'\[BAYES_H2D_COST_V2\]' in mapper
assert r'\\[BAYES_H2D_COST_V2\\]' not in mapper  # no doubled escapes
assert 'BAYES_H2D_COST_V2' in mapper and 'BAYES_H2D_COST_V2' in transfer
assert '_h2d = re.compile(' in mapper
assert 'KV_BAYES_ADAPT_COST' in mapper and 'if not self.adapt_cost:' in mapper
assert '--eviction-trigger-watermark "$LMCACHE_L1_EVICTION_TRIGGER"' in launcher
assert '--eviction-ratio "$LMCACHE_L1_EVICTION_RATIO"' in launcher
assert '--bayes_cpu_gb "$BAYES_CPU_EFFECTIVE_GB"' in launcher
assert '--bayes_disk_gb "$BAYES_DISK_EFFECTIVE_GB"' in launcher
assert 'KV_BAYES_USE_WATERMARK_CAPACITY' in launcher
prior=Path(a.prior)
assert prior.is_file() and prior.stat().st_size>0, f'prior missing: {prior}'
obj=json.loads(prior.read_text())
assert obj.get('score_format')=='expected_ordinal_0_100',obj.get('score_format')
print('CE28G VERIFY PASS: patched cost V2, physical watermarks, capacity toggle, prior')

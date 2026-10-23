#!/usr/bin/env python3
"""Lightweight per-job guards, run before full-root scratch cleanup."""
import argparse
import json
from pathlib import Path
import re
p=argparse.ArgumentParser()
p.add_argument('--repo',default='.')
p.add_argument('--case',required=True)
a=p.parse_args()
r=Path(a.repo)
pack=r/'scripts/h100_next9'
m=(r/'Hierarchical_KV/shortq_placement/bayesian_mapper.py').read_text()
assert 'self.adapt_cost = cost_switch == "1"' in m and 'if not self.adapt_cost:' in m, 'Cost posterior guard not installed'
assert '"cost_adaptation_enabled": self.adapt_cost' in m, 'Missing cost-ablation profile marker'
prior_dir=pack/'priors'
for name in ('uniform_scores_equal_cost_10ms.json','calibrated_scores_equal_cost_10ms.json'):
    v=json.loads((prior_dir/name).read_text())
    assert v['score_format']=='expected_ordinal_0_100'
    assert v['cost_prior_ms']=={'cpu':10.0,'disk':10.0,'recompute':10.0}
assert a.case in ('P01_CAL_SCORE_COSTOFF','P02_UNIF_BOTH_FIXED','P03_UNIF_SCORE_FIXED',
   'P04_UNIF_BOTH_HI','P05_UNIF_BOTH_LO','P06_UNIF_SCORE_HI',
   'P07_UNIF_SCORE_LO','SCAR_NG','SCAR_BY')
sweep=(r/'scripts/h100_next8/run_h100_bayes_sweep_variable.sbatch').read_text()
assert '"$SWEEP_CELL" == b_* || "$SWEEP_CELL" == c_*' in sweep
assert 'export KV_BAYES_DIRECT_COST_FEEDBACK=1 KV_BAYES_USE_COST_PROXIES=0' in sweep
assert 'export KV_BAYES_SCORE_FORMAT=expected_ordinal_0_100' in sweep
assert 'export KV_BAYES_DISK_GB="$LMCACHE_L2_SIZE_GB"' in sweep
assert 'LMCACHE_MP_MAX_INFLIGHT_STORE_GB' in sweep
if a.case=='SCAR_NG':
    n=(r/'scripts/h100_next8/run_h100_nognn_variable.sbatch').read_text()
    assert 'LMCACHE_L2_SIZE_GB' in n and 'max_capacity_gb' in n
print('[NEXT9 VERIFY OK] '+a.case)

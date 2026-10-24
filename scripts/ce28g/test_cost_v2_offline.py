#!/usr/bin/env python3
"""CPU-only CE28G behavioral regression (no torch/cuda needed)."""
from pathlib import Path
import tempfile
import sys

repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'Hierarchical_KV/shortq_placement'))
import bayesian_mapper as bm

expected = (repo / 'Hierarchical_KV/shortq_placement/bayesian_mapper.py').resolve()
actual = Path(bm.__file__).resolve()
print(f"CE28G mapper loaded: {actual}", flush=True)
assert actual == expected, f"Wrong mapper imported: {actual}; expected {expected}"
assert 'BAYES_H2D_COST_V2' in bm.LmcacheCompletionCostReader._h2d.pattern, (
    "Stale pre-CE28G cost reader was imported. Check the source and Python import paths."
)
LmcacheCompletionCostReader = bm.LmcacheCompletionCostReader

class Recorder:
    def __init__(self): self.obs=[]
    def observe_cost(self,tier,ms,count=1): self.obs.append((tier,round(ms,5),count))

with tempfile.TemporaryDirectory() as td:
    log=Path(td)/'lmcache.log'
    log.write_text('''[BAYES_H2D_COMPLETE] request=q1 chunks=2 elapsed_ms=1000.0
[BAYES_H2D_COST_V2] request=q1 chunks=2 elapsed_ms=8.40 transfer_key=t1
[BAYES_H2D_COST_V2] request=q1 chunks=2 elapsed_ms=9.00 transfer_key=t2
[BAYES_L2_LOAD_COMPLETE] external_request=q1 chunks=2 elapsed_ms=80.00
''')
    r=LmcacheCompletionCostReader(log)
    rec=Recorder()
    c=r.collect(rec)
    assert c['h2d_completed']==2 and c['h2d_legacy_diagnostic']==1,c
    assert rec.obs==[('cpu',4.2,2),('cpu',4.5,2),('disk',44.5,2)],rec.obs
    # Incremental reading does not double count previously consumed lines.
    assert r.collect(rec)['h2d_completed']==0
    assert len(rec.obs)==3
    log.write_text('[BAYES_H2D_COMPLETE] request=q2 chunks=2 elapsed_ms=1000.0\n')
    r=LmcacheCompletionCostReader(log)
    try:
        r.collect(Recorder())
        raise AssertionError('Missing V2 cost was silently replaced by old cost')
    except RuntimeError as e:
        assert 'BAYES_H2D_COST_V2' in str(e)
print('CE28G OFFLINE COST V2 PASS: no legacy fallback; L2 uses corrected H2D')

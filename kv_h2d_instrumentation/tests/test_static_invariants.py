#!/usr/bin/env python3
"""No-GPU static checks; run from repo root after applying the patch."""
import ast
from pathlib import Path
root=Path.cwd()
rel=[
    'third_party/LMCache/lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py',
    'third_party/LMCache/lmcache/v1/mp_observability/config.py',
    'third_party/LMCache/lmcache/v1/mp_observability/subscribers/bayes_h2d_diagnostic.py',
]
for name in rel:
    f=root/name
    assert f.exists(), f
    ast.parse(f.read_text(),filename=name)
source=(root/rel[0]).read_text()
assert '"bayes_h2d_done", self._bayes_h2d_complete' in source
assert '"bayes_h2d_diag_done", self._bayes_h2d_diag_complete' in source
assert 'def _bayes_h2d_complete' in source
assert 'BAYES_H2D_COMPLETE' in source
assert 'BAYES_H2D_DIAG' in source
assert 'observe_cost(' not in (root/rel[2]).read_text()
assert 'BAYES_H2D_PHASE' in (root/rel[2]).read_text()
print('PASS: 3 files parse, existing cost feed intact, diagnostics separate')

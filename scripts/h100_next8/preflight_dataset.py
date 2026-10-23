#!/usr/bin/env python3
"""Fast, read-only data validation. Runs before any scratch deletion."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--dataset', choices=['medical','hotpotqa'], required=True)
p.add_argument('--questions', type=int, required=True)
p.add_argument('--policy', choices=['nognn','bayes'], required=True)
a=p.parse_args()
root=a.repo/'Hierarchical_KV'/'LinearRAG'
required = [root/'dataset'/a.dataset/'questions.json',
            root/'import'/a.dataset/'passage_embedding.parquet',
            root/'model'/'all-mpnet-base-v2'/'model.safetensors']
if a.policy == 'bayes':
    required += [root/'import'/a.dataset/'LinearRAG.graphml',
                 root/'import'/a.dataset/'entity_embedding.parquet',
                 a.repo/'Hierarchical_KV'/'shortq_placement'/'model.pt']
missing=[str(x) for x in required if not x.is_file() or x.stat().st_size==0]
if missing:
    raise SystemExit('DATA PREFLIGHT FAILED (not changing scratch). Missing assets:\n'+'\n'.join(missing))
with open(required[0],encoding='utf8') as f: questions=json.load(f)
if not isinstance(questions,list) or len(questions) < a.questions:
    raise SystemExit(f'DATA PREFLIGHT FAILED: {required[0]} has {len(questions) if isinstance(questions,list) else "non-list"} questions; need {a.questions}')
print(f'[NEXT8 DATA PREFLIGHT OK] dataset={a.dataset} count={len(questions)} requested={a.questions} policy={a.policy}',flush=True)

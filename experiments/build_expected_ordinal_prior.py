"""Build an exploratory 0–100 class-expectation score prior from a raw-head probe.

This does NOT train the GNN, measure hardware, or derive priors automatically
from serving-test queries. If supplied the Q650 probe, its output is only
valid for exploratory/diagnostic Q650 experiments (test-distribution leakage).
For future held-out evaluations, use a separate calibration split instead.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import statistics
import zipfile
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Hierarchical_KV.shortq_placement.expected_ordinal import (
    SCORE_FORMAT, expected_ordinal_from_logits,
)


@contextmanager
def open_probe(path: Path):
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            matching = [n for n in archive.namelist()
                        if n.endswith('/placement/gnn_raw_heads_16token.jsonl')]
            if len(matching) != 1:
                raise ValueError(f'Expected exactly one raw-head JSONL in ZIP, found {matching}')
            with archive.open(matching[0]) as stream:
                yield (json.loads(line) for line in stream if line.strip())
    else:
        with path.open(encoding='utf-8') as stream:
            yield (json.loads(line) for line in stream if line.strip())


def quantile_edges(scores: list[float], maximum_bins: int) -> list[float]:
    """Variable-width histogram bins, approximately equal empirical mass.

    Unlike coarse uniform 25-point bins these resolve the strongly clustered
    scores at the lower end of the 0–100 classification expectation range.
    """
    ordered = sorted(scores)
    n = len(ordered)
    bin_count = min(maximum_bins, n)
    edges = [0.0]
    for b in range(1, bin_count):
        offset = b * n // bin_count
        left, right = ordered[offset - 1], ordered[offset]
        candidate = (left + right) / 2
        if left < right and edges[-1] < candidate < 100.0:
            edges.append(candidate)
    edges.append(100.0)
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError('Nonincreasing histogram edges')
    return edges


def create_prior(raw_path: Path, costs_path: Path, *, bins: int) -> dict:
    if bins < 4 or bins > 1024:
        raise ValueError('bin limit must be in [4,1024]')
    original = json.loads(costs_path.read_text(encoding='utf-8'))
    if not {'cpu', 'disk', 'recompute'} <= original.get('cost_prior_ms', {}).keys():
        raise ValueError('Cost prior must contain CPU, disk and recompute values')
    for k in ('cpu', 'disk', 'recompute'):
        if not math.isfinite(float(original['cost_prior_ms'][k])) or float(original['cost_prior_ms'][k]) <= 0:
            raise ValueError(f'Invalid hardware cost prior {k}')
    first_seen: dict[str, float] = {}
    requests = 0
    occurrences = 0
    with open_probe(raw_path) as rows:
        for row in rows:
            requests += 1
            if int(row['native_block_tokens']) != 16 or int(row['lmcache_chunk_tokens']) != 512:
                raise ValueError('Unexpected 16/512 chunk geometry')
            bpc = int(row['blocks_per_chunk'])
            if bpc != 32:
                raise ValueError('Expected 32 native blocks per full chunk')
            hashes = row['chunk_hashes']
            logits = row['block_class_logits']
            if len(hashes) != int(row['prompt_tokens']) // 512:
                raise ValueError('Hashes do not match complete LMCache prompt chunks')
            computed = expected_ordinal_from_logits(logits, bpc, full_chunks=len(hashes))
            if len(computed) != len(hashes):
                raise ValueError(f'Incorrect complete chunk count in request {requests}')
            occurrences += len(hashes)
            for h, v in zip(hashes, computed, strict=True):
                if len(h) != 64 or not 0 <= v <= 100 or not math.isfinite(v):
                    raise ValueError(f'Invalid score/hash: {h} {v}')
                first_seen.setdefault(str(h).lower(), v)
    if not first_seen:
        raise ValueError('No complete chunks in calibration raw heads')
    scores = list(first_seen.values())
    edges = quantile_edges(scores, bins)
    import bisect
    counts = [0] * (len(edges) - 1)
    for score in scores:
        idx = max(0, min(len(counts) - 1, bisect.bisect_right(edges, score) - 1))
        counts[idx] += 1
    n = len(scores)
    prior = dict(original)
    prior.update({
        'score_format': SCORE_FORMAT,
        'score_bin_edges': edges,
        'score_bin_probabilities': [c / n for c in counts],
        'score_calibration': {
            'method': 'first-observed unique prefix-hash expectation of DROP=0, DISK=1, CPU=2, GPU=3',
            'score_range': [0, 100],
            'source': str(raw_path),
            'requests': requests,
            'chunk_occurrences': occurrences,
            'distinct_chunk_hashes': n,
            'bins': len(counts),
            'min': min(scores),
            'max': max(scores),
            'median': statistics.median(scores),
            'test_workload_leakage_warning': (
                'If this raw-head file is from Q650 TEST (such as job 29864), this prior '
                'is ONLY suitable for exploratory analysis, not paper held-out results.'),
        },
        'comment': ('Expected ordinal score distribution from GNN class logits. '
                    'Hardware cost priors were copied UNCHANGED from cost-prior input. '
                    'Do not assume the score histogram is hardware-dependent or test-independent.'),
    })
    return prior


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-heads', type=Path, required=True,
                    help='Probe ZIP from job 29864, or gnn_raw_heads_16token.jsonl')
    ap.add_argument('--cost-prior', type=Path, required=True,
                    help='Hardware cost prior JSON (e.g. successful L40 job 29859)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--bins', type=int, default=256)
    a = ap.parse_args()
    result = create_prior(a.raw_heads, a.cost_prior, bins=a.bins)
    if a.out.resolve() == a.cost_prior.resolve():
        ap.error('Refusing to overwrite calibrated cost-prior input')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print('[EXPECTED_ORDINAL_PRIOR]',a.out,'hashes=',result['score_calibration']['distinct_chunk_hashes'],
          'bins=',result['score_calibration']['bins'], 'costs_ms=',result['cost_prior_ms'],flush=True)


if __name__ == '__main__':
    main()

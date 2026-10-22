"""CPU-only regression tests for class-head score export and compatible priors."""
from __future__ import annotations
import json
import math
from pathlib import Path
import tempfile
import unittest
from Hierarchical_KV.shortq_placement.expected_ordinal import (
    expected_ordinal_from_logits, expected_ordinal_from_torch_logits, SCORE_FORMAT,
)
from experiments.build_expected_ordinal_prior import create_prior
from Hierarchical_KV.shortq_placement.bayesian_mapper import WaveBayesianMapper


class ExpectedOrdinalTests(unittest.TestCase):
    def test_ordinal_math_and_chunk_mean_no_voting(self):
        # One chunk, four equally weighted classes repeated 32 times: score=50.
        uniform = [[0.0] * 4 for _ in range(32)]
        self.assertAlmostEqual(expected_ordinal_from_logits(uniform)[0], 50.0, places=5)
        # Incomplete 16-token tail must never influence persisted chunk score.
        self.assertEqual(expected_ordinal_from_logits(uniform + [[-100, -100, -100, 100]]), [50.0])
        # Probe may export a complete 16-token tail block that does NOT
        # represent a complete 512-token LMCache chunk.
        self.assertEqual(expected_ordinal_from_logits(uniform*2, full_chunks=1), [50.0])
        high_gpu = [[-10, -10, -10, 10] for _ in range(32)]
        low_drop = [[10, -10, -10, -10] for _ in range(32)]
        self.assertGreater(expected_ordinal_from_logits(high_gpu)[0], 99.9)
        self.assertLess(expected_ordinal_from_logits(low_drop)[0], 0.1)
        with self.assertRaises(ValueError):
            expected_ordinal_from_logits([[float('nan'), 0, 0, 0] for _ in range(32)])

    def test_matches_torch_export(self):
        import torch
        values = torch.tensor([[i/10., -i/15., i/12., (-1)**i / 3.] for i in range(64)],dtype=torch.float32)
        a = expected_ordinal_from_logits(values.tolist())
        b = expected_ordinal_from_torch_logits(values,2,32)
        self.assertEqual(len(a), len(b))
        for x,y in zip(a,b): self.assertAlmostEqual(x,y,places=4)

    def test_mapper_cutoff_is_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            prior=d/'p.json';scores=d/'s.jsonl'
            prior.write_text(json.dumps({'score_format':SCORE_FORMAT,
                'score_bin_edges':[0,25,50,75,100],
                'score_bin_probabilities':[0.25]*4,
                'cost_prior_ms':{'cpu':25,'disk':200,'recompute':100}}))
            scores.write_text(json.dumps({'source_index':0,'chunks':[
                {'chunk_hash':'f'*64, 'score':92}]})+'\n')
            m=WaveBayesianMapper(prior,1,10,100,100,scores,d/'m.json',d/'v.json')
            self.assertAlmostEqual(m.score_at_percentile(.625), 62.5)
            bands=m.bands()
            result=m.publish_wave([0], 'cold')
            self.assertEqual(result['thresholds'],{k:bands[k] for k in ('gpu','cpu','disk')})
            self.assertIn('raw_score_cutoffs',result)
            self.assertEqual(result['score_format'],SCORE_FORMAT)

    def test_prior_construction_does_not_change_hardware_costs(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            cost=d/'cost.json'; raw=d/'raw.jsonl'
            base={'score_format': 'mean_32_block_rank_scores',
                  'score_bin_edges':[-1,0,1], 'score_bin_probabilities':[1,1],
                  'cost_prior_ms':{'cpu':2.9,'disk':87,'recompute':73},
                  'cost_calibration':{'recompute_prior_measured':True}}
            cost.write_text(json.dumps(base))
            rows=[]
            for i in range(30):
                logits=[[0,0,i/6, i/10] for _ in range(32)]
                rows.append({'source_index':i, 'prompt_tokens':512, 'native_block_tokens':16,
                             'lmcache_chunk_tokens':512,'blocks_per_chunk':32,
                             'block_class_logits':logits,
                             'chunk_hashes':[f'{i:064x}']})
            raw.write_text('\n'.join(map(json.dumps, rows))+'\n')
            p=create_prior(raw,cost,bins=10)
            self.assertEqual(p['score_format'], SCORE_FORMAT)
            self.assertEqual(p['cost_prior_ms'],base['cost_prior_ms'])
            self.assertEqual(p['cost_calibration'],base['cost_calibration'])
            self.assertEqual(p['score_calibration']['distinct_chunk_hashes'],30)
            self.assertEqual(len(p['score_bin_edges']),len(p['score_bin_probabilities'])+1)
            self.assertAlmostEqual(sum(p['score_bin_probabilities']),1.0)

if __name__ == '__main__': unittest.main()

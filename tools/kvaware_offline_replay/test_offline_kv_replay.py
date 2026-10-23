#!/usr/bin/env python3
"""Independent known-answer CPU tests; no real files, GPU, or vLLM needed."""
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from offline_kv_replay import LRU, _reference_cost_data, simulate_retention


class ToyRun:
    def __init__(self, rows, order=None, prompt_tokens=None):
        self.rows = {k: [(x, 10.) for x in seq] for k, seq in rows.items()}
        self.prompt_tokens = prompt_tokens or {k: len(seq) * 512 for k, seq in rows.items()}
        order = order or list(rows)
        self.windows = [
            ('cold', order, {x: 'cpu' for seq in rows.values() for x in seq}),
            ('warm', order, {x: 'cpu' for seq in rows.values() for x in seq})]


class KnownAnswers(unittest.TestCase):
    def test_lru_eviction_and_touch(self):
        x = LRU(2)
        x.add('a', 0); x.add('b', 1)
        self.assertTrue(x.has('a'))
        x.add('c', 2)
        self.assertEqual(list(x.data), ['a', 'c'])
        self.assertEqual(dict(x.evictions), {1: 1})
        self.assertEqual(x.writes, 3)

    def test_repeated_two_chunk_prefix_cpu_100pct(self):
        run = ToyRun({0: ['a', 'b'], 1: ['a', 'b']})
        out = simulate_retention(run, run.windows, cpu_gb=0.004, disk_gb=0.004,
                                 object_mib=1, l1_watermark=1, l2_watermark=1)
        self.assertEqual(out['warm_cpu_pct_ideal'], 100.0)
        self.assertEqual(out['warm_recompute_pct_ideal'], 0)
        self.assertEqual(out['l1_writes'], 2)
        self.assertEqual(out['l1_evictions'], 0)

    def test_two_chains_thrash_two_slot_cache(self):
        run = ToyRun({0: ['a', 'b'], 1: ['c', 'd']})
        out = simulate_retention(run, run.windows, cpu_gb=2/1024, disk_gb=0,
                                 object_mib=1, l1_watermark=1, l2_watermark=1)
        self.assertEqual(out['warm_cpu_pct_ideal'], 0)
        self.assertEqual(out['warm_recompute_pct_ideal'], 100)
        self.assertEqual(out['l1_evictions'], 6)
        self.assertEqual(out['l1_writes'], 8)

    def test_stranded_later_cache_chunk_is_not_counted(self):
        run = ToyRun({0: ['a', 'b', 'c']})
        labels = {'a': 'cpu', 'b': 'drop', 'c': 'cpu'}
        run.windows = [('cold', [0], labels), ('warm', [0], labels)]
        out = simulate_retention(run, run.windows, cpu_gb=.01, disk_gb=.01,
                                 object_mib=1, l1_watermark=1, l2_watermark=1)
        self.assertAlmostEqual(out['warm_cpu_pct_ideal'], 33.333, places=3)
        self.assertAlmostEqual(out['warm_recompute_pct_ideal'], 66.667, places=3)

    def test_gpu_cpu_backing_and_imperfect_last_chunk(self):
        run = ToyRun({0: ['a', 'b']}, prompt_tokens={0: 1100})
        labels = {'a': 'gpu', 'b': 'cpu'}
        run.windows = [('cold', [0], labels), ('warm', [0], labels)]
        out = simulate_retention(run, run.windows, cpu_gb=.01, disk_gb=.01,
                                 object_mib=1, l1_watermark=1, l2_watermark=1)
        self.assertAlmostEqual(out['warm_gpu_pct_ideal'], round(512/1100*100, 3))
        self.assertAlmostEqual(out['warm_cpu_pct_ideal'], round(512/1100*100, 3))
        self.assertAlmostEqual(out['warm_recompute_pct_ideal'], round(76/1100*100, 3))
        self.assertEqual(out['warm_cpu_only_prefix_available_pct'], round(1024/1100*100, 3))

    def test_disk_lru_ideal_reuse(self):
        run = ToyRun({0: ['d']})
        labels = {'d': 'disk'}
        run.windows = [('cold', [0], labels), ('warm', [0], labels)]
        out = simulate_retention(run, run.windows, cpu_gb=0, disk_gb=.01,
                                 object_mib=1, l1_watermark=1, l2_watermark=1)
        self.assertEqual(out['warm_disk_pct_ideal'], 100)

    def test_reference_cost_inversion_known_result(self):
        prior = {'cost_prior_ms': {'cpu': 20., 'disk': 40., 'recompute': 100.}}
        # w=1; cumulative mean CPU 100 -> posterior 60.
        ref = {'cost_estimate_ms': {'cpu': 60., 'disk': 40., 'recompute': 100.},
               'cost_sample_counts': {'cpu': 10, 'disk': 0, 'recompute': 0}}
        obs, ns = _reference_cost_data(ref, prior, ref_weight=1.)
        self.assertEqual(obs['cpu'], 100.)
        self.assertEqual(ns['cpu'], 10.)
        self.assertEqual(obs['disk'], 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)

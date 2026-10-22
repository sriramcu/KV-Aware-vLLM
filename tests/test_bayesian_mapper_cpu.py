"""Pure Python checks; no GPU, network or vLLM service required."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from Hierarchical_KV.shortq_placement.bayesian_mapper import (
    WaveBayesianMapper, LmcacheLogCostReader,
)


class BayesianMapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.prior = self.dir / 'prior.json'
        self.scores = self.dir / 'scores.jsonl'
        self.metadata = self.dir / 'placement.json'
        self.vpc = self.dir / 'sidecar.json'
        self.prior.write_text(json.dumps({
            'score_bin_edges': [0, 25, 50, 75, 100],
            'score_bin_probabilities': [0.25] * 4,
            'cost_prior_ms': {'cpu': 20, 'disk': 500, 'recompute': 100},
        }))
        rows = [
            {'source_index': 0, 'chunks': [
                {'chunk_hash': 'a' * 64, 'score': 98},
                {'chunk_hash': 'b' * 64, 'score': 78},
                {'chunk_hash': 'c' * 64, 'score': 50},
                {'chunk_hash': 'd' * 64, 'score': 20},
            ]},
            {'source_index': 1, 'chunks': [
                {'chunk_hash': 'a' * 64, 'score': 98},
                {'chunk_hash': 'e' * 64, 'score': 77},
                {'chunk_hash': 'f' * 64, 'score': 10},
            ]},
        ]
        self.scores.write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def mapper(self, weight=1, gpu=10, cpu=100, disk=100):
        return WaveBayesianMapper(self.prior, weight, gpu, cpu, disk,
                                  self.scores, self.metadata, self.vpc)

    def test_recompute_thresholded_cost_tax_and_monotonicity(self):
        m = self.mapper()
        p = m.bands()['fractions']
        self.assertAlmostEqual(p['gpu'], 10 / 210)
        self.assertAlmostEqual(p['cpu'], 100 / 210)  # CPU faster: no tax
        self.assertAlmostEqual(p['disk'], 100 / 210 * .2)  # 5x slower
        self.assertAlmostEqual(sum(p.values()), 1.0)
        old_drop = p['drop']
        m.observe_cost('disk', 200)
        self.assertAlmostEqual(m.estimated_cost('disk'), 350)  # w=1, half prior
        self.assertLess(m.bands()['fractions']['drop'], old_drop)

    def test_prior_weight_is_fixed_even_with_many_observations(self):
        m = self.mapper(weight=1)
        m.observe_cost('cpu', 80, count=10000)
        self.assertAlmostEqual(m.estimated_cost('cpu'), 50)
        a = self.mapper(weight=0)
        a.observe_cost('cpu', 80, count=10000)
        self.assertAlmostEqual(a.estimated_cost('cpu'), 80)
        b = self.mapper(weight=2)
        b.observe_cost('cpu', 80, count=10000)
        self.assertAlmostEqual(b.estimated_cost('cpu'), 40)

    def test_unique_score_samples_and_bayesian_histogram(self):
        m = self.mapper()
        self.assertAlmostEqual(m.percentile(50), 0.5)
        m.observe_scores([0])
        self.assertEqual(len(m.seen_scores), 4)
        m.observe_scores([1])
        self.assertEqual(len(m.seen_scores), 6)  # repeated first hash counted once
        self.assertAlmostEqual(sum(m.probabilities()), 1.0)
        self.assertLess(m.percentile(20), m.percentile(78))

    def test_prefix_closure_favors_discard_over_repair(self):
        b = {'gpu': 0.9, 'cpu': .5, 'disk': .3}
        # GPU, DROP, CPU: repairing the DROP costs .1; discarding CPU costs .03.
        r, cutoff, feasible = WaveBayesianMapper._prefix_closure(
            ['gpu', 'drop', 'cpu'], [.95, .2, .53], ['h1', 'h2', 'h3'], {}, b)
        self.assertTrue(feasible)
        self.assertEqual(cutoff, 1)
        self.assertEqual(r, ['gpu', 'drop', 'drop'])

    def test_fixed_conflict_is_diagnosed_not_migrated(self):
        b = {'gpu': .9, 'cpu': .5, 'disk': .3}
        r, cutoff, feasible = WaveBayesianMapper._prefix_closure(
            ['cpu', 'gpu'], [.6, .99], ['h1', 'h2'],
            {'h1': 'drop', 'h2': 'gpu'}, b)
        self.assertFalse(feasible)
        self.assertEqual(r, ['drop', 'gpu'])

    def test_two_wave_snapshot_and_immutable_hash(self):
        m = self.mapper()
        w1 = m.publish_wave([0], 'cold')
        self.assertEqual(w1['epoch'], 1)
        first = dict(m.assigned)
        sidecar1 = json.loads(self.vpc.read_text())
        self.assertEqual(len(sidecar1['0']), 32 * 4)
        m.observe_scores([0])
        m.observe_cost('disk', 200, count=100)
        w2 = m.publish_wave([1], 'warm')
        self.assertEqual(w2['epoch'], 2)
        self.assertEqual(first['a' * 64], m.assigned['a' * 64])
        self.assertEqual(len(json.loads(self.metadata.read_text())), len(m.assigned))
        self.assertIn('0', json.loads(self.vpc.read_text()))
        self.assertIn('1', json.loads(self.vpc.read_text()))
        self.assertEqual(w2['score_samples'], 4)  # wave-1 observations only

    def test_lmcache_dynamic_hash_loader_refreshes(self):
        # Load the standalone metadata module without importing LMCache native deps.
        import importlib.util
        import logging
        import os
        import sys
        import types
        import time
        from unittest.mock import patch
        root = Path(__file__).resolve().parents[1]
        path = root / 'third_party/LMCache/lmcache/v1/distributed/placement_metadata.py'
        stub = types.ModuleType('lmcache')
        stub_log = types.ModuleType('lmcache.logging')
        stub_log.init_logger = lambda name: logging.getLogger(name)
        spec = importlib.util.spec_from_file_location('bayes_test_metadata', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'lmcache': stub, 'lmcache.logging': stub_log}), \
             patch.dict(os.environ, {'LMCACHE_GNN_PLACEMENT_METADATA': str(self.metadata),
                                     'LMCACHE_GNN_PLACEMENT_DYNAMIC': '1'}):
            spec.loader.exec_module(module)
            from Hierarchical_KV.shortq_placement.bayesian_mapper import atomic_json
            atomic_json(self.metadata, {'aa': 'cpu'})
            self.assertEqual(module.get_chunk_placement(bytes.fromhex('aa')), 'cpu')
            atomic_json(self.metadata, {'aa': 'disk'})
            time.sleep(.28)
            self.assertEqual(module.get_chunk_placement(bytes.fromhex('aa')), 'disk')

    def test_missing_hash_bypasses_refresh_throttle_in_two_workers(self):
        """New-wave hashes must never fall back to disk due to cached metadata."""
        import importlib.util
        import logging
        import os
        import sys
        import types
        from unittest.mock import patch
        from Hierarchical_KV.shortq_placement.bayesian_mapper import atomic_json

        path = (Path(__file__).resolve().parents[1] /
                'third_party/LMCache/lmcache/v1/distributed/placement_metadata.py')
        stub = types.ModuleType('lmcache')
        stub_log = types.ModuleType('lmcache.logging')
        stub_log.init_logger = lambda name: logging.getLogger(name)
        with patch.dict(sys.modules, {'lmcache': stub, 'lmcache.logging': stub_log}), \
             patch.dict(os.environ, {'LMCACHE_GNN_PLACEMENT_METADATA': str(self.metadata),
                                     'LMCACHE_GNN_PLACEMENT_DYNAMIC': '1'}):
            workers = []
            for n in range(2):
                spec = importlib.util.spec_from_file_location(f'test_worker_{n}', path)
                worker = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(worker)
                workers.append(worker)
            a, b, c = ('aa' * 32, 'bb' * 32, 'cc' * 32)
            atomic_json(self.metadata, {a: 'cpu'})
            for worker in workers:
                self.assertEqual(worker.get_chunk_placement(bytes.fromhex(a)), 'cpu')
            # Publish two successive waves without waiting for the 250-ms poll.
            atomic_json(self.metadata, {a: 'cpu', b: 'gpu'})
            for worker in workers:
                self.assertEqual(worker.get_chunk_placement(bytes.fromhex(b)), 'gpu')
            atomic_json(self.metadata, {a: 'cpu', b: 'gpu', c: 'drop'})
            for worker in workers:
                self.assertEqual(worker.get_chunk_placement(bytes.fromhex(c)), 'drop')
                self.assertEqual(len(worker._missed), 0)
            self.assertEqual(workers[0].get_chunk_placement(bytes.fromhex('dd' * 32)), 'disk')

    def test_smoke_prior_discriminates_raw_rank_scores(self):
        root = Path(__file__).resolve().parents[1]
        prior = root / 'Hierarchical_KV/shortq_placement/priors/rank_smoke_diagnostic.json'
        m = WaveBayesianMapper(prior, 1, 10, 225, 100,
                              self.scores, self.metadata, self.vpc)
        candidates = [m._tier(m.percentile(score), m.bands())
                      for score in (-.52, -.4, -.30, -.20)]
        self.assertGreaterEqual(len(set(candidates)), 3)
        self.assertEqual(candidates[0], 'drop')
        self.assertEqual(candidates[-1], 'gpu')

    def test_log_cost_proxies_opt_in(self):
        m = self.mapper()
        log = self.dir / 'lm.log'
        log.write_text('[MP_PREFETCH_DONE] request=2 phase=LOAD lookup_s=0.01 load_wait_s=0.2 load_s=0.8 retained_keys=2 loaded_keys=4 failed_keys=0\n'
                       'Retrieved 1024 tokens in 0.040 seconds\n')
        reader = LmcacheLogCostReader(log)
        counts = reader.collect(m)
        self.assertEqual(counts, {'cpu_proxy': 1, 'disk_proxy': 1})
        self.assertAlmostEqual(m.estimated_cost('cpu'), (20 + 20) / 2)
        self.assertAlmostEqual(m.estimated_cost('disk'), (500 + 200) / 2)
        self.assertEqual(reader.collect(m), {'cpu_proxy': 0, 'disk_proxy': 0})

    def test_pseudo_count_mode_prior_decays_for_costs_and_scores(self):
        m = WaveBayesianMapper(
            self.prior, 1, 10, 100, 100, self.scores, self.metadata, self.vpc,
            posterior_mode='pseudo_count', score_prior_count=4, cost_prior_count=5)
        m.observe_cost('cpu', 80, count=15)
        self.assertAlmostEqual(m.estimated_cost('cpu'), (5*20 + 15*80)/20)
        m.observe_scores([0])
        self.assertEqual(sum(m.hist_counts), 4)
        expected = [(4*.25 + c)/8 for c in m.hist_counts]
        for got, want in zip(m.probabilities(), expected):
            self.assertAlmostEqual(got, want)
        fixed = self.mapper()
        fixed.observe_cost('cpu', 80, count=15)
        self.assertAlmostEqual(fixed.estimated_cost('cpu'), 50)

    def test_readmission_renews_preferences_after_band_update(self):
        m = WaveBayesianMapper(
            self.prior, 0, 10, 100, 100, self.scores, self.metadata, self.vpc,
            readmission=True)
        m.publish_wave([0], 'cold')
        previous = dict(m.assigned)
        self.assertEqual(previous['d'*64], 'drop')
        m.observe_cost('disk', 10, count=15)  # disk now faster than recompute
        status = m.publish_wave([0], 'warm')
        self.assertGreater(status['readmission_preferences_changed'], 0)
        self.assertNotEqual(previous['d'*64], m.assigned['d'*64])
        self.assertEqual(status['new_hashes'], 0)
        self.assertIn('note', status)  # does not claim a physical readmission

    def test_random_scores_shuffle_preserves_marginal_population(self):
        m = WaveBayesianMapper(
            self.prior, 1, 10, 100, 100, self.scores, self.metadata, self.vpc,
            random_score_seed=42)
        old = self.mapper()
        self.assertEqual(sorted({x for row in m.rows.values() for _, x in row}),
                         sorted({x for row in old.rows.values() for _, x in row}))
        self.assertNotEqual(m.rows[0], old.rows[0])
        self.assertEqual(m.rows[0][0][1], m.rows[1][0][1])

    def test_direct_cost_reader_completed_transfer_and_disk_stage(self):
        from Hierarchical_KV.shortq_placement.bayesian_mapper import LmcacheCompletionCostReader
        m = self.mapper()
        log = self.dir / 'direct.log'
        log.write_text('noise\n[BAYES_H2D_COMPLETE] request=a chunks=2 elapsed_ms=40.000\n'
                       '[BAYES_L2_LOAD_COMPLETE] request=7 chunks=4 elapsed_ms=800.000\n')
        r = LmcacheCompletionCostReader(log)
        first = r.collect(m)
        self.assertEqual(first['h2d_completed'], 1)
        self.assertEqual(first['disk_load_completed'], 1)
        self.assertEqual(first['disk_wave_fallback'], 1)
        self.assertEqual(m.cost_counts, {'cpu': 2, 'disk': 4, 'recompute': 0})
        self.assertAlmostEqual(m.estimated_cost('cpu'), 20)
        self.assertAlmostEqual(m.estimated_cost('disk'), 360)
        self.assertEqual(r.collect(m)['h2d_completed'], 0)
        self.assertEqual(r.collect(m)['disk_load_completed'], 0)

    def test_recompute_cost_is_observed_and_weighted(self):
        m = self.mapper()
        self.assertAlmostEqual(m.estimated_cost('recompute'), 100)
        m.observe_cost('recompute', 200, count=2.5)
        m.observe_cost('recompute', 80, count=0.5)
        self.assertAlmostEqual(m.cost_counts['recompute'], 3)
        self.assertAlmostEqual(m.estimated_cost('recompute'),
                               (100 + (200*2.5+80*.5)/3)/2)
        self.assertAlmostEqual(m.bands()['fractions']['cpu'], 100/210)

    def test_matching_l2_h2d_by_external_request(self):
        from Hierarchical_KV.shortq_placement.bayesian_mapper import LmcacheCompletionCostReader
        m = self.mapper()
        log = self.dir / 'matched.log'
        log.write_text(
            '[BAYES_L2_LOAD_COMPLETE] request=19 external_request=req-a chunks=2 elapsed_ms=60\n'
            '[BAYES_H2D_COMPLETE] request=req-a chunks=4 elapsed_ms=80\n'
            '[BAYES_H2D_COMPLETE] request=req-a chunks=4 elapsed_ms=120\n'
            '[BAYES_H2D_COMPLETE] request=req-b chunks=2 elapsed_ms=700\n'
        )
        counts = LmcacheCompletionCostReader(log).collect(m)
        self.assertEqual(counts['disk_matched'], 1)
        self.assertEqual(counts['disk_wave_fallback'], 0)
        # Matched H2D is max(80/4, 120/4) = 30ms, not req-b's 350ms.
        self.assertAlmostEqual(m.estimated_cost('disk'), (500 + 60/2 + 30)/2)

    def test_async_recompute_reader_uses_nonoverlapping_completion_intervals(self):
        from Hierarchical_KV.shortq_placement.bayesian_mapper import PhaseRecomputeCostReader
        import json
        m = self.mapper()
        trace_dir = self.dir / 'async-trace'
        trace_dir.mkdir()
        logfile = trace_dir / 'phase_engine.pid77.jsonl'

        def row(end_ns, submit_ns, prompt, *, decode=0):
            return json.dumps({"kind": "engine_step", "async_scheduling": True,
                               "result_end_ns": end_ns, "submit_end_ns": submit_ns,
                               "prompt_compute_tokens": prompt,
                               "decode_position_tokens": decode}) + '\n'

        # Two pipelined forwards were submitted concurrently. The raw
        # (end-submit) time of the second includes the first and must not
        # be counted a second time. First = 100 ms, second = 80 ms.
        logfile.write_text(row(200_000_000, 100_000_000, 512) +
                           row(280_000_000, 110_000_000, 1024))
        reader = PhaseRecomputeCostReader(trace_dir)
        stats = reader.collect(m)
        self.assertEqual(stats['recompute_steps'], 2)
        self.assertEqual(stats['recompute_async_steps'], 2)
        self.assertEqual(stats['recompute_sync_steps'], 0)
        self.assertEqual(m.cost_counts['recompute'], 3)
        self.assertAlmostEqual(m.estimated_cost('recompute'),
                               (100 + (100 + 80) / 3) / 2)

        # A zero-token completed step moves the completion watermark but
        # does not enter the recomputation average. A new wave must pick up
        # that watermark without replaying old observations.
        with logfile.open('a') as f:
            f.write(row(320_000_000, 270_000_000, 0) +
                    row(400_000_000, 310_000_000, 512))
        stats = reader.collect(m)
        self.assertEqual(stats['recompute_steps'], 1)
        self.assertEqual(stats['recompute_async_steps'], 1)
        self.assertEqual(stats['recompute_skipped'], 1)
        self.assertEqual(m.cost_counts['recompute'], 4)
        self.assertAlmostEqual(m.estimated_cost('recompute'),
                               (100 + (100 + 80 + 80) / 4) / 2)
        self.assertEqual(reader.collect(m)['recompute_steps'], 0)

    def test_phase_recompute_reader_exactly_once_and_partial_line(self):
        from Hierarchical_KV.shortq_placement.bayesian_mapper import PhaseRecomputeCostReader
        m = self.mapper()
        trace_dir = self.dir / 'trace'
        trace_dir.mkdir()
        logfile = trace_dir / 'phase_engine.pid1.jsonl'
        entry = lambda prompt, decode, elapsed_ms: json.dumps({
            'kind': 'engine_step', 'prompt_compute_tokens': prompt,
            'decode_position_tokens': decode, 'submit_end_ns': 1000,
            'result_end_ns': 1000 + int(elapsed_ms * 1e6)}) + '\n'
        logfile.write_text(entry(1024, 0, 200) + entry(1024, 1000, 100) +
                           '{"kind":"engine_step"')
        r = PhaseRecomputeCostReader(trace_dir)
        status = r.collect(m)
        self.assertEqual(status['recompute_steps'], 1)
        self.assertEqual(status['recompute_skipped'], 1)
        self.assertEqual(m.cost_counts['recompute'], 2)
        self.assertAlmostEqual(m.estimated_cost('recompute'), (100+100)/2)
        with logfile.open('a') as f:
            f.write('\n' + entry(512, 0, 60))
        # Previous incomplete JSON line is skipped, but new data is read.
        self.assertEqual(r.collect(m)['recompute_steps'], 1)
        self.assertEqual(r.collect(m)['recompute_steps'], 0)
        self.assertAlmostEqual(m.cost_counts['recompute'], 3)

    def test_inflight_store_pin_survives_snapshot_change(self):
        import importlib.util
        import logging
        import os
        import sys
        import types
        from unittest.mock import patch
        from Hierarchical_KV.shortq_placement.bayesian_mapper import atomic_json
        path = (Path(__file__).resolve().parents[1] /
                'third_party/LMCache/lmcache/v1/distributed/placement_metadata.py')
        stub = types.ModuleType('lmcache')
        stub_log = types.ModuleType('lmcache.logging')
        stub_log.init_logger = lambda name: logging.getLogger(name)
        with patch.dict(sys.modules, {'lmcache': stub, 'lmcache.logging': stub_log}), \
             patch.dict(os.environ, {'LMCACHE_GNN_PLACEMENT_METADATA': str(self.metadata),
                                     'LMCACHE_GNN_PLACEMENT_DYNAMIC': '1'}):
            spec = importlib.util.spec_from_file_location('bayes_pin_test', path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            from dataclasses import dataclass
            @dataclass(frozen=True)
            class Key:
                chunk_hash: bytes
                kv_rank: int
            h = bytes.fromhex('aa'*32)
            key = Key(h, 0)
            other_rank = Key(h, 1)
            atomic_json(self.metadata, {h.hex(): 'disk'})
            self.assertEqual(mod.get_chunk_placement(h), 'disk')
            mod.pin_store_intent(key, 'disk')
            atomic_json(self.metadata, {h.hex(): 'cpu'})
            mod._load_if_needed(force=True)
            self.assertEqual(mod.get_store_intent(key), 'disk')
            self.assertEqual(mod.get_store_intent(other_rank), 'cpu')
            self.assertEqual(mod.get_chunk_placement(h), 'cpu')
            mod.pin_store_intent(key, 'cpu')
            self.assertEqual(mod.get_store_intent(key), 'cpu')

    def test_bayesian_store_task_snapshot_survives_readmission(self):
        """Routing, L1 deletion and budget-skip follow the original task tier."""
        import importlib.util
        import logging
        import os
        import sys
        import types
        from dataclasses import dataclass
        from unittest.mock import patch
        from Hierarchical_KV.shortq_placement.bayesian_mapper import atomic_json
        root = Path(__file__).resolve().parents[1]
        @dataclass(frozen=True)
        class Key:
            chunk_hash: bytes
            kv_rank: int = 0
        h = bytes.fromhex('bb' * 32)
        key = Key(h)
        atomic_json(self.metadata, {h.hex(): 'disk'})
        fake = {name: types.ModuleType(name) for name in (
            'lmcache', 'lmcache.logging', 'lmcache.v1',
            'lmcache.v1.distributed', 'lmcache.v1.distributed.api',
            'lmcache.v1.distributed.l2_adapters',
            'lmcache.v1.distributed.l2_adapters.config',
            'lmcache.v1.distributed.placement_metadata')}
        fake['lmcache.logging'].init_logger = lambda name: logging.getLogger(name)
        fake['lmcache.v1.distributed.api'].ObjectKey = Key
        conf = fake['lmcache.v1.distributed.l2_adapters.config']
        conf.L2AdapterConfigBase = object
        conf.get_type_name_for_config = lambda cfg: 'disk'
        with patch.dict(sys.modules, fake), patch.dict(os.environ, {
            'LMCACHE_GNN_PLACEMENT_METADATA': str(self.metadata),
            'LMCACHE_GNN_PLACEMENT_DYNAMIC': '1',
            'LMCACHE_GNN_L2_BACKING': '0',
        }):
            path = root / 'third_party/LMCache/lmcache/v1/distributed/placement_metadata.py'
            spec = importlib.util.spec_from_file_location('bayes_store_snap_meta', path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules['lmcache.v1.distributed.placement_metadata'] = mod
            mod.pin_store_intent(key, 'disk')
            path = root / 'third_party/LMCache/lmcache/v1/distributed/storage_controllers/store_policy.py'
            spec = importlib.util.spec_from_file_location('bayes_store_snap_policy', path)
            policy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(policy)
            gnn = policy.GNNDynamicStorePolicy()
            descriptor = types.SimpleNamespace(index=0)
            snapshot = gnn._placements([key])
            self.assertEqual(snapshot, {key: 'disk'})
            self.assertEqual(gnn.select_store_targets_with_intents(
                [key], [descriptor], snapshot), {0: [key]})
            # A new admission changes the global pin before the old L2
            # operation finishes. The old completion must use its snapshot.
            mod.pin_store_intent(key, 'cpu')
            self.assertEqual(gnn.select_l1_deletions([key]), [])
            self.assertEqual(gnn.select_l1_deletions_with_intents([key], snapshot), [key])
            self.assertEqual(gnn.select_l1_deletions_on_store_skip_with_intents(
                [key], snapshot), [key])
            self.assertEqual(gnn.select_store_targets_with_intents(
                [key], [descriptor], snapshot), {0: [key]})
            self.assertEqual(gnn.select_store_targets([key], [descriptor]), {})


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""Unit tests: synthetic exact answers + voting/H100 29886 historical integration."""
import os
from pathlib import Path
import unittest
from offline_placement_agreement import (evaluate, parse_pseudo_presets,
    scenario_specs, labels_from_file, make_occurrences)
from offline_kv_replay import RunArchive, load_mapper, replay_waves, reference_diff

HERE=Path(__file__).resolve().parent

class KnownAnswers(unittest.TestCase):
    def test_everything_equal(self):
        labels={'a':'gpu','b':'cpu','c':'disk','d':'drop'}
        m,pos=evaluate(labels,labels,[['a','b','c','d'],['a','b','c','d']])
        self.assertEqual(m['overall_unique_pct'],100.)
        self.assertEqual(m['warm_occurrence_weighted_pct'],100.)
        self.assertEqual(m['prefix_span_weighted_pct'],100.)
        self.assertEqual(m['strict_contiguous_prefix_agreement_pct'],100.)
        self.assertEqual(m['gpu_pos01_pct'],50.)
        self.assertEqual(m['cpu_pos23_pct'],0.)
        self.assertEqual(m['disk_pos45_pct'],None)

    def test_first_prefix_hole_strands_later_agreement(self):
        ref={'a':'gpu','b':'cpu','c':'disk'}
        pred={'a':'cpu','b':'cpu','c':'disk'}
        m,pos=evaluate(ref,pred,[['a','b','c'],['a','b','c']])
        self.assertAlmostEqual(m['overall_unique_pct'],66.667)
        self.assertAlmostEqual(m['warm_occurrence_weighted_pct'],66.667)
        self.assertEqual(m['strict_contiguous_prefix_agreement_pct'],0.)
        self.assertEqual(m['gpu_recall_unique_pct'],0.)

    def test_warm_repetition_weight_and_span_weight(self):
        ref={'a':'gpu','b':'cpu','c':'disk'}
        pred={'a':'gpu','b':'drop','c':'disk'}
        m,_=evaluate(ref,pred,[['a','b','c'],['a'],['a']])
        self.assertAlmostEqual(m['overall_unique_pct'],66.667)
        self.assertEqual(m['warm_occurrence_weighted_pct'],80.0)
        self.assertEqual(m['warm_repeat_excess_weighted_pct'],100.0)
        # span for a 3+1+1=5, b=2, c=1, weight=8; correct a,c=6
        self.assertEqual(m['prefix_span_weighted_pct'],75.)
        # First request matches only a; second and third match a => 3/5
        self.assertEqual(m['strict_contiguous_prefix_agreement_pct'],60.)

    def test_different_hashes_refused(self):
        with self.assertRaisesRegex(ValueError,'hash sets'):
            evaluate({'a':'gpu'},{'b':'gpu'},[['a']])

    def test_grid_125(self):
        sc=scenario_specs([10,15,20,25,30],[225],[300,400,500,600,700],
                          [1.,.5,.2],parse_pseudo_presets('low:10:1,high:1000:100'))
        self.assertEqual(len(sc),125)
        self.assertEqual(len([x for x in sc if x['posterior_mode']=='fixed']),75)
        self.assertEqual(len([x for x in sc if x['posterior_mode']=='pseudo_count']),50)

class HistoricalKnownValues(unittest.TestCase):
    def test_voting_hash_set_exact_and_fixed_baseline(self):
        run=RunArchive(HERE/'h100_29886_minimal_validation_trace.zip').load()
        reference=labels_from_file(HERE/'voting_29756_hash_tiers.json')
        self.assertEqual(len(reference),4857)
        self.assertEqual(sum(map(len,make_occurrences(run))),6831)
        counts={t:sum(v==t for v in reference.values()) for t in ('gpu','cpu','disk','drop')}
        self.assertEqual(counts,{'gpu':258,'cpu':767,'disk':1012,'drop':2820})
        repo=Path(os.environ.get('KVAWARE_REPO','')).resolve() if os.environ.get('KVAWARE_REPO') else HERE/'test_repo'
        if not (repo/'Hierarchical_KV/shortq_placement/bayesian_mapper.py').is_file():
            self.skipTest('Set KVAWARE_REPO to patched repository for the historical mapper integration')
        M=load_mapper(repo)
        result,windows=replay_waves(run,HERE/'h100_29886_prior_validation_only.json',M,
            gpu_gb=10,cpu_gb=225,disk_gb=100,weight=1)
        for delta in reference_diff(result,run.ref_waves):
            self.assertEqual(delta['mapped_abs_count_error'],0,delta)
            self.assertEqual(delta['new_hashes_error'],0,delta)
            self.assertEqual(delta['score_samples_error'],0,delta)
            self.assertAlmostEqual(delta['max_band_error'],0.,places=8)
        observed=labels_from_file(HERE/'bayes_29886_observed_hash_tiers.json')
        self.assertEqual(windows[-1][2], observed)
        summary,_=evaluate(reference,windows[-1][2],make_occurrences(run))
        self.assertAlmostEqual(summary['overall_unique_pct'],61.808,places=3)
        self.assertAlmostEqual(summary['gpu_pos01_pct'],60.462,places=3)
        self.assertAlmostEqual(summary['cpu_pos23_pct'],99.923,places=3)
        self.assertEqual(summary['hash_count'],4857)

if __name__=='__main__':
    unittest.main(verbosity=2)

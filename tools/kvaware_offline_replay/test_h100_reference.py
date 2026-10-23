#!/usr/bin/env python3
"""Known-answer 29886 historical 14-wave regression using frozen real trace."""
import os
from pathlib import Path
import unittest
from offline_kv_replay import (RunArchive, load_mapper, replay_waves, reference_diff,
                               simulate_retention)

HERE = Path(__file__).resolve().parent


def find_repository():
    path = Path(os.environ.get('KVAWARE_REPO', '.')).resolve()
    if (path / 'Hierarchical_KV/shortq_placement/bayesian_mapper.py').is_file():
        return path
    for path in (HERE.parent, HERE.parent.parent, Path.cwd()):
        if (path / 'Hierarchical_KV/shortq_placement/bayesian_mapper.py').is_file():
            return path
    raise RuntimeError('Set KVAWARE_REPO to your patched repository root')


class HistoricalRegressions(unittest.TestCase):
    def test_14_wave_exact_placement_against_job_29889_disk300(self):
        prior = HERE / 'h100_29886_prior_validation_only.json'
        run = RunArchive(HERE / 'h100_29889_disk300_minimal_validation_trace.zip').load()
        Mapper = load_mapper(find_repository())
        actual, _windows = replay_waves(run, prior, Mapper,
            gpu_gb=10, cpu_gb=225, disk_gb=300, weight=1)
        differences = reference_diff(actual, run.ref_waves)
        for entry in differences:
            self.assertEqual(entry['mapped_abs_count_error'], 0, entry)
            self.assertEqual(entry['new_hashes_error'], 0, entry)
            self.assertEqual(entry['score_samples_error'], 0, entry)
            self.assertLess(entry['max_band_error'], 1e-9, entry)
        self.assertEqual(len(actual), 14)

    def test_14_wave_exact_placement_against_job_29886(self):
        prior = HERE / 'h100_29886_prior_validation_only.json'
        run = RunArchive(HERE / 'h100_29886_minimal_validation_trace.zip').load()
        Mapper = load_mapper(find_repository())
        actual, windows = replay_waves(run, prior, Mapper,
            gpu_gb=10, cpu_gb=225, disk_gb=100, weight=1)
        differences = reference_diff(actual, run.ref_waves)
        for entry in differences:
            self.assertEqual(entry['mapped_abs_count_error'], 0, entry)
            self.assertEqual(entry['new_hashes_error'], 0, entry)
            self.assertEqual(entry['score_samples_error'], 0, entry)
            self.assertLess(entry['max_band_error'], 1e-9, entry)
        self.assertEqual(len(actual), 14)
        self.assertEqual(actual[-1]['mapped_counts'],
                         {'gpu': 149, 'cpu': 2150, 'disk': 480, 'drop': 2078})
        ret = simulate_retention(run, windows, cpu_gb=225, disk_gb=100)
        # Simulator-known answers, NOT measured scheduler-useful CPU values.
        self.assertAlmostEqual(ret['warm_cpu_pct_ideal'], 47.489, places=3)
        self.assertAlmostEqual(ret['warm_start_l1_gib'], 180., places=3)
        self.assertAlmostEqual(
            run.measured_diag['useful_contribution']['warm']['l1_pct'],
            16.299397714928585)
        self.assertNotAlmostEqual(ret['warm_cpu_pct_ideal'],
                                  run.measured_diag['useful_contribution']['warm']['l1_pct'],
                                  delta=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)

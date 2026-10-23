#!/usr/bin/env python3
"""CPU-only counterfactual KV replay against the installed WaveBayesianMapper.

Reads existing original-order H100 run archive or run directory and the exact
score+hardware prior JSON. No ML, GPU, LMCache writes, or scratch cleanup.

Wave placement is real mapper logic; physical replay deliberately idealized:
serial requests, immediate stores, hash-granular LRU, 'perfect' GPU label cache,
and ideal disk stores (not actual asynchronous VPC or scheduler timing).
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import csv
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

REQUIRED = (
    'placement/gnn_chunk_scores.jsonl',
    'results/bayes_wave_history.json',
    'results/cold_results.jsonl',
    'results/warm_results_execution_order.jsonl',
    'results/manifest.json',
    'run_config.json',
)


def read_jsonlines(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class RunArchive:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.zip = zipfile.ZipFile(self.path) if self.path.is_file() else None
        if self.zip:
            names = self.zip.namelist()
            suffix = 'placement/gnn_chunk_scores.jsonl'
            matches = [s[:-len(suffix)] for s in names if s.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f'expected one run per archive; found {len(matches)} score files')
            self.root = matches[0]
        else:
            self.root = ''
        for f in REQUIRED:
            self.read(f)

    def read(self, name):
        if self.zip:
            try:
                return self.zip.read(self.root + name).decode('utf-8')
            except KeyError as exc:
                raise FileNotFoundError(f'{self.path}::{self.root + name}') from exc
        return (self.path / name).read_text(encoding='utf-8')

    def load(self):
        self.rows_text = self.read('placement/gnn_chunk_scores.jsonl')
        self.rows = {int(r['source_index']): [(x['chunk_hash'].lower(), float(x['score']))
                                                  for x in r['chunks']]
                     for r in read_jsonlines(self.rows_text)}
        self.ref_waves = json.loads(self.read('results/bayes_wave_history.json'))
        try:
            self.measured_diag = json.loads(self.read('mp_diag_summary.json'))
        except FileNotFoundError:
            self.measured_diag = None
        self.config = json.loads(self.read('run_config.json'))
        self.manifest = json.loads(self.read('results/manifest.json'))
        cold = read_jsonlines(self.read('results/cold_results.jsonl'))
        warm = read_jsonlines(self.read('results/warm_results_execution_order.jsonl'))
        self.cold = self._ordered(cold)
        self.warm = self._ordered(warm)
        self.prompt_tokens = {int(r['source_index']): int(r['prompt_tokens']) for r in cold}
        expected = int(self.manifest['num_requests'])
        if len(self.rows) != expected or len(self.cold) != expected or len(self.warm) != expected:
            raise ValueError('row count mismatch: scores/cold/warm vs manifest')
        if set(self.rows) != set(self.cold) or set(self.rows) != set(self.warm):
            raise ValueError('source indices differ across score and results files')
        if self.manifest.get('passage_order_policy') != 'original':
            raise ValueError('Only original passage ordering is supported by this replay')
        for ix, seq in self.rows.items():
            if len(seq) * 512 > self.prompt_tokens[ix]:
                raise ValueError(f'chunk count exceeds prompt tokens, source {ix}')
        if (len(self.ref_waves) != 14 or
            [w['phase'] for w in self.ref_waves] != ['cold'] * 7 + ['warm'] * 7):
            raise ValueError('reference does not contain expected 7 cold + 7 warm waves')
        return self

    @staticmethod
    def _ordered(records):
        positions = [int(r['phase_position']) for r in records]
        if sorted(positions) != list(range(len(records))):
            raise ValueError('phase_position is not a unique full sequence')
        return [int(r['source_index']) for r in sorted(records, key=lambda r: r['phase_position'])]


def load_mapper(repository):
    module = Path(repository) / 'Hierarchical_KV/shortq_placement/bayesian_mapper.py'
    if not module.is_file():
        raise FileNotFoundError(f'cannot locate actual mapper: {module}')
    spec = importlib.util.spec_from_file_location('kvaware_replay_mapper', module)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if not hasattr(mod, 'WaveBayesianMapper'):
        raise ValueError('installed mapper lacks WaveBayesianMapper')
    return mod.WaveBayesianMapper


def parse_grid(values, positive=True):
    result = [float(x.strip()) for x in values.split(',')]
    if not result or not all(math.isfinite(x) and (x > 0 if positive else x >= 0)
                             for x in result):
        raise ValueError(f'invalid grid: {values}')
    return result


def _reference_cost_data(ref, prior, ref_weight):
    """Infer the *historical cumulative observation mean*, never future costs.

    At each wave, historical cost_estimate = (w*prior+mean)/(1+w),
    when sample count is positive. This inverted observation stream is fed
    back to alternative w values, keeping the reference traffic fixed.
    """
    means, counts = {}, {}
    for tier in ('cpu', 'disk', 'recompute'):
        n = float(ref['cost_sample_counts'].get(tier, 0))
        counts[tier] = n
        old = float(prior['cost_prior_ms'][tier])
        means[tier] = ((1 + ref_weight) * float(ref['cost_estimate_ms'][tier])
                       - ref_weight * old) if n > 0 else 0.0
        if n and means[tier] <= 0:
            raise ValueError(f'bad reconstructed observation mean for {tier}')
    return means, counts


def replay_waves(run, prior_path, mapper_class, *, gpu_gb, cpu_gb, disk_gb,
                 weight=1.0, posterior_mode='fixed', score_prior_count=10,
                 cost_prior_count=5):
    prior = json.loads(Path(prior_path).read_text())
    if prior.get('score_format') != 'expected_ordinal_0_100':
        raise ValueError('prior is not expected_ordinal_0_100')
    if abs(run.config.get('bayes_prior_weight', 1) - 1) > 1e-6:
        # The inversion works for any fixed ref_weight, but not pseudo_count.
        if run.config.get('bayes_posterior_mode') != 'fixed':
            raise ValueError('reference must use fixed-weight cost posteriors')
    ref_weight = float(run.config.get('bayes_prior_weight', 1))
    if run.config.get('bayes_posterior_mode', 'fixed') != 'fixed':
        raise ValueError('reference run must have fixed cost posteriors')
    if not run.config.get('bayes_readmission', 0):
        raise ValueError('reference should have readmission enabled')
    with tempfile.TemporaryDirectory(prefix='kv-replay-') as tmp:
        scores = Path(tmp) / 'scores.jsonl'
        scores.write_text(run.rows_text)
        mapper = mapper_class(
            prior_file=prior_path, weight=weight, gpu_gb=gpu_gb,
            cpu_gb=cpu_gb, disk_gb=disk_gb, score_file=scores,
            metadata_file=Path(tmp) / 'metadata.json',
            vpc_file=Path(tmp) / 'vpc.json', readmission=True,
            posterior_mode=posterior_mode,
            score_prior_count=score_prior_count, cost_prior_count=cost_prior_count,
        )
        outputs = []
        windows = []
        ptrs = {'cold': 0, 'warm': 0}
        for ref in run.ref_waves:
            phase = ref['phase']
            idxs = run.cold if phase == 'cold' else run.warm
            n = int(ref['requests'])
            inds = idxs[ptrs[phase]:ptrs[phase] + n]
            ptrs[phase] += n
            if len(inds) != n:
                raise ValueError(f'not enough requests for {phase} wave')
            # Update cumulative measured-cost observations from the historical
            # same-index wave, before this wave publishes. This is NOT a
            # physically consistent counterfactual cost feedback simulation.
            means, counts = _reference_cost_data(ref, prior, ref_weight)
            for t in means:
                mapper.cost_counts[t] = counts[t]
                mapper.cost_sum_ms[t] = means[t] * counts[t]
            status = mapper.publish_wave(inds, phase)
            outputs.append(status)
            windows.append((phase, inds, dict(mapper.assigned)))
            # The serving driver collects the current wave's score population
            # only after publishing the wave.
            mapper.observe_scores(inds)
        return outputs, windows


class LRU:
    def __init__(self, max_objects):
        self.max_objects = max(0, int(max_objects))
        self.data = OrderedDict()
        self.evictions = Counter()
        self.evicted_keys = set()
        self.reads = 0
        self.writes = 0

    def has(self, key, touch=True):
        if key not in self.data:
            return False
        if touch:
            self.data.move_to_end(key)
            self.reads += 1
        return True

    def add(self, key, pos=0):
        if self.max_objects == 0:
            return
        if key in self.data:
            self.data.move_to_end(key)
            return
        self.data[key] = int(pos)
        self.writes += 1
        while len(self.data) > self.max_objects:
            old_key, old_pos = self.data.popitem(last=False)
            self.evictions[old_pos] += 1
            self.evicted_keys.add(old_key)


def _num_objects(gb, object_mib):
    return math.floor(gb * 1024 / object_mib + 1e-9)


def simulate_retention(run, windows, *, cpu_gb, disk_gb,
                       object_mib=80.0, l1_watermark=0.80, l2_watermark=0.90):
    if not 0 < l1_watermark <= 1 or not 0 < l2_watermark <= 1:
        raise ValueError('invalid cache watermark')
    # Simplification: L1 caps objects at watermark, rather than reproducing
    # native eviction trigger + batch target cycles.
    cpu = LRU(_num_objects(cpu_gb * l1_watermark, object_mib))
    disk = LRU(_num_objects(disk_gb * l2_watermark, object_mib))
    # A perfect, unbounded VPC proxy, not actual vLLM GPU residency.
    gpu = set()
    totals = {'cold': Counter(), 'warm': Counter()}
    occupancy = {}
    evicted_then_requested = Counter()
    for phase, sources, preferences in windows:
        if phase == 'warm' and 'warm_start_l1_gib' not in occupancy:
            occupancy['warm_start_l1_gib'] = len(cpu.data) * object_mib / 1024
            occupancy['warm_start_l1_objects'] = len(cpu.data)
        for source in sources:
            seq = run.rows[source]
            total_tokens = run.prompt_tokens[source]
            ctr = totals[phase]
            ctr['requests'] += 1
            ctr['prompt_tokens'] += total_tokens
            ctr['scored_tokens'] += 512 * len(seq)
            ctr['unscored_tokens'] += total_tokens - 512 * len(seq)
            # Keep a separate CPU-only prefix-availability signal (including
            # GPU labels' CPU backing) regardless of simulated GPU hits.
            cpu_only = 0
            for key, _score in seq:
                if key in cpu.data:
                    cpu_only += 512
                else:
                    break
            ctr['cpu_only_prefix_available_tokens'] += cpu_only
            # Choose hierarchy at request start, then update LRU and admit
            # newly recomputed suffix only. No asynchronous PUT/GET timing.
            found = []
            first_miss = len(seq)
            for i, (key, _score) in enumerate(seq):
                if key in gpu:
                    tier = 'gpu'
                elif key in cpu.data:
                    tier = 'cpu'
                elif key in disk.data:
                    tier = 'disk'
                else:
                    first_miss = i
                    break
                found.append((key, tier))
            for key, tier in found:
                ctr[tier + '_tokens'] += 512
                if tier == 'cpu':
                    cpu.has(key)
                elif tier == 'disk':
                    disk.has(key)
            for i, (key, _score) in enumerate(seq[first_miss:], start=first_miss):
                if key in cpu.evicted_keys:
                    evicted_then_requested[i] += 1
                # Later resident hits are not usable after the first hole.
                ctr['recompute_tokens'] += 512
                label = preferences.get(key, 'drop')
                if label == 'gpu':
                    gpu.add(key)
                    cpu.add(key, pos=i)  # GPU backing to L1
                elif label == 'cpu':
                    cpu.add(key, pos=i)
                elif label == 'disk':
                    disk.add(key, pos=i)
            ctr['recompute_tokens'] += total_tokens - 512 * len(seq)
    occupancy['end_l1_gib'] = len(cpu.data) * object_mib / 1024
    occupancy['end_disk_gib'] = len(disk.data) * object_mib / 1024
    warm = totals['warm']
    denom = warm['prompt_tokens']
    result = {
        'warm_prompt_tokens': denom,
        'warm_gpu_pct_ideal': round(warm['gpu_tokens'] / denom * 100, 3),
        'warm_cpu_pct_ideal': round(warm['cpu_tokens'] / denom * 100, 3),
        'warm_disk_pct_ideal': round(warm['disk_tokens'] / denom * 100, 3),
        'warm_recompute_pct_ideal': round(warm['recompute_tokens'] / denom * 100, 3),
        'warm_cpu_only_prefix_available_pct': round(warm['cpu_only_prefix_available_tokens'] / denom * 100, 3),
        'l1_writes': cpu.writes, 'l1_evictions': sum(cpu.evictions.values()),
        'l1_evictions_by_position': dict(sorted(cpu.evictions.items())),
        'evicted_hash_later_needed_by_position': dict(sorted(evicted_then_requested.items())),
        'l1_capacity_objects_effective': cpu.max_objects,
        'disk_capacity_objects_effective': disk.max_objects,
        **occupancy,
    }
    if abs(sum(result[k] for k in ('warm_gpu_pct_ideal', 'warm_cpu_pct_ideal',
                                  'warm_disk_pct_ideal', 'warm_recompute_pct_ideal')) - 100) > 0.01:
        raise AssertionError('warm fractions fail normalization')
    return result


def reference_diff(outputs, history):
    diffs = []
    for actual, ref in zip(outputs, history):
        count_diff = {t: int(actual.get('mapped_counts', {}).get(t, 0))
                      - int(ref.get('mapped_counts', {}).get(t, 0))
                      for t in ('gpu', 'cpu', 'disk', 'drop')}
        mismatch = sum(abs(x) for x in count_diff.values())
        thr_error = max(abs(actual['thresholds'][t] - ref['thresholds'][t])
                        for t in ('gpu', 'cpu', 'disk'))
        diffs.append({
            'wave': ref['epoch'], 'phase': ref['phase'],
            'mapped_abs_count_error': mismatch,
            'mapped_delta': count_diff,
            'new_hashes_error': actual['new_hashes'] - ref['new_hashes'],
            'score_samples_error': actual['score_samples'] - ref['score_samples'],
            'max_band_error': thr_error,
        })
    return diffs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--repo', default='.', help='repo root; loads YOUR patched WaveBayesianMapper')
    p.add_argument('--run', required=True, help='29886 run directory or ZIP containing one run')
    p.add_argument('--prior', required=True, help='exact H100 expected-ordinal prior JSON')
    p.add_argument('--mode', choices=['validate', 'sweep'], required=True)
    p.add_argument('--cpu-gb', default='100,225')
    p.add_argument('--disk-gb', default='100,300,500,700')
    p.add_argument('--gpu-gb', default='10')
    p.add_argument('--prior-weight', default='1,0.3,0.1')
    p.add_argument('--posterior-mode', choices=['fixed', 'pseudo_count'], default='fixed')
    p.add_argument('--score-prior-count', type=float, default=10)
    p.add_argument('--cost-prior-count', type=float, default=5)
    p.add_argument('--object-mib', type=float, default=80.0, help='H100 TP2 LMCache per-chunk object size')
    p.add_argument('--l1-watermark', type=float, default=0.8)
    p.add_argument('--l2-watermark', type=float, default=0.9)
    p.add_argument('--output', help='CSV for sweep; JSON for validate')
    p.add_argument('--strict', action='store_true', help='validation fails if any wave placement differs')
    return p.parse_args()


def main():
    args = parse_args()
    run = RunArchive(args.run).load()
    if not math.isfinite(args.object_mib) or args.object_mib <= 0:
        raise ValueError('--object-mib must be positive')
    Mapper = load_mapper(args.repo)
    gpu_list, cpu_list, disk_list, weights = (
        parse_grid(args.gpu_gb), parse_grid(args.cpu_gb),
        parse_grid(args.disk_gb), parse_grid(args.prior_weight, positive=False))
    if args.mode == 'validate':
        cfg = run.config
        out, windows = replay_waves(
            run, args.prior, Mapper,
            gpu_gb=float(cfg['bayes_gpu_gb']), cpu_gb=float(cfg['l1_gb']),
            disk_gb=float(cfg['bayes_disk_gb']), weight=float(cfg['bayes_prior_weight']),
            posterior_mode=cfg.get('bayes_posterior_mode', 'fixed'),
            score_prior_count=float(cfg.get('bayes_score_prior_count', 10)),
            cost_prior_count=float(cfg.get('bayes_cost_prior_count', 5)),
        )
        diffs = reference_diff(out, run.ref_waves)
        ret = simulate_retention(run, windows,
                                 cpu_gb=float(cfg['l1_gb']), disk_gb=float(cfg['bayes_disk_gb']),
                                 object_mib=args.object_mib,
                                 l1_watermark=args.l1_watermark, l2_watermark=args.l2_watermark)
        result = {'reference_job': Path(args.run).name, 'wave_comparison': diffs,
                  'waves_exact': all(x['mapped_abs_count_error'] == 0 and
                                     x['new_hashes_error'] == 0 and
                                     x['score_samples_error'] == 0 and
                                     x['max_band_error'] < 1e-9 for x in diffs),
                  'idealized_cache_replay': ret,
                  'measured_reference_l1_warm_start_gib': None,
                  'limitations': 'Placement validation is exact; replayed cache residence is not. '
                                 'Perfect GPU, serial requests, immediate stores, no PUT cap, '
                                 'no asynchronous timing, simplified LRU watermark.'}
        try:
            report = json.loads(run.read('results/l1_warm_start.json'))
            result['measured_reference_l1_warm_start_gib'] = report.get('persistent_object_gib')
        except FileNotFoundError:
            pass
        if run.measured_diag:
            measured = run.measured_diag.get('useful_contribution', {}).get('warm', {})
            cpu_pct = measured.get('l1_pct')
            if cpu_pct is not None:
                result['measured_warm_scheduler_useful_cpu_pct'] = float(cpu_pct)
                result['ideal_cpu_minus_measured_percentage_points'] = round(
                    ret['warm_cpu_pct_ideal'] - float(cpu_pct), 3)
                if abs(result['ideal_cpu_minus_measured_percentage_points']) > 5:
                    result['retention_not_empirically_validated'] = True
                    result['warning'] = (
                        'Idealized CPU estimate differs by >5 percentage points from measured '
                        'scheduler-consumed L1: do not treat offline sweep percentages as '
                        'quantitative performance predictions.')
        payload = json.dumps(result, indent=2)
        if args.output:
            Path(args.output).write_text(payload + '\n')
        print(payload)
        if args.strict and not result['waves_exact']:
            raise SystemExit('VALIDATION FAILED: historical mapper wave decisions differ')
    else:
        if not args.output:
            raise ValueError('--mode sweep requires --output CSV')
        records = []
        for gpu in gpu_list:
            for cpu in cpu_list:
                for disk in disk_list:
                    for w in weights:
                        statuses, windows = replay_waves(
                            run, args.prior, Mapper, gpu_gb=gpu, cpu_gb=cpu, disk_gb=disk,
                            weight=w, posterior_mode=args.posterior_mode,
                            score_prior_count=args.score_prior_count,
                            cost_prior_count=args.cost_prior_count)
                        ret = simulate_retention(run, windows, cpu_gb=cpu, disk_gb=disk,
                                                 object_mib=args.object_mib,
                                                 l1_watermark=args.l1_watermark,
                                                 l2_watermark=args.l2_watermark)
                        records.append({
                            'gpu_gb': gpu, 'cpu_gb': cpu, 'disk_gb': disk,
                            'prior_weight': w, 'posterior_mode': args.posterior_mode,
                            'final_mapped_counts': json.dumps(statuses[-1]['mapped_counts'], sort_keys=True),
                            **{k: v for k, v in ret.items() if not isinstance(v, dict)},
                            'l1_evictions_by_position': json.dumps(ret['l1_evictions_by_position']),
                            'evicted_hash_later_needed_by_position': json.dumps(ret['evicted_hash_later_needed_by_position']),
                        })
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f'Wrote {len(records)} sweep scenarios to {args.output}')
        print('Reminder: estimated hit percentages are idealized bounds, NOT scheduler-consumed KV.')


if __name__ == '__main__':
    main()

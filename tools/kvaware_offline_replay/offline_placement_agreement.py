#!/usr/bin/env python3
"""CPU-only counterfactual *placement* agreement sweep versus voting job 29756.

Uses the installed Bayesian mapper and previously validated replay_waves helper.
No GPU, LMCache, cache-residency model, Slurm, or scratch cleanup.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from offline_kv_replay import RunArchive, load_mapper, replay_waves, parse_grid, reference_diff

TIERS = ('gpu', 'cpu', 'disk', 'drop')
REGIONS = ((0, 1, 'gpu'), (2, 3, 'cpu'), (4, 5, 'disk'))


def labels_from_file(path: str | Path) -> dict[str, str]:
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    labels = data.get('hash_to_tier', data)
    if not isinstance(labels, dict) or not labels:
        raise ValueError(f'bad empty target label map: {path}')
    labels = {str(k).lower(): str(v).lower() for k, v in labels.items()}
    if any(t not in TIERS for t in labels.values()):
        raise ValueError(f'invalid tier in target {path}')
    return labels


def make_occurrences(run):
    """One entry per WARM request/position. Preserve repeats and contiguous order."""
    result = []
    for ix in run.warm:
        row = [h for h, _score in run.rows[ix]]
        result.append(row)
    return result


def evaluate(reference: dict[str, str], predicted: dict[str, str], warm_sequences):
    """Return exact label agreement and occurrence/position/prefix diagnostics.

    These are *placement* comparisons, never effective cache hits.
    """
    used_hashes = {h for seq in warm_sequences for h in seq}
    if used_hashes != set(reference) or set(predicted) != set(reference):
        missing_v = used_hashes - set(reference)
        missing_b = used_hashes - set(predicted)
        extra_v = set(reference) - used_hashes
        extra_b = set(predicted) - used_hashes
        raise ValueError('hash sets do not match: '
                         f'voting missing {len(missing_v)}, Bayesian missing {len(missing_b)}, '
                         f'voting extra {len(extra_v)}, Bayesian extra {len(extra_b)}')
    if any(x not in TIERS for x in predicted.values()):
        raise ValueError('bad predicted tier')

    by_pos = defaultdict(lambda: {'total': 0, 'matched': 0,
                                  'ref': Counter(), 'pred': Counter()})
    warm_freq = Counter(h for seq in warm_sequences for h in seq)
    prefix_span = Counter()
    matched_occ = 0
    strict_matched = 0
    region_count = Counter()
    region_correct = Counter()
    # Positional GPU/CPU/Disk/DROP target-band rates, *not* a claim voting is static.
    target_region = {'0-1': 'gpu', '2-3': 'cpu', '4-5': 'disk', '6+': 'drop'}
    reference_occ_counts = Counter()
    predicted_occ_counts = Counter()
    for seq in warm_sequences:
        still_matching = True
        for pos, h in enumerate(seq):
            ref, pred = reference[h], predicted[h]
            good = ref == pred
            rec = by_pos[pos]
            rec['total'] += 1
            rec['matched'] += int(good)
            rec['ref'][ref] += 1
            rec['pred'][pred] += 1
            region = '0-1' if pos < 2 else '2-3' if pos < 4 else '4-5' if pos < 6 else '6+'
            region_count[region] += 1
            region_correct[region] += int(good)
            reference_occ_counts[ref] += 1
            predicted_occ_counts[pred] += 1
            matched_occ += int(good)
            prefix_span[h] += len(seq) - pos
            if still_matching and good:
                strict_matched += 1
            else:
                still_matching = False

    total_hashes = len(reference)
    total_occ = sum(warm_freq.values())
    correct_hashes = sum(reference[h] == predicted[h] for h in reference)
    total_spans = sum(prefix_span.values())
    repeated_excess = sum(max(0, n-1) for n in warm_freq.values())
    matched_repeat_excess = sum(max(0, n-1) for h,n in warm_freq.items()
                                if reference[h] == predicted[h])

    def pct(num, den):
        return round(num * 100 / den, 3) if den else None

    result = {
        'hash_count': total_hashes,
        'warm_occurrences': total_occ,
        'overall_unique_pct': pct(correct_hashes, total_hashes),
        'warm_occurrence_weighted_pct': pct(matched_occ, total_occ),
        'warm_repeat_excess_weighted_pct': pct(matched_repeat_excess, repeated_excess),
        'prefix_span_weighted_pct': pct(sum(prefix_span[h] for h in reference
                                           if reference[h] == predicted[h]), total_spans),
        'strict_contiguous_prefix_agreement_pct': pct(strict_matched, total_occ),
    }
    for tier in TIERS:
        n = sum(x == tier for x in reference.values())
        tp = sum(reference[h] == tier and predicted[h] == tier for h in reference)
        result[f'{tier}_recall_unique_pct'] = pct(tp, n)
        sel = sum(x == tier for x in predicted.values())
        result[f'{tier}_precision_unique_pct'] = pct(tp, sel)
        result[f'{tier}_hashes'] = sel
    for name, target in target_region.items():
        key = name.replace('-', '_').replace('+','plus')
        result[f'region_{key}_exact_pct'] = pct(region_correct[name], region_count[name])
    for name, lo, hi, tier in (('gpu_pos01',0,1,'gpu'),
                               ('cpu_pos23',2,3,'cpu'),
                               ('disk_pos45',4,5,'disk'),
                               ('drop_pos6plus',6,10**6,'drop'),
                               ('cpu_pos45',4,5,'cpu'),
                               ('cpu_pos6plus',6,10**6,'cpu')):
        denominator = sum(r['total'] for pos,r in by_pos.items() if lo <= pos <= hi)
        numerator = sum(r['pred'][tier] for pos,r in by_pos.items() if lo <= pos <= hi)
        result[name+'_pct'] = pct(numerator, denominator)
    region_values = [result[k] for k in
       ('gpu_pos01_pct','cpu_pos23_pct','disk_pos45_pct','drop_pos6plus_pct')
       if result[k] is not None]
    result['balanced_static_region_pct'] = round(sum(region_values)/len(region_values), 3) if region_values else None
    result['late_cpu_pos4plus_pct'] = pct(
        sum(r['pred']['cpu'] for pos,r in by_pos.items() if pos >= 4),
        sum(r['total'] for pos,r in by_pos.items() if pos >= 4))
    result['voting_counts'] = dict(Counter(reference.values()))
    result['confusion_unique'] = {tier: dict(Counter(predicted[h] for h in reference
                                                   if reference[h] == tier)) for tier in TIERS}
    positions = []
    for pos, rec in sorted(by_pos.items()):
        r = {'position': pos, 'occurrences': rec['total'],
             'exact_agreement_pct': pct(rec['matched'], rec['total'])}
        for tier in TIERS:
            r[f'voting_{tier}_count'] = rec['ref'][tier]
            r[f'bayes_{tier}_count'] = rec['pred'][tier]
        positions.append(r)
    return result, positions


def parse_pseudo_presets(value):
    presets = []
    for item in value.split(',') if value.strip() else []:
        parts = item.strip().split(':')
        if len(parts) != 3:
            raise ValueError('pseudo preset format: label:score_prior_count:cost_prior_count')
        label, score, cost = parts
        if not label or not label.replace('_','').isalnum():
            raise ValueError('invalid pseudo preset label')
        score, cost = float(score), float(cost)
        if not all(math.isfinite(v) and v > 0 for v in (score,cost)):
            raise ValueError('pseudo counts must be positive finite')
        presets.append((label,score,cost))
    if len({x[0] for x in presets}) != len(presets):
        raise ValueError('pseudo preset names must be unique')
    return presets


def scenario_specs(gpus, cpus, disks, weights, pseudo_presets):
    specs = []
    for g in gpus:
        for c in cpus:
            for d in disks:
                for w in weights:
                    specs.append({'gpu_gb':g,'cpu_gb':c,'disk_gb':d,
                                  'posterior_mode':'fixed','prior_weight':w,
                                  'score_prior_count':None,'cost_prior_count':None,
                                  'scenario_name':f'fixed_w{w:g}'})
                for label,a,b in pseudo_presets:
                    specs.append({'gpu_gb':g,'cpu_gb':c,'disk_gb':d,
                                  'posterior_mode':'pseudo_count','prior_weight':None,
                                  'score_prior_count':a,'cost_prior_count':b,
                                  'scenario_name':'pseudo_'+label})
    return specs


def pareto_mark(rows):
    """Nondominated in 4 static-region matches and minimizing late CPU admission.

    Diagnostic only; the 'frontier' is not an optimization of E2E.
    """
    keys=('gpu_pos01_pct','cpu_pos23_pct','disk_pos45_pct',
          'drop_pos6plus_pct')
    for r in rows:
        r['pareto_region_front'] = True
        for other in rows:
            if other is r:
                continue
            if (all(other[k]>=r[k] for k in keys) and
                other['late_cpu_pos4plus_pct']<=r['late_cpu_pos4plus_pct'] and
                (any(other[k]>r[k] for k in keys) or
                 other['late_cpu_pos4plus_pct']<r['late_cpu_pos4plus_pct'])):
                r['pareto_region_front'] = False
                break


def write_csv(path, rows, fields=None):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(rows[0])
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='.',help='source checkout containing patched bayesian_mapper.py')
    p.add_argument('--run',required=True,help='original-order 29886 reference or validated minimal ZIP')
    p.add_argument('--prior',required=True,help='same expected-ordinal score + H100 hardware prior used for original replay')
    p.add_argument('--voting',default=str(Path(__file__).with_name('voting_29756_hash_tiers.json')))
    p.add_argument('--gpu-gb',default='10,15,20,25,30')
    p.add_argument('--cpu-gb',default='225')
    p.add_argument('--disk-gb',default='300,400,500,600,700')
    p.add_argument('--fixed-weights',default='1,0.5,0.2')
    p.add_argument('--pseudo-presets',default='low:10:1,high:1000:100',
                   help='name:score_prior_count:cost_prior_count, comma-separated')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--validate-reference',action='store_true',
                   help='first verify w=1 reproduces all historical waves AND exact 29886 hash decisions')
    p.add_argument('--baseline-map',default=str(Path(__file__).with_name('bayes_29886_observed_hash_tiers.json')),
                   help='observed 29886 map used to verify exact per-hash mapper reproduction')
    p.add_argument('--observed-map',help='optional observed hash map to compare ONE matching scenario')
    p.add_argument('--observed-gpu-gb',type=float)
    p.add_argument('--observed-cpu-gb',type=float)
    p.add_argument('--observed-disk-gb',type=float)
    p.add_argument('--observed-fixed-weight',type=float,default=1)
    args=p.parse_args(argv)
    run=RunArchive(args.run).load()
    voting=labels_from_file(args.voting)
    warm_sequences=make_occurrences(run)
    # Do not partially report when hash identities differ.
    unique_hashes={h for seq in warm_sequences for h in seq}
    if unique_hashes != set(voting):
        raise ValueError(f'voting hash set differs from replay: {len(unique_hashes)} vs {len(voting)}')
    Mapper=load_mapper(args.repo)
    prior=json.loads(Path(args.prior).read_text())
    if prior.get('score_format') != 'expected_ordinal_0_100':
        raise ValueError('wrong prior score_format; expected expected_ordinal_0_100')
    if args.validate_reference:
        statuses, windows=replay_waves(run,args.prior,Mapper,
            gpu_gb=float(run.config['bayes_gpu_gb']),
            cpu_gb=float(run.config['l1_gb']),
            disk_gb=float(run.config['bayes_disk_gb']),
            weight=float(run.config['bayes_prior_weight']))
        diffs=reference_diff(statuses,run.ref_waves)
        if not all(d['mapped_abs_count_error']==0 and d['score_samples_error']==0
                   and d['new_hashes_error']==0 and d['max_band_error']<1e-9 for d in diffs):
            raise RuntimeError('reference mapper failed 14-wave reproduction')
        baseline=labels_from_file(args.baseline_map)
        assigned=windows[-1][2]
        if baseline != assigned:
            disagree=sum(baseline.get(h)!=assigned.get(h) for h in set(baseline)|set(assigned))
            raise RuntimeError(f'historical 29886 final per-hash mapping mismatch: {disagree} hashes')
        print('Validated historical reference: all 14 waves and 4,857 exact hash tiers match')

    specs=scenario_specs(parse_grid(args.gpu_gb),parse_grid(args.cpu_gb),
        parse_grid(args.disk_gb),parse_grid(args.fixed_weights,positive=False),
        parse_pseudo_presets(args.pseudo_presets))
    if not specs:
        raise ValueError('no scenarios selected')
    summaries, positions=[] , []
    observed_report=[]
    observed=labels_from_file(args.observed_map) if args.observed_map else None
    observed_param_list=(args.observed_gpu_gb,args.observed_cpu_gb,args.observed_disk_gb)
    if observed and any(x is None for x in observed_param_list):
        raise ValueError('observed comparison requires --observed-gpu-gb, --observed-cpu-gb and --observed-disk-gb')

    for j,s in enumerate(specs,1):
        print(f'[{j}/{len(specs)}] {s["scenario_name"]} GPU{s["gpu_gb"]:g} CPU{s["cpu_gb"]:g} Disk{s["disk_gb"]:g}', flush=True)
        statuses, windows=replay_waves(run,args.prior,Mapper,
            gpu_gb=s['gpu_gb'],cpu_gb=s['cpu_gb'],disk_gb=s['disk_gb'],
            weight=s['prior_weight'] if s['posterior_mode']=='fixed' else 1,
            posterior_mode=s['posterior_mode'],
            score_prior_count=s['score_prior_count'] or 10,
            cost_prior_count=s['cost_prior_count'] or 5)
        chosen=windows[-1][2]
        measures,bypos=evaluate(voting,chosen,warm_sequences)
        fields={'scenario_index':j,**s,**{k:v for k,v in measures.items() if not isinstance(v,dict)}}
        summaries.append(fields)
        positions.extend({'scenario_index':j,**s,**rec} for rec in bypos)
        if (observed and s['posterior_mode']=='fixed'
             and all(abs(s[k]-t)<1e-8 for k,t in zip(
                 ('gpu_gb','cpu_gb','disk_gb','prior_weight'),
                 (*observed_param_list,args.observed_fixed_weight)))):
            if set(chosen)!=set(observed):
                raise ValueError('observed hash set not identical')
            equal=sum(chosen[h]==observed[h] for h in chosen)
            observed_report.append({'scenario_index':j,**s,
                'counterfactual_vs_actual_exact_hash_pct':round(100*equal/len(chosen),3),
                'warning':'Fixed 29886 cost feedback, unlike live serving observations'})
    pareto_mark(summaries)
    output=Path(args.output_dir)
    write_csv(output/'placement_agreement_sweep.csv',summaries)
    write_csv(output/'agreement_by_position.csv',positions)
    write_csv(output/'pareto_region_candidates.csv',[r for r in summaries if r['pareto_region_front']])
    write_csv(output/'top_by_balanced_regions.csv',sorted(summaries,
      key=lambda r:(r['balanced_static_region_pct'],r['warm_occurrence_weighted_pct']),
      reverse=True)[:min(25,len(summaries))])
    write_csv(output/'top_by_exact_voting_hash.csv',sorted(summaries,
      key=lambda r:(r['overall_unique_pct'],r['warm_occurrence_weighted_pct']),
      reverse=True)[:min(25,len(summaries))])
    if observed_report:
        write_csv(output/'observed_placement_comparison.csv',observed_report)
    info={'source_run':str(args.run),'voting_target':str(args.voting),
          'scenario_count':len(summaries),'distinct_hashes':len(voting),
          'warm_occurrences':sum(map(len,warm_sequences)),
          'pseudo_note':'Counts act on frozen 29886 score+cost observation streams; traffic feedback is NOT simulated.',
          'limits':'Final preference comparisons ONLY. Not L1 occupancy, actual scheduler useful KV or E2E; all offline parameters were evaluated on the medical 650-question workload.',
          'reference_verified':bool(args.validate_reference)}
    output.mkdir(parents=True,exist_ok=True)
    (output/'run_metadata.json').write_text(json.dumps(info,indent=2)+'\n')
    print('\nWrote',len(summaries),'scenarios to',output)
    for metric in ('overall_unique_pct','warm_occurrence_weighted_pct','balanced_static_region_pct'):
        top=max(summaries,key=lambda r:r[metric])
        print('Highest',metric,'=',top[metric],':',top['scenario_name'],
              f'GPU{top["gpu_gb"]:g}/CPU{top["cpu_gb"]:g}/Disk{top["disk_gb"]:g}')
    print('These metrics are placement diagnostics, NOT E2E or scheduler-consumed hit predictions.')
    return summaries


if __name__=='__main__':
    main()

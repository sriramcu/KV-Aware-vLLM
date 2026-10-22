#!/usr/bin/env python3
"""Offline GNN head/vote/aggregation analysis + exact historical Bayesian wave replay.

No GPU, vLLM server, LMCache server, model inference or physical KV needed.
Designed for the prediction-only Q650 L40 job and historical job 29862.
Honest distinctions: raw tier preferences != physical cache residency/useful hits;
matched-population top-k is a hindsight diagnostic, NOT deployable placement.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CLASSES = ("drop", "disk", "cpu", "gpu")
TIER_VALUE = {v: i for i, v in enumerate(CLASSES)}
FIELDS = ("mean_rank", "top8_rank", "top16_rank", "max_rank", "class_expected_mean", "class_gpu_prob_mean")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def csv_write(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def softmax(v):
    m=max(v); e=[math.exp(x-m) for x in v]; s=sum(e)
    return [x/s for x in e]


def pearson(x,y):
    if len(x)!=len(y) or not x:
        return None
    mx=statistics.fmean(x); my=statistics.fmean(y)
    a=sum((i-mx)*(j-my) for i,j in zip(x,y))
    b=sum((i-mx)**2 for i in x)*sum((j-my)**2 for j in y)
    return a/math.sqrt(b) if b>0 else None


def rankdata(a):
    vals=sorted(enumerate(a),key=lambda t:t[1]); out=[0.]*len(a);i=0
    while i<len(vals):
        j=i+1
        while j<len(vals) and vals[j][1]==vals[i][1]:j+=1
        r=(i+j-1)*.5
        for k in range(i,j):out[vals[k][0]]=r
        i=j
    return out


def spearman(x,y):
    return pearson(rankdata(x),rankdata(y))


def vote_counts(classes, cpu_top2, gpu_min):
    n=Counter(classes)
    # Mirror threshold_with_cpu_top2 and threshold_with_drop_plurality.
    # The latter was used in Bayesian launcher but not necessarily in historical GNN.
    if n['gpu']>=gpu_min:
        return ('gpu','gpu','gpu')
    top2=sum(cpu_top2)
    basic='cpu' if n['cpu']>=2 else 'disk'
    cpu2='cpu' if n['cpu']>=2 or top2>=6 else 'disk'
    drop='cpu' if n['cpu']>=2 else ('drop' if n['drop']>max(n['disk'],n['cpu'],n['gpu']) else 'disk')
    return (basic,cpu2,drop)


def unique_rows(rows):
    """First-seen hash scoring, same identity population used by comparisons."""
    d={}
    for r in rows:
        d.setdefault(r['chunk_hash'],r)
    return d


def matched_selection(rows, source_field, vote_field):
    """Equal unique-hash GPU and GPU+CPU population; identity effects only.

    The target counts are from first-seen historical votes, rather than the
    raw occurrence-level population. This is an offline oracle-sized control.
    """
    unique=unique_rows(rows)
    items=list(unique.values())
    target_gpu={r['chunk_hash'] for r in items if r[vote_field]=='gpu'}
    target_cpu={r['chunk_hash'] for r in items if r[vote_field]=='cpu'}
    ordered=sorted(items,key=lambda r:(-r[source_field], r['chunk_hash']))
    candidates_gpu={r['chunk_hash'] for r in ordered[:len(target_gpu)]}
    candidates_cpu={r['chunk_hash'] for r in ordered[len(target_gpu):len(target_gpu)+len(target_cpu)]}
    intersection_gpu=len(candidates_gpu & target_gpu)
    intersection_cpu=len(candidates_cpu & target_cpu)
    union_gpu=len(candidates_gpu | target_gpu)
    return {
      'unique_hashes':len(unique), 'gpu_target_count':len(target_gpu),
      'cpu_target_count':len(target_cpu),
      'gpu_overlap':intersection_gpu,
      'gpu_recall_at_matched_population':intersection_gpu/len(target_gpu) if target_gpu else None,
      'gpu_jaccard':intersection_gpu/union_gpu if union_gpu else None,
      'cpu_overlap':intersection_cpu,
      'cpu_recall_at_matched_population':intersection_cpu/len(target_cpu) if target_cpu else None,
      'cpu_jaccard':intersection_cpu/len(candidates_cpu|target_cpu) if candidates_cpu or target_cpu else None,
      'gpu_by_position':dict(Counter(r['chunk_index'] for r in rows if r['chunk_hash'] in candidates_gpu)),
      'cpu_by_position':dict(Counter(r['chunk_index'] for r in rows if r['chunk_hash'] in candidates_cpu)),
      'gpu_hashes':candidates_gpu, 'cpu_hashes':candidates_cpu,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--heads',required=True)
    p.add_argument('--scores',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--reference-run-dir',default='')
    p.add_argument('--prior-file',default='')
    p.add_argument('--gpu-gb',type=float,default=2)
    p.add_argument('--cpu-gb',type=float,default=64)
    p.add_argument('--disk-gb',type=float,default=100)
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    heads=read_jsonl(args.heads); scores=read_jsonl(args.scores)
    score_by_source={int(r['source_index']):r['chunks'] for r in scores}
    if len(score_by_source)!=len(scores) or len(heads)!=len(scores):
        raise RuntimeError('wrong count or duplicate request identifiers')
    rows=[]; block_rank=[]; block_class_expected=[]
    for request in heads:
        source=int(request['source_index'])
        chunks=score_by_source[source]
        rank=request['block_rank_scores']; logits=request['block_class_logits']
        nblocks=int(request['blocks_per_chunk']);
        assert nblocks==32 and request['native_block_tokens']==16
        assert len(rank)==len(logits)==request['valid_native_blocks']
        assert len(chunks)==len(request['chunk_hashes'])
        assert len(chunks)*nblocks<=len(rank)
        for j, chunk in enumerate(chunks):
            lo=j*nblocks; hi=lo+nblocks
            rr=rank[lo:hi]; ll=logits[lo:hi]
            assert chunk['chunk_hash']==request['chunk_hashes'][j]
            mean=statistics.fmean(rr)
            assert abs(mean-chunk['score'])<1e-5, (source,j,mean,chunk['score'])
            probabilities=[softmax(v) for v in ll]
            classes=[CLASSES[max(range(4),key=lambda k:v[k])] for v in ll]
            cpu2=[2 in sorted(range(4),key=lambda k:-v[k])[:2] for v in ll]
            e=[sum(i*x[i] for i in range(4)) for x in probabilities]
            block_rank+=rr;block_class_expected+=e
            basic8,top8,drop8=vote_counts(classes,cpu2,8)
            basic22,top22,drop22=vote_counts(classes,cpu2,22)
            rows.append({
              'source_index':source,'request_order_index':int(request['request_order_index']),
              'chunk_index':j,'chunk_hash':chunk['chunk_hash'],
              'mean_rank':chunk['score'],
              'top8_rank':statistics.fmean(sorted(rr,reverse=True)[:8]),
              'top16_rank':statistics.fmean(sorted(rr,reverse=True)[:16]),
              'max_rank':max(rr),
              'class_expected_mean':statistics.fmean(e),
              'class_gpu_prob_mean':statistics.fmean(pv[3] for pv in probabilities),
              'gpu_class_count':classes.count('gpu'),
              'cpu_class_count':classes.count('cpu'),
              'disk_class_count':classes.count('disk'),
              'drop_class_count':classes.count('drop'),
              'cpu_top2_count':sum(cpu2),
              'vote_t8_threshold':basic8,'vote_t22_threshold':basic22,
              'vote_t8_cpu_top2':top8,'vote_t22_cpu_top2':top22,
              'vote_t8_drop_plurality':drop8,'vote_t22_drop_plurality':drop22,
            })
    if not rows:raise RuntimeError('no complete 512-token chunks')
    unique=unique_rows(rows)
    position=[]
    for pos in sorted({r['chunk_index'] for r in rows}):
        rr=[r for r in rows if r['chunk_index']==pos]
        pr={'chunk_index':pos,'occurrences':len(rr),
            'unique_hashes':len({r['chunk_hash'] for r in rr})}
        for k in FIELDS:pr[f'{k}_mean']=statistics.fmean(r[k] for r in rr)
        for vote in ('vote_t8_threshold','vote_t22_threshold','vote_t8_cpu_top2','vote_t22_cpu_top2','vote_t22_drop_plurality'):
            for tier in CLASSES:pr[f'{vote}_{tier}_count']=sum(r[vote]==tier for r in rr)
        position.append(pr)
    csv_write(out/'head_chunk_occurrences.csv', rows)
    csv_write(out/'position_profiles.csv', position)
    cm={}
    for vote in ('vote_t8_threshold','vote_t22_threshold','vote_t8_cpu_top2','vote_t22_cpu_top2','vote_t22_drop_plurality'):
        for method in FIELDS:
            v=matched_selection(rows,method,vote)
            v.pop('gpu_hashes');v.pop('cpu_hashes')
            cm[f'{vote}__{method}']=v
    write_json(out/'matched_population_ablations.json',cm)
    # Same-head 'aggregation' vs cross-head change, plus overall head agreement.
    correl={}
    for score in FIELDS:
        correl[score]={
          'against_vote_t22_cpu_top2_spearman':spearman([r[score] for r in rows],[TIER_VALUE[r['vote_t22_cpu_top2']] for r in rows]),
          'against_gpu_vote_count_spearman':spearman([r[score] for r in rows],[r['gpu_class_count'] for r in rows]),
          'against_class_expected_mean_spearman':spearman([r[score] for r in rows],[r['class_expected_mean'] for r in rows]),
        }
    correl['block_raw_rank_vs_class_expected']={
       'pearson':pearson(block_rank,block_class_expected),
       'spearman':spearman(block_rank,block_class_expected),
    }
    write_json(out/'head_agreement_correlations.json',correl)
    provenance={
      'num_requests':len(heads),'chunk_occurrences':len(rows),'unique_prefix_hashes':len(unique),
      'first_two_chunk_mean_rank':{str(i):statistics.fmean(r['mean_rank'] for r in rows if r['chunk_index']==i) for i in (0,1)},
      'note':'All threshold voting and matched-population results are offline counterfactuals. Historical voting policy must be checked against historical run config. No E2E or physical-cache metrics are inferred.',
    }
    if args.reference_run_dir:
        ref=Path(args.reference_run_dir)
        if not ref.is_dir():raise FileNotFoundError(ref)
        old_scores_path=ref/'placement/gnn_chunk_scores.jsonl'
        old_scores=read_jsonl(old_scores_path)
        olds={int(r['source_index']):r['chunks'] for r in old_scores}
        if set(olds)!=set(score_by_source):
            raise RuntimeError('reference source indices do not match the new prediction run')
        different_hashes=[]; score_abs_diff=[]
        for sid, new in score_by_source.items():
            old=olds[sid]
            if len(old)!=len(new):different_hashes.append((sid,'length'));continue
            for i,(n,o) in enumerate(zip(new,old)):
                if n['chunk_hash']!=o['chunk_hash']:different_hashes.append((sid,i))
                else:score_abs_diff.append(abs(n['score']-o['score']))
        provenance['historical_29862_join']={
          'reference_dir':str(ref),
          'hash_mismatches':len(different_hashes),
          'examples':different_hashes[:10],
          'max_rank_score_absolute_difference':max(score_abs_diff,default=None),
        }
        if different_hashes:
            raise RuntimeError(f"Reference hash mismatch: {different_hashes[:5]}; no valid comparison")
        if not args.prior_file:
            raise RuntimeError('--prior-file required for Bayesian wave reconstruction')
        replay_bayesian(args,rows,score_by_source,ref,out,provenance)
    write_json(out/'analysis_summary.json',provenance)
    print('[PREDICTION_HEAD_ANALYSIS_COMPLETE]',json.dumps(provenance,sort_keys=True),flush=True)


def replay_bayesian(args,rows,score_by_source,ref,out,summary):
    # Reconstruct score histogram updates, tier costs, prefix closure, within-wave
    # fixed-hash consistency, and readmission using the source mapper itself.
    # Inject recorded posterior costs (exogenous historical environment) to
    # isolate ranking/mapper without pretending counterfactual costs are causal.
    from Hierarchical_KV.shortq_placement.bayesian_mapper import WaveBayesianMapper
    prior=json.loads(Path(args.prior_file).read_text())
    history=json.loads((ref/'results/bayes_wave_history.json').read_text())
    config=json.loads((ref/'run_config.json').read_text())
    if config['passage_order_policy']!='original' or config['request_order']!='seeded_shuffle':
        raise RuntimeError('unexpected reference prompt/order configuration')
    cold=read_jsonl(ref/'results/cold_results.jsonl')
    warm=read_jsonl(ref/'results/warm_results_execution_order.jsonl')
    cold_sources=[int(r['source_index']) for r in cold]
    warm_sources=[int(r['source_index']) for r in warm]
    if len(cold_sources)!=len(score_by_source) or len(warm_sources)!=len(score_by_source):
        raise RuntimeError('reference wave request count mismatch')
    if len(set(cold_sources))!=len(cold_sources) or len(set(warm_sources))!=len(warm_sources):
        raise RuntimeError('reference source indices nonunique')
    ref_budget=config.get('bayes_gpu_gb',args.gpu_gb),config.get('bayes_cpu_gb',args.cpu_gb),config.get('bayes_disk_gb',args.disk_gb)
    if any(abs(float(a)-b)>1e-7 for a,b in zip(ref_budget,(args.gpu_gb,args.cpu_gb,args.disk_gb))):
        print('[WARN] Mapper capacity differs from reference; reproduction may fail',ref_budget,file=sys.stderr)
    mapper=WaveBayesianMapper(args.prior_file, 1.0,args.gpu_gb,args.cpu_gb,args.disk_gb,
                             args.scores,out/'replay_runtime_metadata.json',out/'replay_vpc.json',readmission=True)
    bysource=defaultdict(dict)
    for r in rows:bysource[r['source_index']][r['chunk_index']]=r
    manifest=json.loads((ref/'results/manifest.json').read_text())
    n_b=int(manifest['submission_batch_size'])
    groups=[('cold',cold_sources[i:i+n_b]) for i in range(0,len(cold_sources),n_b)]
    groups += [('warm',warm_sources[i:i+n_b]) for i in range(0,len(warm_sources),n_b)]
    if len(groups)!=len(history):
        raise RuntimeError('reference wave number mismatch')
    decisions=[];wave_audits=[];threshold_rows=[];prev_sources=None
    for (phase,sources),hist in zip(groups,history):
        if phase!=hist['phase'] or len(sources)!=hist['requests']:
            raise RuntimeError('reference wave phase/order mismatch')
        if prev_sources is not None:mapper.observe_scores(prev_sources)
        # Match the prior/observed cost posterior and counts in historical
        # BEFORE-wave snapshots. This assumes fixed w=1 used in job 29862.
        for tier in ('cpu','disk','recompute'):
            n=float(hist['cost_sample_counts'].get(tier,0))
            posterior=float(hist['cost_estimate_ms'][tier]);p0=mapper.prior_ms[tier]
            if n:
                implied_observed_mean=2*posterior-p0
                if implied_observed_mean<=0:raise RuntimeError('invalid inferred observation mean')
                mapper.cost_counts[tier]=n
                mapper.cost_sum_ms[tier]=implied_observed_mean*n
            else:
                mapper.cost_counts[tier]=mapper.cost_sum_ms[tier]=0
        bands=mapper.bands(); fixed={} # readmission=True resets per-wave constraints
        # Inverse of the piecewise-uniform prior/observed histogram CDF:
        # a *model-implied* cutoff, not a true raw-score quantile.
        posterior_bins=mapper.probabilities()
        def invert_cdf(pct):
            pct=max(0.,min(1.,pct)); cumul=0.
            for b, mass in enumerate(posterior_bins):
                if mass>0 and pct<=cumul+mass+1e-12:
                    alpha=max(0.,min(1.,(pct-cumul)/mass))
                    return mapper.edges[b]+alpha*(mapper.edges[b+1]-mapper.edges[b])
                cumul+=mass
            return mapper.edges[-1]
        threshold_rows.append({
            'epoch':hist['epoch'],'phase':phase,
            'cutoff_gpu_raw_score':invert_cdf(bands['gpu']),
            'cutoff_cpu_raw_score':invert_cdf(bands['cpu']),
            'cutoff_disk_raw_score':invert_cdf(bands['disk']),
            'percentile_gpu':bands['gpu'],'percentile_cpu':bands['cpu'],
            'percentile_disk':bands['disk'],
            'share_gpu':bands['fractions']['gpu'],
            'share_cpu':bands['fractions']['cpu'],
            'share_disk':bands['fractions']['disk'],
            'share_drop':bands['fractions']['drop'],
            'cost_cpu_ms':mapper.estimated_cost('cpu'),
            'cost_disk_ms':mapper.estimated_cost('disk'),
            'cost_recompute_ms':mapper.estimated_cost('recompute'),
            'cpu_to_recompute':mapper.estimated_cost('cpu')/mapper.estimated_cost('recompute'),
            'disk_to_recompute':mapper.estimated_cost('disk')/mapper.estimated_cost('recompute'),
        })
        prior_assign=dict(mapper.assigned)
        for sid in sources:
            seq=mapper.rows[sid];hashes=[h for h,_ in seq]
            percentiles=[mapper.percentile(s) for _,s in seq]
            proposals=[mapper._tier(p,bands) for p in percentiles]
            final,cutoff,feasible=mapper._prefix_closure(proposals,percentiles,hashes,fixed,bands)
            for j,((h,s),pct,proposed,closure_tier) in enumerate(zip(seq,percentiles,proposals,final)):
                if h not in fixed:fixed[h]=closure_tier
                d={'phase':phase,'epoch':hist['epoch'],'source_index':sid,'chunk_index':j,
                    'chunk_hash':h,'mean_rank_score':s,'percentile':pct,
                    'raw_bayes_tier':proposed,'closure_proposal':closure_tier,
                    'published_preference':fixed[h],
                    'previous_preference':prior_assign.get(h),
                    'closure_cutoff':cutoff,'closure_feasible':feasible,
                    'vote_t22_cpu_top2':bysource[sid][j]['vote_t22_cpu_top2'],
                    'vote_t22_threshold':bysource[sid][j]['vote_t22_threshold']}
                decisions.append(d)
        actual=mapper.publish_wave(sources,phase)
        threshold_diff={t:abs(actual['thresholds'][t]-hist['thresholds'][t]) for t in ('gpu','cpu','disk')}
        mapped_hist={t:int(hist['mapped_counts'].get(t,0)) for t in CLASSES}
        mapped_sim={t:int(actual['mapped_counts'].get(t,0)) for t in CLASSES}
        audit={'epoch':hist['epoch'],'phase':phase,'max_threshold_abs_diff':max(threshold_diff.values()),
               'historical_counts':mapped_hist,'replayed_counts':mapped_sim,
               'counts_match':mapped_hist==mapped_sim,
               'score_samples':actual['score_samples'],
               'historical_score_samples':hist['score_samples']}
        wave_audits.append(audit)
        prev_sources=sources
    actual_path=ref/'placement/runtime_hash_to_placement.json'
    if actual_path.exists():
        historical=json.loads(actual_path.read_text())
        changed=[h for h in historical if mapper.assigned.get(h)!=historical[h]]
        summary['bayesian_replay_vs_runtime']={
           'final_metadata_mismatch_hashes':len(changed),
           'first_mismatches':changed[:10],
           'wave_count_mismatches':sum(not r['counts_match'] for r in wave_audits),
           'max_threshold_diff':max(r['max_threshold_abs_diff'] for r in wave_audits),
        }
    with (out/'historical_bayes_decisions_replay.jsonl').open('w') as f:
        for row in decisions:f.write(json.dumps(row,sort_keys=True)+'\n')
    write_json(out/'historical_bayes_wave_replay_audit.json',wave_audits)
    csv_write(out/'bayes_thresholds_and_raw_cutoffs_by_wave.csv',threshold_rows)
    confusion=defaultdict(Counter)
    for r in decisions:
        k=(r['epoch'],r['chunk_index'],r['vote_t22_cpu_top2'],r['raw_bayes_tier'])
        confusion[k]['n']+=1
    csv_write(out/'historical_vote_vs_bayes_raw_by_epoch_position.csv',[
       {'epoch':k[0],'chunk_index':k[1], 'vote_t22_cpu_top2':k[2], 'raw_bayes_tier':k[3], 'count':v['n']}
       for k,v in sorted(confusion.items())])
    # Separate BEFORE mapper (raw percentile tier), AFTER closure, and AFTER
    # hash consistency; count changes, and show per-position preferences.
    summary['bayes_stage_changes']={
      'raw_to_closure_changes':sum(r['raw_bayes_tier']!=r['closure_proposal'] for r in decisions),
      'closure_to_published_changes':sum(r['closure_proposal']!=r['published_preference'] for r in decisions),
      'raw_to_published_changes':sum(r['raw_bayes_tier']!=r['published_preference'] for r in decisions),
    }
    for f in ('raw_bayes_tier','closure_proposal','published_preference'):
        summary['bayes_stage_changes'][f+'_by_position']={str(i):dict(Counter(r[f] for r in decisions if r['chunk_index']==i)) for i in range(10)}

if __name__=='__main__':
    main()

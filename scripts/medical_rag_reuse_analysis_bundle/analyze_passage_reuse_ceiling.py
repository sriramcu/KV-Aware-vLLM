#!/usr/bin/env python3
"""Tokenization-free, passage-level reuse potential for RAG passage reordering.

Models whole-passage reuse under an *ideal passage-aligned, unbounded cache*.
Hashes entire passage texts, then matches only identical ORDERED passage prefixes.
This is NOT an LMCache key simulator, a token/KV-hit estimator, or padding-cost model.

Example:
  python analyze_passage_reuse_ceiling.py --retrieval-json "$PROMPT_RECORDS" \
      --max-questions 650 --topk 5 --output-dir "$OUTPUT_ROOT/passage_ceiling"
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import random
from pathlib import Path
from statistics import mean


def sha_id(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def load(path, max_questions, topk):
    p=Path(path)
    if p.suffix.lower()=='.jsonl':
        with p.open(encoding='utf-8') as f:
            data=[json.loads(line) for line in f if line.strip()]
    else:
        data=json.loads(p.read_text(encoding='utf-8'))
    if isinstance(data,dict):
        for name in ('records','requests','data'):
            if isinstance(data.get(name),list):
                data=data[name];break
    if not isinstance(data,list):
        raise ValueError('Expected a JSON list or JSONL records')
    data=data[:max_questions] if max_questions else data
    result=[]
    for i,r in enumerate(data):
        passages=r.get('sorted_passage')
        if passages is None: passages=r.get('passages',r.get('retrieved_passages'))
        if not isinstance(passages,list) or len(passages)<topk or not all(isinstance(s,str) for s in passages[:topk]):
            raise ValueError(f'Record {i} must contain >= {topk} string sorted_passage entries')
        texts=passages[:topk]
        ids=tuple(sha_id(x) for x in texts)
        result.append(dict(source_index=i,request_id=str(r.get('id',i)),ids=ids,
                           characters={j:len(x) for j,x in zip(ids,texts)}))
    if not result: raise ValueError('No retrieval records loaded')
    return result


def match_prefix(trie, order, chars):
    node=trie
    matched=0
    reused_chars=0
    for x in order:
        if x not in node: break
        node=node[x]
        matched+=1
        reused_chars+=chars[x]
    return matched,reused_chars


def insert(trie,order):
    node=trie
    for x in order:
        node=node.setdefault(x,{})


def choose(ids,chars,trie,strategy):
    if strategy=='original':
        return ids,match_prefix(trie,ids,chars),1
    if strategy=='canonical':
        seq=tuple(sorted(ids))
        return seq,match_prefix(trie,seq,chars),1
    top1fixed=strategy=='optimal_keep_top1'
    # Choose maximal REUSED CONTENT (characters) rather than simply passage count;
    # the two agree when the passage lengths are similar. Tie-break by passage count,
    # then prefer original retrieval order.
    best=ids
    best_match=match_prefix(trie,ids,chars)
    checked=0
    positions=tuple(range(len(ids)))
    all_positions=( (0,)+tail for tail in itertools.permutations(positions[1:]) ) if top1fixed else itertools.permutations(positions)
    for perm in all_positions:
        checked+=1
        seq=tuple(ids[j] for j in perm)
        result=match_prefix(trie,seq,chars)
        if (result[1],result[0])>(best_match[1],best_match[0]):
            best,best_match=seq,result
    return best,best_match,checked


def simulate(records,seq,strategy,progress_every=0):
    trie={}
    rows=[]
    for step,ix in enumerate(seq):
        r=records[ix]
        order,(matched,reused_chars),evaluated=choose(r['ids'],r['characters'],trie,strategy)
        total_chars=sum(r['characters'][x] for x in order)
        rows.append(dict(strategy=strategy,step=step,source_index=ix,request_id=r['request_id'],
                         matched_leading_passages=matched,reused_passage_chars=reused_chars,
                         total_passage_chars=total_chars,passage_count=len(order),prefix_fraction=matched/len(order),
                         reused_char_fraction=(reused_chars/total_chars if total_chars else 0),
                         permuted=int(order!=r['ids']),candidates_evaluated=evaluated))
        insert(trie,order)
        if progress_every and (step+1)%progress_every==0:
            print(f'[{strategy}] {step+1}/{len(seq)}',flush=True)
    return rows


def independent_upper(records,seq):
    seen=set()
    rows=[]
    for step,ix in enumerate(seq):
        r=records[ix]
        count=sum(x in seen for x in r['ids'])
        chars=sum(r['characters'][x] for x in r['ids'] if x in seen)
        total=sum(r['characters'][x] for x in r['ids'])
        rows.append(dict(step=step,source_index=ix,previously_seen_passages=count,
                         previously_seen_chars=chars,total_passage_chars=total,
                         independently_seen_fraction=count/len(r['ids'])))
        seen.update(r['ids'])
    return rows,len(seen)


def summarize(rows,strategy):
    n=len(rows)
    return dict(strategy=strategy,requests=n,
        total_passage_occurrences=sum(r['passage_count'] for r in rows),
        mean_matched_leading_passages=mean(r['matched_leading_passages'] for r in rows),
        requests_with_at_least_one_reused_passage=sum(r['matched_leading_passages']>0 for r in rows)/n,
        requests_with_all_passages_reused=sum(r['prefix_fraction']==1 for r in rows)/n,
        fraction_passage_occurrences_reused=sum(r['matched_leading_passages'] for r in rows)/sum(r['passage_count'] for r in rows),
        fraction_passage_characters_reused=sum(r['reused_passage_chars'] for r in rows)/sum(r['total_passage_chars'] for r in rows),
        requests_reordered=sum(r['permuted'] for r in rows)/n,
        candidate_orders_checked=sum(r['candidates_evaluated'] for r in rows))


def write_csv(p,rows):
    if not rows:return
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--retrieval-json',required=True,help='H100 prompt_records.jsonl with sorted_passage')
    ap.add_argument('--max-questions',type=int,default=650)
    ap.add_argument('--topk',type=int,default=5)
    ap.add_argument('--request-order',choices=['source','random'],default='source')
    ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--progress-every',type=int,default=100)
    ap.add_argument('--output-dir',required=True)
    args=ap.parse_args()
    records=load(args.retrieval_json,args.max_questions,args.topk)
    seq=list(range(len(records)))
    if args.request_order=='random':random.Random(args.seed).shuffle(seq)
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)

    independent,distinct=independent_upper(records,seq)
    ceiling=dict(
        description='Unconstrained earlier-seen PASSAGE TEXTS; does not respect attention/prefix dependence',
        requests=len(records),distinct_passages=distinct,
        total_passage_occurrences=sum(len(records[i]['ids']) for i in seq),
        previously_seen_occurrences=sum(x['previously_seen_passages'] for x in independent),
        fraction_seen_occurrences=sum(x['previously_seen_passages'] for x in independent)/sum(len(records[i]['ids']) for i in seq),
        fraction_seen_passage_characters=sum(x['previously_seen_chars'] for x in independent)/sum(x['total_passage_chars'] for x in independent),
        requests_all_passages_previously_seen=sum(x['previously_seen_passages']==len(records[x['source_index']]['ids']) for x in independent)/len(independent))
    write_csv(output/'independent_upper_bound.csv',independent)

    all_rows=[];summary=[]
    for strategy in ('original','canonical','optimal_keep_top1','optimal'):
        rows=simulate(records,seq,strategy,args.progress_every)
        all_rows.extend(rows)
        s=summarize(rows,strategy)
        summary.append(s)
    write_csv(output/'strategy_requests.csv',all_rows)
    write_csv(output/'strategy_summary.csv',summary)
    report={'assumptions':['Hashes complete passage text, ignores tokenizer and 512-token fragmentation',
                           'Prefix matches require identical leading ordered passage IDs',
                           'Simulates an unlimited, instant-persistence, passage-aligned cache',
                           'Does not model padding compute/storage overhead or answer quality',
                           'Per-request greedy optimal is not a globally joint sequence optimum',
                           'Weights whole-passage content by UTF-8 decoded character counts, not model tokens'],
            'independent_passage_upper_bound':ceiling,'prefix_aligned_scenarios':summary,
            'input':{'requests':len(records),'topk':args.topk,'request_order':args.request_order,'seed':args.seed}}
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nPASSAGE-ONLY REUSE ANALYSIS (no tokenizer)')
    print(f"Independent previously-seen upper bound: {ceiling['fraction_seen_occurrences']*100:.2f}% passage occurrences; "
          f"{ceiling['fraction_seen_passage_characters']*100:.2f}% passage characters")
    for s in summary:
        print(f"{s['strategy']:>18}  full-passage-prefix reuse={s['fraction_passage_occurrences_reused']*100:6.2f}%  "
              f"char coverage={s['fraction_passage_characters_reused']*100:6.2f}%  "
              f"mean matching lead={s['mean_matched_leading_passages']:.2f}/{args.topk}")
    print('Saved to:',output)


if __name__=='__main__':main()

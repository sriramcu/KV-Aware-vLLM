#!/usr/bin/env python3
"""Extensive, leakage-labeled offline RAG passage/prefix-KV reuse study.

Research target: KV-Aware-vLLM / LinearRAG medical Q250/Q650. Python 3.10+.
The ACTUAL serving prompt follows experiments/mp_nognn_project.py.

Input: saved retrieval JSON/JSONL with {question, sorted_passage: [str, ...]};
       optionally build exact dense-import retrieval from original embeddings.
The raw medical questions.json has no retrieval lists. Never infer them.

Outputs: report.json, report.md, request_metrics.csv, passage_frequency.csv,
         pairwise_overlap.csv, chunk_frequency.csv, boundaries.csv,
         scenario_summary.csv, scenario_requests.csv, graphs/*.png.

Counters distinguish dataset-global hindsight, earlier-in-order causal history,
chunk-hash availability, usable contiguous-prefix retrieval and finite LRU.
All token-prefix identities are cryptographic equality surrogates; they are
NOT bit-identical LMCache ObjectKey implementations.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass
from itertools import permutations, combinations
from pathlib import Path
from statistics import mean, median
import csv
import hashlib
import json
import math
import random
import struct
import sys
import time
import zipfile

SYS_PROMPT = (
    'As an advanced reading comprehension assistant, your task is to analyze '
    'text passages and corresponding questions meticulously. Your response '
    'start after "Thought: ", where you will methodically break down the '
    'reasoning process, illustrating how you arrive at conclusions. Conclude '
    'with "Answer: " to present a concise, definitive response, devoid of '
    'additional elaborations.'
)


def average(vals): return sum(vals) / len(vals) if vals else None


def quant(vals):
    if not vals: return {'count': 0}
    s = sorted(vals); n = len(s)
    def q(p):
        x = (n-1)*p; lo = int(x); hi = min(lo+1,n-1)
        return s[lo]*(1-(x-lo))+s[hi]*(x-lo)
    return {'count': n, 'mean': average(vals), 'min': s[0], 'p10': q(.1),
            'p25': q(.25), 'p50': q(.5), 'p75': q(.75), 'p90': q(.9),
            'p95': q(.95), 'p99': q(.99), 'max': s[-1]}


def ratio(n,d): return n/d if d else None


def wilson95(k,n):
    if not n:return None
    z=1.95996398454;phat=k/n;den=1+z*z/n
    center=(phat+z*z/(2*n))/den
    delta=z*math.sqrt((phat*(1-phat)+z*z/(4*n))/n)/den
    return [center-delta,center+delta]


def id_passage(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:24]


def read_json_or_jsonl(path):
    p = Path(path)
    if p.suffix.lower() == '.jsonl':
        with p.open(encoding='utf8') as f:
            return [json.loads(line) for line in f if line.strip()]
    return json.loads(p.read_text(encoding='utf8'))


def find_in_zip(archive: zipfile.ZipFile, suffix: str):
    matches = [x for x in archive.namelist() if x.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f'ZIP has {len(matches)} entries for {suffix!r}')
    return json.loads(archive.read(matches[0]))


def write_csv(path, rows):
    rows = list(rows)
    if not rows: return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for x in rows:
            w.writerow({k: (json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v)
                        for k,v in x.items()})


def build_dense_import_retrieval(args):
    """Replicate mp_nognn_project.py dense-import scores/order, not graph retrieval."""
    try:
        import numpy as np
        import pandas as pd
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit('Dense retrieval requires numpy pandas pyarrow sentence-transformers: '+str(e))
    base = Path(args.import_root) / args.dataset / 'passage_embedding.parquet'
    if not base.is_file(): raise SystemExit(f'Missing dense-import embeddings: {base}')
    questions = read_questions(args)
    if args.max_questions: questions = questions[:args.max_questions]
    df = pd.read_parquet(base)
    texts = df['text'].tolist()
    pe = np.asarray(df['embedding'].tolist(),dtype=np.float32)
    encoder = SentenceTransformer(args.embedding_model, device=args.embed_device)
    rows=[]
    for i,q in enumerate(questions):
        vec = np.asarray(encoder.encode(q['question'],normalize_embeddings=True,
                                        show_progress_bar=False),dtype=np.float32)
        scores = pe @ vec
        order = np.argsort(scores)[::-1][:args.topk]
        rows.append({'question':q['question'], 'id':q.get('id', i),
                     'sorted_passage':[texts[x] for x in order],
                     'sorted_passage_scores':[float(scores[x]) for x in order],
                     'retrieval_mode':'dense_import_fallback',
                     'gold_answer':q.get('answer','')})
        if (i+1)%100==0: print(f'[retrieval] {i+1}/{len(questions)}',flush=True)
    return rows


def read_questions(args):
    if args.questions_json: return read_json_or_jsonl(args.questions_json)
    if args.repo_zip:
        with zipfile.ZipFile(args.repo_zip) as z:
            return find_in_zip(z,f'dataset/{args.dataset}/questions.json')
    path=Path(args.dataset_root)/args.dataset/'questions.json'
    return read_json_or_jsonl(path) if path.is_file() else []


def read_corpus(args):
    if args.chunks_json: return read_json_or_jsonl(args.chunks_json)
    if args.repo_zip:
        with zipfile.ZipFile(args.repo_zip) as z:
            return find_in_zip(z,f'dataset/{args.dataset}/chunks.json')
    path=Path(args.dataset_root)/args.dataset/'chunks.json'
    return read_json_or_jsonl(path) if path.is_file() else []


def normalize_records(rows, topk, max_questions):
    if not isinstance(rows,list):
        raise ValueError('Retrieval input must be JSON list or JSONL records')
    result=[]
    for ix,r in enumerate(rows[:max_questions] if max_questions else rows):
        if not isinstance(r,dict): raise ValueError(f'record {ix} is not an object')
        ps = r.get('sorted_passage',r.get('passages',r.get('retrieved_passages')))
        if not isinstance(ps,list):
            raise ValueError(f'record {ix}: missing sorted_passage list; questions.json alone is insufficient')
        if any(not isinstance(x,str) for x in ps[:topk]):
            raise ValueError(f'record {ix}: sorted_passage must be full passage texts')
        result.append({'index': ix, 'id':str(r.get('id',ix)),
                       'question':str(r['question']), 'passages':tuple(ps[:topk]),
                       'ids':tuple(id_passage(x) for x in ps[:topk]),
                       'scores':r.get('sorted_passage_scores',[]),
                       'answer':r.get('gold_answer',r.get('answer','')),
                       'recorded_llm_prompt':r.get('llm_prompt')})
    if not result: raise ValueError('No records')
    return result


def corpus_profile(questions, corpus):
    if not corpus and not questions: return {}
    text = [x if isinstance(x,str) else str(x) for x in corpus]
    qtypes = Counter(q.get('question_type','unknown') for q in questions if isinstance(q,dict))
    evidence_lens = [len(q.get('evidence',[])) for q in questions
                     if isinstance(q,dict) and isinstance(q.get('evidence'),list)]
    return {'question_count':len(questions), 'corpus_chunk_count':len(corpus),
            'unique_corpus_texts':len(set(text)),
            'question_type_counts':dict(qtypes),
            'evidence_count':quant(evidence_lens),
            'question_char_lengths':quant([len(str(q.get('question',''))) for q in questions if isinstance(q,dict)]),
            'corpus_char_lengths':quant([len(x) for x in text]),
            'corpus_total_characters':sum(map(len,text))}


def lcp(a,b):
    k=0
    for x,y in zip(a,b):
        if x!=y:break
        k+=1
    return k


def jaccard(a,b):
    x,y=set(a),set(b)
    return len(x&y)/len(x|y) if x or y else 0


def reorder_indices(records, mode, seed, depth):
    idx=list(range(len(records)))
    if mode=='source': return idx
    if mode=='seeded_shuffle':
        random.Random(seed).shuffle(idx); return idx
    if mode=='legacy_prefix_hash':
        def key(i):
            prefix='\n'.join(records[i]['passages'][:depth]);
            return (hashlib.sha1(prefix.encode()).hexdigest(),records[i]['question'])
        return sorted(idx,key=key)
    if mode=='hierarchical_prefix':
        return sorted(idx,key=lambda i:(*(records[i]['ids'][x] if x<len(records[i]['ids']) else ''
                                         for x in range(depth)),records[i]['question']))
    raise ValueError(mode)


def passage_analysis(records, execution_order, out, pair_sample=200000, seed=0):
    n=len(records); maxk=max(len(r['ids']) for r in records)
    occ=Counter(pid for r in records for pid in r['ids'])
    ranks=defaultdict(Counter)
    for r in records:
        for k,pid in enumerate(r['ids']):ranks[k+1][pid]+=1
    all_pref=Counter()
    for r in records:
        for k in range(1,len(r['ids'])+1): all_pref[r['ids'][:k]]+=1
    earlier=Counter(); earlier_sets=Counter(); seen_passages=set()
    perreq=[]
    for step,idx in enumerate(execution_order):
        r=records[idx]; ids=r['ids']; st=set(ids)
        prev=set(seen_passages); seen_passages.update(st)
        shared=sum(p in prev for p in st)
        row={'execution_step':step,'source_index':idx,'request_id':r['id'],
             'number_of_passages':len(ids),
             'individual_passages_seen_earlier':shared,
             'individual_passages_seen_earlier_fraction':ratio(shared,len(st)),
             'max_prior_ordered_passage_prefix':max((k for k in range(1,len(ids)+1)
                                                       if earlier[ids[:k]]>0),default=0),
             'shared_passages_with_any_earlier':int(shared>0),
             'all_passages_seen_earlier':int(all(p in prev for p in st))}
        for k in range(1,maxk+1):
            prefix=ids[:k]
            row[f'first_{k}_shared_anywhere']=int(len(ids)>=k and all_pref[prefix]>1)
            row[f'first_{k}_shared_earlier']=int(len(ids)>=k and earlier[prefix]>0)
        row['best_shared_passage_set_with_prior'] = max((len(st & set(other))
                                            for other in earlier_sets),default=0)
        perreq.append(row)
        for k in range(1,len(ids)+1): earlier[ids[:k]]+=1
        earlier_sets[tuple(ids)]+=1
    write_csv(out/'request_metrics.csv',perreq)
    freq=[]
    text_by_id={id_passage(x):x for r in records for x in r['passages']}
    for pid,cnt in occ.most_common():
        t=text_by_id[pid]
        freq.append({'passage_id':pid,'request_occurrences':cnt,
                     'fraction_of_requests':cnt/n,'character_length':len(t),
                     'ranks':dict((j,ranks[j][pid]) for j in sorted(ranks) if ranks[j][pid]),
                     'preview':t[:125].replace('\n',' ')})
    write_csv(out/'passage_frequency.csv',freq)
    groups={}
    for k in range(1,maxk+1):
        have=[r for r in perreq if r['number_of_passages']>=k]
        groups[str(k)]={'questions_eligible':len(have),
                        'fraction_shared_anywhere':average([r[f'first_{k}_shared_anywhere'] for r in have]),
                        'fraction_shared_earlier':average([r[f'first_{k}_shared_earlier'] for r in have]),
                        'shared_anywhere_95pct_wilson_ci':wilson95(sum(r[f'first_{k}_shared_anywhere'] for r in have),len(have)),
                        'shared_earlier_95pct_wilson_ci':wilson95(sum(r[f'first_{k}_shared_earlier'] for r in have),len(have)),
                        'unique_ordered_prefixes':len(set(records[r['source_index']]['ids'][:k] for r in have))}
    rng=random.Random(seed)
    pairs=list(combinations(range(n),2)) if n<=500 else None
    npairs=n*(n-1)//2
    if pairs is None:
        sampled=set()
        while len(sampled)<min(pair_sample,npairs):
            a=rng.randrange(n);b=rng.randrange(n)
            if a!=b: sampled.add(tuple(sorted((a,b))))
        pairs=sorted(sampled)
    pair_rows=[]
    for i,j in pairs:
        a,b=records[i]['ids'],records[j]['ids']; common=len(set(a)&set(b))
        pair_rows.append({'i':i,'j':j,'shared_passages':common,
                          'jaccard':jaccard(a,b), 'ordered_prefix_len':lcp(a,b),
                          'same_retrieved_set':int(set(a)==set(b)),
                          'same_order':int(a==b),
                          'shared_top1_any_rank':int(a[0] in b or b[0] in a)})
    write_csv(out/'pairwise_overlap.csv',pair_rows)
    lengths=[len(r['passages']) for r in records]
    metrics={'requests':n,'retrieved_passages_per_request':quant(lengths),
             'distinct_passages':len(occ),
             'passage_request_frequency':quant(list(occ.values())),
             'passage_singleton_fraction':ratio(sum(v==1 for v in occ.values()),len(occ)),
             'passage_occurrence_top_1pct_coverage':ratio(sum(c for _,c in occ.most_common(max(1,math.ceil(len(occ)*.01)))),sum(occ.values())),
             'ordered_prefix_sharing':groups,
             'fraction_with_any_prior_overlap':average([x['shared_passages_with_any_earlier'] for x in perreq]),
             'prior_shared_passage_fraction':quant([x['individual_passages_seen_earlier_fraction'] for x in perreq]),
             'prior_best_set_overlap_count':quant([x['best_shared_passage_set_with_prior'] for x in perreq]),
             'pairwise_sampling':{'sampled_pairs':len(pair_rows),'all_possible_pairs':npairs,'exhaustive':len(pair_rows)==npairs},
             'pairwise_shared_count':quant([r['shared_passages'] for r in pair_rows]),
             'pairwise_jaccard':quant([r['jaccard'] for r in pair_rows]),
             'pairwise_ordered_lcp':quant([r['ordered_prefix_len'] for r in pair_rows]),
             'pairwise_fraction_any_common':average([r['shared_passages']>0 for r in pair_rows]),
             'pairwise_fraction_same_set':average([r['same_retrieved_set'] for r in pair_rows]),
             'pairwise_fraction_same_order':average([r['same_order'] for r in pair_rows])}
    return metrics,perreq


def prefix_fingerprints(tokens, chunk_size):
    prefix=b'\x00'*16; keys=[]
    for k in range(0,len(tokens)//chunk_size*chunk_size,chunk_size):
        encoded=struct.pack('<%dI'%chunk_size,*tokens[k:k+chunk_size])
        prefix=hashlib.blake2b(prefix+encoded,digest_size=16).digest()
        keys.append(prefix.hex())
    return keys


class PromptEncoder:
    def __init__(self,name,chunk_size=512,max_cache=256,max_key_cache=125000):
        try:from transformers import AutoTokenizer
        except ImportError as e:raise SystemExit('pip install transformers tokenizers ('+str(e)+')')
        self.tokenizer=AutoTokenizer.from_pretrained(name,use_fast=True)
        if not self.tokenizer.is_fast:
            raise SystemExit('Fast tokenizer required for boundary offset analysis.')
        self.size=chunk_size
        self.enc_cache=OrderedDict();self.max_cache=max_cache
        self.key_cache=OrderedDict();self.max_key_cache=max_key_cache
        self.render_count=0;self.encode_time=0.

    def render(self,passages,question):
        user=''.join(p+'\n' for p in passages)+f'Question: {question}\n Thought: '
        return self.tokenizer.apply_chat_template(
                [{'role':'system','content':SYS_PROMPT},{'role':'user','content':user}],
                tokenize=False,add_generation_prompt=True)

    def encode(self,passages,question,need_offsets=False,need_tokens=False):
        key=(tuple(passages),question)
        item=self.enc_cache.get(key)
        if item and (not need_offsets or item['offsets'] is not None):
            self.enc_cache.move_to_end(key)
            return item
        if not need_offsets and not need_tokens and key in self.key_cache:
            self.key_cache.move_to_end(key)
            return self.key_cache[key]
        start=time.perf_counter()
        text=self.render(passages,question)
        result=self.tokenizer(text,add_special_tokens=False,
                              return_offsets_mapping=need_offsets,
                              truncation=False)
        tokens=result['input_ids'];offsets=result.get('offset_mapping')
        # Cryptographic rolling fingerprint of prefix-dependent 512-token chunks.
        # Not byte-identical to LMCache TokenHasher, but exact equality semantics.
        hashes=prefix_fingerprints(tokens,self.size)
        item={'text':text,'tokens':tokens,'offsets':offsets,'keys':hashes,
              'prompt_tokens':len(tokens),'full_chunks':len(hashes),
              'tail_tokens':len(tokens)%self.size}
        self.key_cache[key]={k:item[k] for k in ('keys','prompt_tokens','full_chunks','tail_tokens')}
        self.key_cache.move_to_end(key)
        if len(self.key_cache)>self.max_key_cache:self.key_cache.popitem(last=False)
        if need_offsets or need_tokens:
            self.enc_cache[key]=item;self.enc_cache.move_to_end(key)
        if len(self.enc_cache)>self.max_cache:self.enc_cache.popitem(last=False)
        self.encode_time+=time.perf_counter()-start
        self.render_count+=1
        return item

    def boundaries(self,passages,question):
        item=self.encode(passages,question,True)
        s=item['text'];cursor=0;spans=[]
        for i,p in enumerate(passages):
            start=s.find(p,cursor)
            if start<0:raise RuntimeError('Passage text not found in rendered chat template')
            spans.append((start,start+len(p),i))
            cursor=start+len(p)
        offsets=item['offsets']
        passage_counts=defaultdict(set);lasts=[];token_owner={}
        for ti,(lo,hi) in enumerate(offsets):
            if hi<=lo:continue
            hit=[pid for l,r,pid in spans if hi>l and lo<r]
            if hit:
                passage_counts[ti//self.size].update(hit)
                token_owner[ti]=hit
        for l,r,pid in spans:
            hits=[ti for ti,(lo,hi) in enumerate(offsets)
                  if hi>lo and lo<r and hi>l]
            if hits: lasts.append(max(hits)+1)
        full=item['full_chunks']
        mixed=[j for j in range(full) if len(passage_counts.get(j,set()))>1]
        # True BPE tokenization across a boundary may affect token ownership;
        # boundary counts are diagnostic, not a transformation of live prompts.
        qstring='Question: '+question
        qpos=s.find(qstring,cursor)
        qtoks={i//self.size for i,(lo,hi) in enumerate(offsets)
               if qpos>=0 and hi>lo and hi>qpos and lo<qpos+len(qstring)}
        qmixed=[j for j in qtoks if passage_counts.get(j)]
        ideal_padding=[];added=0
        for end in lasts[:-1]:
            pad=(-(end+added))%self.size
            ideal_padding.append(pad);added+=pad
        return {'source_passage_count':len(passages),
                'prompt_tokens':item['prompt_tokens'],'full_chunks':full,
                'tail_tokens':item['tail_tokens'],
                'mixed_passage_full_chunk_count':len(mixed),
                'mixed_passage_full_chunk_indices':mixed,
                'question_passage_mixed_chunk_count':len(qmixed),
                'mixed_question_chunk_indices':sorted(qmixed),
                'passage_ends_token_offset':lasts,
                'passage_ends_inside_chunk_count':sum(x%self.size!=0 for x in lasts[:-1]),
                'ideal_padding_tokens_before_remaining_passages':sum(ideal_padding),
                'ideal_padding_per_boundary':ideal_padding,
                'full_chunk_passage_occupancy':[len(passage_counts.get(j,set())) for j in range(full)]}


class ChunkCache:
    def __init__(self,capacity=None):
        self.capacity=capacity
        self.lru=OrderedDict();self.evictions=0;self.inserts=0
    def has(self,key):return key in self.lru
    def prefix_hit(self,keys):
        hits=0
        for key in keys:
            if key not in self.lru:break
            hits+=1
        return hits
    def usable(self,keys):return self.prefix_hit(keys)
    def touch(self,keys,n):
        for key in keys[:n]:
            if key in self.lru:self.lru.move_to_end(key)
    def store(self,keys):
        for key in keys:
            if key in self.lru:
                self.lru.move_to_end(key);continue
            if self.capacity==0:continue
            self.lru[key]=None;self.inserts+=1
            if self.capacity is not None and len(self.lru)>self.capacity:
                self.lru.popitem(last=False);self.evictions+=1


def choose_order(record,strategy,encoder,cache,rng,passage_pop,top1_fixed=False,
                 order_size_limit=5):
    ps=record['passages'];ids=record['ids']
    if strategy=='original':return ps,0,1
    if strategy=='canonical':return tuple(x for _,x in sorted(zip(ids,ps))),0,1
    if strategy=='length_asc':return tuple(sorted(ps,key=len)),0,1
    if strategy=='length_desc':return tuple(sorted(ps,key=len,reverse=True)),0,1
    if strategy=='popularity':
        return tuple(x for _,_,x in sorted(zip(ids,range(len(ps)),ps),key=lambda v:(-passage_pop[v[0]],v[1]))),0,1
    if strategy=='random':
        indices=list(range(len(ps)));rng.shuffle(indices)
        return tuple(ps[i] for i in indices),0,1
    if strategy not in ('greedy','greedy_keep_top1'):
        raise ValueError(strategy)
    if len(ps)>order_size_limit:
        raise ValueError(f'greedy exhaustive permutations restricted to {order_size_limit} passages; got {len(ps)}')
    fixed = strategy=='greedy_keep_top1'
    choices=((0,*p) for p in permutations(range(1,len(ps)))) if fixed else permutations(range(len(ps)))
    best=ps;best_hit=-1;visited=0
    for choice in choices:
        trial=tuple(ps[i] for i in choice)
        candidate=encoder.encode(trial,record['question'])
        hit=cache.prefix_hit(candidate['keys'])
        visited+=1
        if hit>best_hit:
            best_hit=hit;best=trial
    return best,best_hit,visited


def scenario(records,execution_order,encoder,strategy,capacity,batch_size,warm_order,
             warm_seed,seed,passage_pop,out_perreq=False):
    cache=ChunkCache(capacity);rng=random.Random(seed+359)
    indices=list(execution_order)
    if warm_order=='same':warm=indices[:]
    elif warm_order=='reverse':warm=indices[::-1]
    else:
        warm=indices[:];random.Random(warm_seed).shuffle(warm)
    rows=[];phase_summaries={}
    for phase,seq in [('cold',indices),('warm',warm)]:
        for step,idx in enumerate(seq):
            if (step + 1) % 25 == 0 or step + 1 == len(seq):
                print(
                    f"[{strategy}/{phase}] "
                    f"{step + 1}/{len(seq)} requests "
                    f"({100 * (step + 1) / len(seq):.1f}%)",
                    flush=True,
                )
            rec=records[idx]
            p,planned_hits,perms=choose_order(rec,strategy,encoder,cache,rng,passage_pop)
            item=encoder.encode(p,rec['question'])
            keys=item['keys'];hit=cache.usable(keys)
            present_any=sum(cache.has(k) for k in keys)
            holes=max(0,present_any-hit)
            cache.touch(keys,hit)
            cache.store(keys)
            rows.append({'strategy':strategy,'capacity':capacity if capacity is not None else 'unlimited',
                         'phase':phase,'step':step,'source_index':idx,
                         'batch':step//batch_size,'prefix_hits':hit,
                         'physical_hash_hits_anywhere':present_any,
                         'stranded_hash_hits':holes,'full_chunks':len(keys),
                         'cached_prefix_tokens':hit*encoder.size,
                         'prompt_tokens':item['prompt_tokens'],
                         'prefix_hit_fraction_of_full_chunks':ratio(hit,len(keys)),
                         'prefix_hit_fraction_of_prompt_tokens':ratio(hit*encoder.size,item['prompt_tokens']),
                         'recompute_prompt_tokens_proxy':item['prompt_tokens']-hit*encoder.size,
                         'permutations_evaluated':perms,
                         'reordered':int(p!=rec['passages']),
                         'best_found_hits':planned_hits if strategy.startswith('greedy') else None,
                         'cache_resident_chunks_after':len(cache.lru),
                         'evictions_cumulative':cache.evictions})
        group=[r for r in rows if r['phase']==phase]
        phase_summaries[phase]={
            'requests':len(group),
            'total_prompt_tokens':sum(r['prompt_tokens'] for r in group),
            'total_full_chunk_occurrences':sum(r['full_chunks'] for r in group),
            'useful_prefix_hit_chunks':sum(r['prefix_hits'] for r in group),
            'physical_chunk_hits_anywhere':sum(r['physical_hash_hits_anywhere'] for r in group),
            'stranded_physical_chunk_hits':sum(r['stranded_hash_hits'] for r in group),
            'useful_cached_prompt_tokens':sum(r['cached_prefix_tokens'] for r in group),
            'recompute_prompt_tokens_proxy':sum(r['recompute_prompt_tokens_proxy'] for r in group),
            'fraction_requests_with_prefix_hit':average([r['prefix_hits']>0 for r in group]),
            'fraction_requests_reordered':average([r['reordered']>0 for r in group]),
            'mean_prefix_hit_chunks':average([r['prefix_hits'] for r in group]),
            'p50_prefix_hit_chunks':median([r['prefix_hits'] for r in group]),
            'prefix_chunk_hit_occurrence_fraction':ratio(sum(r['prefix_hits'] for r in group),sum(r['full_chunks'] for r in group)),
            'cached_prompt_token_fraction':ratio(sum(r['cached_prefix_tokens'] for r in group),sum(r['prompt_tokens'] for r in group)),
            'total_permutations_examined':sum(r['permutations_evaluated'] for r in group),
            'cache_chunks_end_of_phase':len(cache.lru),
            'lru_evictions_cumulative':cache.evictions}
    return phase_summaries,rows


def batch_profile(rows):
    g=defaultdict(list)
    for x in rows:g[(x['strategy'],x['capacity'],x['phase'],x['batch'])].append(x)
    return [{'strategy':k[0],'capacity':k[1],'phase':k[2],'batch':k[3],
             'requests':len(v),'useful_hit_chunks':sum(x['prefix_hits'] for x in v),
             'total_full_chunks':sum(x['full_chunks'] for x in v),
             'prefix_hit_fraction':ratio(sum(x['prefix_hits'] for x in v),sum(x['full_chunks'] for x in v)),
             'reordered_fraction':average([x['reordered'] for x in v]),
             'stranded_hash_hits':sum(x['stranded_hash_hits'] for x in v)}
            for k,v in sorted(g.items(),key=lambda x:str(x[0]))]


def chunk_profile(records,order,encoder,out,boundary_limit=None):
    count=Counter();positions=defaultdict(Counter);seen=set()
    reqrows=[];boundrows=[];template_audits=0;template_mismatches=0
    for step,idx in enumerate(order):
        rec=records[idx];item=encoder.encode(rec['passages'],rec['question'],need_tokens=True)
        if rec.get('recorded_llm_prompt') is not None:
            template_audits+=1
            template_mismatches+=int(item['text'] != rec['recorded_llm_prompt'])
        keys=item['keys'];cached=sum(x in seen for x in keys)
        prefix=0
        for x in keys:
            if x not in seen:break
            prefix+=1
        for k,key in enumerate(keys):
            count[key]+=1;positions[k][key]+=1
        seen.update(keys)
        reqrows.append({'source_index':idx,'step':step,'total_prompt_tokens':item['prompt_tokens'],
                        'full_chunks':len(keys),'trailing_partial_tokens':item['tail_tokens'],
                        'global_cumulative_prefix_hits':prefix,
                        'global_cumulative_present_anywhere':cached,
                        'global_cumulative_stranded':cached-prefix})
        if boundary_limit is None or step<boundary_limit:
            row=encoder.boundaries(rec['passages'],rec['question'])
            boundrows.append({'source_index':idx,'step':step,**row})
    write_csv(out/'chunk_frequency.csv',({'hash':k,'occurrences':v,
                                'appears_at_chunk_positions':{j:cnt[k] for j,cnt in positions.items() if k in cnt}}
                               for k,v in count.most_common()))
    write_csv(out/'boundaries.csv',boundrows)
    write_csv(out/'baseline_chunk_requests.csv',reqrows)
    total=sum(count.values());ret=sum(v-1 for v in count.values())
    return {
        'tokenizer_exact_512_token_equality':True,
        'recorded_prompt_count_audited':template_audits,
        'recorded_prompt_mismatch_count':template_mismatches,
        'full_chunk_occurrences':total,'unique_full_chunk_prefix_hashes':len(count),
        'repeated_full_chunk_occurrences':ret,
        'global_repeated_chunk_occurrence_fraction':ratio(ret,total),
        'chunk_frequency':quant(list(count.values())),
        'max_occurrence_chunk':max(count.values(),default=0),
        'baseline_causal_prefix_chunks_total':sum(r['global_cumulative_prefix_hits'] for r in reqrows),
        'baseline_causal_cumulative_anywhere_hits':sum(r['global_cumulative_present_anywhere'] for r in reqrows),
        'baseline_causal_stranded_hashes':sum(r['global_cumulative_stranded'] for r in reqrows),
        'baseline_causal_prefix_hit_chunk_fraction':ratio(sum(r['global_cumulative_prefix_hits'] for r in reqrows),total),
        'baseline_causal_cached_prompt_token_fraction':ratio(sum(r['global_cumulative_prefix_hits'] for r in reqrows)*encoder.size,sum(r['total_prompt_tokens'] for r in reqrows)),
        'prompt_lengths':quant([r['total_prompt_tokens'] for r in reqrows]),
        'complete_chunk_counts':quant([r['full_chunks'] for r in reqrows]),
        'trailing_chunk_lengths':quant([r['trailing_partial_tokens'] for r in reqrows]),
        'boundary_sample_requests':len(boundrows),
        'mixed_passage_full_chunks':sum(r['mixed_passage_full_chunk_count'] for r in boundrows),
        'fraction_sampled_requests_with_mixed_passage_chunk':average([r['mixed_passage_full_chunk_count']>0 for r in boundrows]),
        'mixed_passage_chunk_count_per_request':quant([r['mixed_passage_full_chunk_count'] for r in boundrows]),
        'mixed_question_chunk_count_per_request':quant([r['question_passage_mixed_chunk_count'] for r in boundrows]),
        'ideal_token_level_padding_tokens_per_request':quant([r['ideal_padding_tokens_before_remaining_passages'] for r in boundrows]),
        'ideal_padding_fraction_of_prompt':quant([ratio(r['ideal_padding_tokens_before_remaining_passages'],r['prompt_tokens']) for r in boundrows]),
        'note_padding':'Ideal token-level padding budget only: NOT a valid rendered-space-padding simulation; it does not measure padded KV hits or answer quality.'}



def chunk_size_sweep(records, order, encoder, chunk_sizes):
    rows=[]
    tokenized=[encoder.encode(records[i]['passages'],records[i]['question'],need_tokens=True)['tokens'] for i in order]
    for size in chunk_sizes:
        seen=set();all_hash=Counter();hit=0;seen_any=0;stranded=0;full=0
        for tokens in tokenized:
            keys=prefix_fingerprints(tokens,size)
            n=0
            for k in keys:
                if k not in seen:break
                n+=1
            physical=sum(k in seen for k in keys)
            hit+=n;seen_any+=physical;stranded+=physical-n;full+=len(keys)
            all_hash.update(keys);seen.update(keys)
        rows.append({'chunk_size':size,'full_chunk_occurrences':full,
                     'unique_prefix_hashes':len(all_hash),
                     'repeated_occurrence_fraction':ratio(full-len(all_hash),full),
                     'causal_usable_prefix_hit_chunks':hit,
                     'causal_usable_prefix_hit_tokens':hit*size,
                     'causal_usable_fraction_of_prompt_tokens':ratio(hit*size,sum(len(x) for x in tokenized)),
                     'causal_prefix_hit_chunk_fraction':ratio(hit,full),
                     'physically_present_chunk_hits_anywhere':seen_any,
                     'stranded_chunk_hits':stranded})
    return rows


def hindsight_all_original(records,order,encoder,limit=None):
    # Contains FUTURE requests, but no a request's own unique hashes. No cache capacity.
    original=[encoder.encode(r['passages'],r['question'])['keys'] for r in records]
    population=Counter(x for keys in original for x in keys)
    detail=[]
    for j,idx in enumerate(order[:limit] if limit else order):
        r=records[idx];old=original[idx]
        available=set(k for k in population if population[k]-(k in old)>0)
        def prefix(keys):
            n=0
            for h in keys:
                if h not in available:break
                n+=1
            return n
        base=prefix(old);best=base;best_order=r['ids'];seen=set()
        for perm in permutations(range(len(r['passages']))):
            passage_order=tuple(r['passages'][i] for i in perm)
            perm_ids=tuple(r['ids'][i] for i in perm)
            if perm_ids in seen:continue
            seen.add(perm_ids)
            keys=encoder.encode(passage_order,r['question'])['keys']
            h=prefix(keys)
            if h>best:best=h;best_order=perm_ids
        detail.append({'request_index':idx,'hindsight_original_order_hits':base,
                       'hindsight_permutation_optimal_hits':best,
                       'additional_hits':best-base,'changed_order':int(best_order!=r['ids']),
                       'candidates_examined':len(seen)})
        if (j+1)%50==0:print(f'[hindsight-oracle] {j+1} / {limit or len(order)}',flush=True)
    return {'requests':len(detail),
            'mean_baseline_hits':average([x['hindsight_original_order_hits'] for x in detail]),
            'mean_permutation_optimal_hits':average([x['hindsight_permutation_optimal_hits'] for x in detail]),
            'mean_additional_hits':average([x['additional_hits'] for x in detail]),
            'requests_with_improvement_fraction':average([x['additional_hits']>0 for x in detail]),
            'extra_hits':sum(x['additional_hits'] for x in detail),
            'warning':'Hindsight: includes other requests executed in the future; all reference prompts are in their original passage order. Not realizable causal performance and not a global optimum.'},detail


def figures(out,passage,chunk,scenario_rows,batches):
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError: return
    figdir=out/'graphs';figdir.mkdir(exist_ok=True)
    groups=passage.get('ordered_prefix_sharing',{})
    if groups:
        ks=list(groups);y1=[groups[k]['fraction_shared_anywhere']*100 for k in ks]
        y2=[groups[k]['fraction_shared_earlier']*100 for k in ks]
        plt.figure(figsize=(8,4.5));plt.plot(ks,y1,'o-',label='Anywhere in dataset (hindsight)')
        plt.plot(ks,y2,'s-',label='Previously in execution order')
        plt.xlabel('Top-k ordered passage prefix');plt.ylabel('Questions sharing prefix (%)')
        plt.legend();plt.tight_layout();plt.savefig(figdir/'ordered_passage_prefix_sharing.png',dpi=160);plt.close()
    if scenario_rows:
        cs=[x for x in scenario_rows if x['phase']=='cold']
        ws=[x for x in scenario_rows if x['phase']=='warm']
        labels=[x['strategy']+'\nC='+str(x['capacity']) for x in cs]
        xs=list(range(len(labels)))
        plt.figure(figsize=(max(9,len(labels)*1.1),4.8))
        plt.bar([v-.19 for v in xs],[x['cached_prompt_token_fraction']*100 for x in cs],width=.38,label='Cold')
        plt.bar([v+.19 for v in xs],[x['cached_prompt_token_fraction']*100 for x in ws],width=.38,label='Warm')
        plt.xticks(xs,labels,rotation=35,ha='right');plt.ylabel('Usable prefix tokens / prompt tokens (%)')
        plt.legend();plt.tight_layout();plt.savefig(figdir/'causal_strategy_comparison.png',dpi=160);plt.close()
    if batches:
        plt.figure(figsize=(9,5))
        for k in sorted(set((x['strategy'],x['capacity'],x['phase']) for x in batches),key=str):
            if k[2]!='warm':continue
            vals=sorted((x for x in batches if (x['strategy'],x['capacity'],x['phase'])==k),key=lambda r:r['batch'])
            plt.plot([x['batch']+1 for x in vals],[x['prefix_hit_fraction']*100 for x in vals],marker='.',label=f'{k[0]} C={k[1]}')
        plt.xlabel('Warm submission wave (not scheduler concurrency)');plt.ylabel('Usable 512-token prefix hit fraction (%)')
        if len(set((x['strategy'],x['capacity']) for x in batches))<=12:plt.legend()
        plt.tight_layout();plt.savefig(figdir/'warm_batch_reuse.png',dpi=160);plt.close()


def markdown_report(r):
    lines=['# Offline passage and prefix-KV reuse profile','',
           '**Interpretation:** passage-level overlap is not chunk-prefix reuse. '
           'Warm/cold simulations below are causal, use exact token-sequence equality fingerprints, '
           'and assume deterministic persistence of all full chunks admitted to the simulated cache. '
           'They do not simulate CPU/disk GET timing, vLLM 16-token VPC, cache-tier weights, or answer quality.','']
    m=r.get('metadata',{})
    lines += ['## Inputs','',f"- Requests: {m.get('requests','N/A')}; top-k: {m.get('topk')}",
              f"- Request order: {m.get('request_order')}; cold seed: {m.get('seed')}",
              f"- Tokenizer: {m.get('tokenizer') or 'not available'}; full chunk size: {m.get('chunk_size')}",
              f"- Retrieval provenance: {m.get('retrieval_source')}",'']
    p=r.get('passage_overlap',{});groups=p.get('ordered_prefix_sharing',{})
    if groups:
        lines += ['## Ordered passage-prefix repetition','',
                  '| First k passages | Share with any question | Share with earlier request | Distinct ordered prefixes |',
                  '|---|---:|---:|---:|']
        for k,g in groups.items():
            lines.append(f"| {k} | {g['fraction_shared_anywhere']:.2%} | {g['fraction_shared_earlier']:.2%} | {g['unique_ordered_prefixes']} |")
        lines+=['',f"Distinct passage texts: **{p.get('distinct_passages')}**",'']
    c=r.get('token_chunks')
    if c:
        lines += ['## Exact-token chunk conversion','',
                  f"- Full chunk occurrences / unique hashes: **{c['full_chunk_occurrences']:,} / {c['unique_full_chunk_prefix_hashes']:,}**",
                  f"- Repeated full chunk occurrence share (global, retrospective): **{c['global_repeated_chunk_occurrence_fraction']:.2%}**",
                  f"- Causal usable contiguous-prefix share (cold, unlimited cache): **{c['baseline_causal_prefix_hit_chunk_fraction']:.2%}**",
                  f"- Physically present but stranded full-chunk hashes: **{c['baseline_causal_stranded_hashes']:,}**",
                  f"- Sampled requests with mixed-passage 512-token chunks: **{c['fraction_sampled_requests_with_mixed_passage_chunk']:.2%}**",
                  f"- First passage shared earlier but zero full-chunk hits: **{c.get('first_passage_shared_earlier_but_zero_full_chunk_hits',0)}**",
                  f"- Recorded prompts failing reconstruction audit: **{c.get('recorded_prompt_mismatch_count',0)} / {c.get('recorded_prompt_count_audited',0)}**",
                  '- Padding estimates are ideal token-count overheads only, not a live-text padding experiment.','']
    rows=r.get('scenarios',[])
    if rows:
        lines+=['## Causal simulation (unlimited or fixed LRU)','',
                '| Strategy | Capacity (full chunks) | Phase | Useful hit chunks | Token reuse | Reordered | Evictions cumulative |',
                '|---|---:|---|---:|---:|---:|---:|']
        for x in rows:
            lines.append(f"| {x['strategy']} | {x['capacity']} | {x['phase']} | {x['useful_prefix_hit_chunks']:,} | {x['cached_prompt_token_fraction']:.2%} | {x['fraction_requests_reordered']:.2%} | {x['lru_evictions_cumulative']:,} |")
    lines+=['','## Validity boundaries','',
            '- `shared_anywhere` includes future requests; `shared_earlier` respects the specified replay order.',
            '- `greedy` optimizes each request over currently simulated available hashes, not a globally optimal sequence.',
            '- Both cold and warm phases update the same simulated cache; no store latency or stage overlap is modeled.',
            '- Token fingerprints are prefix-dependent. Hash serialization differs from LMCache ObjectKey; verify live exact prompt token IDs on the cluster.',
            '- Whole-passage reordering changes evidence order and may affect quality; measure answers separately.',
            '- The raw `dataset/medical/questions.json` does **not** contain sorted passages. Matching the historical 90.4%/49.6% requires the original retrieval method, top-k, subset, and sampling order.','']
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--retrieval-json',help='Saved JSON/JSONL records containing sorted_passage (preferred)')
    ap.add_argument('--build-dense-from-import',action='store_true',help='Use original dense-import embedding matrix and SentenceTransformer, never approximate with corpus text')
    ap.add_argument('--import-root',default='Hierarchical_KV/LinearRAG/import')
    ap.add_argument('--embedding-model',default='sentence-transformers/all-mpnet-base-v2')
    ap.add_argument('--embed-device',default='cpu')
    ap.add_argument('--repo-zip',help='Repository ZIP providing original dataset/medical/*.json')
    ap.add_argument('--dataset-root',default='dataset')
    ap.add_argument('--questions-json');ap.add_argument('--chunks-json')
    ap.add_argument('--dataset',default='medical')
    ap.add_argument('--topk',type=int,default=5)
    ap.add_argument('--max-questions',type=int,default=650)
    ap.add_argument('--tokenizer',default=None,help='HF tokenizer path (e.g. meta-llama/Llama-3.3-70B-Instruct); omit for passage-only analysis')
    ap.add_argument('--chunk-size',type=int,default=512)
    ap.add_argument('--batch-size',type=int,default=50)
    ap.add_argument('--request-order',choices=['source','seeded_shuffle','legacy_prefix_hash','hierarchical_prefix'],default='seeded_shuffle')
    ap.add_argument('--request-order-seed',type=int,default=0)
    ap.add_argument('--prefix-sort-depth',type=int,default=2)
    ap.add_argument('--warm-order',choices=['same','reverse','random'],default='random')
    ap.add_argument('--warm-seed',type=int,default=1)
    ap.add_argument('--strategies',nargs='+',default=['original','canonical','greedy'],
                    choices=['original','canonical','length_asc','length_desc','popularity','random','greedy','greedy_keep_top1'])
    ap.add_argument('--capacities',nargs='+',default=['unlimited'],
                    help='Max full-chunk entries in shared simulated LRU cache; unlimited or integer. Not per-tier simulation.')
    ap.add_argument('--boundary-sample',type=int,default=650,help='Number of original-order prompts for detailed offset/boundary diagnostics')
    ap.add_argument('--candidate-key-cache',type=int,default=125000,help='Maximum cached candidate prefix-hash summaries (speeds warm/strategy experiments)')
    ap.add_argument('--pair-sample',type=int,default=200000)
    ap.add_argument('--chunk-sizes',type=int,nargs='+',default=[128,256,512,1024],help='Chunk-size sensitivity sweep using identical token IDs')
    ap.add_argument('--hindsight-oracle',action='store_true',help='Expensive future-aware counterfactual: choose best of 120 permutations against all OTHER original prompts')
    ap.add_argument('--hindsight-limit',type=int,default=None,help='Limit hindsight oracle requests for a pilot')
    ap.add_argument('--output-dir',default='medical_reuse_analysis')
    ap.add_argument('--skip-scenarios',action='store_true')
    args=ap.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if args.topk<1 or args.chunk_size<1 or args.batch_size<1:ap.error('topk, chunk-size, batch-size must be positive')
    questions=read_questions(args)
    corpus=read_corpus(args)
    data={'metadata':{'dataset':args.dataset,'questions_in_raw_dataset':len(questions),
                      'raw_source_corpus_chunks':len(corpus),'topk':args.topk,
                      'chunk_size':args.chunk_size,'batch_size':args.batch_size,
                      'request_order':args.request_order,'seed':args.request_order_seed,
                      'warm_order':args.warm_order,'warm_seed':args.warm_seed,
                      'tokenizer':args.tokenizer,
                      'retrieval_source':args.retrieval_json or ('dense_import_exact' if args.build_dense_from_import else None)},
          'corpus_profile':corpus_profile(questions,corpus)}
    if args.retrieval_json:
        rows=read_json_or_jsonl(args.retrieval_json)
    elif args.build_dense_from_import:
        if not questions: ap.error('Need questions.json for dense retrieval')
        rows=build_dense_import_retrieval(args)
        with (out/'reconstructed_dense_retrieval.json').open('w',encoding='utf8') as f:
            json.dump(rows,f,ensure_ascii=False,indent=1)
    else:
        data['warnings']=['Passage/chunk reuse cannot be computed: raw questions.json and chunks.json contain no ranked retrieval assignments. Pass --retrieval-json or --build-dense-from-import.']
        (out/'report.json').write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')
        (out/'report.md').write_text(markdown_report(data),encoding='utf8')
        print(json.dumps(data,indent=2));return
    recs=normalize_records(rows,args.topk,args.max_questions)
    data['metadata']['requests']=len(recs)
    idx=reorder_indices(recs,args.request_order,args.request_order_seed,args.prefix_sort_depth)
    print(f'[passage] {len(recs)} requests; order={args.request_order}',flush=True)
    passage,passage_rows=passage_analysis(recs,idx,out,args.pair_sample,args.request_order_seed)
    data['passage_overlap']=passage
    if questions:
        question_type_by_text={str(q.get('question','')):q.get('question_type','unknown')
                              for q in questions if isinstance(q,dict)}
        by_type=defaultdict(list)
        for r in passage_rows:
            qtext=recs[r['source_index']]['question']
            by_type[question_type_by_text.get(qtext,'unknown')].append(r)
        data['question_type_breakdown']={kind:{
            'requests':len(items),
            'first_passage_shared_anywhere':average([x['first_1_shared_anywhere'] for x in items]),
            'first_passage_shared_earlier':average([x['first_1_shared_earlier'] for x in items]),
            'first_two_shared_anywhere':average([x.get('first_2_shared_anywhere',0) for x in items]),
            'mean_prior_distinct_passage_overlap_fraction':average([x['individual_passages_seen_earlier_fraction'] for x in items])
        } for kind,items in by_type.items()}
        write_csv(out/'question_type_breakdown.csv',
                  [{'question_type':k,**v} for k,v in data['question_type_breakdown'].items()])
    print('[passage] original prefix sharing:',{k:round(v['fraction_shared_anywhere']*100,2) for k,v in passage['ordered_prefix_sharing'].items()},flush=True)
    if not args.tokenizer:
        figures(out,passage,{},[],[])
    if args.tokenizer:
        encoder=PromptEncoder(args.tokenizer,args.chunk_size,max_key_cache=args.candidate_key_cache)
        print('[chunk] tokenizing original prompts and boundaries...',flush=True)
        data['token_chunks']=chunk_profile(recs,idx,encoder,out,args.boundary_sample)
        import csv as _csv
        with (out/'baseline_chunk_requests.csv').open(newline='',encoding='utf8') as _f:
            baseline_chunks=list(_csv.DictReader(_f))
        prior_by_index={x['source_index']:x for x in passage_rows}
        comparison=[]
        for rr in baseline_chunks:
            p=prior_by_index[int(rr['source_index'])]
            comparison.append({'source_index':int(rr['source_index']),
                'previously_shared_first_passage':p.get('first_1_shared_earlier',0),
                'previously_shared_top_two':p.get('first_2_shared_earlier',0),
                'earlier_ordered_passage_prefix_length':p['max_prior_ordered_passage_prefix'],
                'useful_full_chunk_prefix_hits':int(rr['global_cumulative_prefix_hits']),
                'full_chunk_hits_anywhere':int(rr['global_cumulative_present_anywhere']),
                'passage_shared_but_zero_chunks':int(p.get('first_1_shared_earlier',0)==1 and int(rr['global_cumulative_prefix_hits'])==0)})
        write_csv(out/'passage_to_chunk_conversion.csv',comparison)
        for k in range(1,args.topk+1):
            subset=[r for r in comparison if r['earlier_ordered_passage_prefix_length']>=k]
            data['token_chunks'][f'top_{k}_ordered_prefix_earlier_to_chunk']={
                'requests':len(subset),
                'fraction_with_at_least_one_full_chunk_hit':average([r['useful_full_chunk_prefix_hits']>0 for r in subset]),
                'usable_prefix_hit_chunks':quant([r['useful_full_chunk_prefix_hits'] for r in subset])}
        data['token_chunks']['first_passage_shared_earlier_but_zero_full_chunk_hits']=sum(r['passage_shared_but_zero_chunks'] for r in comparison)
        data['chunk_size_sensitivity']=chunk_size_sweep(recs,idx,encoder,args.chunk_sizes)
        write_csv(out/'chunk_size_sensitivity.csv',data['chunk_size_sensitivity'])
        if args.hindsight_oracle:
            if any(len(r['passages'])>5 for r in recs):
                raise ValueError('Hindsight oracle intended for up to 5 passages per request')
            data['hindsight_all_original_oracle'],oracle_rows=hindsight_all_original(
                recs,idx,encoder,args.hindsight_limit)
            write_csv(out/'hindsight_oracle_requests.csv',oracle_rows)
        scenario_rows=[];scenario_requests=[]
        if not args.skip_scenarios:
            popularity=Counter(p for r in recs for p in r['ids'])
            for capstr in args.capacities:
                capacity=None if capstr.lower()=='unlimited' else int(capstr)
                for strategy in args.strategies:
                    print(f'[scenario] {strategy}, capacity={capstr}',flush=True)
                    result,detail=scenario(recs,idx,encoder,strategy,capacity,args.batch_size,
                                           args.warm_order,args.warm_seed,args.request_order_seed,popularity)
                    for phase,s in result.items():scenario_rows.append({'strategy':strategy,'capacity':capstr,'phase':phase,**s})
                    scenario_requests.extend(detail)
            write_csv(out/'scenario_summary.csv',scenario_rows)
            write_csv(out/'scenario_requests.csv',scenario_requests)
            batches=batch_profile(scenario_requests)
            write_csv(out/'batch_summary.csv',batches)
            data['scenarios']=scenario_rows
            data['tokenizer_runtime']={'encoded_candidates':encoder.render_count,
                                       'encoding_seconds':encoder.encode_time}
            figures(out,passage,data['token_chunks'],scenario_rows,batches)
        else:figures(out,passage,data['token_chunks'],[],[])
    data['warnings']=[
      'Repeated first-passage statistics are dependent on retrieval model, first-N sampling, and question order.',
      'Permutations are whole passages; prompt is rebuilt before tokenization and prefix hashing.',
      'SHA-based rolling identity fingerprints match exact token-prefix equality, but are not serialized LMCache keys.',
      'Simulated cold+warm unlimited/LRU retention does not model multi-tier timing, PUT caps, prefix holes from DROP, VPC block eviction or answer quality.',
      'Finite LRU capacities count logical 512-token hash entries, not GPU reserved active memory; use as sensitivity, not a claim about production occupancy.',
      'Question suffix differs between requests; short prefixes may nevertheless be token-identical, while final mixed question chunks usually cannot match.',
      'Estimated ideal padding is token-count-only, not physically rendered prompt padding.'
    ]
    (out/'report.json').write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')
    (out/'report.md').write_text(markdown_report(data),encoding='utf8')
    print(f'[done] {out.resolve()}')
    if 'token_chunks' in data:
        print('useful baseline cold prefix hits:',data['token_chunks']['baseline_causal_prefix_chunks_total'])
    print('outputs:',', '.join(x.name for x in sorted(out.iterdir())))

if __name__=='__main__': main()

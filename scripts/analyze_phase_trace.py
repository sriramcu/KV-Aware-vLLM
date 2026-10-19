#!/usr/bin/env python3
"""CPU-only offline analysis of opt-in phase_engine.pid*.jsonl traces.

The results distinguish scheduled prompt/decode-position token counts, KV
remote-wait transitions, and wall-clock engine durations. The executor-result
interval is NOT exclusive GPU time. 'decode interstep gap' is an estimate of
time between batch completions and is not a causal scheduler-wait attribution.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median


def summary(values):
    if not values:
        return {"count": 0}
    values = sorted(values)
    return {
        "count": len(values), "sum": sum(values),
        "mean": mean(values), "median": median(values),
        "p95": values[min(len(values)-1, int(0.95 * len(values)))],
        "max": values[-1],
    }


def phase_of(req_id):
    if 'kvaware-cold-' in req_id:
        return 'cold'
    if 'kvaware-warm-' in req_id:
        return 'warm'
    return 'unknown'


def run(trace_dir: Path):
    events = []
    files = sorted(trace_dir.glob('phase_engine.pid*.jsonl'))
    for f in files:
        for num, line in enumerate(f.read_text().splitlines(), 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # End-of-job flush and killing a process can truncate last line.
                continue
            event['_pid'] = f.stem.split('pid')[-1]
            events.append(event)
    events.sort(key=lambda e: e.get('ts_ns', 0))
    periods = defaultdict(lambda: defaultdict(list))
    request_steps = defaultdict(list)
    remote_pending = {}
    remote_complete = defaultdict(list)
    totals = Counter()
    for e in events:
        kind = e.get('kind')
        if kind in ('kv_wait_enter', 'kv_wait_exit'):
            rid = str(e['request_id'])
            key = (e['_pid'], rid)
            if kind == 'kv_wait_enter':
                remote_pending[key] = e['ts_ns']
            elif key in remote_pending:
                remote_complete[key].append((e['ts_ns'] - remote_pending.pop(key))/1e6)
            continue
        if kind != 'engine_step':
            continue
        totals['steps'] += 1
        pre = e.get('prompt_compute_tokens', 0)
        dec = e.get('decode_position_tokens', 0)
        batch_kind = 'mixed' if pre and dec else 'prefill' if pre else 'decode' if dec else 'idle'
        totals[f'{batch_kind}_steps'] += 1
        for k, a, b in (
            ('schedule_wall_ms', 'schedule_start_ns', 'schedule_end_ns'),
            ('dispatch_wall_ms', 'schedule_end_ns', 'submit_end_ns'),
            ('executor_result_wall_ms', 'submit_end_ns', 'result_end_ns'),
            ('output_update_wall_ms', 'result_end_ns', 'update_end_ns'),
        ):
            periods['all'][k].append((e[b]-e[a])/1e6)
            periods['all'][f'{batch_kind}_{k}'].append((e[b]-e[a])/1e6)
        periods['all']['waiting_remote_snapshot'].append(e.get('waiting_remote_requests',0))
        for row in e.get('requests', []):
            rid = row['request_id']
            p = phase_of(rid)
            for key in ('scheduled','prompt_compute','decode_position'):
                periods[p][key].append(row.get(key,0))
            request_steps[(e['_pid'], rid)].append({
                'schedule_ns':e['schedule_start_ns'],
                'result_ns':e['result_end_ns'],
                'prompt':row.get('prompt_compute',0),
                'decode':row.get('decode_position',0),
                'batch_kind':batch_kind,
                'batch_result_ms': (e['result_end_ns']-e['submit_end_ns'])/1e6,
            })
    request_aggregate = {}
    for key, steps in request_steps.items():
        _, rid = key
        steps.sort(key=lambda e:e['schedule_ns'])
        p=phase_of(rid)
        a=request_aggregate.setdefault(rid, {"phase":p,"prompt_compute":0,
            "decode_position":0,"prefill_steps":0,"decode_steps":0,
            "decode_gap_ms":0.,"decode_batch_result_exposure_ms":0.})
        last_decode_result = None
        for step in steps:
            a['prompt_compute'] += step['prompt']
            a['decode_position'] += step['decode']
            a['prefill_steps'] += bool(step['prompt'])
            a['decode_steps'] += bool(step['decode'])
            if step['decode']:
                a['decode_batch_result_exposure_ms'] += step['batch_result_ms']
                if last_decode_result is not None:
                    # A proxy, not an exclusive wait attribution. Async steps
                    # can overlap, so negative gaps are truncated.
                    a['decode_gap_ms'] += max(0,(step['schedule_ns']-last_decode_result)/1e6)
                last_decode_result = step['result_ns']
    by_phase = {}
    for phase in ('cold','warm','unknown'):
        group=[a for a in request_aggregate.values() if a['phase']==phase]
        if not group and phase == 'unknown': continue
        by_phase[phase] = {
            'requests':len(group),
            **{k:summary([a[k] for a in group])
                for k in ('prompt_compute','decode_position','prefill_steps',
                           'decode_steps','decode_gap_ms',
                           'decode_batch_result_exposure_ms')},
            'kv_remote_wait_ms':summary([
                x for (_,rid), durations in remote_complete.items()
                if phase_of(rid)==phase for x in durations]),
        }
    device_files=sorted(trace_dir.glob('phase_device.pid*.jsonl'))
    device={}
    for f in device_files:
        duration=[]
        for line in f.read_text().splitlines():
            try: ev=json.loads(line)
            except json.JSONDecodeError: continue
            if ev.get('kind')=='device_forward':
                duration.append(ev['duration_ms'])
        device[f.stem]=summary(duration)
    return {
        'note': 'Measured executor wait is NOT GPU compute. Device timings are per worker, not per-request. Decode gaps are descriptive, not causal.',
        'trace_files':[str(p) for p in files],
        'steps':dict(totals),
        'engine_step_wall_ms': {k:summary(v) for k,v in periods['all'].items()},
        'by_phase':by_phase,
        'device_forward_per_worker_ms':device,
        'request_traces_count':len(request_aggregate),
        'unclosed_remote_wait_intervals':len(remote_pending),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trace-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.trace_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'steps':result['steps'], 'by_phase':result['by_phase'],
                      'trace_files':result['trace_files']},indent=2))


if __name__=='__main__': main()

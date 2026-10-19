#!/usr/bin/env python3
"""Plot request participation in scheduler batches (CPU engine timings).

Orange = scheduled prompt positions; teal = scheduled decode positions;
gray = intervals between that request's participating batches. A single
GPU batch can contain both types; drawing separate bars is NOT per-request
exclusive GPU kernel timing. No GPU is needed to generate this plot.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace-dir', required=True, type=Path)
    ap.add_argument('--request-id', required=True)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    import matplotlib.pyplot as plt
    matches = []
    for filename in sorted(args.trace_dir.glob('phase_engine.pid*.jsonl')):
        for line in filename.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('kind') != 'engine_step':
                continue
            for req in row.get('requests', []):
                if req['request_id'] == args.request_id:
                    matches.append((row, req))
    if not matches:
        raise SystemExit(f'No batch steps found for request {args.request_id!r}')
    matches.sort(key=lambda x: x[0]['schedule_start_ns'])
    t0 = matches[0][0]['schedule_start_ns']
    fig, ax = plt.subplots(figsize=(13, 2.4))
    last_end = None
    for row, req in matches:
        begin = (row['submit_end_ns'] - t0) / 1e9
        end = (row['result_end_ns'] - t0) / 1e9
        if last_end is not None and begin > last_end:
            ax.broken_barh([(last_end, begin-last_end)], (0, 0.9), facecolors='#b9bec7')
        kind = ('mixed' if req.get('prompt_compute') and req.get('decode_position')
                else 'prompt' if req.get('prompt_compute') else 'decode')
        color = {'prompt':'#e7a437', 'decode':'#1a9da8', 'mixed':'#7559bb'}[kind]
        ax.broken_barh([(begin, max(0,end-begin))], (0, 0.9), facecolors=color)
        last_end = max(last_end or end, end)
    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([])
    ax.set_xlabel('Elapsed engine time (s), from first scheduled step')
    ax.set_title(f'{args.request_id} — participating batch wall intervals (NOT exclusive GPU time)')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=k) for k,c in [
        ('Prompt scheduled','#e7a437'),('Decode position scheduled','#1a9da8'),
        ('Mixed','#7559bb'),('Between observed steps','#b9bec7')]],
        ncol=4,loc='upper center', bbox_to_anchor=(.5,-.35), fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160,bbox_inches='tight')
    print(args.output)


if __name__ == '__main__': main()

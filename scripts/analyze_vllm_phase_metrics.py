#!/usr/bin/env python3
"""Derive cold/warm queue/prefill/decode timing from Prometheus snapshots."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

METRICS = [
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:request_inference_time_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:e2e_request_latency_seconds",
]

def parse(path:Path):
    out={}
    text=path.read_text(errors="replace")
    for base in METRICS:
        for suffix in ("sum","count"):
            # Prometheus histograms may have labels before the numeric value.
            pat=re.compile(rf"^{re.escape(base)}_{suffix}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$",re.M)
            vals=[float(m.group(1)) for m in pat.finditer(text)]
            out[f"{base}_{suffix}"]=sum(vals) if vals else 0.0
    return out

def delta(a,b):
    result={}
    for base in METRICS:
        s=max(0.0,b[f"{base}_sum"]-a[f"{base}_sum"]); c=max(0.0,b[f"{base}_count"]-a[f"{base}_count"])
        result[base]={"sum_seconds":s,"count":int(round(c)),"mean_seconds_per_request":s/c if c else None}
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--before",type=Path,required=True); ap.add_argument("--after-cold",type=Path,required=True); ap.add_argument("--after-warm",type=Path,required=True); ap.add_argument("--output",type=Path); args=ap.parse_args()
    b=parse(args.before); c=parse(args.after_cold); w=parse(args.after_warm)
    result={"cold":delta(b,c),"warm":delta(c,w),"note":"Per-request interval sums are not phase wall-clock decomposition; concurrent requests overlap."}
    rendered=json.dumps(result,indent=2,sort_keys=True); print(rendered)
    if args.output: args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(rendered+"\n")
if __name__=="__main__": main()

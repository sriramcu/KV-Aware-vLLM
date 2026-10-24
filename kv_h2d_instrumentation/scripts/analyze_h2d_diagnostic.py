#!/usr/bin/env python3
"""Summarize [BAYES_H2D_*] records from an LMCache diagnostic log.

Stdlib-only. Native phase records are paired with diagnostic transfer keys.
CUDA staging samples are per step (NOT per request or per 512-token chunk).
"""
import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

TAGS = ("BAYES_H2D_DIAG", "BAYES_H2D_PHASE", "BAYES_H2D_COMPLETE", "BAYES_H2D_DIAG_ERROR", "BAYES_H2D_DIAG_INIT")
KV = re.compile(r"([\w_]+)=([^\s]+)")

def q(xs):
    if not xs:
        return "n/a"
    xs=sorted(xs)
    return f"n={len(xs)} p50={statistics.median(xs):.3f} p90={xs[int(.9*(len(xs)-1))]:.3f} p99={xs[int(.99*(len(xs)-1))]:.3f} max={xs[-1]:.3f}"

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", type=Path, help="LMCache log (e.g. logs/lmcache.log)")
    p.add_argument("--csv",type=Path,help="Optional parsed diagnostic+phase CSV")
    a=p.parse_args()
    rows=[]
    for line in a.log.open(errors="replace"):
        for tag in TAGS:
            if f"[{tag}]" in line:
                fields=dict(KV.findall(line.partition(f"[{tag}]")[2]))
                fields["tag"]=tag
                rows.append(fields)
                break
    print(f"Parsed {len(rows)} tagged records from {a.log}")
    for tag in TAGS:
        found=[r for r in rows if r["tag"]==tag]
        print(f"{tag}: {len(found)}")
    for field in ("elapsed_ms", "wall_to_dispatch_ms", "host_enqueue_ms", "gpu_producer_wait_ms", "gpu_postwait_gap_ms", "gpu_transfer_region_ms", "gpu_stream_region_ms"):
        vals=[float(r[field]) for r in rows if field in r and r["tag"]!="BAYES_H2D_PHASE"]
        if vals:
            print(f"{field}: {q(vals)}")
    for phase in ("staging_dma", "scatter_kernel"):
        dat=[r for r in rows if r["tag"]=="BAYES_H2D_PHASE" and r.get("phase")==phase]
        vals=[float(r["elapsed_ms"]) for r in dat if "elapsed_ms" in r]
        print(f"native {phase} CUDA-event duration (ms): {q(vals)}")
        byte_rate=[float(r["nbytes"])/(float(r["elapsed_ms"])*1e6) for r in dat if float(r.get("elapsed_ms","0"))>0]
        if byte_rate and phase=="staging_dma":
            print(f"native staging GB/s: {q(byte_rate)}")
    diag_keys={(r.get("transfer_key"),r.get("device")) for r in rows if r["tag"]=="BAYES_H2D_DIAG"}
    phase_keys={(r.get("transfer_key"),r.get("device")) for r in rows if r["tag"]=="BAYES_H2D_PHASE"}
    print(f"Diagnostic transfers: {len(diag_keys)}; transfers matched to native stage samples: {len(diag_keys & phase_keys)}")
    if a.csv:
        keys=sorted(set().union(*(r.keys() for r in rows)))
        a.csv.parent.mkdir(parents=True,exist_ok=True)
        with a.csv.open("w",newline="") as fh:
            out=csv.DictWriter(fh,fieldnames=keys);out.writeheader();out.writerows(rows)
        print(f"Wrote {a.csv}")

if __name__=="__main__":
    main()

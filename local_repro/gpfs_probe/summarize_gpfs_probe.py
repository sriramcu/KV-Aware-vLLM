#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("summary"); a=p.parse_args(); phases=json.loads(Path(a.summary).read_text())
    print("phase writers mode read_MiB/s median_file_s p95_file_s writer_GiB app_writer_MiB/s deferred_fsync_s")
    for ph in phases:
        reader=next((x for k,x in ph["results"] if k=="reader"),{})
        writers=[x for k,x in ph["results"] if k=="writer"]
        wb=sum(int(x.get("written_bytes",0)) for x in writers); we=sum(float(x.get("elapsed_sum_s",0)) for x in writers); wr=wb/1024/1024/we if we else 0.0
        print(f'{ph["phase_index"]:>5} {ph["writer_count"]:>7} {str(reader.get("mode","-")):>8} {float(reader.get("aggregate_read_mib_per_s") or 0):>10.2f} '
              f'{float(reader.get("median_file_read_s") or 0):>13.3f} {float(reader.get("p95_file_read_s") or 0):>10.3f} {wb/1024**3:>10.2f} {wr:>18.2f} '
              f'{float(ph["flush"].get("total_fsync_s") or 0):>16.3f}')
    return 0
if __name__=="__main__": raise SystemExit(main())

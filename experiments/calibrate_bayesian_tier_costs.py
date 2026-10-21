"""Lightweight ONE-TIME hardware prior generator (no LMCache, no vLLM).

CPU prior: completed H2D transfer of a per-rank 512-token KV-sized buffer.
Disk prior: buffered file read plus completed H2D transfer, per rank.
For TP2 use max across individual-rank times; measurements are NOT concurrent
serving latencies. The recompute reference remains the input JSON value.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import tempfile
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prior-in', type=Path, required=True)
    ap.add_argument('--prior-out', type=Path, required=True)
    ap.add_argument('--scratch', type=Path, default=Path('/scratch2/sriramc2'))
    ap.add_argument('--mib-per-rank', type=int, default=80)
    ap.add_argument('--repeats', type=int, default=8)
    args = ap.parse_args()
    if args.repeats < 3 or args.mib_per_rank <= 0:
        ap.error('at least 3 repeats and a positive chunk size required')

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPUs required for calibration')
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Expected exactly 2 visible GPUs for TP2 hardware calibration')
    size = args.mib_per_rank * 1024 * 1024
    args.scratch.mkdir(parents=True, exist_ok=True)
    prior = json.loads(args.prior_in.read_text())
    all_cpu, all_disk = [], []
    with tempfile.TemporaryDirectory(prefix='kvbayes_probe_', dir=args.scratch) as tmp:
        for device in range(torch.cuda.device_count()):
            # Byte tensor has the exact transfer size without storing actual KV.
            host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            host.fill_(17)
            with torch.cuda.device(device):
                dst = torch.empty(size, device=f'cuda:{device}', dtype=torch.uint8)
                stream = torch.cuda.Stream(device=device)
                device_cpu, device_disk = [], []
                fpath = Path(tmp) / f'rank_{device}.bin'
                # One file per rank. Cleanup is automatic, no LMCache pollution.
                with fpath.open('wb') as f:
                    f.write(bytes([17]) * size)
                    f.flush()
                    os.fsync(f.fileno())
                for trial in range(args.repeats + 2):
                    with torch.cuda.stream(stream):
                        stream.synchronize()
                        start = time.perf_counter()
                        dst.copy_(host, non_blocking=True)
                        stream.synchronize()
                        h2d_ms = (time.perf_counter() - start) * 1000
                    # This is a buffered filesystem read, not O_DIRECT.
                    # POSIX fadvise reduces page-cache reuse where available.
                    with fpath.open('rb', buffering=0) as f:
                        if hasattr(os, 'posix_fadvise') and hasattr(os, 'POSIX_FADV_DONTNEED'):
                            try:
                                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                            except OSError:
                                pass
                        start = time.perf_counter()
                        buf = f.read(size)
                        disk_read_ms = (time.perf_counter() - start) * 1000
                    if len(buf) != size:
                        raise RuntimeError('short synthetic file read')
                    if trial >= 2:
                        device_cpu.append(h2d_ms)
                        device_disk.append(disk_read_ms + h2d_ms)
                all_cpu.append(statistics.median(device_cpu))
                all_disk.append(statistics.median(device_disk))
                print(f'gpu={device} cpu_median_ms={all_cpu[-1]:.3f} '
                      f'disk+cpu_median_ms={all_disk[-1]:.3f}', flush=True)
    prior['cost_prior_ms']['cpu'] = round(max(all_cpu), 5)
    prior['cost_prior_ms']['disk'] = round(max(all_disk), 5)
    # Deliberately not pretending to calibrate recomputation.
    prior['comment'] = ('Score histogram is unchanged from the input prior (Q12-derived diagnostic). '
                        'CPU and disk costs are standalone transfer benchmarks; '
                        'recompute cost is unchanged and remains a reference assumption.')
    prior['cost_calibration'] = {
        'method': 'standalone buffered disk read + completed pinned H2D',
        'mib_per_tp_rank': args.mib_per_rank,
        'rank_count': torch.cuda.device_count(),
        'repeats_per_rank': args.repeats,
        'cpu_rank_medians_ms': all_cpu,
        'disk_rank_medians_ms': all_disk,
        'tp_wall_approximation': 'max of independent per-rank medians, not a simultaneous TP2 benchmark',
        'recompute_prior_measured': False,
        'disk_read_cache_note': 'page cache may still affect buffered reads',
    }
    args.prior_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.prior_out.with_suffix(args.prior_out.suffix + '.tmp')
    tmp_path.write_text(json.dumps(prior, indent=2, sort_keys=True) + '\n')
    os.replace(tmp_path, args.prior_out)
    print('calibrated_prior=' + str(args.prior_out), flush=True)
    print('recompute_prior_ms=' + str(prior['cost_prior_ms']['recompute']) + ' (UNCHANGED; not calibrated)')


if __name__ == '__main__':
    main()

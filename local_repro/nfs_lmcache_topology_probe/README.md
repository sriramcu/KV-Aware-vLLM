# NFS / LMCache topology probe

This is the second-stage storage benchmark. It reproduces the topology that the
first standalone test did not cover:

- two readers;
- six buffered writers;
- 80 MiB files;
- 120 seconds of sustained stress;
- fixed-path overwrites;
- one Python `readinto()` call per buffered read;
- an O_DIRECT control phase;
- deliberate same-file read/write overlap in one phase;
- NFS RPC, TCP retransmission, network, Dirty, and Writeback logging.

The benchmark does not use CUDA. The Slurm job requests one H100 only to place
the process on the same H100 node and NFS client used by the LMCache experiment.

## Files created

At the default settings, preparation creates:

- 256 immutable reader files: 20 GiB
- 48 fixed writer files: 3.75 GiB
- 48 overlap files: 3.75 GiB

Total: 27.5 GiB.

## 1. Install

```bash
mkdir -p \
  /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/nfs_lmcache_topology_probe

cp nfs_lmcache_topology_probe.py \
   run_nfs_lmcache_topology_h100.sbatch \
   /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/nfs_lmcache_topology_probe/
```

## 2. Prepare on the login node

```bash
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/nfs_lmcache_topology_probe

mkdir -p \
  /mnt/shared/gpfs/home/sriramc2/runs/nfs_lmcache_topology_probe

nice -n 10 python3 nfs_lmcache_topology_probe.py prepare \
  --root /mnt/shared/gpfs/home/sriramc2/runs/nfs_lmcache_topology_probe \
  --file-mib 80 \
  --immutable-files 256 \
  --writer-files 48 \
  --overlap-files 48 \
  2>&1 | tee \
  /mnt/shared/gpfs/home/sriramc2/runs/nfs_lmcache_topology_probe/prepare_console.log
```

Preparation calls `fdatasync()` before recording each file as complete. Existing
correctly-sized files are reused unless `--overwrite` is supplied.

## 3. Submit

```bash
sbatch run_nfs_lmcache_topology_h100.sbatch
```

The default phase matrix is:

1. `buffered_baseline`
   - two buffered readers
   - no writers
   - 30 seconds

2. `buffered_6w_separate`
   - two buffered readers
   - six buffered writers
   - readers and writers use different fixed files
   - 120 seconds

3. `buffered_6w_same_files`
   - two buffered readers
   - six buffered writers
   - both sides cycle through the same fixed files
   - short reads are expected to be possible because writers use `O_TRUNC`
   - a 10 ms backoff follows each short read to prevent a tight log-flood loop
   - 120 seconds

4. `direct_6w_separate`
   - two O_DIRECT readers
   - six buffered writers
   - separate fixed files
   - 120 seconds

5. `buffered_recovery`
   - two buffered readers
   - no writers
   - 30 seconds

Writers begin 10 seconds before readers in stressed phases.

## Buffered reader semantics

The buffered path is intentionally:

```python
with open(path, "rb") as handle:
    nread = handle.readinto(bytearray_80_mib)
```

The default also issues `POSIX_FADV_DONTNEED` before and after each read. This
does not turn the operation into O_DIRECT; it is a best-effort hint to prevent
the repeated 20 GiB scan from becoming a pure local-page-cache benchmark.

Use `--no-cold-hint` to match LMCache more literally, accepting that later scans
may become cache hits.

## Detailed logs

Each phase directory contains:

- `parent.jsonl`
- `sampler.jsonl`
- `reader_0.jsonl`
- `reader_1.jsonl`
- `writer_0.jsonl` through `writer_5.jsonl`

`reader_operation` records include:

- `open_s`
- `read_s`
- `close_s`
- `read_mib_per_s`
- `short_read`
- `/proc/self/io` deltas

`writer_operation` records include:

- `open_s`
- `write_s`
- `close_s`
- application-level MiB/s
- `/proc/self/io` deltas

`system_sample` records include:

- network RX/TX totals and one-second deltas
- per-interface deltas
- Dirty and Writeback
- selected `/proc/vmstat` counters
- TCP retransmission and timeout deltas
- raw `/proc/net/rpc/nfs` client counters

After each stressed phase, the parent calls `fsync()` on every fixed writer file
and records the remaining synchronization cost.

## Interpretation

### Buffered separate-file phase approaches ~10 MiB/s

This reproduces the physical collapse with no same-file race. The main cause is
sustained shared NFS/TCP contention plus the buffered read path.

### Only same-file phase collapses

NFS cache coherence, truncate/rewrite races, or same-inode serialization is a
major missing ingredient in the original three-writer test.

### O_DIRECT stays fast while buffered collapses

The Linux/NFS buffered page-cache or read-ahead/writeback interaction is central.

### Both buffered and O_DIRECT collapse

The bottleneck is lower than the compute-node page cache: NFS transport, server,
or backing storage.

### Many short reads in the same-file phase

That is a real semantic race caused by opening an existing path with `O_TRUNC`.
It is useful diagnostically but should not be interpreted as valid cache data.

### Large TCP retransmission or timeout deltas

Network loss or congestion is materially contributing.

### Network traffic approaches the link ceiling without retransmissions

Bandwidth sharing, rather than packet loss, is the stronger explanation.

### Large post-phase fsync time

Writers reported completion while substantial NFS commit work remained.

## Quick reduced smoke test

Before the full run:

```bash
python3 nfs_lmcache_topology_probe.py run \
  --root /mnt/shared/gpfs/home/sriramc2/runs/nfs_lmcache_topology_probe \
  --stress-duration-s 10 \
  --baseline-duration-s 5 \
  --writer-headstart-s 2
```

## Important throughput detail

The summary reports aggregate reader bandwidth across both reader processes and
aggregate writer bandwidth across all six writer processes. Each aggregate uses
the corresponding reader or writer wall interval, including completion of the
last operation already in progress when the stop signal arrives. The summary
also records `stop_signal_elapsed_s` and `join_elapsed_s` separately.

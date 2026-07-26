# Standalone GPFS mixed read/write probe

This benchmark tests whether one 80 MiB physical reader collapses toward the
~10 MiB/s seen by LMCache as concurrent buffered writers increase from 0 to 3.
It imports neither vLLM nor LMCache.

## What it measures

- Five phases: `0,1,2,3,0` concurrent writers.
- One serialized reader in every phase.
- Each phase uses 25 different immutable 80 MiB reader files (2,000 MiB).
- Writers create ordinary buffered 80 MiB files and do **not** call fsync before
  reporting application-level completion.
- After the reader finishes, the parent fsyncs only that phase's writer files.
  This measures deferred work remaining after `write()` and `close()` returned.
- The reader attempts O_DIRECT with an aligned 8 MiB buffer. If GPFS rejects
  O_DIRECT, the fallback and reason are logged, and buffered reads plus
  `POSIX_FADV_DONTNEED` are used.
- A one-second sampler records `Dirty`, `Writeback`, `Cached`, `MemAvailable`,
  selected `/proc/vmstat` counters, and network byte deltas.

## 1. Copy the tools into the repository

```bash
mkdir -p /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/gpfs_probe
cp prepare_gpfs_probe.py \
   gpfs_mixed_io_probe.py \
   summarize_gpfs_probe.py \
   run_gpfs_probe_h100.sbatch \
   /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/gpfs_probe/
```

## 2. Login-node preparation

This creates 128 x 80 MiB = 10 GiB of immutable, fdatasync'd reader data.
It is sequential and does not use CUDA. Run it only if login-node policy permits
bulk sequential GPFS I/O.

```bash
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/gpfs_probe
mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/gpfs_io_probe

nice -n 10 python3 prepare_gpfs_probe.py \
  --root /mnt/shared/gpfs/home/sriramc2/runs/gpfs_io_probe \
  --num-files 128 \
  --file-mib 80 \
  2>&1 | tee /mnt/shared/gpfs/home/sriramc2/runs/gpfs_io_probe/prep.log
```

Re-running the command validates and reuses correctly sized files. Add
`--overwrite` only when intentionally replacing them.

## 3. Submit on the H100 node

```bash
sbatch run_gpfs_probe_h100.sbatch
```

The job requests one H100 only to select the same node used by the LMCache
experiments. The benchmark itself does not call CUDA.

## 4. Output

The Slurm log ends with:

```text
phase writers mode read_MiB/s median_file_s p95_file_s writer_GiB app_writer_MiB/s deferred_fsync_s
```

Per-phase JSONL logs are under:

```text
/mnt/shared/gpfs/home/sriramc2/runs/gpfs_io_probe/runs/<run-id>/
```

Important events:

- `reader_file_done`: per-file open/read/close time and `/proc/self/io` delta.
- `writer_file_done`: application-level write time and apparent throughput.
- `system_sample`: dirty/writeback memory, VM counters, and network traffic.
- `phase_file_fsync_done`: remaining flush latency per writer file.
- `phase_flush_done`: total deferred flush time after writers said “done”.
- `reader_direct_fallback`: O_DIRECT was rejected and buffered fallback was used.

## Interpretation

- Baseline fast, three-writer phase near 10 MiB/s:
  mixed GPFS read/write contention is reproduced outside LMCache.
- Every direct-read phase near 10 MiB/s:
  the cold/direct GPFS path is already slow on that node.
- Read rate falls as `Dirty`/`Writeback` and writer count rise:
  background writeback pressure is strongly implicated.
- Application writes look fast but `deferred_fsync_s` is large:
  `write()`/`close()` returned before backing-store work was complete.
- Direct reads stay fast with three writers:
  LMCache-specific buffering, duplicate same-key rewrites, or its task topology
  is needed to reproduce the collapse.
- Final zero-writer phase remains slow:
  queued/background writeback continues affecting reads after writer processes
  have stopped.

## Resource footprint

- Permanent reader data: 10 GiB.
- Maximum temporary writer data in the three-writer phase: about 30 GiB.
- Temporary writer files are fsync'd and deleted after each phase.
- Process memory: roughly 80 MiB per writer plus overhead.

## Caveats

- O_DIRECT bypasses normal client page cache, but GPFS server-side caches may
  still serve data.
- `/proc/self/io` may not fully account for network filesystem traffic. Treat
  syscall timing and phase-to-phase comparisons as primary evidence.
- Do not run this benchmark at the same time as the LMCache workload.

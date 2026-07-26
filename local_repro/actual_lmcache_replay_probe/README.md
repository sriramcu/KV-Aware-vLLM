# Actual LMCache file replay probe

This benchmark reads the **real files from an existing LMCache cache tree**.
It does not import LMCache or vLLM, and it never modifies the cache tree.

The default experiment is:

1. Two serial buffered readers, no writers.
2. Two serial buffered readers plus six writers creating unique 80 MiB files.
3. Two serial buffered readers after the writers stop.

Each reader corresponds to one tensor-parallel rank. Rank-0 and rank-1 files are
paired by their LMCache filename:

```text
model@2@0@chunk_hash@dtype.pt
model@2@1@chunk_hash@dtype.pt
```

Synthetic writes use a different, uniquely marked scratch directory on the same
NFS mount.

## Safety

The script:

- never opens an LMCache file for writing;
- refuses to place scratch inside the cache directory;
- checks that the LMCache directory remains unchanged for five seconds before
  starting;
- creates writer files with `O_EXCL`;
- deletes only a run-specific scratch directory containing its own marker;
- refuses scratch cleanup if that marker is missing.

Run only after the original vLLM/LMCache job has exited.

## 1. Install

```bash
mkdir -p \
  /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/actual_lmcache_replay_probe

cp actual_lmcache_replay_probe.py \
   run_actual_lmcache_replay_h100.sbatch \
   /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/actual_lmcache_replay_probe/
```

## 2. Build the manifest on the login node

For job 17383:

```bash
TOOLS=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/actual_lmcache_replay_probe
CACHE=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/lmcache_vllm/hotpotqa_wo_gnn_17383
OUT=/mnt/shared/gpfs/home/sriramc2/runs/actual_lmcache_replay_probe

mkdir -p "$OUT"

python3 "$TOOLS/actual_lmcache_replay_probe.py" inventory \
  --cache-root "$CACHE" \
  --source-log /path/to/kvaware_h100_wo_gnn_17383.out \
  --output-root "$OUT" \
  --expected-file-mib 80
```

`--source-log` is optional. When supplied, file pairs observed in
`KVIO_FILE_READ` are placed first in their original log order. Remaining pairs
follow in creation-time order.

If the original Slurm output is still in the repository or run directory,
replace `/path/to/...` with that location. Otherwise copy the uploaded job
output to the cluster first.

The inventory operation reads filenames and metadata only. It does not read the
80 MiB payloads.

## 3. Inspect inventory output

The command reports:

```text
pair_count
observed_source_log_pairs
total_paired_gib
incomplete_pair_groups
wrong_size_files
```

The default run needs 1,200 pairs for three phases of 400 pairs. If fewer pairs
remain, the script automatically uses one third of the available pairs per
phase.

## 4. Submit the H100-node test

```bash
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/actual_lmcache_replay_probe
sbatch run_actual_lmcache_replay_h100.sbatch
```

The GPU is not used. The allocation places the process on the same H100/NFS
client class used by the LMCache run.

## Default data volume

Reader side:

```text
400 logical pairs per phase
= 400 × 80 MiB per rank
= 31.25 GiB per reader
= 62.5 GiB total attempted reads per phase
```

A phase stops after 240 seconds if those files have not all completed.

Writer side:

```text
6 writers
up to 160 GiB of unique 80 MiB files in total
10-second head start
```

Writer files remain in place during the recovery phase so deletion traffic does
not contaminate recovery. They are removed only after all measurements finish.

For a smaller first run, edit the Slurm command to:

```bash
--pairs-per-phase 150
--phase-timeout-s 120
--writer-max-scratch-gib 64
```

## Why round-robin partitioning is the default

The manifest is ordered by original `KVIO_FILE_READ` occurrence when available,
then by file creation time. The default distributes adjacent entries across the
three phases:

```text
baseline: pair 0, 3, 6, ...
stress:   pair 1, 4, 7, ...
recovery: pair 2, 5, 8, ...
```

This balances file age and original order across phases while preserving
relative order within each phase. Use `--partition contiguous` only when exact
contiguous early/middle/late slices are desired.

## Reader semantics

The timed operation is the same ordinary buffered call used by the observed
LMCache path:

```python
with open(path, "rb") as handle:
    nread = handle.readinto(bytearray_80_mib)
```

By default, `POSIX_FADV_DONTNEED` is issued before and after the timed read to
reduce compute-node page-cache reuse. It does not turn the read into O_DIRECT.
Use `--no-cold-hint` for a more literal LMCache call sequence.

Every operation logs `/proc/self/io.read_bytes`. It is classified as
`physical=true` when at least half the 80 MiB appeared as actual read I/O.

## Key output columns

```text
read_MiB/s
physical
cached
phys_med_MiB/s
phys_med_s
p95_s
max_s
writer_GiB
netRX
netTX
peakWB
tcpRet
nfsRet
```

`read_MiB/s` is aggregate across both rank readers.

`phys_med_MiB/s` is the median per-file rate among reads classified as physical.

## Interpretation

### Baseline actual files are already around 10 MiB/s

The actual LMCache file population is cold/slow without synthetic writers.
Server-cache misses or backing-store placement are sufficient.

### Baseline is fast, stress falls toward 10 MiB/s

Unique sustained writes reproduce the physical collapse independently of
LMCache's scheduler and queues.

### Stress falls only to hundreds of MiB/s

The actual file population alone is insufficient. LMCache's exact burst shape,
larger write volume, per-rank worker pool, or other application behavior remains
necessary.

### Recovery remains slow

Outstanding NFS/server/backing-store work survives after writer processes stop.

### Recovery immediately returns to baseline

The slow reads require active mixed traffic, rather than a long deferred flush.

### Physical count is small

The test is still seeing client/server caching. Increase the pair count, use
different actual files, or run after more time has elapsed.

## Detailed logs

Each phase contains:

```text
parent.jsonl
sampler.jsonl
reader_rank0.jsonl
reader_rank1.jsonl
writer_0.jsonl ... writer_5.jsonl
```

The run directory also contains:

```text
environment.json
run.jsonl
summary.json
```

Upload `summary.json`, `environment.json`, all three `parent.jsonl` files, and
all three `sampler.jsonl` files for analysis.

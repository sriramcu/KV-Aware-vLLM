# KV async I/O logging-only instrumentation

This patch adds observability only. It does not add cancellation, admission control,
draining, sleeps, cache-policy changes, timeout changes, or concurrency-limit changes.
It does add logging overhead, so use a 32- or 64-request diagnostic job first.

## Apply

```bash
cd /path/to/kvio_logging_patch
./apply_kvio_logging.sh
```

The script defaults to:

- repo: `/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM`
- site-packages: `/mnt/shared/gpfs/home/sriramc2/venvs/kvaware/lib/python3.12/site-packages`

Override with `REPO=...` or `SITE_PACKAGES=...` when needed. It creates backups,
runs both patches in dry-run mode, applies them, and syntax-checks the edited files.

## Run

```bash
MAX_QUESTIONS=32 \
SUBMISSION_BATCH_SIZE=8 \
SRIRAM_KV_IO_TRACE=1 \
bash /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/local_repro/run_driver.sh wo_gnn
```

Keep the existing LMCache YAML and async settings unchanged for a clean diagnostic.

## Important log tags

- `KVIO_LOOKUP_REQUEST_SEND`, `KVIO_LOOKUP_TIMEOUT`, `KVIO_LOOKUP_ALL_RESPONSES`:
  scheduler-visible lookup lifetime and late responses.
- `KVIO_ENGINE_SUBMIT`, `KVIO_ENGINE_FUTURE_DONE`:
  worker coroutine lifetime.
- `KVIO_CONTAINS_DONE`:
  lookup/pinning time per tier.
- `KVIO_SERIALIZER_ENQUEUE`, `KVIO_SERIALIZER_ACQUIRE`, `KVIO_SERIALIZER_RELEASE`:
  per-rank serializer queue wait versus lock-held time.
- `KVIO_STAGE_ALLOC`, `KVIO_STAGE_BATCH_DONE`:
  CPU staging allocation time.
- `KVIO_DISKQ_ENQUEUE`, `KVIO_DISKQ_START`, `KVIO_DISKQ_DONE`:
  disk executor queue wait versus service time; includes active puts/prefetches.
- `KVIO_FILE_READ`, `KVIO_FILE_WRITE`:
  per-file open and `readinto`/write time, byte counts, `/proc/self/io` deltas.
- `KVIO_PUT_TASK_INSERT`, `KVIO_PUT_TASK_REMOVE`:
  cold-write backlog size.
- `KVIO_PHASE_BOUNDARY`, `KVIO_DRIVER_BATCH_START`, `KVIO_DRIVER_BATCH_DONE`:
  alignment with cold/warm phases and generate waves.
- `KVIO_ENGINE_CLEANUP_DEFER`, `KVIO_ENGINE_CLEANUP_DONE`:
  whether aborted lookup cleanup was requested before or after completion.
- `KVIO_DISKQ_CLOSE_ENTER`:
  unfinished disk work at shutdown.

## Extract a compact trace

```bash
grep -E 'KVIO_(PHASE_BOUNDARY|LOOKUP_TIMEOUT|LOOKUP_ALL_RESPONSES|SERIALIZER_|STAGE_|DISKQ_|FILE_READ|FILE_WRITE|PUT_TASK_|ENGINE_CLEANUP)' job.out \
  > kvio_trace.log
```

Send the complete job output and `kvio_trace.log` for correlation.

## Interpretation

- Large `KVIO_SERIALIZER_ACQUIRE queue_wait`, small `KVIO_DISKQ_START queue_wait`:
  head-of-line blocking at `AsyncSingleSerializer`.
- Small serializer wait, large disk-queue wait with active `put` tasks:
  cold writes occupying the four disk-worker threads.
- Small queue waits, large `KVIO_FILE_READ read`:
  physical/page-cache-backed GPFS read is the bottleneck.
- Large `KVIO_STAGE_ALLOC elapsed` or failed allocation before disk submission:
  CPU staging/eviction pressure is the bottleneck.
- A timeout followed much later by `KVIO_LOOKUP_ALL_RESPONSES` and cleanup:
  obsolete work is surviving the scheduler timeout.

`/proc/self/io` is process-wide. When multiple disk-worker threads overlap, its deltas
include I/O from the other threads and should be interpreted together with the queue and
per-file timers, not as an isolated per-file hardware counter.

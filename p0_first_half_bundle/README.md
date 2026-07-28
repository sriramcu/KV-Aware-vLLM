# LMCache P0 first-half patch

Baseline parent commit: `3f9342de59adb5fe649f6bc3823a153fd3ed507d`

This bundle changes only:

- `third_party/LMCache/lmcache/v1/storage_backend/storage_manager.py`
- `third_party/LMCache/lmcache/v1/storage_backend/local_disk_backend.py`

## Scope

**P0-A — pre-pin lookup admission**

A per-worker async semaphore is acquired before `batched_async_contains(..., pin=True)` can pin backend keys or allocate staging buffers. The default window is one lookup. The slot is held until physical prefetch work completes, so lookups waiting behind the existing serializer do not pin early.

**P0-B — bounded disk-put admission**

A per-worker bounded semaphore is acquired before `memory_obj.ref_count_up()` and before the put enters the executor's unbounded priority queue. The default limit is eight pending/active disk puts. Submission blocks outside the storage-manager event-loop thread, providing backpressure to the producer.

This patch deliberately does **not** implement the P0 second half: cancellation-safe resource cleanup and timeout-aware cancellation/discard behavior.

## Apply

```bash
cd /path/to/extracted/p0_first_half_bundle
bash apply_p0_first_half.sh /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
```

The apply script verifies the exact baseline hashes, applies the patch, runs `py_compile`, and runs `git diff --check`. LMCache is installed editable, so no pip reinstall or C++ rebuild is needed.

## Run

```bash
sbatch 03_h100_p0_first_half.sbatch
```

Defaults:

```text
MAX_QUESTIONS=250
SUBMISSION_BATCH_SIZE=8
LMCACHE_P0_LOOKUP_MAX_INFLIGHT=1
LMCACHE_P0_DISK_PUT_MAX_PENDING=8
CLEAN_OLD_LMCACHE=0
```

For a smaller smoke run while preserving the same patch:

```bash
MAX_QUESTIONS=32 SUBMISSION_BATCH_SIZE=8 \
  sbatch --export=ALL,MAX_QUESTIONS=32,SUBMISSION_BATCH_SIZE=8 \
  03_h100_p0_first_half.sbatch
```

## Summarize

```bash
bash summarize_p0_first_half.sh JOB_ID
```

Expected invariants:

- lookup `inflight` never exceeds `1` per worker;
- put `inflight` never exceeds `8` per worker;
- acquire/release counts balance by PID at clean shutdown;
- no event-loop-thread rejection;
- no failed/cancelled put completion;
- the job reaches `=== P0 FIRST HALF JOB PASSED ===`.

`P0_PUT_ADMISSION_STALLED` is not itself a failure. It proves producer backpressure was exercised when a put waited at least five seconds for a slot.

## Revert

```bash
bash revert_p0_first_half.sh /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
```

The reverse script refuses to proceed unless the patch can be reversed cleanly.

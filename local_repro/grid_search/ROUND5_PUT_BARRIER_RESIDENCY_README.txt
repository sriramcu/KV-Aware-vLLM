Round 5: cold->warm PUT barrier x completed-resident PUT dedup
================================================================

Purpose
-------
Isolate two mechanisms suggested by Round 4:
  1. cold asynchronous disk PUTs spilling into warm generation;
  2. warm recomputation offering already-resident disk keys for persistence again.

New independent controls
------------------------
Functional:
  SC_LMCACHE_COLD_WARM_PUT_BARRIER_ENABLE=0|1
  SC_LMCACHE_DISK_RESIDENT_PUT_DEDUP_ENABLE=0|1

Observability:
  SC_LMCACHE_DISK_PUT_RESIDENCY_TRACE_ENABLE=0|1

Barrier tuning (kept fixed by this grid):
  SC_LMCACHE_PUT_BARRIER_TIMEOUT_S=1200
  SC_LMCACHE_PUT_BARRIER_POLL_S=0.25
  SC_LMCACHE_PUT_BARRIER_STABLE_S=1.0
  SC_LMCACHE_PUT_BARRIER_EXPECTED_WORKERS=2

The barrier status directory is created automatically under the job's /tmp.
It does not live on the NFS or scratch cache tier being measured.

Resident-dedup semantics
------------------------
When enabled, LocalDiskBackend checks its existing in-memory resident-key
metadata before taking a disk-PUT MemoryObj reference, before capacity
accounting, and before enqueuing physical I/O. A resident duplicate skips the
write. To minimize semantic change, it preserves the existing behavior's cache
policy refresh (e.g. LRU move-to-MRU) but does not pin/unpin the disk entry.

The existing in-flight same-key PUT suppression remains intact. A second
resident recheck after the put-task claim closes the race where residency could
change between the fast check and admission.

Residency logging
-----------------
SC_LMCACHE_DISK_PUT_RESIDENCY_TRACE_ENABLE=1 emits lines such as:
  [SC_DISK_PUT_RESIDENCY] ... action=enqueue_new resident=False ...
  [SC_DISK_PUT_RESIDENCY] ... action=enqueue_resident_rewrite resident=True ...
  [SC_DISK_PUT_RESIDENCY] ... action=skip_resident resident=True ...
  [SC_DISK_PUT_RESIDENCY] ... action=skip_inflight ...

Thus the no-dedup cells directly count confirmed completed-resident rewrites,
while the dedup cells count the physical writes avoided.

Barrier logging
---------------
Barrier-enabled cells emit:
  [SC_PUT_BARRIER_START]
  [SC_PUT_BARRIER_POLL]
  [SC_PUT_BARRIER_DONE] elapsed=...
  [SC_PUT_BARRIER_TIMEOUT] ...   (failure case)

Each TP rank publishes rank_<rank>.json with pending_puts, executor queue depth,
and active task counts. The barrier requires both ranks to report
pending_puts=0 continuously for 1 second before warm timing starts. Immediately
after release, the driver writes a node-local sentinel so workers stop publishing
barrier status before the measured warm phase.

Frozen experiment settings
--------------------------
  dataset                    medical
  GNN                        off
  CPU LMCache                20 GiB per TP rank
  max_num_seqs               16
  submission batch           250
  scheduler admission        enabled
  scheduler max inflight     12
  scheduler wait             0 ms
  worker lookup admission    off
  disk-put admission         off
  TP                         2
  default node               codenimbus-003-1

Scheduler max=12 is deliberately frozen for all six cells. It matches the
Round-4 NFS-B / scratch-D comparison and the scratch setting where PUT/read
executor interference was prominent, avoiding a scheduler-width confound.

Six cells
---------
  A  NFS      no barrier   no resident dedup
  B  NFS      barrier      no resident dedup
  C  NFS      barrier      resident dedup
  D  scratch  no barrier   no resident dedup
  E  scratch  barrier      no resident dedup
  F  scratch  barrier      resident dedup

All cells enable residency tracing, including the baseline cells.

Default scratch path
--------------------
  /scratch/sriramc2/lmcache_round5_put_barrier_residency

run_driver.sh clears the exact data directory at each job start. The grid
refuses dangerous scratch roots such as /scratch or /scratch/sriramc2.

Apply and static-check
----------------------
From the repo root:

  git apply --check /path/to/round5_put_barrier_residency.patch
  git apply /path/to/round5_put_barrier_residency.patch

No CUDA/C++ rebuild is required: the LMCache changes are Python-only and the
existing environment uses the vendored LMCache editable source.

Then:

  python -m py_compile \
    local_repro/cpu_offload_lmcache_sriram.py \
    local_repro/grid_search/run_round5_put_barrier_residency_grid.py \
    third_party/LMCache/lmcache/v1/storage_backend/local_disk_backend.py

  bash -n local_repro/run_driver.sh
  bash -n local_repro/sc_lmcache_knobs.sh
  bash -n local_repro/grid_search/submit_round5_put_barrier_residency_grid.sbatch

The Round-5 master also refuses to submit if the existing CPU-size override
SC_LMCACHE_MAX_LOCAL_CPU_SIZE is not visible in the active driver.

Smoke grid: 1 question per cell
-------------------------------
First inspect:

  python local_repro/grid_search/run_round5_put_barrier_residency_grid.py \
    --max-questions 1 --print-grid

Then run all six smoke cells sequentially:

  python local_repro/grid_search/run_round5_put_barrier_residency_grid.py \
    --max-questions 1

Ctrl-C stops only the foreground controller and leaves the active Slurm job
alone. Re-run the exact command to resume.

Useful smoke checks after completion:

  grep -R "SC_DISK_PUT_RESIDENCY\|SC_PUT_BARRIER_" \
    ~/runs/kvaware_repro/pressure_grid/sc-round5-put-barrier-residency-v1-cpu20_maxq_1/runs/*/slurm-*.out

For barrier-enabled cells verify there are two rank files reported in barrier
poll/done records and a [SC_PUT_BARRIER_DONE]. For baseline cells verify
[SC_PUT_BARRIER_DISABLED].

Full grid: 50 questions per cell
--------------------------------
After the smoke grid passes:

  python local_repro/grid_search/run_round5_put_barrier_residency_grid.py \
    --max-questions 50

Persistent state:
  ~/runs/kvaware_repro/pressure_grid/sc-round5-put-barrier-residency-v1-cpu20_maxq_50/

Final archive:
  ~/runs/kvaware_repro/pressure_grid/sc-round5-put-barrier-residency-v1-cpu20_maxq_50.zip

The per-run summary.json additionally records:
  put_residency_actions
  put_residency_actions_by_phase
  put_task_insert_count_by_phase
  put_barrier_elapsed_seconds
  put_barrier_timeout_count

Primary comparisons
-------------------
A vs B: cost/benefit of draining cold PUTs on NFS.
B vs C: incremental effect of resident-write suppression after cold PUTs drain.
D vs E: cost/benefit of draining cold PUTs on scratch.
E vs F: incremental effect of resident-write suppression on scratch.
A vs D, B vs E, C vs F: storage-dependent behavior under matched controls.

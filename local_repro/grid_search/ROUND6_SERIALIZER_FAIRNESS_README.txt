Round 6: CPU-preferred single-serializer fairness
================================================

Purpose
-------
Test whether backend-aware ordering of the existing single LMCache async
serializer reduces avoidable CPU-behind-disk head-of-line blocking and improves
end-to-end warm inference on shared NFS.

This patch DOES NOT add serializer parallelism. Exactly one backend operation is
active at a time, preserving the original race-avoidance intent. It changes only
which waiting backend operation gets the serializer next.

Knobs
-----
SC_LMCACHE_SERIALIZER_FAIRNESS_ENABLE=0|1
    0: preserve the existing asyncio.Lock serializer behavior exactly.
    1: use CPU-preferred deterministic weighted selection.

SC_LMCACHE_SERIALIZER_CPU_BURST_RATIO=N
    Positive integer. N means: while CPU and disk are both continuously waiting,
    allow at most N CPU selections before forcing one disk selection. The ':1'
    denominator is implied. If only one tier is waiting, it runs immediately.

Examples:
    N=2  -> up to 2 CPU selections, then 1 disk selection under sustained contention
    N=4  -> up to 4 CPU selections, then 1 disk selection under sustained contention

Important: fairness disabled is the true job-19109 baseline. Fairness enabled
with N=1 is an alternating 1:1 policy under contention and is NOT exactly the
same as the old FIFO-ish asyncio.Lock behavior.

Round-6 frozen environment
--------------------------
All cells use:
  shared NFS (driver default LMCache data directory)
  Medical dataset
  batch=250
  max_num_seqs=16
  CPU LMCache=20 GiB/rank
  TP=2
  GNN off
  scheduler admission on, max inflight=12, wait=0 ms
  worker lookup gate off
  disk-PUT admission gate off
  cold->warm PUT barrier on
  resident PUT dedup on
  existing IO/load/memory/residency tracing on

Only SC_LMCACHE_SERIALIZER_CPU_BURST_RATIO changes across cells:
  R2  = 2
  R4  = 4
  R8  = 8
  R16 = 16

The historical job 19109 is the fairness-disabled Q250 baseline.

Fairness tracing
----------------
The patch preserves existing SC_IO_SERIALIZER_* traces and adds:
  [SC_IO_FAIR_ENQUEUE]
  [SC_IO_FAIR_SELECT]
  [SC_IO_FAIR_RELEASE]
  [SC_IO_FAIR_WAIT_CANCELLED]

SC_IO_FAIR_SELECT includes:
  backend/tier
  reason (cpu_only, cpu_preferred, disk_only, disk_starvation_guard)
  CPU/disk waiter counts
  current CPU burst
  configured ratio
  queue wait

The resumable runner also records warm-only fairness selection counts,
scheduler admit/reject counts, lookup timeout counts, and serializer queue-wait
p50/p90/max by backend into each summary.json.

Recommended use
---------------
1. Apply the incremental patch on top of the already-applied Round-5 v3 patch.

2. Smoke-test all four ratios with one question:

   python local_repro/grid_search/run_round6_serializer_fairness_grid.py \
       --max-questions 1 --print-grid

   python local_repro/grid_search/run_round6_serializer_fairness_grid.py \
       --max-questions 1

3. If smoke is clean, run Q250:

   python local_repro/grid_search/run_round6_serializer_fairness_grid.py \
       --max-questions 250

State/output directory:
  ~/runs/kvaware_repro/pressure_grid/
    sc-round6-serializer-fairness-v1-cpu20_maxq_<Q>/

When all four cells are terminal the runner automatically creates a sibling ZIP.

Resume semantics
----------------
The Python controller is foreground and resumable. Ctrl-C stops only the
controller; it does NOT cancel the currently active Slurm job. Rerun the same
command to resume from grid_state.json.

Expected proof-of-concept signal
--------------------------------
Compared with job 19109, a useful fairness policy should primarily reduce
LocalCPUBackend serializer queue wait while allowing disk to continue making
progress. Secondary expected effects are fewer warm lookup timeouts, shorter
scheduler-permit lifetimes, fewer scheduler rejections, more useful LMCache
contribution, less recomputation, and lower warm wall time.

Do not require NFS disk queueing itself to disappear. The target is to stop slow
disk traffic from unnecessarily imposing its queue latency on fast CPU hits.

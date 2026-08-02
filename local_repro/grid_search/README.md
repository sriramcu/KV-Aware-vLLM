# SC LMCache pressure grid

This directory contains a foreground, resumable Slurm grid runner. It submits
one H100 job at a time; the master process itself is not a Slurm job.

## Smoke grid

From the repository root:

```bash
./local_repro/grid_search/run_smoke_grid.sh
```

The smoke wrapper runs all 16 configurations with `MAX_QUESTIONS=1`. To inspect
the grid without submitting anything:

```bash
./local_repro/grid_search/run_smoke_grid.sh --print-grid
```

The default persistent directory is:

```text
~/runs/kvaware_repro/pressure_grid/sc-pressure-grid-v1_maxq_1/
```

Important files:

```text
grid_state.json       authoritative resumable state
grid_progress.txt     human-readable status, atomically refreshed every poll
active_job_id.txt     master-managed copy of the current Slurm job ID
runs/*/config.json    exact configuration for each grid point
runs/*/summary.json   simple metrics parsed after a terminal job
runs/*/slurm-*.out    stdout for each submitted attempt
runs/*/slurm-*.err    stderr for each submitted attempt
```

When all configurations are terminal, the runner creates:

```text
~/runs/kvaware_repro/pressure_grid/sc-pressure-grid-v1_maxq_1.zip
```

## Interrupt and resume

Between jobs the runner prints an interruptible four-second warning countdown.
After the countdown it briefly defers Ctrl-C while it writes the current state,
submits with `sbatch`, captures the job ID, and persists that ID. It then prints
that Ctrl-C is safe again.

Ctrl-C while waiting stops only the foreground master. It does not cancel the
Slurm job. Run the same command again to resume. The master uses
`grid_state.json` as the authority and repairs `active_job_id.txt` when needed.

A real Slurm job that reaches `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, `CANCELLED`,
or another terminal failure is recorded and the grid advances without retrying.
Only an unfinished run whose persisted job ID cannot be associated with any
`squeue` or `sacct` record after the recovery grace is resubmitted.

A run is successful only when Slurm reports `COMPLETED` and its stdout contains
`=== DONE ===`.

## Real grid

After validating the smoke archive, use a distinct automatically selected state
directory for 250 questions:

```bash
python local_repro/grid_search/run_pressure_grid.py --max-questions 250
```

The grid is one-factor-at-a-time around the fresh refactored baseline:

- scheduler admission: default, max 1, max 12, wait 100 ms, wait 1000 ms;
- worker admission per TP rank: max 1, 4, 12;
- disk-put admission: max pending 8, 1, 32;
- dataset: Medical versus HotpotQA;
- submission batch size: 50 versus 10 and 250;
- vLLM `max_num_seqs`: 4 versus 12.

GNN remains disabled. The fixed diagnostics are IO, load, memory pressure, and
the driver resource monitor; heavier request, lookup-stack, tier, and snapshot
traces remain disabled.

## Simple structured metrics

When `SC_LMCACHE_IO_TRACE_ENABLE=1`, the driver emits JSON log records for
exact prompt/output-token counts, finish reasons, phase throughput, and
cold/warm output consistency. Setting that existing switch to `0` disables all
four lightweight driver-metric records as well as the detailed IO trace:

```text
[SC_DRIVER_REQUEST_METRIC] {...}
[SC_DRIVER_PHASE_SUMMARY] {...}
[SC_DRIVER_OUTPUT_CONSISTENCY] {...}
[SC_DRIVER_METRICS_DONE]
```

The master copies the phase summaries and a few inexpensive failure counters
into each `summary.json`. Detailed pressure and contention analysis should still
be performed from the complete `.out` logs.

# CodeNimbus H100 Q650 ten-cell matrix (original passage ordering)

This package queues P1–P8 plus P11 and P12, *not* P9/P10. It works against the
**current patched repository**, rather than bundling old copies of the launchers.

## First: install on the login node

From the repo root:

```bash
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
unzip -o ~/h100_ten_job_launch.zip -d .
python scripts/h100_10batch/install.py --repo "$PWD"
```

The installer **does not overwrite** either original launcher. It reads and
clones your current `scripts/run_h100_bayes_sweep_cell.sbatch` and
`scripts/run_h100_mp_nognn_q650_current_cleanall.sbatch` into
`scripts/h100_10batch/`, after applying two targeted changes:

* The Bayesian sweep clone propagates the requested fixed/pseudo-count posterior.
* Both Bayesian and no-GNN modes honor **`CLEAN_OLD_LMCACHE=1` (the default)**.
  Before every experiment, the guarded cleaner removes all children of
  `/scratch2/sriramc2` but preserves the root directory. No-GNN performs this
  in the generated no-GNN clone; Bayesian performs it in the existing common
  launcher. Unique L2 paths are still used.

**WARNING:** Singleton serializes this batch only. The full-root cleanup can
remove scratch files belonging to **other concurrently running jobs** under
`/scratch2/sriramc2`. Don't run unrelated scratch-using experiments concurrently.

The installer will refuse to proceed if required current physical L2 capacity
patches are absent. For Bayesian jobs it also verifies that the ordinal
expected-score prior exists at:
`/mnt/shared/gpfs/home/sriramc2/runs/kvaware/h100_expected_ordinal_exploratory.json`.

## Second: submit all ten to the queue

```bash
bash scripts/h100_10batch/submit_10_sbatch_only.sh
```

`submit_10_sbatch_only.sh` contains **exactly ten `sbatch` commands and no
other lines**, so you can run any line separately for re-submissions.

Every submission uses `--dependency=singleton --job-name=kvq650-10batch`.
Consequently **all ten jobs enter the queue immediately** but **at most one
of this batch runs at a time**; each next job can run after the previous ends,
including if the previous fails. That prevents port collisions and keeps the
batch's scratch usage nonconcurrent. Jobs from elsewhere using the same scratch
root should still avoid touching the job-specific L2 paths.

Each successive experiment clears the prior run's scratch cache. The final
experiment's cache remains until another clean run starts or you remove it.
All logs/results under the GPFS run directory are preserved.

All runs: H100 partition `gpu-h100-dchen`, TP2 Llama-3.3-70B FP8,
`cpus-per-task=16`, 6h walltime, Q650, B100, original passage order,
seeded cold shuffle seed0, warm random seed1, 128 output tokens,
max_num_seqs16, L1 CPU setting as below; physical L2 capacity equals the
specified Disk budget, and L2 watermark is 0.9. No `--mem` is included.

## Jobs

| Job | Experiment | GPU policy | L1 CPU GiB | Physical L2 GiB | Disk PUT cap GiB | Posterior | Prior parameters |
|---|---|---:|---:|---:|---:|---|---|
| P1 | no GNN vs 29920 | n/a | 225 | 500 | 0.25 | n/a | n/a |
| P2 | ordinal GNN | 25 | 225 | 500 | 0.25 | fixed | w=0.5 |
| P3 | ordinal GNN | 30 | 225 | 300 | 0.25 | pseudo_count | score=1000, cost=100 |
| P4 | no GNN vs 29918/29919 | n/a | 225 | 700 | 0.25 | n/a | n/a |
| P5 | no GNN vs 29889 | n/a | 225 | 300 | 0.25 | n/a | n/a |
| P6 | ordinal GNN | 25 | 225 | 500 | 0.25 | fixed | w=0.2 |
| P7 | ordinal GNN | 30 | 225 | 300 | 0.25 | fixed | w=0.2 |
| P8 | no GNN vs 29890 | n/a | 100 | 100 | **0** | n/a | n/a |
| P11 | ordinal GNN vs 29920 | 25 | 225 | 500 | **0** | fixed | w=1 |
| P12 | no GNN vs P1 | n/a | 225 | 500 | **0** | n/a | n/a |

An L2 PUT cap of `0` means **uncapped**. No-GNN has no Bayesian GPU *policy*
budget; vLLM TP2/GPU configuration is held fixed for comparison.

## Inspection

```bash
squeue -u "$USER" -o '%.18i %.22j %.2t %.12M %.30R'
```

Logs: `/mnt/shared/gpfs/home/sriramc2/runs/kvaware/kvq650-10batch_<SLURM_JOB_ID>.out` and `.err`.

The run directory names retain labels such as `c_P2_bayes` and `c_P1_nognn`,
so you can associate runs with the experiment grid.

Check each completed run's `run_config.json` before analysis, particularly
`bayes_posterior_mode`, `bayes_score_prior_count`, `bayes_cost_prior_count`,
`lmcache_l2_size_gb`, `store_max_inflight_gb`, and original passage order.

## Methodological cautions

* This runs all ten as requested, but it does not guarantee their queue start
  times. Singleton serializes only jobs with this job name submitted by you.
* Physically capped L2 should be installed in both the current Bayesian/common
  and no-GNN baseline launchers (the installer verifies required markers).
* No-GNN is **not** identical to the GNN policy with the model removed: learned
  retention bias is disabled but ordinary vLLM prefix caching is still enabled.
* The score prior was built from the exploratory medical workload; do not
  present this ten-run group as independent test-set generalization.
* Do **not** use older submission scripts that may clear the whole scratch root
  concurrently with these jobs.

## Default scratch cleanup and reruns

`scripts/h100_10batch/run_h100_10batch_cell.sbatch` exports
`CLEAN_OLD_LMCACHE=1`. Set to `0` only if you need to preserve scratch contents.
The 10 sbatch lines remain unchanged and use `singleton`, which waits for jobs
with the same name/user to finish, **even after a job fails**. `afterok` instead
blocks on failure, and is not suitable for this overnight sweep.

If you previously submitted jobs using the old wrapper, Slurm may have already
spooled the old wrapper with `CLEAN_OLD_LMCACHE=0`; do not assume editing the
file on disk changes those submissions. Re-submit those jobs after verifying
their currently queued settings.

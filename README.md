## v3 installer correction (after v2 assertion failure)

The v2 installer might successfully patch the mapper but then fail a test
because it incorrectly demands the **entire** `cost_counts` dictionary contain
*only* `cpu` and `disk`. Some mapper variants track extra cost tiers.
The v3 behavior test records all counters/sums and verifies:

- OFF: no cost counter, accumulated cost, or estimated CPU/Disk cost changes.
- ON: CPU count increases exactly 12 and Disk count exactly 3, with nonzero effect on estimates.
- Profile flag agrees with switch, score observations still work.

The v3 installer is idempotent: if the v2 patch is present, it does not
re-patch or overwrite the `bayesian_mapper.py.next9_before_cost_guard.bak` file.
If a real adaptation failure exists, v3 aborts with **before/after diagnostics**.
No Slurm jobs or scratch cleanups happen during installation or checks.

After replacing the package:

```
unzip -o ~/h100_next9_launch_v3.zip -d .
python scripts/h100_next9/install.py --repo "$PWD"
python scripts/h100_next9/install.py --repo "$PWD" --check-only
```

# H100 next-nine: arbitrary-prior robustness (7) + proportional scarcity (2)

This bundle **reuses your existing, working `scripts/h100_next8/*_variable.*` launchers** and does not reinstall or overwrite them. It changes only the pure-Python Bayesian mapper, installing a guarded `KV_BAYES_ADAPT_COST` switch. It intentionally fixes the former `Invalid cell: next8_*` error by using `c_next9_*` names accepted by the shared dispatcher.

## Install (CodeNimbus login node, no H100 required)

Copy the zip into your repo/home and run:

```bash
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
unzip -o ~/h100_next9_launch.zip -d .
python scripts/h100_next9/install.py --repo "$PWD"
```

The installer creates `scripts/h100_next9/priors/calibrated_scores_equal_cost_10ms.json` by **preserving the calibrated histogram** in `/mnt/shared/gpfs/home/sriramc2/runs/kvaware/h100_expected_ordinal_exploratory.json` and changing *only* the cost prior to 10/10/10 ms. The original calibrated file remains unmodified. The packaged `uniform_scores_equal_cost_10ms.json` has 256 equal-width bins on [0,100], each probability 1/256, and costs 10/10/10 ms.

The v2 installer uses syntax-aware matching for the cost-update guard, to tolerate harmless source differences in `observe_cost()` while refusing ambiguous changes. The first v1 attempt may have already created the hybrid JSON; rerunning is safe and does not overwrite it.

The installer backs up the original `Hierarchical_KV/shortq_placement/bayesian_mapper.py` as `bayesian_mapper.py.next9_before_cost_guard.bak`, adds `KV_BAYES_ADAPT_COST=0|1` (default ON), and tests ON/OFF posterior behavior with a synthetic single-wave score source. The switch **ignores observations in `observe_cost()`** when OFF; the direct-cost instrumentation and LMCache log parsing stay ON. It also writes a `cost_adaptation_enabled` boolean in every `bayes_wave_history.json` profile. No vLLM/LMCache C++ or storage code is changed.

To rerun the checks later without changing anything:

```bash
python scripts/h100_next9/install.py --repo "$PWD" --check-only
```

Optional case-by-case login-node dry run, with **no Slurm allocation and no scratch deletion**:

```bash
DRY_RUN=1 CASE_ID=P01_CAL_SCORE_COSTOFF bash scripts/h100_next9/run_h100_next9_cell.sbatch
DRY_RUN=1 CASE_ID=SCAR_NG bash scripts/h100_next9/run_h100_next9_cell.sbatch
```

## Run

Submit exactly nine serialized jobs:

```bash
bash scripts/h100_next9/submit_9_sbatch_only.sh
```

They share `--dependency=singleton --job-name=kvq650-10batch` with the preceding test jobs so there are no concurrent *same-job-name* scratch cleanups. **Other active users/jobs of `/scratch2/sriramc2` are NOT protected.** Every run intentionally deletes *all contents* of `/scratch2/sriramc2` and preserves the root directory. Results/logs reside on GPFS. This is destructive; confirm no unrelated jobs are using that scratch before submitting. No `--mem` directive is used. Each job requests 2×H100 TP2, 16 CPU cores, 6 hours.

### Experimental grid

All runs are Medical Q650/B100, original passage ordering, cold seed0/warm seed1, 512-token chunks, max_num_seqs16, capped Disk PUT=0.25GiB, full cold-to-warm store barrier, all GNN using expected-ordinal scores. All prior ablations use GPU25/CPU225/Disk500 and **score adaptation ON**. Fixed-weight cells use `w=0.5`. Pseudo-count cells retain `w=0.5` in their logs but the pseudo-count equations use `kA/kB`, not `w`.

| Cell | Score prior | Cost prior | Score adaptation | Cost adaptation | Posterior | GPU/CPU/Disk GiB |
|---|---|---|---|---|---|---|
| P01_CAL_SCORE_COSTOFF | Calibrated medical histogram | Equal 10/10/10 | ON | OFF | Fixed w=0.5 | 25/225/500 |
| P02_UNIF_BOTH_FIXED | Uniform | Equal 10/10/10 | ON | ON | Fixed w=0.5 | 25/225/500 |
| P03_UNIF_SCORE_FIXED | Uniform | Equal 10/10/10 | ON | OFF | Fixed w=0.5 | 25/225/500 |
| P04_UNIF_BOTH_HI | Uniform | Equal 10/10/10 | ON | ON | Pseudo kA=1000/kB=100 | 25/225/500 |
| P05_UNIF_BOTH_LO | Uniform | Equal 10/10/10 | ON | ON | Pseudo kA=10/kB=1 | 25/225/500 |
| P06_UNIF_SCORE_HI | Uniform | Equal 10/10/10 | ON | OFF | Pseudo kA=1000/kB=100† | 25/225/500 |
| P07_UNIF_SCORE_LO | Uniform | Equal 10/10/10 | ON | OFF | Pseudo kA=10/kB=1† | 25/225/500 |
| SCAR_NG | n/a | n/a | n/a | n/a | No GNN | n/a/100/222 |
| SCAR_BY | 29954's original calibrated histogram | 29954's original calibrated cost prior | ON | ON | Fixed w=0.5 | 25/100/222 |

† kB is provided as the same nominal setting in the matched cost-ON cell but has **no effect** with cost adaptation OFF.

**Important methodology note:** P01 and P03 hold cost prior and cost adaptation constant, isolating *score-histogram calibration*. P01 is **not** the original calibrated cost-prior configuration; the sole baseline retaining all original calibration is existing job **29954**. SCAR_BY is an exact policy continuation of 29954 with only CPU/Disk capacities changed. SCAR_NG matches physical CPU/Disk capacities. The GPU policy budget is fixed at 25 GiB, so the three tiers are not proportionally scaled; only CPU and Disk are scaled together from 225/500 to 100/222. The cost prior 10ms scale is arbitrary and affects posterior cost updates; report it transparently.

The current Bayesian implementation always freezes `recompute` cost at the prior; only CPU and Disk cost estimates adapt, so "cost adaptation ON" means *CPU/Disk* cost adaptation only.

## Logs & sanity checks

All jobs write GPFS Slurm logs `runs/kvaware/kvq650-10batch_<JOBID>.out` and `.err`, and preserve the normal detailed result folders with unique `c_next9_<CASE_ID>_<POLICY>` labels. `bayes_wave_history.json` includes `cost_adaptation_enabled`: it must be false for P01/P03/P06/P07, true for P02/P04/P05/SCAR_BY. OFF cells should keep mapper `cost_sample_counts={"cpu":0,"disk":0}` but still have direct LMCache timing records in their LMCache logs.

The sampler still updates only after the normal B100 waves. This bundle does **not** attempt intra-batch Bayesian updates and does not change request concurrency or schedule.

If a local H100 env differs from the current known-good next8 launcher, the installer checks fail rather than falling back to the old unpatched launcher.

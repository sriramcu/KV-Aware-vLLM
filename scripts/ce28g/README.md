# CE28G — Cost Estimator 2.0 eight-run grid

**Target:** UIUC CodeNimbus / 2 × H100 TP2 / vLLM 0.29.0 / vendored LMCache MP. This patch is designed to apply **after** the previously installed `kv_h2d_instrumentation/bayes_h2d_diagnostic.patch` and Next9 v3 cost guard. It was assembled against the source snapshot supplied October 23, 2026. It has not been H100-runtime-tested in its CE28G form.

## The experiment, exactly eight cells

All: Medical Short-Q 650 requests, B100, original passage ordering, 128 generated tokens, `max_num_seqs=16`, GPU preference budget 25 GiB, physical CPU L1 225 GiB, physical Disk L2 500 GiB, PUT cap 0.25 GiB, original retrieval / request ordering, frozen old GNN, score posterior ON. Physical eviction trigger watermark remains **80% L1 / 90% L2** throughout. Nominal GPU policy budget remains 25 GiB (not watermark-adjusted).

| Order | CASE_ID | Prior | Posterior | Cost adaptation | Physical eviction ratio L1/L2 | Bayesian capacity adjusted to watermark |
|---|---|---|---|---|---|---|
| 1 | `CAL_ON_BASE` | Original 29954 calibrated scores+costs | fixed w=0.5 | ON | 0.20 / 0.10 | OFF (CPU225/Disk500) |
| 2 | `CAL_OFF_BASE` | Original 29954 calibrated scores+costs | fixed w=0.5 | OFF | 0.20 / 0.10 | OFF (CPU225/Disk500) |
| 3 | `CAL_ON_WM5` | Original 29954 calibrated scores+costs | fixed w=0.5 | ON | 0.05 / 0.05 | ON (CPU180/Disk450) |
| 4 | `CAL_OFF_WM5` | Original 29954 calibrated scores+costs | fixed w=0.5 | OFF | 0.05 / 0.05 | ON (CPU180/Disk450) |
| 5 | `P04_HI_ON` | Uniform scores, equal 10ms costs | pseudo kA=1000, kB=100 | ON | 0.05 / 0.05 | ON |
| 6 | `P05_LO_ON` | Uniform scores, equal 10ms costs | pseudo kA=10, kB=1 | ON | 0.05 / 0.05 | ON |
| 7 | `P06_HI_OFF` | Uniform scores, equal 10ms costs | pseudo kA=1000, kB=100 | OFF | 0.05 / 0.05 | ON |
| 8 | `P07_LO_OFF` | Uniform scores, equal 10ms costs | pseudo kA=10, kB=1 | OFF | 0.05 / 0.05 | ON |

**Causal scope:** 1 vs 2 isolates cost adaptation with corrected cost V2 and original cache policy. 3 vs 4 isolates adaptation with combined watermark adjustment + smaller eviction batches. 1 vs 3 and 2 vs 4 change **both** mapper effective budgets and physical eviction ratios, so cannot isolate either independently. P04/P05 still change both score and cost pseudo-counts; P06/P07 isolate score pseudo-count with cost updates frozen.

## Source changes

- **Permanent H2D estimator:** `[BAYES_H2D_COST_V2]` uses CuPy CUDA-event time for the `gpu_transfer_region_ms` interval (includes staging DMA + scatter). There is no old/new estimator switch and no hardcoded 3.24ms constant. Legacy `[BAYES_H2D_COMPLETE]` stays in LMCache logs for diagnostics **but no longer enters the Bayesian posterior**. Same `gpu_transfer_region_ms` had been validated in job 30048. V2 timing is automatically recorded whenever `LMCACHE_BAYES_TIMING=1`; CE28G enables native timing diagnostics for corroboration.
- **Disk estimate:** retains measured L2-load time and existing request-match/wave-fallback logic; replaces the H2D *component* with the new GPU-region timing. Disk/load overlap and non-identical cost semantics remain limitations.
- **Cost OFF:** `KV_BAYES_ADAPT_COST=0` preserves the Next9 v3 early guard: cost counts and estimates do not change, while all cost measurements still appear in the logs.
- **Actual physical eviction:** L1 CLI now explicitly receives `--eviction-trigger-watermark` and `--eviction-ratio`; L2 FS adapter's existing `trigger_watermark` and `eviction_ratio` are still populated from the same launcher variables.
- **Mapper capacities:** single `KV_BAYES_USE_WATERMARK_CAPACITY=0/1` switch. With 1, the **actual physical** `LMCACHE_L1_EVICTION_TRIGGER` and `LMCACHE_L2_EVICTION_TRIGGER` multiply the configured 225GiB/500GiB budgets. There are **no separate mapper watermark settings**. Physical allocations and watermark values are recorded in the run manifest and `[CE28G_CAPACITY]` line.

## Installation

On login node, from repo root:

```bash
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
# Extract provided bundle to your home or repo, e.g. `unzip CE28G_patch_bundle.zip`.
PATCH=/path/to/ce28g/ce28g.patch
git apply --check "$PATCH"
git apply "$PATCH"
python scripts/ce28g/test_cost_v2_offline.py
python scripts/ce28g/verify_ce28g.py --repo "$PWD" \
  --prior /mnt/shared/gpfs/home/sriramc2/runs/kvaware/h100_expected_ordinal_exploratory.json
python scripts/ce28g/verify_ce28g.py --repo "$PWD" \
  --prior "$PWD/scripts/h100_next9/priors/uniform_scores_equal_cost_10ms.json"
```

If `git apply --check` fails: **do not** use `--reject`, overwrite source, or restore `bayesian_mapper.py.next9_before_cost_guard.bak`. Report the `git apply --check` error and current diff first. No `--mem` Slurm flag, C++ rebuild, or new virtual environment is required by this patch.

## Case audits before any scratch wipe

```bash
for case_id in CAL_ON_BASE CAL_OFF_BASE CAL_ON_WM5 CAL_OFF_WM5 \
               P04_HI_ON P05_LO_ON P06_HI_OFF P07_LO_OFF; do
  DRY_RUN=1 CASE_ID="$case_id" bash scripts/ce28g/run_ce28g_cell.sbatch || exit
 done
```

Case audits validate prerequisites and print settings; they **do not** launch vLLM, allocate GPUs, or touch scratch. The underlying workload does not start until the next step.

## Submit exactly eight jobs, serialized with singleton

Make sure unrelated jobs/processes are not using `/scratch2/sriramc2`. `CLEAN_OLD_LMCACHE=1` recursively removes **all contents** but preserves that root directory; logs, metrics, and summaries are stored on GPFS.

```bash
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
bash scripts/ce28g/submit_8_sbatch_only.sh | tee /mnt/shared/gpfs/home/sriramc2/runs/kvaware/ce28g_submit_ids.txt
```

It uses `--dependency=singleton --job-name=kvq650-10batch` on each case, one node, `--gres=gpu:2`, `--cpus-per-task=16`, six hours, **no** `--mem`. The job IDs are not known until submission; capture `Submitted batch job …` outputs in the file above. Because singleton also waits behind other jobs of the same Slurm job name, do not combine with an uncoordinated scratch-cleanup series.

## Required runtime evidence per job

- `[CE28G_CONFIG]` and `[CE28G_CAPACITY]` lines with intended effective mapper budgets.
- `logs/lmcache.log` contains `[BAYES_H2D_COST_V2]`, legacy `[BAYES_H2D_COMPLETE]` (diagnostic only), `[BAYES_H2D_DIAG]`, and native `[BAYES_H2D_PHASE]` records. Estimate the corresponding event medians / chip rank correlations; compare event counts to detect any lost diagnostic callbacks. Verify **no** `[BAYES_H2D_COST_V2_ERROR]` or `[BAYES_H2D_DIAG_ERROR]`.
- `results/bayes_wave_history.json`: with cost ON, CPU+Disk cost observations should be in the transfer-stage ms/chunk range, with recompute still its separate existing proxy; with cost OFF, CPU/Disk/recompute posteriors and counts must stay at prior/zero.
- Actual L1 eviction triggers, keys evicted, L2 evictions, L1 occupancy, PUT cap skips, TP-shard routing errors, store barrier and failed stores.
- Scheduler-useful GPU/CPU/Disk prompt KV and recomputation **not** raw hit rate; physical reachable prefix gaps and cold/warm E2E. GNN preprocessing cost separately reported.

## Limits / warnings

The H2D corrected estimator is a CUDA transfer-region measure, **not** a proven incremental serving critical-path time. The Disk estimator still adds possibly overlapping L2 and H2D stages. The old input's accidental admission selectivity may be lost, causing greater cache churn. Watermark budget adjustment uses **static soft eviction triggers**, not a live free-space/occupancy predictor. Lower eviction ratios may increase eviction frequency. The diagnostic run is not interchangeable with job 29954 without acknowledging differences in code revision and observations. **This code has passed CPU-only checks, not an H100 integration run**; audit the first cell's V2 marker and actual posterior before interpreting the entire grid.

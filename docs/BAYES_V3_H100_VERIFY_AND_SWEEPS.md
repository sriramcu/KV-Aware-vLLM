# Bayesian KV v3 — calibrated starting costs, completed transfer feedback, renewable admission preferences

This is an **incremental patch on top of bayesian_wave_kv.patch + bayesian_smoke_fix_v2.patch**, themselves on top of canonical_q650_phase_trace.patch. It is opt-in; the original GNN and no-GNN serving paths remain available. Tested **CPU-only**; H100 testing is explicitly required before a sweep.

## Summary of behavioral changes

- `KV_BAYES_POSTERIOR_MODE=fixed` remains the default. `KV_BAYES_PRIOR_WEIGHT=1` continues to average the prior and empirical mean 50/50 even after many observations.
- Optional `KV_BAYES_POSTERIOR_MODE=pseudo_count` uses `KV_BAYES_SCORE_PRIOR_COUNT` and `KV_BAYES_COST_PRIOR_COUNT` (defaults 10 and 5). The score posterior uses Dirichlet-style bin counts, and CPU/disk cost estimates use pseudo-observation weighted means. Neither mode learns recomputation time; this remains a configured reference.
- `KV_BAYES_READMISSION=1` (for Bayesian launcher) recalculates admission *preferences* for hashes encountered in later waves. Actual GPU/CPU/disk cache hits are still served normally regardless of preference. A newly computed KV object uses the current preference when host storage reservation succeeds. Store-controller policy decisions use a captured **staged-tier intent**, not the potentially newer live snapshot. No explicit tier migration or new eviction policy.
- `KV_BAYES_DIRECT_COST_FEEDBACK=1` uses `BAYES_H2D_COMPLETE` stream-completion callbacks (no serving CUDA synchronization), and `BAYES_L2_LOAD_COMPLETE` when L2->L1 load finishes. The disk cost combines an empirical L2 stage mean with the H2D-completion mean. These are **better-timed but still approximate and not exclusive device/E2E costs**. Old `KV_BAYES_USE_COST_PROXIES=1` remains only as legacy fallback; do not enable both.
- New simple one-time synthetic prior utility independently benchmarks pinned H2D and buffered disk read + H2D for the two GPUs; **recompute prior is NOT calibrated** and remains the original configured value. Score histogram bin edges/probabilities are copied unchanged. OS page caching can affect buffered disk reads.
- Wave history includes changed hash admission preferences (not the same as actual renewed physical stores). LMCache logs real reservation events `BAYES_NEW_STORE_INTENT` and `BAYES_READMISSION_STORE`. Job root now includes `slurm_and_source.txt` recording CPU allocation, Slurm TRES and git revision/dirty state.
- CPU/disk/GPU budgets in the mapper are policy budgets, **not automatically enforced physical cache capacity limits**. No workload-size inference or dynamic eviction optimization was added.

## 1. Apply only AFTER Bayesian v2

From the login node, copy `bayesian_kv_v3_incremental.patch` to the cluster repository root. Then:

```bash
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
git branch --show-current
git status --short                  # protect your current work first
git apply --check bayesian_kv_v3_incremental.patch
git apply bayesian_kv_v3_incremental.patch
```

**Stop on any `--check` failure**. No manual `patch -p1 -f`. The patch assumes the earlier Bayesian v2 smoke fix was already applied.

## 2. CPU-only verification on login node

```bash
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029/bin/activate
python -m unittest discover -s tests -p test_bayesian_mapper_cpu.py -v
python -m compileall -q Hierarchical_KV/shortq_placement/bayesian_mapper.py \
  experiments/mp_nognn_project.py experiments/calibrate_bayesian_tier_costs.py \
  third_party/LMCache/lmcache/v1/distributed/placement_metadata.py \
  third_party/LMCache/lmcache/v1/distributed/storage_controllers/prefetch_controller.py \
  third_party/LMCache/lmcache/v1/distributed/storage_controllers/store_policy.py \
  third_party/LMCache/lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py
for f in scripts/run_shortq_gnn_dynamic_vpc_common.sh \
  scripts/run_h100_bayes_targeted_smoke.sbatch \
  scripts/run_h100_bayes_cost_calibrate.sbatch \
  scripts/run_h100_bayes_sweep_cell.sbatch \
  scripts/submit_bayes_group_{A,B,C}.sh; do bash -n "$f" || exit 1; done
```

Expected unit tests: **15 tests, OK** (other project tests remain separate).

## 3. Calibration — one short Slurm job, required before smoke

`mkdir -p /mnt/shared/gpfs/home/sriramc2/runs/kvaware` on the login node first.

```bash
sbatch scripts/run_h100_bayes_cost_calibrate.sbatch
```

Wait for completion (job ID printed by sbatch). Output prior:

```
/mnt/shared/gpfs/home/sriramc2/runs/kvaware/rank_h100_calibrated.json
```

The score histogram is **the unchanged Q12 diagnostic histogram**. The recompute prior is **unchanged/unmeasured**. Inspect its `cost_calibration` metadata. This is a simple starting point, not a physically exact hardware-cost oracle. Reuse this prior file throughout matched experiments.

## 4. Targeted H100 smoke — do NOT skip

```bash
sbatch scripts/run_h100_bayes_targeted_smoke.sbatch
```

This uses Q80/B20, 16 generated tokens, a fixed prior weight of 1, completed-transfer feedback and renewable readmission preferences. Existing LMCache scratch-cleanup still deletes **everything under `/scratch2/sriramc2`**; preserve anything needed beforehand.

Inspect the run directory printed as `run_dir=...`:

```bash
RUNDIR=/path/printed/by/job
cat "$RUNDIR/slurm_and_source.txt"
grep -E '\[BAYES_(WAVE|H2D_COMPLETE|L2_LOAD_COMPLETE|READMISSION_STORE)' "$RUNDIR"/logs/{driver,lmcache}.log | tail -60
python - "$RUNDIR" <<'PY'
import json, sys, pathlib
p=pathlib.Path(sys.argv[1]); h=json.loads((p/'results/bayes_wave_history.json').read_text())
print('epochs:',len(h))
print('cost samples:',h[-1]['cost_sample_counts'])
print('preference changes:',sum(e['readmission_preferences_changed'] for e in h))
print('final estimates:',h[-1]['cost_estimate_ms'])
l=(p/'logs/lmcache.log').read_text(errors='replace')
print('metadata misses:',l.count('[GNN_PLACEMENT_METADATA_MISS]'))
print('physical changed store intents:',l.count('[BAYES_READMISSION_STORE]'))
assert len(h)>=2 and '[GNN_PLACEMENT_METADATA_MISS]' not in l
PY
```

**Do not assert that Q80 necessarily produces a physical changed-tier readmission**; changes depend on misses and band crossings. Check that H2D observations are plausible (160 MiB logical per TP2 chunk / 80 MiB per rank), that CPU/disk sample counts grow when those paths run, and that no asynchronous errors or TP disagreements occur. If no renewed physical store was observed, the CPU unit test exercises that state transition, but a targeted miss-forcing H100 test would still be needed to claim runtime validation of that case.

## 5. Launch Group A — seven configs

From repo root *after* smoke review:

```bash
bash scripts/submit_bayes_group_A.sh
```

Individual arms:

```bash
sbatch --export=ALL,SWEEP_CELL=bayes_original,SWEEP_ORDER=original scripts/run_h100_bayes_sweep_cell.sbatch
sbatch --export=ALL,SWEEP_CELL=bayes_canonical,SWEEP_ORDER=canonical scripts/run_h100_bayes_sweep_cell.sbatch
```

Arms: no-GNN original/canonical, existing GNN original/canonical, Bayesian GNN original/canonical, **one** random-score Bayesian canonical (seed 42). The random control permutes unique-hash scores while preserving the score distribution; it does not sample random tier labels.

## 6. Launch Group B — four budgets x two policies

```bash
bash scripts/submit_bayes_group_B.sh
```

Rows: `(GPU policy GB, CPU cache GiB, disk policy GB, disk PUT cap GiB)` = `(10,225,100,.25)`, `(10,100,100,.25)`, `(10,225,300,.25)`, `(5,100,100,0)`.

**Interpretation limitation:** `GPU` and `disk` numbers are percentile-mapper *policy budgets*. With the current code they **do not physically shrink GPU VPC or disk storage**, and no-GNN ignores them. CPU GiB is physically configured and the 0 cap means **unbounded PUT admission**, not zero disk storage. Therefore the rows are policy-budget sensitivity plus CPU-capacity/store-cap stress tests, NOT a clean hardware-capacity elasticity study. Group B no-GNN arms with equal CPU/store-cap may intentionally be duplicates; do not claim that those no-GNN configurations vary physically with a Bayesian-only budget.

## 7. Launch Group C — four weights (pick best arm from A/B first)

Choose the winning policy order/capacity values. For example:

```bash
GROUP_C_GPU_GB=10 GROUP_C_CPU_GB=225 GROUP_C_DISK_GB=100 \
GROUP_C_STORE_CAP_GB=0.25 GROUP_C_ORDER=canonical \
bash scripts/submit_bayes_group_C.sh
```

Weights: 0, 0.25, 1, 4. `fixed` posterior mode for all. Can omit weight=1 if a fully matched run already exists (edit submit script before launch). The optional `pseudo_count` mode is implemented but deliberately not in this sweep.

## 8. Shared operational rules

- **All jobs:** partition `gpu-h100-dchen`, TP2 H100, `--cpus-per-task=16`, **no `--mem` directive**, B100, Q650, cold seed 0 / warm seed 1, 128 generated tokens, cold/warm store barrier and phase trace, no CUDA device trace. Resource allocation recorded in `slurm_and_source.txt`.
- Group launcher scripts submit jobs **sequentially with `afterok`** dependencies because each arm cleans the same scratch root. If one arm fails, later ones stay pending; check `squeue -u "$USER"` and `scontrol show job JOBID`.
- Logs: `/mnt/shared/gpfs/home/sriramc2/runs/kvaware/kvbayes-cell_JOBID.out` and `.err` (`%x` is `kvbayes-cell`). Run artifact paths print on completion.
- Calibrated prior is a simple transfer measurement; recompute reference is unmeasured, and runtime disk cost combines two nonexclusive stages. **A small performance gap cannot prove Bayesian superiority** without matched repeated trials.
- Do not compare B100 directly with old B200 headline runs or mix different CPU allocations.

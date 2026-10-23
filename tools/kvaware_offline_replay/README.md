# CPU-only ordinal Bayesian placement + idealized cache replay

**Version:** 2026-10-22 exploratory. Python 3.10+, standard library only; no GPU, Slurm, vLLM, LMCache, dataset loader or KV tensors required. Nothing in this package edits the repository, changes the mapper, removes scratch files, or submits jobs.

## What is exact, and what is not

- **Exact:** Placement decisions. This tool imports **your installed** `Hierarchical_KV/shortq_placement/bayesian_mapper.py`, reads original-order Q650 per-request ordinal scores and prefix hashes, and replays 14 chronological submission waves using their recorded score-population updates and **recorded cost-observation means**. The included validation fixtures reproduce *every* recorded wave's mapped counts, new-hash counts, observed score counts and percentile boundaries for **jobs 29886 and 29889**.
- **Not exact:** The CPU/Disk/VPC residency and theoretical hit percentages. This exploratory model serializes originally concurrent requests, instantly admits computed chunks, approximates L1 as hash-granular LRU at a hard 80% watermark, L2 as hash-granular LRU at 90%, assumes GPU-placed hashes remain in a **perfect, unbounded GPU cache**, and assumes ideal disk PUTs despite the runtime's 0.25 GiB inflight cap. Real vLLM VPC eviction, requests-in-flight, LMCache async PUT/GET, scheduler-admitted contiguous prefixes, staging, memory locking and eviction batches are not replayed.
- **Frozen-feedback counterfactual:** For each wave, the reference job's *cumulative observed* retrieval/recompute means are inferred from its recorded posterior and calibrated prior. Changing cost prior weight reweights those fixed observations; changing admission/capacity **does not** change the simulated cost observations to reflect different congestion. Do not interpret counterfactual cost curves or resulting hit percentages as an E2E projection.
- **Evidence that the physical model is not yet calibrated:** For job 29886, replay gives **47.489% idealized warm CPU vs 16.299% actual scheduler-consumed CPU**, and **180.0 GiB estimated warm-start L1 vs 153.672 GiB measured**. The replay is therefore appropriate for mechanism debugging/relative hypothesis exploration, **not for estimating actual CPU hit rates, speedups, or selecting an optimal Disk budget without another serving validation**. An explicit warning appears in `validate` output.

The user-provided `h100_29886_prior_validation_only.json` was **reconstructed** from the existing Q650 expected-ordinal histogram and measured H100 CPU/Disk/recompute priors. It exactly replays the two archived experiments; **it is not a replacement for your own H100 calibration output in serving**.

## Install on CodeNimbus

From the existing repository root:

```bash
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029/bin/activate
mkdir -p tools/kvaware_offline_replay
unzip -o ~/kvaware_offline_replay_bundle.zip -d tools/kvaware_offline_replay
export KVAWARE_REPO="$PWD"
```

Change the ZIP path in `unzip` to where you uploaded it. Leave the existing mapper patches installed. This package does **not** install new patches.

## Step 1 — known-answer tests (9 tests)

```bash
cd "$KVAWARE_REPO/tools/kvaware_offline_replay"
KVAWARE_REPO="$KVAWARE_REPO" python -m unittest -v \
  test_offline_kv_replay test_h100_reference
```

Expect `Ran 9 tests` and `OK`.

The 7 toy tests independently check LRU eviction/touching, repeat-hit CPU100%, total thrash CPU0%, prefix holes that strand later cached chunks, GPU->CPU backing, ideal Disk retrieval, and fixed-weight cost observation inversion. The 2 integration tests check every wave of original-order **29886 Disk100** and **29889 Disk300** against real `bayes_wave_history.json`, plus the known simulated-vs-measured L1 discrepancy.

## Step 2 — verify the historical placement replay

From the repository root:

```bash
cd "$KVAWARE_REPO"
mkdir -p /tmp/kvaware_ordinal_replay_$USER
python tools/kvaware_offline_replay/offline_kv_replay.py \
  --repo "$PWD" \
  --run tools/kvaware_offline_replay/h100_29886_minimal_validation_trace.zip \
  --prior tools/kvaware_offline_replay/h100_29886_prior_validation_only.json \
  --mode validate --strict \
  --output /tmp/kvaware_ordinal_replay_$USER/validate_29886.json \
  >/tmp/kvaware_ordinal_replay_$USER/validate_29886.stdout

python - <<'PY'
import json, os
p = '/tmp/kvaware_ordinal_replay_' + os.environ['USER'] + '/validate_29886.json'
r = json.load(open(p))
print('All 14 historical placement waves exact:', r['waves_exact'])
print('Idealized CPU %, actual scheduler-useful CPU %:',
      r['idealized_cache_replay']['warm_cpu_pct_ideal'],
      r['measured_warm_scheduler_useful_cpu_pct'])
print('Warning:', r.get('warning', 'none'))
assert r['waves_exact']
PY
```

It must say `True` for the 14 waves. The large CPU-hit discrepancy is **expected** and must *not* be hidden. To validate your local full results instead, replace `--run` with:

```bash
--run /mnt/shared/gpfs/home/sriramc2/runs/kvaware/h100_shortq_gnn_dynamic_vpc_q650_bayes_original_29886 \
--prior /mnt/shared/gpfs/home/sriramc2/runs/kvaware/h100_expected_ordinal_exploratory.json
```

`--run` can be one original-order run directory or a ZIP containing **exactly one run**; it cannot directly read the multi-run Group A/B aggregate ZIP. The included minimal ZIP eliminates the need to supply the original archives for basic validation.

## Step 3 — run CPU-only parameter sweep

```bash
cd "$KVAWARE_REPO"
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware
mkdir -p "$RUN_ROOT/offline_ordinal_replay"

python tools/kvaware_offline_replay/offline_kv_replay.py \
  --repo "$PWD" \
  --run "$RUN_ROOT/h100_shortq_gnn_dynamic_vpc_q650_bayes_original_29886" \
  --prior "$RUN_ROOT/h100_expected_ordinal_exploratory.json" \
  --mode sweep \
  --gpu-gb 10 \
  --cpu-gb 100,225 \
  --disk-gb 100,300,500,700 \
  --prior-weight 1,0.3,0.1 \
  --object-mib 80 \
  --l1-watermark 0.8 --l2-watermark 0.9 \
  --output "$RUN_ROOT/offline_ordinal_replay/ordinal_sweep.csv"
```

This generates **24 scenarios** (2 CPU x 4 Disk x 3 weights) and should normally finish in **tens of seconds to a few minutes** on a login node. If the full run directory has been cleaned, substitute `--run tools/kvaware_offline_replay/h100_29886_minimal_validation_trace.zip` with the same `--prior`; the minimal trace contains everything this simulator needs.

For an apples-to-apples **Disk-only** experiment with unchanged prior weight:

```bash
python tools/kvaware_offline_replay/offline_kv_replay.py \
  --repo "$PWD" \
  --run tools/kvaware_offline_replay/h100_29886_minimal_validation_trace.zip \
  --prior tools/kvaware_offline_replay/h100_29886_prior_validation_only.json \
  --mode sweep --gpu-gb 10 --cpu-gb 225 \
  --disk-gb 100,300,500,700 --prior-weight 1 \
  --output /tmp/ordinal_disk_only.csv
```

## Reading the CSV

`final_mapped_counts` shows exact Bayesian mapper preferences under frozen historical cost observations. These are the **most trustworthy outputs**. `warm_cpu_pct_ideal`, `warm_cpu_only_prefix_available_pct`, `warm_gpu_pct_ideal`, `warm_disk_pct_ideal` and `warm_recompute_pct_ideal` are **theoretical cache-model quantities**, not observed scheduler-useful contributions. `l1_writes`, `l1_evictions`, `warm_start_l1_gib`, `l1_evictions_by_position`, and `evicted_hash_later_needed_by_position` refer to **simulated events**, not actual LMCache events. The warm CPU-only metric asks how much of each prefix would be contiguous in this hypothetical L1, ignoring GPU and Disk.

Most useful immediate comparison: check how capacity/weight changes **the exact mapper's mapped counts** and whether the lightweight model predicts reduced admissions and churn. If a candidate looks promising, the next step is an actual H100 experiment or a more faithful trace-driven physical cache simulator before making a performance claim.

## No-side-effects / troubleshooting

- Requires only Python standard library. Reads existing files and writes only designated `--output` / OS temporary files.
- Fails if the prior is not `expected_ordinal_0_100`, the run is not `original`, the expected trace files are absent, or `source_index` is inconsistent.
- If historical 14-wave validation fails with your repo, do not trust sweeps until you reconcile patched mapper version, score histogram, readmission and wave logs.
- The `--posterior-mode pseudo_count` option is available for exploratory counterfactuals, but its score/cost strengths are applied using the **fixed historical observation stream**. It is *not* a real pseudo-count-vs-fixed serving comparison.
- Does not run Slurm, touch GPUs, open network ports, or clean `/scratch2/sriramc2`.

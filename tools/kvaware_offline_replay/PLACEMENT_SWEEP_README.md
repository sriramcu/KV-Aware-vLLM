# Expected-ordinal Bayesian vs voting GNN — offline placement-agreement sweep

**2026-10-22.** These scripts are an *add-on* to the previously validated `tools/kvaware_offline_replay/offline_kv_replay.py`. They run on the CodeNimbus login node with ordinary Python, **no GPU, Slurm, KV loading, model inference, or scratch cleanup**. Nothing changes the repository implementation.

## What is in this ZIP

- `offline_placement_agreement.py` — imports the existing `offline_kv_replay` and **your installed** `WaveBayesianMapper`. Replays Bayesian score and cost observations from **job 29886** for counterfactual capacities/weights, compares every final hashed tier preference against actual **voting GNN job 29756**, and writes detailed CSVs.
- `test_placement_agreement.py` — synthetic known-answer tests plus exact historical validation.
- `voting_29756_hash_tiers.json` — 4,857 actual final hash→tier preferences from the original-ordering voting GNN, extracted from `group_a(1).zip` (`placement/runtime_hash_to_placement.json`). Not a position-only approximation.
- `bayes_29886_observed_hash_tiers.json` — 4,857 reference hash→tier preferences. The test must reproduce every one exactly.
- `bayes_29918_observed_hash_tiers.json`, `bayes_29919_observed_hash_tiers.json`, `bayes_29920_observed_hash_tiers.json` — actual final policies, for **optional** counterfactual-vs-serving sanity comparisons.

**Prerequisite:** You already have the original simulator bundle in `tools/kvaware_offline_replay`, with `offline_kv_replay.py`, `h100_29886_minimal_validation_trace.zip`, and `h100_29886_prior_validation_only.json`. Do not remove these files or replace your patched Bayesian mapper.

## 1. Install

Copy the downloaded `kvaware_voting_placement_sweep.zip` to your CodeNimbus login home directory, then:

```bash
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029/bin/activate
export KVAWARE_REPO="$PWD"
mkdir -p tools/kvaware_offline_replay
unzip -o ~/kvaware_voting_placement_sweep.zip -d tools/kvaware_offline_replay
```

The ZIP adds new scripts and voting labels; it **does not overwrite** the old replay simulator or your prior JSON.

## 2. Known-answer + historical tests

```bash
cd "$KVAWARE_REPO/tools/kvaware_offline_replay"
python -m unittest -v test_placement_agreement
```

Expected: **6 tests, OK**. The tests check exact/partial agreement, reuse-frequency weighting, strict prefix mismatches, mismatched hash identity rejection, the intended 125-scenario matrix, and **exact final hash-level replay of job 29886**.

## 3. Run a small smoke sweep (4 scenarios)

```bash
cd "$KVAWARE_REPO"
OUT=/tmp/kvaware_placement_smoke_$USER
python tools/kvaware_offline_replay/offline_placement_agreement.py \
  --repo "$PWD" \
  --run tools/kvaware_offline_replay/h100_29886_minimal_validation_trace.zip \
  --prior tools/kvaware_offline_replay/h100_29886_prior_validation_only.json \
  --validate-reference \
  --gpu-gb 10,25 --cpu-gb 225 --disk-gb 100 \
  --fixed-weights 1 --pseudo-presets low:10:1 \
  --output-dir "$OUT"
```

The first console line should be:

```text
Validated historical reference: all 14 waves and 4,857 exact hash tiers match
```

The baseline **GPU10/CPU225/Disk100, fixed w=1** must show **61.808%** exact agreement against voting job 29756 (4,857 unique hashes). If validation fails, **do not trust the sweep**; check installed mapper, input prior/trace and patches.

## 4. Full proposed sweep — 125 configurations

**5 GPU × 5 Disk × (3 fixed-weight + 2 pseudo-count) = 125**; CPU remains 225 GiB:

```bash
cd "$KVAWARE_REPO"
RUN_ROOT=/mnt/shared/gpfs/home/sriramc2/runs/kvaware
OUT="$RUN_ROOT/offline_voting_agreement_125"
python tools/kvaware_offline_replay/offline_placement_agreement.py \
  --repo "$PWD" \
  --run tools/kvaware_offline_replay/h100_29886_minimal_validation_trace.zip \
  --prior tools/kvaware_offline_replay/h100_29886_prior_validation_only.json \
  --voting tools/kvaware_offline_replay/voting_29756_hash_tiers.json \
  --validate-reference \
  --gpu-gb 10,15,20,25,30 \
  --cpu-gb 225 \
  --disk-gb 300,400,500,600,700 \
  --fixed-weights 1,0.5,0.2 \
  --pseudo-presets low:10:1,high:1000:100 \
  --output-dir "$OUT" \
  2>&1 | tee /tmp/kvaware_voting_placement_sweep_$USER.log
```

Pseudo presets have the form `name:score_prior_count:cost_prior_count`. `low:10:1` means score pseudo count 10 and **cost** pseudo count 1. `high:1000:100` means score count 1000 and cost count 100. In fixed mode, `--fixed-weights` adjusts the **score and cost posteriors** jointly. In pseudo mode the pseudo-counts govern the two posteriors and the fixed-weight grid is not applied.

The script uses the chronological medical Q650 cold/warm wave schedule and **job 29886's observed congestion/cost stream**, which is held fixed when changing the policy. Unlike actual serving, it does not regenerate counterfactual cost observations from the changed traffic. It is a controlled placement-policy screen, **not an E2E prediction**. The 125 runs can take a few minutes depending on login-node contention.

## 5. Interpret the results

Generated in `$OUT`:

| File | Purpose |
|---|---|
| `placement_agreement_sweep.csv` | One row per setting; the complete 125-scenario table. |
| `agreement_by_position.csv` | Per-position voting/Bayesian counts (GPU/CPU/Disk/DROP) and exact per-position agreement. |
| `top_by_exact_voting_hash.csv` | 25 settings with highest *unique-hash* exact tier agreement. |
| `top_by_balanced_regions.csv` | 25 settings with highest average of positional GPU(0–1), CPU(2–3), Disk(4–5), DROP(6+) percentages. This is **not** E2E optimization. |
| `pareto_region_candidates.csv` | Configurations nondominated on four positional matches **and** minimizing late CPU admission at 4+. Informational; may have many rows. |
| `run_metadata.json` | Input provenance, assumptions and scenario count. |

Important columns:

- `overall_unique_pct`: voting label == Bayesian label for each of **4,857** distinct hashes, each counted once.
- `warm_occurrence_weighted_pct`: exact tier agreement across **6,831** chunk occurrences in the 650 warm requests; repeated hashes count more.
- `warm_repeat_excess_weighted_pct`: repeated-use-only weighting `max(0, warm_occurrences_per_hash - 1)`; does not count the first warm appearance of a hash.
- `prefix_span_weighted_pct`: hash-tier agreement weighted by the number of remaining scored chunks *dependent on that prefix position*. This is a **structural weighting**, not observed prefix hits.
- `strict_contiguous_prefix_agreement_pct`: per warm request, count correct tier labels only until the **first disagreement with voting**. Different tier labels don't necessarily mean the prefix was actually inaccessible. **This is not scheduler-reachable KV.**
- `{gpu,cpu,disk,drop}_recall_unique_pct`: for voting hashes of a tier, how often Bayesian selects the same tier.
- `{gpu,cpu,disk,drop}_hashes`: how many hashes the Bayesian policy selects for each tier.
- `gpu_pos01_pct`, `cpu_pos23_pct`, `disk_pos45_pct`, `drop_pos6plus_pct`: Bayesian positional *assignment rates*, rather than direct match to voting's exceptional cases.
- `cpu_pos45_pct`, `late_cpu_pos4plus_pct`: CPU over-admission diagnostics.
- `balanced_static_region_pct`: arithmetic mean of the four positional assignment rates; **a diagnostic proxy for historical positional placement**, not an importance metric justified by hardware cost.

**Do not** treat `top_by_*` as a list of expected performance winners. Our real H100 runs already showed that the highest voting agreement needn't have the best warm E2E.

## 6. Optional: compare counterfactual 29920 with actual job 29920

This is useful to see the inevitable error introduced by freezing job 29886's cost stream:

```bash
cd "$KVAWARE_REPO"
python tools/kvaware_offline_replay/offline_placement_agreement.py \
  --repo "$PWD" \
  --run tools/kvaware_offline_replay/h100_29886_minimal_validation_trace.zip \
  --prior tools/kvaware_offline_replay/h100_29886_prior_validation_only.json \
  --validate-reference \
  --gpu-gb 25 --cpu-gb 225 --disk-gb 500 \
  --fixed-weights 1 --pseudo-presets '' \
  --observed-map tools/kvaware_offline_replay/bayes_29920_observed_hash_tiers.json \
  --observed-gpu-gb 25 --observed-cpu-gb 225 --observed-disk-gb 500 \
  --output-dir /tmp/kvaware_29920_crosscheck_$USER
cat /tmp/kvaware_29920_crosscheck_$USER/observed_placement_comparison.csv
```

With the reference mapper version used to build this bundle, the counterfactual differs from 29920's real final hash preferences: only **90.632%** match. This is *expected* because Bayesian observations are endogenous to runtime demand. (By contrast, using the original 29886 capacities/cost stream reproduces all **4,857** actual 29886 hash assignments exactly.)

You can do the same optional check for 29918 and 29919 with their included maps and respective GPU10/Disk700 or GPU25/Disk700 settings. These observational checks don't establish which policy is better.

## Key limitations

1. **No cache-residency simulator is used.** The previous simulator's L1/Disk usefulness estimates were not calibrated against real LMCache and vLLM scheduler behavior. This script only predicts which hashes the Bayesian **preference mapper** assigns.
2. **Frozen feedback:** tier-cost observations are reconstructed from job 29886; changed capacities or posterior strength can alter actual cost/queueing feedback. Early score distributions are also sourced from the same Q650 workload. Results are *conditional counterfactuals*, not precise predictions of new runs.
3. **Vote imitation is not serving optimization:** voting was tuned on this medical workload and H100 stack. Matching it should be used to select **diverse** experimental hypotheses (e.g., similar 2–3 CPU retention with less 4+ admission), not automatically to maximize a single score.
4. No Slurm jobs, GPU inference, NFS access, scratch cleanup, simulator calibration, model patches, or E2E claims are performed.

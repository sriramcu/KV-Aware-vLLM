# L40 Q650 prediction-only head/aggregation/mapper diagnostic

## Intent and restrictions

One prediction-only L40 GPU job, with frozen Short-Q ranker and 8B feature model, to distinguish the original **class head + 16-token voting** from **rank head + 512-token mean**, independently of the Bayesian percentile/capacity decision rule. There is **no vLLM serving, no LMCache, no cache cleanup, and no additional H100 source modification**. No earlier Bayesian ranking-diagnostics patch is required. The export is a standalone copy of the already-existing `experiments/precompute_shortq_placements.py` pipeline with one optional raw-head export path; this avoids touching the scripts used by the queued H100 validation chain.

## Copy to CodeNimbus

Download all 3 files and put them in your repository:

- `precompute_shortq_heads_probe.py` -> `experiments/precompute_shortq_heads_probe.py`
- `analyze_shortq_heads_probe.py` -> `experiments/analyze_shortq_heads_probe.py`
- `run_l40_gnn_prediction_only_q650.sbatch` -> `scripts/run_l40_gnn_prediction_only_q650.sbatch`

```bash
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029/bin/activate
python -m py_compile experiments/precompute_shortq_heads_probe.py experiments/analyze_shortq_heads_probe.py
bash -n scripts/run_l40_gnn_prediction_only_q650.sbatch
sbatch scripts/run_l40_gnn_prediction_only_q650.sbatch
```

Defaults expect the old full run at:

```
/mnt/shared/gpfs/home/sriramc2/runs/kvaware/l40_bayesian_dynamic_vpc_q650_l40_bayes_three_cost_original_b200_mns10_29862
```

and the measured L40 prior at:

```
/mnt/shared/gpfs/home/sriramc2/runs/kvaware/l40_three_cost_preflight_29859/rank_l40_calibrated.json
```

If those are elsewhere, override with `sbatch --export=ALL,REFERENCE_RUN_DIR=/actual/full/run,L40_PRIOR_PATH=/actual/prior.json scripts/run_l40_gnn_prediction_only_q650.sbatch`.

**Do not apply** the earlier `kvaware_bayes_rank_diagnostics_incremental.patch` for this experiment. It isn't needed.

## Important verification gate

The job recalculates prefix-dependent 512-token chunk hashes and chunk-mean rank scores for exactly 650 **original-order medical** prompts, seeded request order 0, with the L40 8B tokenizer and frozen checkpoint. The offline analysis compares the output against job 29862's saved scores. It **fails** rather than providing misleading comparisons if the hashes differ, the ranks drift substantially, or a historical Bayesian replay cannot reproduce its wave thresholds, final tier-preference metadata and population counts.

No changes to GNN checkpoint, scoring, posterior estimator, caching, or Bayesian mapper. The historical Bayes wave replay uses the **recorded** 29862 CPU/disk/recompute posterior costs and request order; it isn't a counterfactual serving experiment and does not pretend that hypothetical placements leave congestion unchanged.

## Key artifacts

All artifacts are under:

```
/mnt/shared/gpfs/home/sriramc2/runs/kvaware/l40_gnn_head_probe_original_q650_<NEW_SLURM_ID>/
```

A ZIP is also written alongside the directory, including when the job fails after creating partial artifacts:

```
.../runs/kvaware/l40_gnn_head_probe_original_q650_<NEW_SLURM_ID>.zip
```

### Raw prediction data

- `placement/gnn_raw_heads_16token.jsonl`: per request source/request index, prompt token count, full hash list, all valid 16-token raw rank values and four-way raw class logits (class order DROP, DISK, CPU, GPU), including blocks of the incomplete trailing chunk.
- `placement/gnn_chunk_scores.jsonl`: **exact current Bayesian mean-of-32** score format.
- `placement/gnn_precompute_summary.json`: GNN model/checkpoint, token/shape, preprocessing diagnostics.

### Offline analysis

- `analysis/head_chunk_occurrences.csv`: ~6831 chunk occurrences with mean/top-8/top-16/max rank, class-probability expected score, GPU-class counts and old vote outcomes for `gpu_min=8/22`, with pure threshold, CPU-top2 rescue and DROP-plurality variants.
- `analysis/position_profiles.csv`: 512-token chunk-position summary of both heads and each voting policy.
- `analysis/head_agreement_correlations.json`: block-level and chunk-level correlations (including Spearman).
- `analysis/matched_population_ablations.json`: rank-head vs class-derived rank alternatives at **the exact same number of unique hash GPU and CPU preferences** as historical voting; position distribution and identity overlap. This top-K population is an offline diagnostic and does **not** claim deployable no-future placement.
- `analysis/historical_bayes_decisions_replay.jsonl`: per request/wave chunk score, percentile, Bayesian proposal, prefix-closure change, final published preference and corresponding old t22 votes.
- `analysis/historical_bayes_wave_replay_audit.json`: checks match to job 29862.
- `analysis/bayes_thresholds_and_raw_cutoffs_by_wave.csv`: all eight Q650 policy snapshots; GPU/CPU/Disk percentile thresholds and histogram-interpolated **raw score cutoffs**, all 3 posterior costs and normalized ratios. These cutoffs are model-implied bin interpolations, not exact empirical quantiles.
- `analysis/historical_vote_vs_bayes_raw_by_epoch_position.csv`: vote-vs-raw-mapper confusion counts, by epoch and chunk position.
- `analysis/analysis_summary.json`: reference score hash verification, score deviations, actual replay agreement, and prefix-closure/duplicate hash accounting.

## Interpretation constraints

Historical job 29428's **exact** chunk-vote implementation/flag configuration should be verified from that run's manifest/log if available. The analyzer explicitly supplies multiple candidates instead of silently assuming which historical policy was used. The first-seen-per-prefix-hash convention is used for matched-population counts. Recomputed hypothetical class/rank assignments do not prove a speedup or actual physical GPU residency, CPU hits, prefix consumption, or E2E change; that requires a controlled serving experiment.

The historical Q650 Bayesian 29862 replay is exact with respect to its recorded **policy decisions and observations** if replay audits match. But the class-head/alternative score choices were never served, and cannot be assigned retrospective latency benefits.

## Tests run before delivery

- Standalone predictor and offline analyzer compile with Python.
- Slurm shell passes `bash -n`.
- Offline analyzer replayed the **uploaded job 29862** using a synthetic raw-head fixture derived from 29862's actual exact hash/mean-score file. It reproduced all eight epoch threshold values, every epoch's tier counts, and the final per-hash preferences with zero discrepancies. This tests replay math and file formats; it is **not** a real GNN inference or proof that the new live L40 job will complete.

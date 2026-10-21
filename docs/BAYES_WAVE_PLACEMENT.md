# Experimental wave-level Bayesian KV placement (Python-only)

This is an **opt-in research prototype** layered on top of the earlier canonical
passage-order + phase-trace changes. It does not replace the current GNN
checkpoint, vLLM GPU allocator, LMCache LRU, disk PUT cap or safety fallbacks.
The patch has **CPU-only tests**, but has **not yet run on an H100**.

## 1. Pipeline and semantics

Precomputation computes a **mean of 32 existing 16-token `rank_head` scores**
for each complete, real 512-token LMCache prefix-hash chunk. It writes
`placement/gnn_chunk_scores.jsonl` with source indices and ordered `(hash, score)`
records. In Bayesian mode **no placements are precomputed**. Incomplete tail
chunks are omitted because LMCache doesn't persist them at 512-token boundaries.

The existing submission-wave driver maps each wave *just before* submitting
HTTP requests, using the prior and observations from **completed earlier waves**.
The same hash always keeps its first assignment across cold and warm. The
metadata files are atomically replaced between waves, and version-aware readers
in LMCache and vLLM reload them. No hash migration or eviction changes.

Score posterior: for each histogram bin `p=(w*prior_prob + observed_prob)/(w+1)`.
Count **unique prefix hashes** only, after each completed wave. Before any
samples, use the prior. Percentile interpolation is uniform *within* each
histogram bin. Scores outside bin edges are clamped, so validate bin coverage.

Cost posterior: `estimated_ms=(w*prior_ms + observed_mean_ms)/(w+1)`.
A tier without observations retains its prior. `w=1` keeps exactly 50/50
prior/empirical average **regardless of sample count**; this is *fixed-weight
prior regularization*, not a conventional conjugate posterior. `recompute_ms`
remains a fixed reference until reliable timing is available.

For tiers `gpu,cpu,disk`, first allocate capacity shares
`a_i=C_i/sum(C)`. Set retention `r_gpu=1` and
`r_i=min(1, recompute_ms/retrieve_ms_i)` for CPU and disk.
Final shares: `rho_i=a_i*r_i`; `rho_DROP=sum(a_i*(1-r_i))`.
Map percentile bands GPU (high) -> CPU -> disk -> DROP (low).
Then apply O(n) **minimum-deviation prefix closure** to the ordered chunks,
constrained by first-seen hash decisions. An irreconcilable fixed DROP/store
hole is logged as `fixed_conflict_requests`; we do not silently migrate an
existing hash. Disk repairs are still subject to **existing** LMCache store
throttling and may fail to persist. Closure is admission consistency, not a
physical or post-eviction guarantee of complete reusable prefixes.

**Capacities** are total *effective budgets*, not current free space or an
estimate of total workload size. GPU budget must be conservative reusable
VPC capacity, not all HBM. Disk budget must be a configured useful disk-cache
budget, not the entire filesystem. No automatic disk/cpu capacity-ratio
clipping is implemented yet. Backing GPU-intended chunks in CPU (existing
`KV_GNN_L1_BACKING=1`) can make physical occupancy differ from tier shares.

## 2. Prior files, costs and critical caveats

`Hierarchical_KV/shortq_placement/priors/rank_example.json` is an
**illustrative, uncalibrated** prior for the existing unbounded rank head.
`importance_0_100_example.json` demonstrates the *same* interface for future
native chunk importance scores. Both include **hypothetical** 20/500/100 ms
CPU/disk/recompute costs; these are **not actual H100 measurements** and must
not support performance claims. Calibrate priors from external/training data
or independent probes. A 0–100 score producer does **not** exist in this patch.

`KV_BAYES_PRIOR_WEIGHT` is the single weight, default 1. Set
`KV_BAYES_PRIOR_FILE` to the JSON to select your score scale/prior. Verify
**score direction** (larger raw rank must mean more important), raw-score
coverage within bin edges, and score/placement correlations before trusting
mapping.

**Crucial timing limitation:** `KV_BAYES_USE_COST_PROXIES=0` (default) means
no live measured retrieval costs are fed into the mapper; only the score
histogram adapts between waves, while tier costs retain configured priors.
This is the safe integration smoke mode. You may set
`KV_BAYES_USE_COST_PROXIES=1` for a **mechanism-only adaptive-cost smoke**.
That opt-in mode uses the diagnostic LMCache log's `MP_PREFETCH_DONE load_s /
loaded_keys` as a disk proxy and `Retrieved N tokens in S seconds` as a CPU
proxy. These are **NOT** isolated GPU-complete CPU/disk retrieval times: they
can include queuing, asynchronous submission and cross-tier staging. Results
with these proxies should **not** be presented as a validated adaptive
cost-benefit algorithm. Reliable synchronized tier timing is the next
instrumentation task; the mapper interface does not need to change for it.

## 3. Applying the patch on a new branch

This patch was generated against the uploaded repo ZIP **after applying**
`canonical_q650_phase_trace.patch`, which your existing H100 runs used.
Inside the cluster checkout:

```bash
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
git status --short                # commit/stash anything important first
git switch -c feat/bayes-wave-kv # choose another name if it already exists
# Download bayesian_wave_kv.patch from this chat, then scp/copy into this repo.
git apply --check bayesian_wave_kv.patch
git apply bayesian_wave_kv.patch
```

If `git apply --check` fails, **STOP**; don't use `--force` or patch
individual hunks blindly. Send the error and `git status --short` for
reconciliation with your current checkout.

Run CPU checks **before** starting any GPU server:

```bash
source /mnt/shared/gpfs/home/sriramc2/venvs/kvaware-vllm029/bin/activate
python -m unittest discover -s tests -p test_bayesian_mapper_cpu.py -v
bash -n scripts/run_h100_bayes_canonical_smoke.sh \
  scripts/run_shortq_gnn_dynamic_vpc_common.sh
python -m compileall -q Hierarchical_KV/shortq_placement/bayesian_mapper.py \
  experiments/precompute_shortq_placements.py experiments/mp_nognn_project.py \
  vllm/v1/request.py \
  third_party/LMCache/lmcache/v1/distributed/placement_metadata.py
python - <<'PY'
import vllm
print('vllm import:', vllm.__file__)
import lmcache
print('lmcache import:', lmcache.__file__)
PY
```

The import paths must resolve to the modified checkout and vendored LMCache.
Python-only edits require a **server restart**, not a vLLM CUDA rebuild.

## 4. First H100 smoke, INSIDE the existing `srun` allocation

**CAUTION: the common launcher defaults to `CLEAN_OLD_LMCACHE=1` and removes
all contents directly beneath `/scratch2/sriramc2`.** Copy old results off
scratch or set `CLEAN_OLD_LMCACHE=0` only when you can ensure isolation.

```bash
cd /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM
KV_BAYES_PRIOR_WEIGHT=1 \
KV_BAYES_GPU_GB=10 KV_BAYES_DISK_GB=100 \
NUM_QUESTIONS=12 SUBMISSION_BATCH_SIZE=4 \
KV_BAYES_USE_COST_PROXIES=0 \
bash scripts/run_h100_bayes_canonical_smoke.sh
```

`h100_smoke` uses Q12, canonical order, cold shuffle seed 0, warm random seed
1, 16 output tokens and three wave updates per phase. It deliberately does
**not** require artificially forcing every tier into the smoke workload.
For later cost-feedback *wiring validation only* (not an accuracy benchmark),
repeat with `KV_BAYES_USE_COST_PROXIES=1`. Observe `cost_sample_counts` and
`cost_estimate_ms` change in the `BAYES_WAVE_PUBLISH` logs, if tier hits occur.

After the launcher prints `run_dir=...`:

```bash
RUNDIR=/path/printed/by/launcher
cat "$RUNDIR/results/bayes_wave_history.json"
grep '\[BAYES_' "$RUNDIR/logs/driver.log" | tail -25
cat "$RUNDIR/placement/runtime_hash_to_placement.json" | head
cat "$RUNDIR/results/summary.json"
grep -E 'GNN_PLACEMENT_METADATA_MISS|Traceback|ERROR' \
  "$RUNDIR/logs/lmcache.log" "$RUNDIR/logs/vllm.log" | tail -30
```

Check `BAYES_WAVE_RUN_PASS`, no placement metadata misses, multiple epochs,
new hashes before first request of each wave, stable repeated hashes, no
request failures, and completed cold-to-warm store barrier.

## 5. Scope and known limitations

- This is **not** an H100-validated patch. CPU tests validate the mapper,
  not actual multiserver startup/refresh, thread timing or throughput.
- The GNN still runs **offline**, but emits scores only in Bayesian mode;
  a future online score producer can write the same request-indexed data.
- The `prior_weight=1` average does not converge to the empirical mean as
  samples grow; it converges to the midpoint of prior and empirical means.
- At most four B200 waves per 650-request phase; thresholds change only
  between completed waves, not per chunk/within a wave.
- LMCache readers throttle dynamic metadata version checks to 250ms;
  the wave driver waits 300ms after atomically publishing both snapshots
  before submitting requests, preventing stale metadata at a wave boundary.
- The GPU `window` VPC remains unchanged in this integration. Benchmark
  comparisons must account for its existing CPU scheduling overhead.
- No change to LRU, disk PUT cap, starvation fallback or canonical ordering.
- Full 0–100 score inference, reliable exclusive cost timing, and global
  workload-capacity knowledge are intentionally out of scope.

# VPC occupancy, prefix-hole diagnostics, and hindsight placement

## Scope and safety

This is an additive patch against the **latest uploaded repository**. Existing
`vanilla`, `window`, `tierwise`, `selective` eviction semantics remain unchanged.
All diagnostics are disabled by default, and hindsight oracle placement is
**disabled** by default (`KV_CHUNK_PLACEMENT_MODE=gnn`). The oracle is a static
**per-512-token chunk-hash label policy**, not a pinned cache, not precomputed KV,
and not an online predictor. It reuses the existing LMCache runtime placement
map and vLLM per-request importance sidecar. Selective is the existing
**last-reference GPU-only admission + upstream LRU** policy.

### Module boundaries (v2)

`vllm/v1/core/vpc_policies/diagnostics.py` contains the pure read-only
occupancy/hash-presence/chunk-shape functions.
`vllm/v1/core/vpc_policies/runtime_diagnostics.py` owns request-path indexes,
first GPU-reclaim capture, pre/post-reclaim comparisons, throttled sampling,
and the per-pool diagnostics lifecycle. `block_pool.py` only constructs this
optional object and calls it at existing allocator/lookup/reset hooks.
`vpc_policies/stats.py` still owns and serializes the existing stable counters
and JSONL schema. None of the four eviction policies is replaced.

`Hierarchical_KV/shortq_placement/hindsight_oracle.py` owns the ranking,
budget parsing and sidecar label rewriting. The precompute script simply
calls the helper after normal GNN chunk discovery and records its summary.

### Enable VPC diagnostics for a future run

```bash
VLLM_VPC_DIAGNOSTICS=1
VLLM_VPC_DIAG_INTERVAL_S=5
VLLM_VPC_STATS_JSONL="$RUN_DIR/logs/vpc_periodic.jsonl"
VLLM_VPC_STATS_INTERVAL_S=5
```

The H100 sweep wrapper passes these environment variables through (set them
in the `sbatch --export` list). Inspect `vpc_periodic.pid*.jsonl` under
`vpc_diagnostics`. The sampler scans on the **scheduler thread** no more often
than its throttle; the async JSONL thread only writes an immutable snapshot.
Check `diagnostic_scan_cpu_ms` because this opt-in scan costs scheduler CPU time.
`sample_engine_elapsed_s` tells you how old the latest diagnostic observation
is compared with the surrounding periodic JSONL `elapsed_s`.

### Definitions / interpretation

`physical_blocks` is a partition of the actual block objects: `active`
(`ref_cnt>0`), `idle_cached` (`ref_cnt==0` with primary or extra indexed hash),
`uncached_available` (unreferenced with no searchable hash), and
`reserved_special` (null block). Both active and idle are split by carried
importance label. **None** is interpreted as unlabelled. A physical block can
only occupy one state. `accounting_ok` validates their sum.

`observed_chunk_universe` is a deduplicated set of *full* 32×16-token logical
paths seen at first `get_computed_blocks` call for a request, using the
request's 16-token block hashes and cache-group 0 (H100 Q650 TP2 one-group
setting). It reports the number of observed paths and how many are fully,
partly, or not resident **at snapshot time**. `complete_any` counts cached
hashes even if actively referenced; `complete_idle` requires all 32 positions
to be searchable in unreferenced blocks. `resident_*_blocks` is a
request-path sum and may double-count a physical hash shared by several paths.
`stranded_*_blocks` within each observed logical chunk counts positions after
its first missing block; gaps *before* its chunk are handled separately.
A chunk can be complete but still unreachable from the beginning of a request.

`request_first_lookup_cumulative` counts the contiguous GPU hash prefix and
physical hashes beyond the first gap, over **one observation per distinct
request ID**. It is not actual scheduler GPU-hit tokens, does not perform LRU
touches, and includes both physically hashed active+idle and idle-only
viewpoints. The observation is made at the first manager lookup/skip, so the
sample can be earlier than later L2 offload/reload decisions. Check original
CHTHM for useful, scheduler-consumed contributions. In mixed/multigroup
architectures the probe is explicitly group 0 only.

`gpu_chunk_reclaim_diagnostics` counts which **previously observed GPU-labelled
logical chunk hashes** are affected by later cached-block reclaims, and the
number of complete->incomplete transitions (idle-only and active-or-idle).
These counts can exceed the physical victim count because one hashed block
may belong to multiple observed prefix paths, and exclude content never
observed as part of a full chunk. A reclaim does **not** prove that a chunk
became incomplete if another duplicate physical block retains the same hash.

`first_gpu_reclaim` records the earliest allocation-time cached GPU-labelled
victim, the engine monotonic elapsed time, active/idle census, and number of
first-seen cold/warm request IDs. Cold/warm classification uses the runner's
`kvaware-cold-` and `kvaware-warm-` IDs and otherwise falls back to `other`.
A reclaim indicates **pressure against a cached victim**, not a fixed
persistent-memory reservation becoming full. Request counts refer to first
manager observations and not completed questions.

Sidecar labels map `DROP` to low-priority `disk` at 16-token VPC level; this
is the existing behavior. For group-0 `observed_chunk_universe`, `disk`
therefore includes DROP unless the sidecar format is later extended. The
LMCache `runtime_hash_to_placement.json` still retains explicit `drop`.

### Hindsight oracle: build a reusable ranking

Generate a budget-independent rank JSON from the **original Q650 occurrence
trace** (e.g., 29265's `placement/gnn_chunk_placements.jsonl`):

```bash
python scripts/build_hindsight_rank.py \
  --placement-trace /path/to/gnn_chunk_placements.jsonl \
  --output /path/to/q650_hindsight_rank.json
```

Ranking uses descending full-chunk occurrence count with deterministic hash
lexical tie-break. It is a **retrospective frequency heuristic** and does not
promise optimal dynamic E2E, discount first touches, or prefill cache contents.
It does not encode budgets. The precompute step currently **still runs the GNN
inference pipeline** to produce the same exact prompt population, then replaces
final placements from the oracle ranking. This wastes offline precompute time,
but guarantees compatibility with existing hashing, runtime metadata,
per-request sidecars, and analysis. It does not change serving-time KV logic.

### Where to store the supplied Q650 ranking JSON

Keep the ranking on persistent GPFS storage rather than `/scratch2/sriramc2`
(the runner can delete scratch when `CLEAN_OLD_LMCACHE=1`). A suggested path is:

```bash
mkdir -p /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/experiments/oracle_rankings
cp q650_hindsight_frequency_rank_29358.json \
  /mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/experiments/oracle_rankings/
```

The ZIP also contains this file under the same `experiments/oracle_rankings/`
relative path. Set `KV_ORACLE_RANKING_FILE` to its **absolute** path in the
sbatch environment. Do not use `/mnt/data/...` on the cluster. The provenance
is window run 29358; the same ranking is reusable for different GPU/CPU/DISK
label budgets but remains tied to the Q650 hash population. `assign_oracle`
checks that the ranking hashes match the current workload exactly.

### Invoke a later oracle experiment (not now)

The existing H100 wrapper accepts the following additional exports (same
H100 true-DROP runner defaults; explicit oracle **overrides final labels**):

```bash
KV_CHUNK_PLACEMENT_MODE=oracle
KV_ORACLE_RANKING_FILE=/mnt/shared/gpfs/home/sriramc2/KV-Aware-vLLM/experiments/oracle_rankings/q650_hindsight_frequency_rank_29358.json
KV_ORACLE_GPU_CHUNKS=258
KV_ORACLE_CPU_CHUNKS=767
KV_ORACLE_DISK_CHUNKS=1012
VLLM_VPC_POLICY=selective
VLLM_VPC_DIAGNOSTICS=1
```

Use unique Slurm experiment labels, independent L2 storage, and ensure failed
stores are zero. For alternative oracle class budgets (e.g. GPU=50, 100, 150),
change only the cutoff values, **not** the ranking file. Classes are assigned:
first `GPU`, next `CPU`, next `DISK`, remainder `DROP`. The ranks must exactly
cover all unique hashes in the run; a mismatch aborts rather than silently
mislabels missing hashes. The precompute summary records the budgets and file.

With `VLLM_VPC_POLICY=selective`, only GPU-labelled idle blocks retain cached
hashes, and they remain evictable by vanilla LRU when new allocations need
capacity. Active blocks are never evicted by admission filtering. CPU/DISK/DROP
placements use the unchanged LMCache backend and existing store cap. This is a
**different treatment** from comparing GNN W256 with oracle selective; use
matched VPC policies to isolate predictor/classification quality.

### Validations and limitations

Python compilation and standalone oracle/census tests can run on login nodes.
The repository-wide vLLM pytest suite requires a correctly provisioned vLLM
environment and GPU platform (the portable container may lack `tblib`,
`msgspec`, CUDA and compiled `vllm._C`). Run the vLLM core policy and new
diagnostic tests on a compute node before an H100 sweep. Recommended: run a
short cold/warm Q24 smoke with `VLLM_VPC_DIAGNOSTICS=1`, compare outputs and
E2E with diagnostics off, and verify every `accounting_ok`. The periodic
sample introduces measurable scheduler overhead; scientific comparisons must
either apply it to all arms or disable it for all arms.

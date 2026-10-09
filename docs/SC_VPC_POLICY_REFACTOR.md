# [SC] Modular VPC policy refactor (vLLM 0.29.0)

This patch is limited to the vLLM V1 GPU prefix-cache policy. It does **not**
implement the future hindsight/oracle placement provider, its ranking file,
or any LMCache L1/L2 policy changes.

## Configuration

Set `VLLM_VPC_POLICY` in the **vLLM scheduler/engine process**:

| Value | Prefix-cache admission after final unref | Allocation / victim policy |
| --- | --- | --- |
| `vanilla` | Upstream | Original `FreeKVCacheBlockQueue.popleft_n`, O(1) per block |
| `window` | Upstream | Existing Short-Q min-rank among oldest W cached blocks, O(W) |
| `tierwise` | Upstream | Unlabeled -> disk -> cpu -> gpu; oldest within class, O(1) cached-victim selection |
| `selective` | Retain only GPU-labelled blocks | Original upstream LRU bulk allocator |

Default is **backward-compatible**: if `VLLM_VPC_POLICY` is omitted,
`VLLM_GNN_AWARE_VPC=1` continues to select `window` and absence of this flag
selects `vanilla`. The explicit mode always takes precedence. `window` uses
`VLLM_GNN_AWARE_VPC_WINDOW=256` by default, as before. `enable_kv_importance`
remains active for `window`, `tierwise`, and `selective` and inactive for
`vanilla`. The existing request-to-block importance hook is retained.

The physical block size remains **16 tokens**. The placement producer still
assigns one class to each **512-token LMCache chunk** and propagates it to the
native blocks through `request.kv_importance_placements`. Selective admission
makes no independent blockwise prediction; unknown/unlabelled blocks are
conservatively *not* retained after `ref_cnt` reaches zero.

Tierwise uses secondary indexes over the stock free queue. Fixed-class victim
selection and removal are O(1). The rare operation of changing the label of
an already-free cached block may reinsert it in LRU order with O(N) metadata
maintenance; that does not happen during victim selection. Explicit hash
invalidations (including free-list interior entries) are handled by a separate
uncached index. The original free list owns block availability/refcounts.

## Logging

`[VPC_POLICY_INIT]` appears in normal vLLM logs at initialization.
`[GNN_AWARE_VPC_INIT]` and the first `[GNN_AWARE_VPC_BIAS]` are retained for
window-mode backward compatibility.

At **normal interpreter shutdown**, `[VPC_SUMMARY]` emits a cumulative JSON
record in the normal vLLM log. The summary includes cached allocation-time
reclaims by victim tier (counts and percentages), uncached allocations, total
selection CPU time, approximate p95 (logarithmic histogram *upper boundary*),
mean number of scanned candidates, LRU bypass rate, selective admission
rejections, and explicit hash invalidations. The victim percentage denominator
is *only cached blocks reclaimed for new allocations*: it excludes uncached
free-slot reuse, explicit hash invalidation, and selective admission rejection.

For snapshots every few seconds in a **separate file**, opt in with:

```bash
export VLLM_VPC_STATS_JSONL=/path/to/run/logs/vpc_periodic.jsonl
export VLLM_VPC_STATS_INTERVAL_S=5
```

The writer is a daemon thread, not a write in the scheduler's eviction path.
The actual output filename contains `.pid<PID>` before `.jsonl`, avoiding
clobbering across processes. Snapshots are cumulative. The free-block count
is included. IO errors are reported in the main log.

**Reporting caveats:** Neither the BlockPool nor this patch knows the cold/warm
phase boundary. Per-phase deltas can be calculated by subtracting snapshots,
or a future runner can invoke `BlockPool.get_vpc_stats_snapshot()` at phase
boundaries. Existing `results/summary.json` analyzers are untouched. Abrupt
SIGKILL/SIGTERM may bypass the interpreter's shutdown hook, so keep periodic
JSONL enabled for diagnostic runs or explicitly snapshot before stopping the
engine. The periodic reader does not acquire scheduler locks; snapshots are
best-effort and can momentarily reflect partially updated counters.

## Safety and validation

The patch adds no launch-script, GNN predictor, LMCache, GPU kernel, or
external-cache changes. `KVCacheManager.cache_blocks()` and `.free()` additionally
ensure importance propagation before release even on delayed paths. An
existing cached-hash copy-on-write transfer now transfers the old content's
importance label to the destination physical block.

On CodeNimbus after applying the patch to the matching project commit:

```bash
$VENV/bin/python -m pytest -q tests/v1/core/test_sc_vpc_policies.py
$VENV/bin/python -m pytest -q tests/v1/core/test_prefix_caching.py
$VENV/bin/python -m pytest -q tests/v1/core/test_kv_cache_utils.py
```

Start with smoke requests before Q650. Check that all modes initialize with
the intended configuration and all cached block accounting remains correct.
Do not interpret local unit-test passing as production/TP2 validation.

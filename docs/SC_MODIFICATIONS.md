# [SC] modifications to vendored vLLM / LMCache

This file is the audit index for project-specific changes kept after the 2026-10 refactor. The refactor deliberately leaves unrelated upstream vLLM/LMCache code alone; surviving edits in existing upstream files carry an `[SC]` marker.

## Upstream baselines

- vLLM: `v0.29.0`, commit `98dff2a81d747d1dba01a47f939f48c3526d4206`
- LMCache: commit `60480f0e3defbce935dcf2bc3356b85decfb31c6`

## vLLM changes kept

### External-KV load failure rewind repair

Files: `vllm/v1/core/sched/output.py`, `vllm/v1/core/sched/scheduler.py`, `vllm/v1/worker/gpu/model_runner.py`.

When a KV load fails and vLLM rewinds `num_computed_tokens`, the scheduler now sends explicit rewind metadata and the GPU worker restores the authoritative request state instead of retaining a stale device-side frontier. The local recovery path also sanitizes speculative/output state and guards invalid effective sequence lengths.

Upstream context:

- https://github.com/vllm-project/vllm/issues/49250
- https://github.com/vllm-project/vllm/pull/49252
- https://github.com/vllm-project/vllm/pull/53298

### Two-pass Stage-1 starvation fallback

File: `vllm/v1/core/sched/scheduler.py`.

Only the progress-based two-pass policy remains: after two consecutive complete WAITING scans with no model scheduling progress, free execution slots, and pending Stage-1 LMCache lookups, the scheduler non-blockingly abandons the oldest eligible Stage-1 dependencies up to the free-slot budget. Stage 2 is never force-abandoned by this policy. The retired GPU-occupancy watchdog was removed.

Related LMCache timeout/recompute design:

- https://github.com/LMCache/LMCache/issues/4945
- https://github.com/LMCache/LMCache/issues/2585

### Mutual-prefix / VPC integration and bounded GNN VPC bias

Files: `vllm/v1/request.py`, `vllm/v1/core/block_pool.py`, `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/sched/scheduler.py`.

The retained project hooks carry semantic `gpu/cpu/disk` placement metadata, support the `[VPC prefix] | [LMCache suffix] | recompute tail` boundary, and optionally bias VPC victim selection within a bounded oldest-candidate window. The older request-local importance/release-order experiment was removed.

## LMCache changes kept

### Late async completion filtering

File: `third_party/LMCache/lmcache/integration/vllm/lmcache_mp_connector.py`.

Late receive completions for requests already finished by the engine are suppressed at the connector boundary while legitimate async send/store completion is preserved.

Upstream context:

- https://github.com/vllm-project/vllm/issues/49089
- https://github.com/vllm-project/vllm/pull/49278

### Scheduler-authorized load endpoint accounting

Files: `third_party/LMCache/lmcache/integration/vllm/lmcache_mp_metadata.py`, `third_party/LMCache/lmcache/integration/vllm/vllm_v1_adapter.py`, and connector/adapter plumbing.

The adapter distinguishes the full LMCache lookup hit from the exact external token endpoint authorized by vLLM, preventing an `N`-token cache match from becoming an `N`-token retrieve when the scheduler intentionally authorizes only `N-1` tokens.

Related LMCache issue:

- https://github.com/LMCache/LMCache/issues/4614

### Stage-1 async lookup state, freshness, and logical abandonment cleanup

Files: `third_party/LMCache/lmcache/integration/vllm/vllm_multi_process_adapter.py`, `third_party/LMCache/lmcache/integration/vllm/lmcache_mp_connector.py`, `third_party/LMCache/lmcache/v1/multiprocess/modules/lookup.py`, plus diagnostics plumbing.

This keeps the nonblocking lookup/status path, freshness guard, logical Stage-1 abandonment hook, and background reaper that eventually consumes stale prefetch results and releases held locks. Logical abandonment still does not physically cancel daemon-side work.

Related RFC/issues:

- https://github.com/LMCache/LMCache/issues/4945
- https://github.com/LMCache/LMCache/issues/2585

### Separate lookup and load admission

Files: `third_party/LMCache/lmcache/v1/distributed/config.py`, `storage_manager.py`, `storage_controllers/prefetch_controller.py`.

Lookup/discovery concurrency and slow L2-load concurrency are explicit independent limits. The current launcher uses lookup PF 16 and load PF 2. Omitting the lookup limit preserves the upstream whole-prefetch admission behavior. `--l2-prefetch-max-in-flight` remains a CLI alias for the explicit `--l2-load-max-in-flight` name so archived launchers do not immediately break.

### fs_native dedicated operation workers

Files: `third_party/LMCache/csrc/storage_backends/fs/connector.{h,cpp}`, `pybind.cpp`, `lmcache/v1/distributed/l2_adapters/{native_connector_l2_adapter.py,fs_native_l2_adapter.py}`, and config/tests.

Dedicated lookup/retrieve worker counts are self-enabling: zero means the operation uses the shared pool. The separate experimental boolean gate was removed. The selected project topology is 5 shared + 1 lookup + 2 retrieve workers.

### Rolling L2 store-byte admission

Files: `third_party/LMCache/lmcache/v1/distributed/config.py`, `storage_manager.py`, `storage_controllers/store_controller.py`.

The store-byte limit is explicit controller configuration (`max_inflight_store_bytes`) rather than a hidden environment read inside `StoreController`. Completed stores return budget; zero means unlimited.

### Semantic GNN placement and store policy

Files: `third_party/LMCache/lmcache/v1/distributed/placement_metadata.py` (project-owned new file) and `storage_controllers/store_policy.py` plus store/controller plumbing.

Short-Q metadata uses semantic `gpu/cpu/disk` intent. CPU/GPU-labelled chunks may remain in L1 and optionally receive L2 safety backing; disk-labelled chunks use L1 as staging and are removed according to the GNN dynamic store policy. Physical LMCache tier numbering is no longer overloaded as the prediction vocabulary.

### Diagnostics retained

Optional congestion, CHTHM, L1 reservation-lifetime, prefix-diagnostic, and warm-start L1 snapshot instrumentation remains gated and off unless explicitly enabled. These are experimental observability hooks, not changes to the default upstream policy.

## Removed project code

The refactor removes project-added persistent GPU-L0 implementation/smokes, the GPU-occupancy Stage-1 fallback, the old request-local importance hook, obsolete patch artifacts/backups, and superseded launch scripts. Upstream LMCache/vLLM references to their own native L0 terminology are not rewritten.

## Short-Q configuration surface

`Hierarchical_KV/shortq_placement/chunk_voting.py` keeps the source-controlled selection style:

- change `_selected_vote_policy` to choose the Python default policy;
- change `DEFAULT_VOTE_CONFIG` to change default thresholds.

Optional launcher overrides are:

- `GNN_CHUNK_VOTE_POLICY`
- `GNN_CHUNK_EXPECTED_BLOCKS`
- `GNN_CHUNK_GPU_MIN`
- `GNN_CHUNK_CPU_MIN`
- `GNN_CHUNK_CPU_TOP2_MIN`

Precedence is explicit caller configuration > environment > source defaults. The refactor preserves the latest tree's effective default policy: GPU argmax count `>= 8`, otherwise CPU argmax count `>= 2`, otherwise CPU top-2 count `>= 6`, otherwise disk.

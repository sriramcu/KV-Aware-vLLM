# SPDX-License-Identifier: Apache-2.0
"""Opt-in VPC diagnostic lifecycle, separated from production block allocation.

The BlockPool owns physical blocks, hashes, LRU and victim selection; this
module only observes them. All hash reads bypass recency/touch operations.
Invoked synchronously on the scheduler thread at existing allocator hooks;
periodic writer only serializes completed sample dictionaries.
"""
from __future__ import annotations

from collections import Counter
import time
from typing import TYPE_CHECKING, Any

from vllm.v1.core.kv_cache_utils import (
    make_block_hash_with_group_id, resolve_block_hashes,
)
from vllm.v1.core.vpc_policies.diagnostics import (
    occupancy, chunk_summary, hash_presence, prefix_shape,
)

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request


class VPCDiagnostics:
    """Scheduler-owned observations for one BlockPool; no cache mutation."""

    def __init__(self, pool: BlockPool, interval_s: float = 5):
        if interval_s <= 0:
            raise ValueError("VLLM_VPC_DIAG_INTERVAL_S must be > 0")
        self.pool = pool
        self.interval_s = interval_s
        self._last_sample_s = 0.0
        self._pending_reclaim: tuple[set[tuple], dict, dict] | None = None
        self.reset()

    def reset(self) -> None:
        self.seen_requests: set[str] = set()
        self.request_count = 0
        self.request_counts_by_phase = {"cold": 0, "warm": 0, "other": 0}
        self.prefix_totals: dict[str, int] = {
            "lookup_samples": 0, "total_blocks": 0,
            "reachable_any_blocks": 0, "stranded_any_blocks": 0,
            "reachable_idle_blocks": 0, "stranded_idle_blocks": 0,
        }
        # Index complete paths of 32 logical block hashes, not physical IDs.
        self.observed_chunks: dict[tuple, str] = {}
        self.gpu_chunks_by_hash: dict[Any, set[tuple]] = {}
        self._pending_reclaim = None

    def observe_request_prefix(
        self, request: Request, block_size: int, kv_cache_group_id: int = 0,
    ) -> None:
        """Record the prefix shape once per request at first cache lookup.

        This Q650-specific observation supports one 16-token block group.
        Unlike normal cache lookups it does not touch/allocate physical blocks.
        """
        pool = self.pool
        if kv_cache_group_id != 0:
            return
        request_id = request.request_id
        if request_id in self.seen_requests:
            return
        self.seen_requests.add(request_id)
        self.request_count += 1
        phase = ("cold" if "kvaware-cold-" in request_id else
                 "warm" if "kvaware-warm-" in request_id else "other")
        self.request_counts_by_phase[phase] += 1
        if block_size != 16 or pool.hash_block_size != 16:
            return
        hashes = resolve_block_hashes(
            request.block_hashes, pool.hash_block_size, block_size)
        n = min(len(hashes), request.num_tokens // block_size)
        keys = [make_block_hash_with_group_id(h, kv_cache_group_id)
                for h in hashes[:n]]
        bits = [hash_presence(pool.cached_block_hash_to_block, k) for k in keys]
        counters = self.prefix_totals
        counters["lookup_samples"] += 1
        counters["total_blocks"] += n
        for kind, offset in (("any", 0), ("idle", 1)):
            shape = prefix_shape([b[offset] for b in bits])
            counters[f"reachable_{kind}_blocks"] += shape["prefix_blocks"]
            counters[f"stranded_{kind}_blocks"] += shape["stranded_after_gap"]
        labels = getattr(request, "kv_importance_placements", None) or {}
        for start in range(0, n - n % 32, 32):
            chunk = tuple(keys[start:start + 32])
            label = labels.get(start, "unlabeled")
            previous = self.observed_chunks.get(chunk)
            if previous is None:
                self.observed_chunks[chunk] = label
                if label == "gpu":
                    for key in chunk:
                        self.gpu_chunks_by_hash.setdefault(key, set()).add(chunk)
            elif previous != label:
                counters["chunk_label_conflicts"] = (
                    counters.get("chunk_label_conflicts", 0) + 1)
        self.maybe_sample()

    def before_reclaim(self, block: KVCacheBlock, label: str | None) -> None:
        """Capture the first GPU victim and pre-invalidation chunk presence."""
        pool = self.pool
        stats = pool.vpc_stats
        if block.block_hash is not None and label == "gpu" and stats.first_gpu_reclaim is None:
            stats.first_gpu_reclaim = {
                "engine_elapsed_s": round((time.monotonic_ns() - stats.start_ns) / 1e9, 3),
                "requests_first_seen": self.request_count,
                "requests_first_seen_by_phase": dict(self.request_counts_by_phase),
                "physical_blocks_before_reclaim": occupancy(
                    pool.blocks, pool.kv_importance_by_block_id,
                    pool.cached_block_hashes_by_block),
            }
        self._pending_reclaim = None
        keys = set(pool.cached_block_hashes_by_block.get(block.block_id, ()))
        if block.block_hash is not None:
            keys.add(block.block_hash)
        chunks: set[tuple] = set()
        for key in keys:
            chunks.update(self.gpu_chunks_by_hash.get(key, ()))
        if not chunks:
            return
        cache_map = pool.cached_block_hash_to_block
        before_idle = {
            path: all(hash_presence(cache_map, key)[1] for key in path)
            for path in chunks
        }
        before_any = {
            path: all(hash_presence(cache_map, key)[0] for key in path)
            for path in chunks
        }
        self._pending_reclaim = (chunks, before_idle, before_any)

    def after_reclaim(self) -> None:
        pending = self._pending_reclaim
        self._pending_reclaim = None
        if not pending:
            return
        chunks, before_idle, before_any = pending
        cache_map = self.pool.cached_block_hash_to_block
        stats = self.pool.vpc_stats
        stats.gpu_chunk_reclaim_physical_events += 1
        for path in chunks:
            stats.gpu_chunk_reclaims[path[-1]] += 1
            if before_idle[path] and not all(
                hash_presence(cache_map, key)[1] for key in path
            ):
                stats.gpu_chunk_complete_idle_breaks += 1
            if before_any[path] and not all(
                hash_presence(cache_map, key)[0] for key in path
            ):
                stats.gpu_chunk_complete_any_breaks += 1

    def maybe_sample(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_sample_s < self.interval_s:
            return
        self._last_sample_s = now
        sample_t0 = time.perf_counter_ns()
        pool = self.pool
        stats = pool.vpc_stats
        blocks = occupancy(
            pool.blocks, pool.kv_importance_by_block_id,
            pool.cached_block_hashes_by_block,
        )
        stats.diagnostic_sample = {
            "sample_engine_elapsed_s": round(
                (time.monotonic_ns() - stats.start_ns) / 1e9, 3),
            "sample_request_count": self.request_count,
            "sample_requests_by_phase": dict(self.request_counts_by_phase),
            "group_id": 0,
            "native_block_tokens": pool.hash_block_size,
            "logical_chunk_blocks": 32,
            "physical_blocks": blocks,
            "observed_chunk_universe": chunk_summary(
                pool.cached_block_hash_to_block, self.observed_chunks),
            "request_first_lookup_cumulative": dict(self.prefix_totals),
            "free_queue_count": pool.get_num_free_blocks(),
            "accounting_ok": blocks["accounted_total"] == pool.num_gpu_blocks,
            "diagnostic_scan_cpu_ms": round(
                (time.perf_counter_ns() - sample_t0) / 1e6, 3),
        }

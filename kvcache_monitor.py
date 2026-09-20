"""
kv_cache_monitor.py
--------------------
Monkey-patches vLLM's FullAttentionManager.find_longest_cache_hit to record
per-block hit/miss events.

Because find_longest_cache_hit has no request_id parameter, we use a
thread-local context variable. Set it from your calling code:

    import kv_cache_monitor as mon
    mon.set_current_request_id("req-42")
    # vLLM now calls find_longest_cache_hit internally
    # hits/misses are attributed to "req-42"

With AsyncLLMEngine you can set it inside a request wrapper. With the
synchronous LLM class a simple incrementing counter is set automatically.

Usage
-----
    from kv_cache_monitor import install, report, to_dataframe
    install()

    from vllm import LLM
    llm = LLM(model="...", enable_prefix_caching=True)
    llm.generate(prompts)

    report()
    df = to_dataframe()
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Thread-local request-id context
# ---------------------------------------------------------------------------

_ctx = threading.local()
_auto_req_counter = 0
_counter_lock = threading.Lock()


def set_current_request_id(request_id: str) -> None:
    """Call this before each request to tag events with the correct id."""
    _ctx.request_id = request_id


def _get_current_request_id() -> str:
    rid = getattr(_ctx, "request_id", None)
    if rid is None:
        global _auto_req_counter
        with _counter_lock:
            _auto_req_counter += 1
            rid = f"auto-{_auto_req_counter}"
        _ctx.request_id = rid
    return rid


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BlockEvent:
    ts: float
    request_id: str
    block_hash_repr: str
    block_index: int    # position in the prefix chain (0 = first page)
    hit: bool
    group_id: int


@dataclass
class RequestSummary:
    request_id: str
    ts_start: float
    ts_end: float
    total_blocks_checked: int
    hit_blocks: int
    miss_blocks: int
    block_size: int
    hit_token_count: int
    total_token_count: int


# ---------------------------------------------------------------------------
# Central store (thread-safe)
# ---------------------------------------------------------------------------

class _Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.block_hit_counts:   dict[str, int]       = defaultdict(int)
        self.block_check_counts: dict[str, int]       = defaultdict(int)
        self.block_hit_requests: dict[str, set]       = defaultdict(set)
        self.events:             list[BlockEvent]      = []
        self.request_summaries:  dict[str, RequestSummary] = {}

    def record_lookup(self, ts, request_id, block_hash_repr,
                      block_index, hit, group_id) -> None:
        with self._lock:
            self.block_check_counts[block_hash_repr] += 1
            if hit:
                self.block_hit_counts[block_hash_repr] += 1
                self.block_hit_requests[block_hash_repr].add(request_id)
            self.events.append(BlockEvent(
                ts=ts, request_id=request_id,
                block_hash_repr=block_hash_repr,
                block_index=block_index, hit=hit, group_id=group_id,
            ))

    def record_request(self, summary: RequestSummary) -> None:
        with self._lock:
            existing = self.request_summaries.get(summary.request_id)
            if existing is None or summary.hit_blocks > existing.hit_blocks:
                self.request_summaries[summary.request_id] = summary


_STORE = _Store()


# ---------------------------------------------------------------------------
# Monkey-patch
# ---------------------------------------------------------------------------

_original_find_longest_cache_hit = None


def install(vllm_import_path: str | None = None) -> None:
    """Patch FullAttentionManager.find_longest_cache_hit in-place."""
    global _original_find_longest_cache_hit
    if _original_find_longest_cache_hit is not None:
        print("[kv_cache_monitor] Already installed.")
        return

    if vllm_import_path:
        import importlib
        FullAttentionManager = getattr(
            importlib.import_module(vllm_import_path), "FullAttentionManager"
        )
    else:
        from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
       
    _original_find_longest_cache_hit = FullAttentionManager.find_longest_cache_hit

    # ---- patched method — signature matches your vLLM version exactly ----
    @classmethod  # type: ignore[misc]
    def _patched(
        cls,
        block_hashes,
        max_length: int,
        kv_cache_group_ids: list,
        block_pool,
        kv_cache_spec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ):
        request_id = _get_current_request_id()
        ts_start = time.time()

        block_size = kv_cache_spec.block_size
        if dcp_world_size * pcp_world_size > 1:
            block_size *= dcp_world_size * pcp_world_size
        max_num_blocks = max_length // block_size

        computed_blocks: tuple[list, ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )

        total_checked = 0
        for block_index, block_hash in enumerate(
            itertools.islice(block_hashes, max_num_blocks)
        ):
            cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
            hit = cached_block is not None
            bh_repr = repr(block_hash)

            for group_id in kv_cache_group_ids:
                _STORE.record_lookup(
                    ts=time.time(),
                    request_id=request_id,
                    block_hash_repr=bh_repr,
                    block_index=block_index,
                    hit=hit,
                    group_id=group_id,
                )

            total_checked += 1
            if hit:
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break

        # eagle / alignment trimming — identical to original
        if use_eagle and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
        while (
            block_size != alignment_tokens
            and len(computed_blocks[0]) * block_size % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()

        hit_blocks = len(computed_blocks[0]) if computed_blocks else 0
        _STORE.record_request(RequestSummary(
            request_id=request_id,
            ts_start=ts_start,
            ts_end=time.time(),
            total_blocks_checked=total_checked,
            hit_blocks=hit_blocks,
            miss_blocks=total_checked - hit_blocks,
            block_size=block_size,
            hit_token_count=hit_blocks * block_size,
            total_token_count=total_checked * block_size,
        ))

        return computed_blocks

    FullAttentionManager.find_longest_cache_hit = _patched
    print("[kv_cache_monitor] Installed ✓")


def uninstall() -> None:
    global _original_find_longest_cache_hit
    if _original_find_longest_cache_hit is None:
        return
    try:
        from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    except ImportError:
        from vllm.core.kv_cache_manager import FullAttentionManager  # type: ignore
    FullAttentionManager.find_longest_cache_hit = _original_find_longest_cache_hit
    _original_find_longest_cache_hit = None
    print("[kv_cache_monitor] Uninstalled")


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def hottest_blocks(top_n: int = 20) -> list[dict]:
    with _STORE._lock:
        checks = dict(_STORE.block_check_counts)
        hits   = dict(_STORE.block_hit_counts)
        reqs   = {k: len(v) for k, v in _STORE.block_hit_requests.items()}
    rows = [
        {
            "block_hash": bh,
            "hit_count": hits.get(bh, 0),
            "check_count": n,
            "hit_rate": hits.get(bh, 0) / n,
            "unique_request_hits": reqs.get(bh, 0),
        }
        for bh, n in checks.items()
    ]
    rows.sort(key=lambda r: r["hit_count"], reverse=True)
    return rows[:top_n]


def per_request_stats() -> list[dict]:
    with _STORE._lock:
        summaries = list(_STORE.request_summaries.values())
    summaries.sort(key=lambda s: s.ts_start)
    return [
        {
            "request_id":           s.request_id,
            "hit_blocks":           s.hit_blocks,
            "miss_blocks":          s.miss_blocks,
            "total_blocks_checked": s.total_blocks_checked,
            "hit_rate":             s.hit_blocks / s.total_blocks_checked
                                    if s.total_blocks_checked else 0.0,
            "hit_tokens":           s.hit_token_count,
            "total_tokens":         s.total_token_count,
            "block_size":           s.block_size,
            "latency_ms":           (s.ts_end - s.ts_start) * 1000,
            "ts_start":             s.ts_start,
        }
        for s in summaries
    ]


def global_hit_rate() -> dict:
    with _STORE._lock:
        total_hits   = sum(_STORE.block_hit_counts.values())
        total_checks = sum(_STORE.block_check_counts.values())
    return {
        "total_block_lookups": total_checks,
        "total_block_hits":    total_hits,
        "total_block_misses":  total_checks - total_hits,
        "global_hit_rate":     total_hits / total_checks if total_checks else 0.0,
    }


def hit_rate_over_time(bucket_seconds: float = 10.0) -> list[dict]:
    with _STORE._lock:
        events = list(_STORE.events)
    if not events:
        return []
    t0 = events[0].ts
    buckets: dict[int, list] = defaultdict(list)
    for ev in events:
        buckets[int((ev.ts - t0) / bucket_seconds)].append(ev.hit)
    return [
        {
            "time_offset_s": bid * bucket_seconds,
            "lookups": len(evs),
            "hits": sum(evs),
            "hit_rate": sum(evs) / len(evs),
        }
        for bid, evs in sorted(buckets.items())
    ]


def to_dataframe():
    """Return all raw events as a pandas DataFrame."""
    import pandas as pd
    with _STORE._lock:
        events = list(_STORE.events)
    return pd.DataFrame([
        {
            "ts":           e.ts,
            "request_id":   e.request_id,
            "block_hash":   e.block_hash_repr,
            "block_index":  e.block_index,
            "hit":          e.hit,
            "group_id":     e.group_id,
        }
        for e in events
    ])


def reset() -> None:
    global _auto_req_counter
    with _STORE._lock:
        _STORE.block_hit_counts.clear()
        _STORE.block_check_counts.clear()
        _STORE.block_hit_requests.clear()
        _STORE.events.clear()
        _STORE.request_summaries.clear()
    with _counter_lock:
        _auto_req_counter = 0


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def report(top_n: int = 10) -> None:
    print("\n" + "=" * 60)
    print("  KV CACHE BLOCK HIT REPORT")
    print("=" * 60)
    g = global_hit_rate()
    print(f"\n[Global]\n"
          f"  Block lookups : {g['total_block_lookups']}\n"
          f"  Hits          : {g['total_block_hits']}\n"
          f"  Misses        : {g['total_block_misses']}\n"
          f"  Hit rate      : {g['global_hit_rate']:.1%}")

    print(f"\n[Hottest {top_n} blocks]")
    fmt = "  {:>6}  {:>6}  {:>8}  {:>10}  {}"
    print(fmt.format("hits", "checks", "hit_rate", "uniq_reqs", "block_hash (truncated)"))
    print("  " + "-" * 58)
    for row in hottest_blocks(top_n):
        bh = row["block_hash"]
        bh_str = bh[:46] + "…" if len(bh) > 46 else bh
        print(fmt.format(row["hit_count"], row["check_count"],
                         f"{row['hit_rate']:.1%}", row["unique_request_hits"], bh_str))

    print(f"\n[Per-request stats (first {top_n})]")
    fmt2 = "  {:>20}  {:>4}  {:>4}  {:>8}  {:>10}"
    print(fmt2.format("request_id", "hits", "miss", "hit_rate", "latency_ms"))
    print("  " + "-" * 55)
    for r in per_request_stats()[:top_n]:
        print(fmt2.format(r["request_id"][-20:], r["hit_blocks"], r["miss_blocks"],
                          f"{r['hit_rate']:.1%}", f"{r['latency_ms']:.1f}"))
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Self-test (no vLLM needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    print("Injecting synthetic data …")
    t0 = time.time()
    for req_i in range(20):
        set_current_request_id(f"req-{req_i:03d}")
        n_check = random.randint(4, 10)
        hit_blocks = min(4, req_i)
        for blk_i in range(n_check):
            bh = f"hash_{blk_i:04d}" if blk_i < 4 else f"hash_{blk_i:04d}_req{req_i}"
            hit = blk_i < hit_blocks
            _STORE.record_lookup(
                ts=t0 + req_i * 0.15 + blk_i * 0.01,
                request_id=f"req-{req_i:03d}",
                block_hash_repr=bh,
                block_index=blk_i,
                hit=hit,
                group_id=0,
            )
            if not hit:
                break
        _STORE.record_request(RequestSummary(
            request_id=f"req-{req_i:03d}",
            ts_start=t0 + req_i * 0.15,
            ts_end=t0 + req_i * 0.15 + 0.05,
            total_blocks_checked=min(hit_blocks + 1, n_check),
            hit_blocks=hit_blocks,
            miss_blocks=1,
            block_size=16,
            hit_token_count=hit_blocks * 16,
            total_token_count=n_check * 16,
        ))
    report()
# SPDX-License-Identifier: Apache-2.0
"""Read-only VPC occupancy and logical-prefix diagnostic functions.

Never touch LRU links, call a mutating cache lookup, or modify refcounts.
A hash may map to multiple physical blocks: inspect *all* duplicates when
classifying idle/active residency. Totals for request paths are observations,
not a global count of independent cache objects.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

LABELS = ("gpu", "cpu", "disk", "unlabeled")


def occupancy(blocks: list[Any], importance: dict[int, str], extra_hashes: dict) -> dict:
    active = 0
    active_labels = Counter()
    idle = Counter()
    uncached_available = 0
    special = 0
    for block in blocks:
        if block.is_null:
            special += 1
        elif block.ref_cnt > 0:
            active += 1
            label = importance.get(block.block_id, "unlabeled")
            active_labels[label if label in LABELS else "unlabeled"] += 1
        elif block.block_hash is not None or extra_hashes.get(block.block_id):
            label = importance.get(block.block_id, "unlabeled")
            idle[label if label in LABELS else "unlabeled"] += 1
        else:
            uncached_available += 1
    return {
        "total": len(blocks), "active": active,
        "active_by_label": {label: active_labels[label] for label in LABELS},
        "idle_cached": {label: idle[label] for label in LABELS},
        "idle_cached_total": sum(idle.values()),
        "uncached_available": uncached_available, "reserved_special": special,
        "accounted_total": active + sum(idle.values()) + uncached_available + special,
    }


def hash_presence(cache_map: Any, key: Any) -> tuple[bool, bool]:
    """(resident_any, resident_idle); includes all physical hash duplicates."""
    entry = cache_map._cache.get(key)  # direct read, NOT get_cached_block/touch
    if entry is None:
        return False, False
    blocks = entry.values() if isinstance(entry, dict) else (entry,)
    present = False
    idle = False
    for block in blocks:
        present = True
        idle |= block.ref_cnt == 0
    return present, idle


def prefix_shape(bits: list[bool] | tuple[bool, ...]) -> dict[str, int | bool]:
    first_gap = next((idx for idx, present in enumerate(bits) if not present), len(bits))
    retained = sum(bits)
    return {
        "positions": len(bits), "retained": retained,
        "prefix_blocks": first_gap,
        "first_gap_block": first_gap if first_gap < len(bits) else -1,
        "stranded_after_gap": sum(bits[first_gap + 1:]) if first_gap < len(bits) else 0,
        "complete": retained == len(bits),
    }


def chunk_summary(cache_map: Any, chunks: dict[tuple, str]) -> dict:
    """Snapshot distinct observed full chunks, each with 32 native block keys.

    Counts are for the *observed hash universe*, not an exhaustive census of
    all cache objects ever encountered. Each chunk is deduplicated by its full
    group-specific hash path, not by physical block ID or pointer adjacency.
    """
    labels = ("gpu", "cpu", "disk", "drop", "unlabeled")
    result = {label: {"observed": 0, "complete_any": 0,
                      "complete_idle": 0, "partial_any": 0,
                      "partial_idle": 0, "absent_any": 0,
                      "resident_any_blocks": 0, "resident_idle_blocks": 0,
                      "stranded_any_blocks": 0, "stranded_idle_blocks": 0}
              for label in labels}
    for path, raw_label in chunks.items():
        label = raw_label if raw_label in result else "unlabeled"
        counters = result[label]
        counters["observed"] += 1
        presences = [hash_presence(cache_map, k) for k in path]
        for kind, offset in (("any", 0), ("idle", 1)):
            bits = [x[offset] for x in presences]
            shape = prefix_shape(bits)
            counters[f"resident_{kind}_blocks"] += shape["retained"]
            counters[f"stranded_{kind}_blocks"] += shape["stranded_after_gap"]
            if shape["complete"]:
                counters[f"complete_{kind}"] += 1
            elif shape["retained"]:
                counters[f"partial_{kind}"] += 1
            elif kind == "any":
                counters["absent_any"] += 1
    return result

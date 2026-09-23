#!/usr/bin/env python3
"""Summarize learned-placement intent and observed physical hierarchy activity."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

STORE_RE = re.compile(
    r"\[GNN_EXCLUSIVE_STORE\].*target_L0=(\d+) target_L1=(\d+) target_L2=(\d+) "
    r"L0_reserved=(\d+) L1_stage_reserved=(\d+) l0_bytes=(\d+) host_stage_bytes=(\d+) "
    r"logical_stored=(\d+) success=(\S+)"
)
POLICY_RE = re.compile(
    r"\[GNN_EXCLUSIVE_L2_POLICY\] candidates=(\d+) persistent_l1=(\d+) l2_targets=(\d+) adapters=(\d+)"
)
L2_COMMIT_RE = re.compile(r"\[GNN_EXCLUSIVE_L2_COMMIT\] l2_keys=(\d+) delete_l1_staging=(\d+)")
MP_STORE_RE = re.compile(r"\[MP_STORE_DONE\].*success=(\S+) keys=(\d+) bytes=(\d+)")
MISS_RE = re.compile(r"\[GNN_PLACEMENT_METADATA_MISS\]")
EVICT_RE = re.compile(r"\[(?:L0_SMOKE_EVICT|L0_EVICT)[^]]*\]")
L1_FAIL_RE = re.compile(r"failed.*allocat|allocation.*fail", re.I)
WATERMARK_RE = re.compile(r"watermark", re.I)


def tree(path: Path | None):
    files = bytes_ = 0
    if path and path.exists():
        for p in path.rglob("*"):
            if p.is_file():
                files += 1
                try:
                    bytes_ += p.stat().st_size
                except OSError:
                    pass
    return {"files": files, "bytes": bytes_}


def _prom_value(path: Path | None, metric: str):
    if path is None or not path.exists():
        return None
    normalized = metric.replace(".", "[_\\.]")
    pat = re.compile(rf"^(?:{normalized})(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$", re.M)
    vals = [float(m.group(1)) for m in pat.finditer(path.read_text(errors="replace"))]
    return sum(vals) if vals else None


def _hash_dependence(path: Path | None):
    by_hash: dict[str, Counter[str]] = defaultdict(Counter)
    occurrences = 0
    if path is not None and path.exists():
        for line in path.open(errors="replace"):
            try:
                row = json.loads(line)
            except Exception:
                continue
            h = str(row.get("chunk_hash", ""))
            pred = str(row.get("prediction", ""))
            if h and pred:
                by_hash[h][pred] += 1
                occurrences += 1
    repeated = {h: c for h, c in by_hash.items() if sum(c.values()) > 1}
    conflicting = {h: c for h, c in repeated.items() if len(c) > 1}
    return {
        "occurrences": occurrences,
        "unique_hashes": len(by_hash),
        "repeated_hashes": len(repeated),
        "conflicting_hashes": len(conflicting),
        "conflicting_hash_fraction_among_repeated": (
            len(conflicting) / len(repeated) if repeated else 0.0
        ),
        "examples": [
            {"chunk_hash": h, "prediction_counts": dict(c)}
            for h, c in list(conflicting.items())[:20]
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precompute-summary", type=Path, required=True)
    ap.add_argument("--hash-prediction-trace", type=Path)
    ap.add_argument("--lmcache-log", type=Path, required=True)
    ap.add_argument("--lmcache-metrics", type=Path)
    ap.add_argument("--l2-dir", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    pre = json.loads(args.precompute_summary.read_text())
    target = Counter()
    reserved = Counter()
    bytes_seen = Counter()
    policies = Counter()
    commits = 0
    l2_done_keys = l2_done_bytes = 0
    misses = evicts = l1fails = watermarks = 0
    calls = ok = 0
    for line in args.lmcache_log.open(errors="replace"):
        if m := STORE_RE.search(line):
            vals = list(map(int, m.groups()[:8]))
            calls += 1
            ok += m.group(9).lower() == "true"
            for tier, val in zip(("L0", "L1", "L2"), vals[:3]):
                target[tier] += val
            reserved["L0"] += vals[3]
            reserved["L1_or_L2_staging"] += vals[4]
            bytes_seen["L0_reserved_bytes"] += vals[5]
            bytes_seen["host_stage_reserved_bytes"] += vals[6]
            reserved["logical"] += vals[7]
        if m := POLICY_RE.search(line):
            policies["candidates"] += int(m.group(1))
            policies["persistent_l1"] += int(m.group(2))
            policies["l2_targets"] += int(m.group(3))
        if m := L2_COMMIT_RE.search(line):
            commits += int(m.group(1))
        if m := MP_STORE_RE.search(line):
            if m.group(1).lower() == "true":
                l2_done_keys += int(m.group(2))
                l2_done_bytes += int(m.group(3))
        misses += bool(MISS_RE.search(line))
        evicts += bool(EVICT_RE.search(line))
        l1fails += bool(L1_FAIL_RE.search(line))
        watermarks += bool(WATERMARK_RE.search(line))

    final_metrics = {
        "l0_memory_usage_bytes": _prom_value(args.lmcache_metrics, "lmcache_mp.l0_memory_usage_bytes"),
        "l0_capacity_bytes": _prom_value(args.lmcache_metrics, "lmcache_mp.l0_capacity_bytes"),
        "l1_memory_usage_bytes": _prom_value(args.lmcache_metrics, "lmcache_mp.l1_memory_usage_bytes"),
        "l1_usage_ratio": _prom_value(args.lmcache_metrics, "lmcache_mp.l1_usage_ratio"),
    }

    result = {
        "precompute": {
            "chunk_occurrences": pre.get("chunk_occurrences"),
            "unique_chunk_hashes": pre.get("unique_chunk_hashes"),
            "occurrence_candidate_tier_counts": pre.get("occurrence_candidate_tier_counts"),
            "unique_runtime_tier_counts": pre.get("unique_runtime_tier_counts"),
            "block_class_counts": pre.get("block_class_counts"),
            "repeated_hashes": pre.get("repeated_hashes"),
            "conflicting_hashes": pre.get("conflicting_hashes"),
            "conflict_fraction": pre.get("conflicting_hash_fraction_among_repeated"),
            "prediction_timing": pre.get("prediction_timing"),
            "smoke_runtime_overrides": pre.get("smoke_runtime_overrides", []),
        },
        "hash_request_dependence": _hash_dependence(args.hash_prediction_trace),
        "runtime": {
            "store_calls": calls,
            "successful_store_calls": ok,
            "target_counts_seen_by_workers": dict(target),
            "reserved_counts_seen_by_workers": dict(reserved),
            "reserved_bytes_seen_by_workers": dict(bytes_seen),
            "store_policy_counts": dict(policies),
            "successful_l2_commit_keys_seen": commits,
            "successful_l2_store_done_keys": l2_done_keys,
            "successful_l2_store_done_bytes": l2_done_bytes,
            "metadata_miss_markers": misses,
            "l0_eviction_markers": evicts,
            "l1_allocation_failure_lines": l1fails,
            "l1_watermark_lines": watermarks,
        },
        "final_lmcache_metrics": final_metrics,
        "final_l2_footprint": tree(args.l2_dir),
        "note": (
            "Worker/rank store counters are physical observations and may include TP/object-group "
            "multiplicity; precompute counts are logical chunk decisions. L2 traffic bytes are "
            "successful physical store bytes, while final_l2_footprint is final residency."
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

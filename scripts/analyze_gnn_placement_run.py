#!/usr/bin/env python3
"""Summarize Short-Q placement intent and observed CPU-L1 / disk-L2 activity."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

DYNAMIC_STORE_RE = re.compile(
    r"\[GNN_DYNAMIC_STORE\].*target_gpu=(\d+) target_cpu=(\d+) target_disk=(\d+) "
    r"l1_backing=(\S+) host_reserved=(\d+) reserved_gpu=(\d+) "
    r"reserved_cpu=(\d+) reserved_disk=(\d+) host_bytes=(\d+) success=(\S+)"
)
DYNAMIC_STORE_DROP_RE = re.compile(
    r"\[GNN_DYNAMIC_STORE\].*target_drop=(\d+) reserved_drop=(\d+)"
)
POLICY_RE = re.compile(
    r"\[GNN_DYNAMIC_L2_POLICY\] candidates=(\d+) persistent_l1=(\d+) "
    r"disk_targets=(\d+) backing_targets=(\d+) l2_backing=(\S+) adapters=(\d+)"
)
L2_COMMIT_RE = re.compile(
    r"\[GNN_DYNAMIC_L2_COMMIT\] l2_keys=(\d+) delete_l1_staging=(\d+) "
    r"keep_l1_backing=(\d+)"
)
L2_SKIP_RE = re.compile(
    r"\[GNN_DYNAMIC_L2_SKIP\] skipped_l2_keys=(\d+) delete_l1_staging=(\d+) "
    r"keep_l1_backing=(\d+)"
)
MP_STORE_RE = re.compile(r"\[MP_STORE_DONE\].*success=(\S+) keys=(\d+) bytes=(\d+)")
MISS_RE = re.compile(r"\[GNN_PLACEMENT_METADATA_MISS\]")
L1_FAIL_RE = re.compile(r"failed.*allocat|allocation.*fail", re.I)
WATERMARK_RE = re.compile(r"watermark", re.I)


def tree(path: Path | None) -> dict[str, int]:
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


def _prom_value(path: Path | None, metric: str) -> float | None:
    if path is None or not path.exists():
        return None
    normalized = metric.replace(".", "[_\\.]")
    pat = re.compile(rf"^(?:{normalized})(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$", re.M)
    vals = [float(m.group(1)) for m in pat.finditer(path.read_text(errors="replace"))]
    return sum(vals) if vals else None


def _hash_dependence(path: Path | None) -> dict:
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


def _precompute_counts(pre: dict, new_key: str, old_key: str):
    """Read current placement names while tolerating pre-refactor summaries."""
    return pre.get(new_key, pre.get(old_key))


def main() -> None:
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
    commits = Counter()
    skips = Counter()
    l2_done_keys = l2_done_bytes = 0
    misses = l1fails = watermarks = 0
    calls = ok = 0

    for line in args.lmcache_log.open(errors="replace"):
        if m := DYNAMIC_STORE_RE.search(line):
            calls += 1
            ok += m.group(10).lower() == "true"
            target["gpu"] += int(m.group(1))
            target["cpu"] += int(m.group(2))
            target["disk"] += int(m.group(3))
            reserved["host_total"] += int(m.group(5))
            reserved["gpu"] += int(m.group(6))
            reserved["cpu"] += int(m.group(7))
            reserved["disk"] += int(m.group(8))
            bytes_seen["host_reserved_bytes"] += int(m.group(9))
        if m := DYNAMIC_STORE_DROP_RE.search(line):
            target["drop"] += int(m.group(1))
            reserved["drop"] += int(m.group(2))
        if m := POLICY_RE.search(line):
            policies["candidates"] += int(m.group(1))
            policies["persistent_l1"] += int(m.group(2))
            policies["disk_targets"] += int(m.group(3))
            policies["backing_targets"] += int(m.group(4))
        if m := L2_COMMIT_RE.search(line):
            commits["l2_keys"] += int(m.group(1))
            commits["delete_l1_staging"] += int(m.group(2))
            commits["keep_l1_backing"] += int(m.group(3))
        if m := L2_SKIP_RE.search(line):
            skips["l2_keys"] += int(m.group(1))
            skips["delete_l1_staging"] += int(m.group(2))
            skips["keep_l1_backing"] += int(m.group(3))
        if m := MP_STORE_RE.search(line):
            if m.group(1).lower() == "true":
                l2_done_keys += int(m.group(2))
                l2_done_bytes += int(m.group(3))
        misses += bool(MISS_RE.search(line))
        l1fails += bool(L1_FAIL_RE.search(line))
        watermarks += bool(WATERMARK_RE.search(line))

    final_metrics = {
        "l1_memory_usage_bytes": _prom_value(
            args.lmcache_metrics, "lmcache_mp.l1_memory_usage_bytes"
        ),
        "l1_usage_ratio": _prom_value(args.lmcache_metrics, "lmcache_mp.l1_usage_ratio"),
    }

    result = {
        "precompute": {
            "chunk_occurrences": pre.get("chunk_occurrences"),
            "unique_chunk_hashes": pre.get("unique_chunk_hashes"),
            "occurrence_candidate_placement_counts": _precompute_counts(
                pre,
                "occurrence_candidate_placement_counts",
                "occurrence_candidate_tier_counts",
            ),
            "unique_runtime_placement_counts": _precompute_counts(
                pre, "unique_runtime_placement_counts", "unique_runtime_tier_counts"
            ),
            "chunk_vote": pre.get("chunk_vote"),
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
            "target_placement_counts_seen_by_workers": dict(target),
            "host_reserved_counts_seen_by_workers": dict(reserved),
            "host_reserved_bytes_seen_by_workers": dict(bytes_seen),
            "store_policy_counts": dict(policies),
            "l2_commit_counts": dict(commits),
            "l2_store_cap_skip_counts": dict(skips),
            "successful_l2_store_done_keys": l2_done_keys,
            "successful_l2_store_done_bytes": l2_done_bytes,
            "metadata_miss_markers": misses,
            "l1_allocation_failure_lines": l1fails,
            "l1_watermark_lines": watermarks,
        },
        "final_lmcache_metrics": final_metrics,
        "final_l2_footprint": tree(args.l2_dir),
        "note": (
            "Worker/rank store counters are physical observations and may include "
            "TP/object-group multiplicity; precompute counts are logical chunk "
            "placement decisions. L2 traffic bytes are successful physical store "
            "bytes, while final_l2_footprint is final residency."
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

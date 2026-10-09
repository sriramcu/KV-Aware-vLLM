# SPDX-License-Identifier: Apache-2.0
"""Hindsight static 512-token hash ranking, no preloading/pinning/extra eviction.

The ranking is deliberately independent of budget. Only assign labels to
existing hashes seen in the *current trace* and never create KV contents.
A retrospective ranking is NOT a causal inference-time model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import os
from typing import Any, Mapping
import json
from pathlib import Path

TIERS = ("gpu", "cpu", "disk")


def load_ranking(path: str | Path) -> list[str]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = obj["ranking"] if isinstance(obj, dict) else obj
    if not isinstance(entries, list) or not entries:
        raise ValueError("oracle ranking must be a nonempty list")
    hashes = [entry["chunk_hash"] if isinstance(entry, dict) else entry
              for entry in entries]
    if any(not isinstance(h, str) or not h for h in hashes):
        raise ValueError("oracle ranking entries must have nonempty chunk_hash")
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate chunk hashes in oracle ranking")
    return hashes


def assign_oracle(runtime_hashes: set[str] | list[str], ranking: list[str],
                  gpu: int, cpu: int, disk: int, *, strict: bool = True) -> dict[str, str]:
    """Population-bounded static oracle; unselected hashes are DROP.

    Counts are class-label budgets, not resident chunks or admission quotas.
    Explicit budget cutoffs never preallocate or pin physical VPC blocks.
    """
    if min(gpu, cpu, disk) < 0:
        raise ValueError("oracle class budgets must be nonnegative")
    unique = set(runtime_hashes)
    if strict and unique != set(ranking):
        missing = unique - set(ranking)
        extra = set(ranking) - unique
        raise ValueError(f"oracle ranking/trace mismatch: missing={len(missing)} extra={len(extra)}")
    if gpu + cpu + disk > len(unique):
        raise ValueError("oracle budgets exceed unique chunk population")
    ordered = [h for h in ranking if h in unique]
    if not strict:
        ordered.extend(sorted(unique - set(ordered)))
    out = {}
    for index, h in enumerate(ordered):
        out[h] = ("gpu" if index < gpu else
                  "cpu" if index < gpu + cpu else
                  "disk" if index < gpu + cpu + disk else "drop")
    return out


def rank_from_occurrences(rows: list[dict]) -> dict:
    freq = Counter(row["chunk_hash"] for row in rows)
    return {
        "format": "kvaware-hindsight-ranking-v1",
        "source": "retrospective chunk-hash occurrence counts; not causal",
        "unique_chunks": len(freq), "chunk_occurrences": sum(freq.values()),
        "ranking": [{"chunk_hash": h, "occurrences": count}
                    for h, count in sorted(freq.items(), key=lambda pair: (-pair[1], pair[0]))],
    }


def oracle_mode_from_environment(environ: Mapping[str, str] | None = None) -> str:
    """Validate opt-in mode before the expensive GNN preprocessing starts."""
    env = os.environ if environ is None else environ
    mode = env.get("KV_CHUNK_PLACEMENT_MODE", "gnn").strip().lower()
    if mode not in ("gnn", "oracle"):
        raise ValueError(f"unknown KV_CHUNK_PLACEMENT_MODE={mode!r}")
    if mode == "oracle":
        if env.get("VLLM_VPC_POLICY") != "selective":
            raise ValueError("hindsight oracle currently requires VLLM_VPC_POLICY=selective")
        if not env.get("KV_ORACLE_RANKING_FILE"):
            raise ValueError("KV_ORACLE_RANKING_FILE is required for oracle")
    return mode


def apply_oracle_placements(
    runtime: dict[str, str],
    occurrences: list[dict[str, Any]],
    hash_prediction_rows: list[dict[str, Any]],
    *,
    smoke_force_min_unique_per_placement: int = 0,
    environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any] | None, list[str] | None, dict | None]:
    """Replace GNN labels with hindsight labels, updating existing sidecars.

    No mutation in normal GNN mode. The return values are the optional oracle
    run metadata, occurrence label vector, and per-hash label counts.
    """
    env = os.environ if environ is None else environ
    if oracle_mode_from_environment(env) != "oracle":
        return None, None, None
    if smoke_force_min_unique_per_placement:
        raise ValueError("oracle is incompatible with smoke forcing")
    ranking_file = env["KV_ORACLE_RANKING_FILE"]
    budgets = {
        tier: int(env.get(f"KV_ORACLE_{tier.upper()}_CHUNKS", default))
        for tier, default in (("gpu", 258), ("cpu", 767), ("disk", 1012))
    }
    assigned = assign_oracle(
        set(runtime), load_ranking(ranking_file),
        budgets["gpu"], budgets["cpu"], budgets["disk"],
    )
    # Compute/validate all assignments before modifying the live mappings.
    runtime.clear()
    runtime.update(assigned)
    occurrence_placements: list[str] = []
    candidate_by_hash: dict[str, Counter[str]] = defaultdict(Counter)
    for row in occurrences:
        chosen = runtime[row["chunk_hash"]]
        row.update(candidate_placement=chosen, runtime_placement=chosen,
                   duplicate_conflict=False)
        occurrence_placements.append(chosen)
        candidate_by_hash[row["chunk_hash"]][chosen] += 1
    for row in hash_prediction_rows:
        chosen = runtime[row["chunk_hash"]]
        row.update(prediction=chosen, runtime_placement=chosen,
                   duplicate_conflict=False)
    meta = {
        "ranking_file": ranking_file, "budgets": budgets,
        "mode": "hindsight_static_chunk_hash", "unique_hashes": len(runtime),
    }
    return meta, occurrence_placements, candidate_by_hash

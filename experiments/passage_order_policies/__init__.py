"""Deterministic post-retrieval ordering policies for RAG prompt construction.

The input is the *retriever-ranked* list. The original policy returns the
same list; canonical uses SHA-256 of the exact passage text (not Python hash).
All related per-passage fields are permuted together. No tokenizer or KV
operations occur here.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

POLICIES = ("original", "canonical")


def passage_order_indices(passages: Sequence[str], policy: str) -> list[int]:
    if policy == "original":
        return list(range(len(passages)))
    if policy != "canonical":
        raise ValueError(f"Unknown passage ordering policy {policy!r}; choose {POLICIES}")
    # Explicit tie-breaker avoids relying on sort implementation details.
    return sorted(
        range(len(passages)),
        key=lambda i: (
            hashlib.sha256(str(passages[i]).encode("utf-8")).hexdigest(),
            str(passages[i]),
            i,
        ),
    )


def apply_passage_order(
    retrieval_results: list[dict[str, Any]], policy: str
) -> list[dict[str, Any]]:
    if policy not in POLICIES:
        raise ValueError(f"Unknown passage ordering policy {policy!r}")
    if policy == "original":
        return retrieval_results  # Preserve the baseline exactly.
    out = []
    for record in retrieval_results:
        passages = record["sorted_passage"]
        indices = passage_order_indices(passages, policy)
        updated = dict(record)
        updated["sorted_passage"] = [passages[i] for i in indices]
        for name in ("sorted_passage_scores", "sorted_passage_ids"):
            data = record.get(name)
            if data is not None:
                if len(data) != len(passages):
                    raise ValueError(f"{name} length disagrees with sorted_passage")
                updated[name] = [data[i] for i in indices]
        updated["retrieval_rank_after_order"] = indices
        out.append(updated)
    return out

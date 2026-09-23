"""Resolve request-conditioned duplicate-hash placement conflicts.

Runtime metadata is intentionally only ``chunk_hash -> final tier``.  We still
score every request occurrence while the prediction cache is dormant, record
conflicts in diagnostics, then use this small policy to pick the one runtime
value for a repeated hash.
"""

from __future__ import annotations


def first_seen_wins(existing: str | None, candidate: str) -> str:
    return candidate if existing is None else existing


# Edit this one binding if a later experiment wants majority/coldest/etc.
_selected_duplicate_policy = first_seen_wins


def resolve_duplicate(existing: str | None, candidate: str) -> str:
    return _selected_duplicate_policy(existing, candidate)

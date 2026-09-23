"""Aggregate 32 block-tier predictions into one 512-token chunk placement.

To change chunk aggregation later, change only ``_selected_vote_policy``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

VALID_TIERS = ("L0", "L1", "L2")


def plurality_cold_tiebreak(block_tiers: Sequence[str]) -> str:
    """Plurality vote; ties prefer colder storage: L2 > L1 > L0."""
    if not block_tiers:
        raise ValueError("cannot vote an empty block tier list")
    bad = [tier for tier in block_tiers if tier not in VALID_TIERS]
    if bad:
        raise ValueError(f"invalid tier(s): {bad[:4]}")
    counts = Counter(block_tiers)
    best = max(counts.values())
    for tier in ("L2", "L1", "L0"):
        if counts[tier] == best:
            return tier
    raise AssertionError("unreachable")


def hottest_wins(block_tiers: Sequence[str]) -> str:
    if "L0" in block_tiers:
        return "L0"
    if "L1" in block_tiers:
        return "L1"
    return "L2"


# Edit this one binding for future block->chunk aggregation experiments.
_selected_vote_policy = plurality_cold_tiebreak


def vote_chunk_placement(block_tiers: Sequence[str]) -> str:
    return _selected_vote_policy(block_tiers)

"""Aggregate 32 block-tier predictions into one 512-token chunk placement.

To change chunk aggregation later, change only `_selected_vote_policy`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


VALID_TIERS = ("L0", "L1", "L2")


class BlockTierVote(str):
    """String-compatible tier vote with optional block-level metadata.

    The value itself is still exactly "L0", "L1", or "L2", so existing
    policies using Counter(), equality, membership, etc. continue to work
    unchanged.

    Extra metadata is available to policies that need it.
    """

    def __new__(
        cls,
        tier: str,
        *,
        cpu_top2: bool = False,
    ) -> "BlockTierVote":
        if tier not in VALID_TIERS:
            raise ValueError(f"invalid tier: {tier}")

        obj = str.__new__(cls, tier)
        obj.cpu_top2 = bool(cpu_top2)
        return obj


def make_block_tier_vote(
    tier: str,
    *,
    cpu_top2: bool = False,
) -> BlockTierVote:
    """Construct one enriched block-tier vote."""
    return BlockTierVote(
        tier,
        cpu_top2=cpu_top2,
    )


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


def gpu6_cpu5_else_l2(block_tiers: Sequence[str]) -> str:
    """
    Threshold promotion policy for one 512-token / 32-block chunk.

    Block mapping before this function:
        gpu       -> L0
        cpu       -> L1
        disk/drop -> L2

    Policy:
        >= 6 GPU/L0 blocks -> L0
        else >= 5 CPU/L1 blocks -> L1
        else -> L2

    L0 has priority if both thresholds are satisfied.
    """
    if len(block_tiers) != 32:
        raise ValueError(
            f"expected exactly 32 block predictions for a 512-token chunk, "
            f"got {len(block_tiers)}"
        )

    bad = [tier for tier in block_tiers if tier not in VALID_TIERS]
    if bad:
        raise ValueError(f"invalid tier(s): {bad[:4]}")

    counts = Counter(block_tiers)

    if counts["L0"] >= 6:
        return "L0"

    if counts["L1"] >= 5:
        return "L1"

    return "L2"


def gpu6_cpu5_cpu_top2_8_else_l2(
    block_tiers: Sequence[str],
) -> str:
    """
    Threshold + CPU top-2 rescue policy.

    Policy:
        >= 6 GPU/L0 argmax blocks -> L0
        else >= 5 CPU/L1 argmax blocks -> L1
        else CPU is top-2 for >= 8 blocks -> L1
        else -> L2

    `block_tiers` still contains the normal argmax-derived physical tier.
    The CPU top-2 flag is attached as metadata to each BlockTierVote.

    Rank score is not used.
    """
    if len(block_tiers) != 32:
        raise ValueError(
            f"expected exactly 32 block predictions for a 512-token chunk, "
            f"got {len(block_tiers)}"
        )

    bad = [tier for tier in block_tiers if tier not in VALID_TIERS]
    if bad:
        raise ValueError(f"invalid tier(s): {bad[:4]}")

    counts = Counter(block_tiers)

    if counts["L0"] >= 8:
        return "L0"

    if counts["L1"] >= 2:
        return "L1"

    # Fail loudly if somebody tries to use this policy without the
    # one-time top-2 metadata plumbing.
    missing_metadata = [
        i
        for i, tier in enumerate(block_tiers)
        if not hasattr(tier, "cpu_top2")
    ]
    if missing_metadata:
        raise RuntimeError(
            "gpu6_cpu5_cpu_top2_8_else_l2 requires enriched "
            "BlockTierVote inputs carrying cpu_top2 metadata; "
            f"missing metadata at block(s) {missing_metadata[:4]}"
        )

    cpu_top2_count = sum(
        1
        for tier in block_tiers
        if bool(tier.cpu_top2)
    )

    if cpu_top2_count >= 6:
        return "L1"

    return "L2"


# ----------------------------------------------------------------------
# Edit ONLY this binding for block -> chunk aggregation experiments.
# ----------------------------------------------------------------------

_selected_vote_policy = gpu6_cpu5_cpu_top2_8_else_l2


def selected_vote_policy_name() -> str:
    """Canonical policy name for diagnostics / experiment metadata."""
    return _selected_vote_policy.__name__


def vote_chunk_placement(block_tiers: Sequence[str]) -> str:
    return _selected_vote_policy(block_tiers)
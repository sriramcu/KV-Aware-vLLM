# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared logical KV-tier voting and block-to-chunk coarsening helpers."""

from collections.abc import Mapping, Sequence

_VALID_TIERS = frozenset({"disk", "cpu", "gpu"})


def _normalize_block_tiers(block_tiers: Sequence[str]) -> list[str]:
    if not block_tiers:
        raise ValueError("block_tiers must not be empty")

    normalized = [str(tier).lower() for tier in block_tiers]
    invalid = set(normalized) - _VALID_TIERS
    if invalid:
        raise ValueError(f"Unknown KV tier(s): {sorted(invalid)}")
    return normalized


def vote_chunk_tier_legacy_hot_priority(block_tiers: Sequence[str]) -> str:
    """Legacy policy: any GPU label wins, then any CPU label, else disk."""
    normalized = _normalize_block_tiers(block_tiers)
    if "gpu" in normalized:
        return "gpu"
    if "cpu" in normalized:
        return "cpu"
    return "disk"


def vote_chunk_tier_disk_majority(block_tiers: Sequence[str]) -> str:
    """Current policy used for one LMCache-sized logical chunk.

    Disk wins only with a strict majority (> 50%). Otherwise CPU and GPU
    compete directly. An exact CPU/GPU tie goes to GPU, preserving the old
    hot-tier tie-break without changing GPU storage semantics.
    """
    normalized = _normalize_block_tiers(block_tiers)

    disk_count = normalized.count("disk")
    if 2 * disk_count > len(normalized):
        return "disk"

    cpu_count = normalized.count("cpu")
    gpu_count = normalized.count("gpu")
    return "gpu" if gpu_count >= cpu_count else "cpu"


def vote_chunk_tier(block_tiers: Sequence[str]) -> str:
    """Single selection point for the experiment's chunk voting policy.

    To try a different policy, define it above and change only this call. Both
    LMCache placement and vLLM GPU retention consume this same selection point.
    """
    return vote_chunk_tier_disk_majority(block_tiers)
    # Legacy alternative:
    # return vote_chunk_tier_legacy_hot_priority(block_tiers)


def coarsen_block_tiers_to_chunks(
    block_tiers: Mapping[int, str],
    *,
    first_token: int,
    num_tokens: int,
    block_size: int,
    chunk_size: int,
    missing_tier: str = "cpu",
) -> list[str]:
    """Vote fine-grained block tiers into consecutive chunk tiers.

    Chunk boundaries start at ``first_token`` and advance by ``chunk_size``;
    this intentionally matches the LMCache placement loop. The final chunk is
    not padded. If a block index inside a real chunk extent has no label, use
    ``missing_tier`` (currently CPU, matching the pre-refactor LMCache path).
    """
    if first_token < 0:
        raise ValueError(f"first_token must be non-negative, got {first_token}")
    if num_tokens < first_token:
        raise ValueError(
            f"num_tokens ({num_tokens}) must be >= first_token ({first_token})"
        )
    if block_size <= 0 or chunk_size <= 0:
        raise ValueError(
            f"block_size and chunk_size must be positive; got "
            f"block={block_size}, chunk={chunk_size}"
        )

    missing_tier = str(missing_tier).lower()
    if missing_tier not in _VALID_TIERS:
        raise ValueError(f"Unknown missing_tier {missing_tier!r}")

    chunk_tiers: list[str] = []
    for chunk_start in range(first_token, num_tokens, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_tokens)
        first_block = chunk_start // block_size
        last_block = (chunk_end - 1) // block_size
        fine_tiers = [
            block_tiers.get(block_index, missing_tier)
            for block_index in range(first_block, last_block + 1)
        ]
        chunk_tiers.append(vote_chunk_tier(fine_tiers))

    return chunk_tiers

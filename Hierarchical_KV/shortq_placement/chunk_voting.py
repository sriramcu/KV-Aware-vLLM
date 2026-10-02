"""Aggregate Short-Q block predictions into one LMCache chunk placement.

The normal experiment configuration remains Python-native: choose the default
vote function by changing ``_selected_vote_policy`` and change numeric knobs in
``DEFAULT_VOTE_CONFIG``. Launchers may optionally override the policy and knobs
through environment variables without editing this module.

Policy precedence: explicit caller policy > ``GNN_CHUNK_VOTE_POLICY`` >
``_selected_vote_policy``.
Config precedence: explicit caller config > environment overrides >
``DEFAULT_VOTE_CONFIG``.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

VALID_PLACEMENTS = ("gpu", "cpu", "disk")


class BlockPlacementVote(str):
    """String-compatible placement vote with optional block metadata."""

    def __new__(
        cls,
        placement: str,
        *,
        cpu_top2: bool = False,
    ) -> "BlockPlacementVote":
        if placement not in VALID_PLACEMENTS:
            raise ValueError(f"invalid placement: {placement}")
        obj = str.__new__(cls, placement)
        obj.cpu_top2 = bool(cpu_top2)
        return obj


def make_block_placement_vote(
    placement: str,
    *,
    cpu_top2: bool = False,
) -> BlockPlacementVote:
    return BlockPlacementVote(placement, cpu_top2=cpu_top2)


@dataclass(frozen=True)
class ChunkVoteConfig:
    """Numeric knobs shared by chunk-voting policies."""

    expected_blocks: int = 32
    gpu_min: int = 8
    cpu_min: int = 2
    cpu_top2_min: int = 6


# Edit this object for the normal Python-defined experiment thresholds.
DEFAULT_VOTE_CONFIG = ChunkVoteConfig(
    gpu_min=8,
    cpu_min=2,
    cpu_top2_min=6,
)


def _validate_votes(
    block_placements: Sequence[str], expected_blocks: int | None
) -> None:
    if expected_blocks is not None and len(block_placements) != expected_blocks:
        raise ValueError(
            f"expected exactly {expected_blocks} block predictions for one chunk, "
            f"got {len(block_placements)}"
        )
    if not block_placements:
        raise ValueError("cannot vote an empty block placement list")
    bad = [value for value in block_placements if value not in VALID_PLACEMENTS]
    if bad:
        raise ValueError(f"invalid placement(s): {bad[:4]}")


def plurality_cold_tiebreak(
    block_placements: Sequence[str], config: ChunkVoteConfig
) -> str:
    """Plurality vote; ties prefer colder storage: disk > cpu > gpu."""
    _validate_votes(block_placements, None)
    counts = Counter(block_placements)
    best = max(counts.values())
    for placement in ("disk", "cpu", "gpu"):
        if counts[placement] == best:
            return placement
    raise AssertionError("unreachable")


def hottest_wins(block_placements: Sequence[str], config: ChunkVoteConfig) -> str:
    """Choose the hottest placement present in the chunk."""
    _validate_votes(block_placements, None)
    if "gpu" in block_placements:
        return "gpu"
    if "cpu" in block_placements:
        return "cpu"
    return "disk"


def threshold(block_placements: Sequence[str], config: ChunkVoteConfig) -> str:
    """Promote a chunk when enough argmax block votes target GPU or CPU."""
    _validate_votes(block_placements, config.expected_blocks)
    counts = Counter(block_placements)
    if counts["gpu"] >= config.gpu_min:
        return "gpu"
    if counts["cpu"] >= config.cpu_min:
        return "cpu"
    return "disk"


def threshold_with_cpu_top2(
    block_placements: Sequence[str], config: ChunkVoteConfig
) -> str:
    """Threshold policy with a CPU top-2 rescue path.

    Current defaults preserve the effective policy used by the latest uploaded
    tree: GPU argmax >= 8 -> gpu; CPU argmax >= 2 -> cpu; otherwise CPU top-2
    >= 6 -> cpu; else disk.
    """
    _validate_votes(block_placements, config.expected_blocks)
    counts = Counter(block_placements)
    if counts["gpu"] >= config.gpu_min:
        return "gpu"
    if counts["cpu"] >= config.cpu_min:
        return "cpu"

    missing_metadata = [
        i for i, vote in enumerate(block_placements) if not hasattr(vote, "cpu_top2")
    ]
    if missing_metadata:
        raise RuntimeError(
            "threshold_with_cpu_top2 requires BlockPlacementVote inputs carrying "
            f"cpu_top2 metadata; missing metadata at block(s) {missing_metadata[:4]}"
        )
    if sum(bool(vote.cpu_top2) for vote in block_placements) >= config.cpu_top2_min:
        return "cpu"
    return "disk"


VotePolicy = Callable[[Sequence[str], ChunkVoteConfig], str]
POLICIES: dict[str, VotePolicy] = {
    "plurality_cold_tiebreak": plurality_cold_tiebreak,
    "hottest_wins": hottest_wins,
    "threshold": threshold,
    "threshold_with_cpu_top2": threshold_with_cpu_top2,
}

# ---------------------------------------------------------------------------
# Normal source-controlled selection point. Change only this binding to switch
# policies in Python. GNN_CHUNK_VOTE_POLICY is an optional launcher override.
# ---------------------------------------------------------------------------
_selected_vote_policy: VotePolicy = threshold_with_cpu_top2


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def resolve_vote_config(explicit: ChunkVoteConfig | None = None) -> ChunkVoteConfig:
    """Resolve numeric knobs using explicit > environment > source defaults."""
    if explicit is not None:
        config = explicit
    else:
        config = replace(
            DEFAULT_VOTE_CONFIG,
            expected_blocks=_env_int(
                "GNN_CHUNK_EXPECTED_BLOCKS", DEFAULT_VOTE_CONFIG.expected_blocks
            ),
            gpu_min=_env_int("GNN_CHUNK_GPU_MIN", DEFAULT_VOTE_CONFIG.gpu_min),
            cpu_min=_env_int("GNN_CHUNK_CPU_MIN", DEFAULT_VOTE_CONFIG.cpu_min),
            cpu_top2_min=_env_int(
                "GNN_CHUNK_CPU_TOP2_MIN", DEFAULT_VOTE_CONFIG.cpu_top2_min
            ),
        )
    if config.expected_blocks <= 0:
        raise ValueError("expected_blocks must be > 0")
    for name in ("gpu_min", "cpu_min", "cpu_top2_min"):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be >= 0")
    return config


def resolve_vote_policy(
    explicit: str | VotePolicy | None = None,
) -> VotePolicy:
    """Resolve policy using explicit > environment > source binding."""
    if callable(explicit):
        return explicit
    name = explicit or os.getenv("GNN_CHUNK_VOTE_POLICY", "").strip()
    if not name:
        return _selected_vote_policy
    if name not in POLICIES:
        raise ValueError(
            f"unknown chunk vote policy {name!r}; choose one of {sorted(POLICIES)}"
        )
    return POLICIES[name]


def selected_vote_policy_name() -> str:
    return resolve_vote_policy().__name__


def selected_vote_config() -> ChunkVoteConfig:
    return resolve_vote_config()


def vote_chunk_placement(
    block_placements: Sequence[str],
    config: ChunkVoteConfig | None = None,
    policy: str | VotePolicy | None = None,
) -> str:
    resolved_config = resolve_vote_config(config)
    resolved_policy = resolve_vote_policy(policy)
    return resolved_policy(block_placements, resolved_config)

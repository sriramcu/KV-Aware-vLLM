"""Aggregate Short-Q block predictions into one LMCache chunk placement.

The normal experiment configuration remains Python-native: choose the default
vote function by changing ``_selected_vote_policy`` and change numeric knobs in
``DEFAULT_VOTE_CONFIG``. Launchers may optionally override the policy and knobs
through environment variables without editing this module.

Policy precedence: explicit caller policy > ``GNN_CHUNK_VOTE_POLICY`` >
``_selected_vote_policy``.
Config precedence: explicit caller config > environment overrides >
``DEFAULT_VOTE_CONFIG``.

Random-exclusive policies are also selected through ``GNN_CHUNK_VOTE_POLICY``.
They are assigned once per unique LMCache chunk hash by the precompute driver,
not by ``vote_chunk_placement``. This keeps random placement stable for repeated
hashes and lets matched-random baselines control the exact unique-hash tier
population.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

VALID_PLACEMENTS = ("gpu", "cpu", "disk")
RAW_SHORTQ_CLASSES = ("drop", "disk", "cpu", "gpu")
ALL_FINAL_PLACEMENTS = ("gpu", "cpu", "disk", "drop")


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


def _validate_raw_class_votes(
    block_classes: Sequence[str], expected_blocks: int | None
) -> None:
    if expected_blocks is not None and len(block_classes) != expected_blocks:
        raise ValueError(
            f"expected exactly {expected_blocks} raw block predictions for one chunk, "
            f"got {len(block_classes)}"
        )
    if not block_classes:
        raise ValueError("cannot vote an empty raw block-class list")
    bad = [value for value in block_classes if value not in RAW_SHORTQ_CLASSES]
    if bad:
        raise ValueError(f"invalid raw Short-Q class(es): {bad[:4]}")


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


def threshold_with_drop_plurality(
    raw_block_classes: Sequence[str], config: ChunkVoteConfig
) -> str:
    """Threshold 8/2, then true DROP only on a strict raw-class plurality.

    This policy intentionally operates on the checkpoint-native 4-way argmax
    classes before ``drop`` is collapsed into ``disk`` by the legacy mapping:

      * GPU argmax count >= ``gpu_min`` -> ``gpu``
      * else CPU argmax count >= ``cpu_min`` -> ``cpu``
      * else if raw DROP has a strict plurality -> ``drop``
      * else -> ``disk``

    A DROP/DISK tie therefore falls back to ``disk``.
    """
    _validate_raw_class_votes(raw_block_classes, config.expected_blocks)
    counts = Counter(raw_block_classes)
    if counts["gpu"] >= config.gpu_min:
        return "gpu"
    if counts["cpu"] >= config.cpu_min:
        return "cpu"
    if counts["drop"] > max(counts["disk"], counts["cpu"], counts["gpu"]):
        return "drop"
    return "disk"


VotePolicy = Callable[[Sequence[str], ChunkVoteConfig], str]


def _random_policy_requires_unique_hash_assignment(
    block_placements: Sequence[str], config: ChunkVoteConfig
) -> str:
    del block_placements, config
    raise RuntimeError(
        "random-exclusive chunk placement must be assigned by "
        "assign_random_exclusive_placements() after the precompute driver has "
        "collected the complete unique chunk-hash set"
    )


def random_exclusive_no_drop(
    block_placements: Sequence[str], config: ChunkVoteConfig
) -> str:
    """Marker policy: exact balanced random GPU/CPU/DISK assignment."""
    return _random_policy_requires_unique_hash_assignment(block_placements, config)


def random_exclusive_with_drop(
    block_placements: Sequence[str], config: ChunkVoteConfig
) -> str:
    """Marker policy: exact balanced random GPU/CPU/DISK/DROP assignment."""
    return _random_policy_requires_unique_hash_assignment(block_placements, config)


def random_exclusive_matched(
    block_placements: Sequence[str], config: ChunkVoteConfig
) -> str:
    """Marker policy: random assignment with an exact configured distribution."""
    return _random_policy_requires_unique_hash_assignment(block_placements, config)


POLICIES: dict[str, VotePolicy] = {
    "plurality_cold_tiebreak": plurality_cold_tiebreak,
    "hottest_wins": hottest_wins,
    "threshold": threshold,
    "threshold_with_cpu_top2": threshold_with_cpu_top2,
    "threshold_with_drop_plurality": threshold_with_drop_plurality,
    "random_exclusive_no_drop": random_exclusive_no_drop,
    "random_exclusive_with_drop": random_exclusive_with_drop,
    "random_exclusive_matched": random_exclusive_matched,
}

RANDOM_EXCLUSIVE_POLICIES = {
    "random_exclusive_no_drop",
    "random_exclusive_with_drop",
    "random_exclusive_matched",
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


def is_random_exclusive_policy(policy_name: str) -> bool:
    return policy_name in RANDOM_EXCLUSIVE_POLICIES


def _parse_matched_distribution(raw: str) -> dict[str, float]:
    if not raw.strip():
        raise ValueError(
            "random_exclusive_matched requires GNN_RANDOM_MATCH_DISTRIBUTION, "
            "for example gpu=258,cpu=767,disk=1012,drop=2820"
        )
    values = {placement: 0.0 for placement in ALL_FINAL_PLACEMENTS}
    for item in raw.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(
                "GNN_RANDOM_MATCH_DISTRIBUTION entries must use name=value"
            )
        name, value = item.split("=", 1)
        name = name.strip().lower()
        if name not in values:
            raise ValueError(
                f"unknown placement {name!r} in GNN_RANDOM_MATCH_DISTRIBUTION"
            )
        number = float(value.strip())
        if number < 0:
            raise ValueError("random matched-distribution weights must be >= 0")
        values[name] = number
    if sum(values.values()) <= 0:
        raise ValueError("GNN_RANDOM_MATCH_DISTRIBUTION must have positive mass")
    return values


def _allocate_exact_counts(
    total: int, weights: dict[str, float]
) -> dict[str, int]:
    """Largest-remainder allocation so counts sum exactly to ``total``."""
    if total < 0:
        raise ValueError("total must be >= 0")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("random placement weights must have positive mass")
    raw = {
        placement: total * float(weights.get(placement, 0.0)) / weight_sum
        for placement in ALL_FINAL_PLACEMENTS
    }
    counts = {placement: int(raw[placement]) for placement in ALL_FINAL_PLACEMENTS}
    remaining = total - sum(counts.values())
    placement_order = {
        placement: i for i, placement in enumerate(ALL_FINAL_PLACEMENTS)
    }
    order = sorted(
        ALL_FINAL_PLACEMENTS,
        key=lambda placement: (
            -(raw[placement] - counts[placement]),
            placement_order[placement],
        ),
    )
    for placement in order[:remaining]:
        counts[placement] += 1
    return counts


def assign_random_exclusive_placements(
    chunk_hashes: Sequence[str], policy_name: str
) -> tuple[dict[str, str], dict[str, object]]:
    """Assign reproducible random exclusive placements to unique chunk hashes.

    Assignment is performed over the complete unique-hash set, so repeated
    hashes always get the same placement and the requested tier population is
    exact rather than only correct in expectation.

    Environment knobs:

    ``GNN_RANDOM_CHUNK_SEED``
        Integer seed, default 0.

    ``GNN_RANDOM_MATCH_DISTRIBUTION``
        Required only for ``random_exclusive_matched``. Comma-separated
        ``name=value`` weights/counts, e.g. ``gpu=258,cpu=767,disk=1012,drop=2820``.
        Values are normalized and converted to exact unique-hash counts by the
        largest-remainder method.
    """
    if policy_name not in RANDOM_EXCLUSIVE_POLICIES:
        raise ValueError(f"not a random-exclusive policy: {policy_name!r}")

    unique_hashes = sorted(set(str(h) for h in chunk_hashes))
    seed = _env_int("GNN_RANDOM_CHUNK_SEED", 0)
    if policy_name == "random_exclusive_no_drop":
        weights = {"gpu": 1.0, "cpu": 1.0, "disk": 1.0, "drop": 0.0}
        source = "uniform_gpu_cpu_disk"
    elif policy_name == "random_exclusive_with_drop":
        weights = {"gpu": 1.0, "cpu": 1.0, "disk": 1.0, "drop": 1.0}
        source = "uniform_gpu_cpu_disk_drop"
    else:
        raw = os.getenv("GNN_RANDOM_MATCH_DISTRIBUTION", "")
        weights = _parse_matched_distribution(raw)
        source = raw

    counts = _allocate_exact_counts(len(unique_hashes), weights)
    labels: list[str] = []
    for placement in ALL_FINAL_PLACEMENTS:
        labels.extend([placement] * counts[placement])
    if len(labels) != len(unique_hashes):
        raise AssertionError("random placement schedule length mismatch")

    rng = random.Random(seed)
    rng.shuffle(labels)
    assignments = dict(zip(unique_hashes, labels, strict=True))
    summary: dict[str, object] = {
        "policy": policy_name,
        "seed": seed,
        "distribution_source": source,
        "unique_hashes": len(unique_hashes),
        "assigned_counts": counts,
    }
    return assignments, summary

# SPDX-License-Identifier: Apache-2.0
# [SC] Project-specific changes in this upstream file are marked with [SC];
# see repo-root docs/SC_MODIFICATIONS.md for rationale and provenance.

"""Short-Q hash -> semantic KV placement metadata.

The values are placement intents (``gpu``, ``cpu``, ``disk``, ``drop``), not
LMCache physical tier numbers. In the current architecture GPU intent is
consumed by vLLM's prefix-cache policy, CPU intent receives L1 backing, disk
intent targets L2, and drop receives no LMCache persistence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

from lmcache.logging import init_logger

logger = init_logger(__name__)
_VALID_PLACEMENTS = {"gpu", "cpu", "disk", "drop"}
_lock = Lock()
_loaded_path: str | None = None
_placements: dict[str, str] = {}
_missed: set[str] = set()


def _metadata_path() -> str:
    return os.getenv("LMCACHE_GNN_PLACEMENT_METADATA", "").strip()


def _load_if_needed() -> None:
    global _loaded_path, _placements
    path = _metadata_path()
    if path == _loaded_path:
        return
    with _lock:
        if path == _loaded_path:
            return
        if not path:
            _placements = {}
            _loaded_path = path
            return
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("GNN placement metadata must be a JSON object")
        parsed: dict[str, str] = {}
        for key, value in raw.items():
            placement = str(value).lower()
            if placement not in _VALID_PLACEMENTS:
                raise ValueError(f"invalid placement for {key}: {value!r}")
            parsed[str(key).lower()] = placement
        _placements = parsed
        _loaded_path = path
        logger.info("[GNN_PLACEMENT_METADATA] path=%s entries=%d", path, len(parsed))


def get_chunk_placement(
    chunk_hash: bytes, *, missing_default: str = "disk"
) -> str:
    """Return semantic placement intent for one rolling LMCache chunk hash."""
    _load_if_needed()
    key = bytes(chunk_hash).hex().lower()
    placement = _placements.get(key)
    if placement is not None:
        return placement
    default = missing_default.lower()
    if default not in _VALID_PLACEMENTS:
        raise ValueError(f"invalid missing_default {missing_default!r}")
    if key not in _missed:
        _missed.add(key)
        logger.warning(
            "[GNN_PLACEMENT_METADATA_MISS] chunk_hash=%s fallback=%s", key, default
        )
    return default

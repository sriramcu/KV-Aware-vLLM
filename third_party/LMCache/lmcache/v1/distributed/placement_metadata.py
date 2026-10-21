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
import time
from pathlib import Path
from threading import Lock

from lmcache.logging import init_logger

logger = init_logger(__name__)
_VALID_PLACEMENTS = {"gpu", "cpu", "disk", "drop"}
_lock = Lock()
_loaded_path: str | None = None
_loaded_stamp: tuple[int, int] | None = None
_last_stat_at = 0.0
_placements: dict[str, str] = {}
_missed: set[str] = set()
# Capture the semantic destination when a newly computed object is staged in
# host L1. Background L2 writes may complete after a new policy wave begins;
# they must use this pinned decision, never a newer map snapshot.
_staged_intents: dict[object, str] = {}
_pin_lock = Lock()


def pin_store_intent(key: object, placement: str) -> None:
    """Pin an admission for one *full ObjectKey* (rank, salt and group).

    The pin is a staging hint. The asynchronous L2 controller additionally
    snapshots this hint on each submitted task so later readmissions cannot
    redirect the old task's completion/deletion decisions.
    """
    if placement not in _VALID_PLACEMENTS:
        raise ValueError(placement)
    chunk_hash = getattr(key, "chunk_hash", None)
    if not isinstance(chunk_hash, bytes):
        raise TypeError("store intent requires a full ObjectKey")
    with _pin_lock:
        previous = _staged_intents.get(key)
        _staged_intents[key] = placement
    if previous is None:
        logger.info("[BAYES_NEW_STORE_INTENT] chunk_hash=%s tier=%s", chunk_hash.hex(), placement)
    elif previous != placement:
        logger.info("[BAYES_READMISSION_STORE] chunk_hash=%s from=%s to=%s",
                    chunk_hash.hex(), previous, placement)


def get_store_intent(key: object) -> str:
    chunk_hash = getattr(key, "chunk_hash", None)
    if not isinstance(chunk_hash, bytes):
        raise TypeError("store intent requires a full ObjectKey")
    with _pin_lock:
        old = _staged_intents.get(key)
    return old if old is not None else get_chunk_placement(chunk_hash)



def _metadata_path() -> str:
    return os.getenv("LMCACHE_GNN_PLACEMENT_METADATA", "").strip()


def _load_if_needed(*, force: bool = False) -> None:
    global _loaded_path, _loaded_stamp, _last_stat_at, _placements
    path = _metadata_path()
    dynamic = os.getenv("LMCACHE_GNN_PLACEMENT_DYNAMIC", "0") == "1"
    # Existing static behavior is unchanged unless the new env flag is on.
    if not dynamic and path == _loaded_path:
        return
    if dynamic and not force and path == _loaded_path and time.monotonic() - _last_stat_at < 0.25:
        return
    with _lock:
        if not dynamic and path == _loaded_path:
            return
        if not path:
            _placements, _loaded_path, _loaded_stamp = {}, path, None
            return
        try:
            st = Path(path).stat()
            stamp = (st.st_ino, st.st_mtime_ns)
        except FileNotFoundError:
            if _loaded_path is not None:
                return  # Avoid discarding the last good mapping on a transient error.
            raise
        _last_stat_at = time.monotonic()
        if path == _loaded_path and stamp == _loaded_stamp:
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
        _placements, _loaded_path, _loaded_stamp = parsed, path, stamp
        logger.info("[GNN_PLACEMENT_METADATA] path=%s entries=%d stamp=%s", path, len(parsed), stamp)


def get_chunk_placement(
    chunk_hash: bytes, *, missing_default: str = "disk"
) -> str:
    """Return semantic placement intent for one rolling LMCache chunk hash."""
    _load_if_needed()
    key = bytes(chunk_hash).hex().lower()
    placement = _placements.get(key)
    if placement is None and os.getenv("LMCACHE_GNN_PLACEMENT_DYNAMIC", "0") == "1":
        # A new wave may be published inside the 250-ms stat throttle.
        # Unknown hashes must not fall back to disk based on a stale snapshot.
        # Re-stat immediately (serialized under _lock) and retry the lookup.
        _load_if_needed(force=True)
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

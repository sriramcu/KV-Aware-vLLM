# SPDX-License-Identifier: Apache-2.0
"""Minimal hash -> persistent-tier metadata for learned placement experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

from lmcache.logging import init_logger

logger = init_logger(__name__)
_VALID_TIERS = {"L0", "L1", "L2"}
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
            tier = str(value).upper()
            if tier not in _VALID_TIERS:
                raise ValueError(f"invalid placement tier for {key}: {value!r}")
            parsed[str(key).lower()] = tier
        _placements = parsed
        _loaded_path = path
        logger.info("[GNN_PLACEMENT_METADATA] path=%s entries=%d", path, len(parsed))


def get_chunk_placement(chunk_hash: bytes, *, missing_default: str = "L2") -> str:
    """Return the persistent target for one rolling LMCache chunk hash.

    Unknown keys safely fall back to L2.  The fallback is observable and is
    intended only for sanity/unknown requests; benchmark workloads should have
    zero metadata misses.
    """
    _load_if_needed()
    key = bytes(chunk_hash).hex().lower()
    tier = _placements.get(key)
    if tier is not None:
        return tier
    default = missing_default.upper()
    if default not in _VALID_TIERS:
        raise ValueError(f"invalid missing_default {missing_default!r}")
    if key not in _missed:
        _missed.add(key)
        logger.warning("[GNN_PLACEMENT_METADATA_MISS] chunk_hash=%s fallback=%s", key, default)
    return default

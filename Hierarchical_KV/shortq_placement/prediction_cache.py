"""Prediction-cache seam.

The cache is deliberately dormant for the first request-conditioned experiment:
every occurrence is scored independently.  Enable a real cache later by changing
only ``_selected_lookup`` / ``_selected_store`` in this file.
"""

from __future__ import annotations


def dormant_lookup(chunk_hash_hex: str) -> str | None:
    del chunk_hash_hex
    return None


def dormant_store(chunk_hash_hex: str, tier: str) -> None:
    del chunk_hash_hex, tier


_selected_lookup = dormant_lookup
_selected_store = dormant_store


def lookup_prediction(chunk_hash_hex: str) -> str | None:
    return _selected_lookup(chunk_hash_hex)


def store_prediction(chunk_hash_hex: str, tier: str) -> None:
    _selected_store(chunk_hash_hex, tier)

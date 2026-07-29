from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any


_MODE_LEVELS = {
    "off": 0,
    "summary": 1,
    "full": 2,
}

_FALSE_VALUES = {"", "0", "false", "no", "off"}
_SUMMARY_VALUES = {"1", "summary"}
_FULL_VALUES = {"2", "true", "yes", "on", "full", "debug", "trace"}


@lru_cache(maxsize=1)
def _configuration() -> tuple[int, frozenset[str]]:
    raw_mode = os.getenv("KVWARE_TRACE", "off").strip().lower()

    if raw_mode in _FALSE_VALUES:
        mode = _MODE_LEVELS["off"]
    elif raw_mode in _SUMMARY_VALUES:
        mode = _MODE_LEVELS["summary"]
    elif raw_mode in _FULL_VALUES:
        mode = _MODE_LEVELS["full"]
    else:
        raise ValueError(
            "KVWARE_TRACE must be one of "
            "'off', 'summary', or 'full'; "
            f"got {raw_mode!r}"
        )

    raw_categories = os.getenv(
        "KVWARE_TRACE_CATEGORIES",
        "",
    )

    categories = frozenset(
        value.strip().lower()
        for value in raw_categories.split(",")
        if value.strip()
    )

    return mode, categories


def kvtrace_enabled(
    category: str,
    *,
    detail: int = 2,
) -> bool:
    """Return whether a custom trace should be emitted.

    detail=1: summary
    detail=2: full/per-operation tracing
    """
    mode, categories = _configuration()

    if mode < detail:
        return False

    normalized_category = category.strip().lower()

    return (
        not categories
        or "all" in categories
        or normalized_category in categories
    )


def kvtrace(
    logger: logging.Logger,
    category: str,
    message: str,
    *args: Any,
    detail: int = 2,
    level: int = logging.INFO,
) -> None:
    """Log a custom KV-Aware trace when enabled."""
    if not kvtrace_enabled(category, detail=detail):
        return

    logger.log(level, message, *args)
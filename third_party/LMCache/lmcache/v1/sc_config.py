# SPDX-License-Identifier: Apache-2.0
"""SC experiment controls for the vendored LMCache build.

All controls are read from environment variables so Slurm jobs can compose
independent gate and observability settings without editing Python source.
Unset controls preserve vanilla LMCache behavior: custom gates and custom
tracing are disabled.
"""

from __future__ import annotations

import os
from typing import Final


TRUE_VALUES: Final = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES: Final = frozenset({"0", "false", "no", "off", ""})


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of 0/1, false/true, no/yes, off/on; got {raw!r}"
    )


def env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    value = default if raw is None or raw.strip() == "" else int(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    value = default if raw is None or raw.strip() == "" else float(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def scheduler_lookup_admission_enabled() -> bool:
    return env_flag("SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE")


def worker_lookup_admission_enabled() -> bool:
    return env_flag("SC_LMCACHE_WORKER_LOOKUP_ADMISSION_ENABLE")


def disk_put_admission_enabled() -> bool:
    return env_flag("SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE")


def io_trace_enabled() -> bool:
    return env_flag("SC_LMCACHE_IO_TRACE_ENABLE")


def load_trace_enabled() -> bool:
    return env_flag("SC_LMCACHE_LOAD_TRACE_ENABLE")


def memory_trace_enabled() -> bool:
    return env_flag("SC_LMCACHE_MEMORY_TRACE_ENABLE")


def memory_snapshot_enabled() -> bool:
    return env_flag("SC_LMCACHE_MEMORY_SNAPSHOT_ENABLE")


def lookup_trace_enabled() -> bool:
    return env_flag("SC_LMCACHE_LOOKUP_TRACE_ENABLE")


def request_trace_enabled() -> bool:
    return env_flag("SC_LMCACHE_REQUEST_TRACE_ENABLE")


def tier_trace_enabled() -> bool:
    return env_flag("SC_LMCACHE_TIER_TRACE_ENABLE")


def gpu_assert_snapshot_enabled() -> bool:
    return env_flag("SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE")

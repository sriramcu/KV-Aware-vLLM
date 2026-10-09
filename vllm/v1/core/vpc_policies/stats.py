# SPDX-License-Identifier: Apache-2.0
# [SC] Cheap in-memory VPC accounting; optional off-path periodic JSONL writer.
from __future__ import annotations
import logging
import sys
import atexit
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from vllm.logger import init_logger

logger = init_logger(__name__)

def _safe_shutdown_log(level: int, message: str) -> None:
    """Log normally unless a handler's output stream has been closed.

    pytest and Python shutdown can close logging handlers before VPCStats
    atexit callbacks execute. Fall back to the original stderr stream.
    """
    current = logger

    while current is not None:
        for handler in current.handlers:
            stream = getattr(handler, "stream", None)

            if (
                getattr(handler, "_closed", False)
                or getattr(stream, "closed", False)
            ):
                stderr = sys.__stderr__

                if stderr is not None and not stderr.closed:
                    stderr.write(message + "\n")
                    stderr.flush()

                return

        if not current.propagate:
            break

        current = current.parent

    logger.log(level, "%s", message)

class VPCStats:
    """Counters are scheduler-owned; the reporter reads best-effort snapshots.

    The periodic writer never runs in the eviction call stack. Every JSONL
    path gains a PID suffix to avoid clobbering other engine processes.
    """

    def __init__(self, mode: str, window: int | None = None):
        self.mode = mode
        self.window = window
        self.start_ns = time.monotonic_ns()
        self.cached_reclaims: Counter[str] = Counter()
        self.admission_rejects: Counter[str] = Counter()
        self.uncached_allocations = 0
        self.explicit_hash_invalidations = 0
        self.free_blocks_provider: Callable[[], int] | None = None
        self.select_calls = 0
        self.scanned = 0
        self.bypassed_lru = 0
        self.select_ns = 0
        # Coarse logarithmic histogram (nanosecond-scale buckets) for
        # constant-space latency distribution, including approximated p95.
        self.latency_hist: Counter[int] = Counter()
        self._closed = False
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        target = os.environ.get("VLLM_VPC_STATS_JSONL", "").strip()
        self.periodic_path: Path | None = None
        if target:
            requested = Path(target).expanduser()
            self.periodic_path = requested.with_name(
                f"{requested.stem}.pid{os.getpid()}{requested.suffix or '.jsonl'}"
            )
            interval = float(os.environ.get("VLLM_VPC_STATS_INTERVAL_S", "5"))
            if interval <= 0:
                raise ValueError("VLLM_VPC_STATS_INTERVAL_S must be > 0")
            self._thread = threading.Thread(
                target=self._periodic_writer, args=(interval,),
                name="vpc-periodic-stats", daemon=True,
            )
            self._thread.start()
        atexit.register(self.emit_final)

    def record_selection(self, elapsed_ns: int, candidates: int, bypass: bool) -> None:
        self.select_calls += 1
        self.select_ns += elapsed_ns
        self.scanned += candidates
        self.bypassed_lru += int(bypass)
        self.latency_hist[max(0, elapsed_ns).bit_length()] += 1

    def record_reclaim(self, label: str | None, cached: bool) -> None:
        if cached:
            self.cached_reclaims[label or "unlabeled"] += 1
        else:
            self.uncached_allocations += 1

    def record_reject(self, label: str | None) -> None:
        self.admission_rejects[label or "unlabeled"] += 1

    def snapshot(self, free_blocks: int | None = None) -> dict[str, Any]:
        n = sum(self.cached_reclaims.values())
        histogram = sorted(self.latency_hist.items())
        cutoff = max(1, int(self.select_calls * 0.95 + 0.9999))
        passed = 0
        p95_upper_us: float | None = None
        for bit_length, count in histogram:
            passed += count
            if passed >= cutoff:
                p95_upper_us = (2 ** bit_length) / 1000
                break
        return {
            "policy": self.mode,
            "window": self.window,
            "pid": os.getpid(),
            "elapsed_s": round((time.monotonic_ns() - self.start_ns) / 1e9, 3),
            "free_blocks": free_blocks if free_blocks is not None
                else (self.free_blocks_provider() if self.free_blocks_provider else None),
            "explicit_hash_invalidations": self.explicit_hash_invalidations,
            "cached_reclaims": n,
            "victim_counts": {tier: self.cached_reclaims[tier]
                              for tier in ("unlabeled", "disk", "cpu", "gpu")},
            "victim_pct": {tier: round(100 * self.cached_reclaims[tier] / n, 3)
                           if n else 0.0
                           for tier in ("unlabeled", "disk", "cpu", "gpu")},
            "uncached_allocations": self.uncached_allocations,
            "select_calls": self.select_calls,
            "selection_time_ms": round(self.select_ns / 1e6, 3),
            "selection_p95_upper_us": p95_upper_us,
            "mean_candidates": round(self.scanned / self.select_calls, 3)
                               if self.select_calls else 0.0,
            "different_from_lru_pct": round(
                100 * self.bypassed_lru / self.select_calls, 3
            ) if self.select_calls else 0.0,
            "admission_reject_counts": dict(self.admission_rejects),
        }

    def _periodic_writer(self, interval: float) -> None:
        assert self.periodic_path is not None
        while not self._shutdown.wait(interval):
            try:
                self.periodic_path.parent.mkdir(parents=True, exist_ok=True)
                with self.periodic_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(self.snapshot(), sort_keys=True) + "\n")
            except OSError as exc:
                logger.warning("[VPC_STATS_IO] periodic=%s error=%s",
                               self.periodic_path, exc)

    def emit_final(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._shutdown.set()

        report = self.snapshot()
        report_json = json.dumps(report, sort_keys=True)

        # Use normal vLLM logging when available.
        # Fall back to stderr if logging has already shut down.
        _safe_shutdown_log(
            logging.INFO,
            f"[VPC_SUMMARY] {report_json}",
        )

        # Preserve the final JSONL snapshot independently of logging.
        if self.periodic_path is not None:
            try:
                self.periodic_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with self.periodic_path.open(
                    "a",
                    encoding="utf-8",
                ) as stream:
                    stream.write(
                        json.dumps(
                            {**report, "final": True},
                            sort_keys=True,
                        ) + "\n"
                    )

            except OSError as exc:
                _safe_shutdown_log(
                    logging.WARNING,
                    f"[VPC_STATS_IO] final={self.periodic_path} "
                    f"error={exc}",
                )

"""Opt-in, JSONL, CPU-side scheduler/engine phase observability.

One buffered file per engine process; timestamps are monotonic nanoseconds.
Worker-side device event measurements, when enabled, use a separate file.
Do NOT interpret engine model-executor durations as GPU kernel durations.
"""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

_ENABLED = os.getenv("VLLM_PHASE_TRACE", "0") == "1"
_lock = threading.Lock()
_file = None
_counter = 0


def enabled() -> bool:
    return _ENABLED


def now_ns() -> int:
    return time.monotonic_ns()


def emit(kind: str, **values: Any) -> None:
    global _file, _counter
    if not _ENABLED:
        return
    try:
        with _lock:
            if _file is None:
                target = Path(os.environ.get("VLLM_PHASE_TRACE_DIR", "/tmp/vllm-phase-trace"))
                target.mkdir(parents=True, exist_ok=True)
                _file = (target / f"phase_engine.pid{os.getpid()}.jsonl").open(
                    "a", encoding="utf-8", buffering=1
                )
            _file.write(json.dumps({"kind": kind, "ts_ns": now_ns(), **values},
                                   separators=(",", ":")) + "\n")
            _counter += 1
    except OSError:
        # Observability must never affect scheduling correctness.
        pass


def close() -> None:
    if _file is not None:
        with _lock:
            _file.flush()


atexit.register(close)


def capture_schedule(scheduler: Any, output: Any) -> None:
    """Snapshot request positions BEFORE _update_after_schedule advances them.

    Attached attributes are used only in the engine process; they are not
    dataclass fields and must not become part of worker-serialized payloads.
    """
    if not _ENABLED:
        return
    rows = []
    prompt_sum = decode_sum = 0
    for rid, scheduled in output.num_scheduled_tokens.items():
        request = scheduler.requests[rid]
        prompt_total = int(request.num_prompt_tokens)
        position = int(request.num_computed_tokens)
        scheduled = int(scheduled)
        prompt_n = min(scheduled, max(0, prompt_total - position))
        decode_n = scheduled - prompt_n
        prompt_sum += prompt_n
        decode_sum += decode_n
        rows.append({
            "request_id": rid, "before": position,
            "prompt_len": prompt_total, "scheduled": scheduled,
            "prompt_compute": prompt_n, "decode_position": decode_n,
            "preemptions": int(request.num_preemptions),
        })
    output._sc_phase_rows = rows
    output._sc_phase_totals = (prompt_sum, decode_sum)
    # NOTE: Waiting state sampled at each schedule call, not timed wait events.
    waiting_remote = sum(
        1 for r in scheduler.requests.values()
        if getattr(r.status, "name", "") == "WAITING_FOR_REMOTE_KVS"
    )
    output._sc_phase_waiting_remote = waiting_remote
    output._sc_phase_running_count = len(scheduler.running)


def record_engine_step(output: Any, schedule_start_ns: int,
                       schedule_end_ns: int, submit_end_ns: int,
                       result_end_ns: int, update_end_ns: int) -> None:
    if not _ENABLED:
        return
    prompt_n, decode_n = getattr(output, "_sc_phase_totals", (0, 0))
    emit(
        "engine_step", schedule_start_ns=schedule_start_ns,
        schedule_end_ns=schedule_end_ns, submit_end_ns=submit_end_ns,
        result_end_ns=result_end_ns, update_end_ns=update_end_ns,
        prompt_compute_tokens=prompt_n, decode_position_tokens=decode_n,
        scheduled_total=output.total_num_scheduled_tokens,
        running_requests=getattr(output, "_sc_phase_running_count", None),
        waiting_remote_requests=getattr(output, "_sc_phase_waiting_remote", None),
        requests=getattr(output, "_sc_phase_rows", []),
    )

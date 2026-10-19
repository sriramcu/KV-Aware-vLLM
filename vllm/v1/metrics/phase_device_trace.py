"""Optional deferred PyTorch CUDA-event timing for model-forward regions.

Off by default. Timing is per worker/TP rank and per *mixed batch*, never
exclusive per-request kernel time. No synchronization is introduced; unready
records are buffered and may be lost if the process exits without draining.
Use on a small GPU validation run before paper-level timing claims.
"""
from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path

_ENABLED = os.getenv("VLLM_PHASE_TRACE_DEVICE_TIMING", "0") == "1"
_pending = deque()
_file = None


def begin():
    if not _ENABLED:
        return None
    import torch
    if not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
        return None
    # Avoid retaining a backlog of CUDA events when instrumentation is enabled.
    drain()
    if len(_pending) >= 4096:
        return None
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    start.record()
    return start, finish


def end(state, scheduler_output):
    if state is None:
        return
    start, finish = state
    finish.record()
    _pending.append((start, finish, scheduler_output.total_num_scheduled_tokens,
                     len(scheduler_output.num_scheduled_tokens)))
    if len(_pending) % 32 == 0:
        drain()


def drain():
    global _file
    if not _ENABLED:
        return
    while _pending and _pending[0][1].query():
        start, finish, tokens, requests = _pending.popleft()
        try:
            if _file is None:
                directory = Path(os.getenv("VLLM_PHASE_TRACE_DIR", "/tmp/vllm-phase-trace"))
                directory.mkdir(parents=True, exist_ok=True)
                _file = (directory / f"phase_device.pid{os.getpid()}.jsonl").open(
                    "a", encoding="utf-8", buffering=256 * 1024)
            _file.write(json.dumps({"kind": "device_forward", "duration_ms":
                                    start.elapsed_time(finish),
                                    "scheduled_tokens": tokens,
                                    "num_requests": requests}) + "\n")
        except OSError:
            pass

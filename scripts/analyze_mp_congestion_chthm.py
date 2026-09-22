#!/usr/bin/env python3
"""Post-hoc parser for KV-Aware LMCache MP research diagnostics.

Consumes the opt-in log markers emitted by the Stage-1 freshness/congestion
patch and, when benchmark result JSONL files are available, reports prompt-token
CHTHM split by phase:

  local_gpu_cache / external_l1 / external_l2 / local_compute

For L0-off PREFIX experiments, L1 is a prefix and L2 extends it, so the useful
external source split is the overlap of the scheduler-authorized external-load
interval with the server-reported L1/L2 boundary.  This intentionally uses
*consumed/admitted* external tokens rather than raw lookup hits.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Iterable

KV_LOOKUP_RE = re.compile(
    r"\[MP_CHTHM_LOOKUP\] request=(\S+) requested_tokens=(\d+) "
    r"total_hit_tokens=(\d+) l1_hit_tokens=(\d+) l2_hit_tokens=(\d+)"
)
KV_ADMIT_RE = re.compile(
    r"\[MP_CHTHM_ADMIT\] request=(\S+) decision=(\S+) prompt_tokens=(\d+) "
    r"vllm_hit_tokens=(\d+) lmcache_hit_tokens=(\d+) "
    r"external_load_tokens=(\d+) lookup_age_s=([0-9.]+)"
)
FRESH_RE = re.compile(
    r"\[MP_STAGE1_FRESHNESS_GUARD\] request=(\S+) lookup_age_s=([0-9.]+) "
    r"guard_s=([0-9.]+) hit_tokens=(\d+)"
)
PREFETCH_ADMIT_RE = re.compile(
    r"\[MP_PREFETCH_ADMIT\].*queue_wait_s=([0-9.]+).*pending_after=(\d+) "
    r"inflight_before=(\d+) max_in_flight=(\d+)"
)
PREFETCH_PHASE_RE = re.compile(
    r"\[MP_PREFETCH_PHASE\].*lookup_s=([0-9.]+).*pending=(\d+) inflight=(\d+)"
)
PREFETCH_DONE_RE = re.compile(
    r"\[MP_PREFETCH_DONE\].*total_s=([0-9.]+).*lookup_s=([0-9.]+) "
    r"load_s=([0-9.]+).*pending=(\d+) inflight=(\d+)"
)
STORE_DONE_RE = re.compile(
    r"\[MP_STORE_DONE\].*service_s=([0-9.]+) inflight_after=(\d+) listener_pending=(\d+)"
)
NATIVE_DONE_RE = re.compile(
    r"\[MP_NATIVE_IO_DONE\].*op=(\S+).*service_s=([0-9.]+) "
    r"pending_store=(\d+) pending_lookup=(\d+) pending_load=(\d+) pending_total=(\d+)"
)
LIFETIME_RE = re.compile(
    r"\[L1_READ_LIFETIME\] unsafe_read (missing|unlocked).*reserved_age_s=([-0-9.]+)"
)
RECOVERY_RE = re.compile(r"Recovered from KV load failure:\s*(\d+) request")
STARVE_RE = re.compile(r"\[KV_STAGE1_STARVATION_FALLBACK\].*abandoning (\d+)")


def pctile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    x = (len(vals) - 1) * p
    lo = math.floor(x)
    hi = math.ceil(x)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - x) + vals[hi] * (x - lo)


def dist(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "p50": pctile(values, 0.50),
        "p95": pctile(values, 0.95),
        "p99": pctile(values, 0.99),
        "max": max(values) if values else None,
    }


def iter_lines(path: Path | None) -> Iterable[str]:
    if path is None or not path.exists():
        return []
    return path.open("r", errors="replace")


def load_phase_map(results_dir: Path | None) -> dict[str, str]:
    phases: dict[str, str] = {}
    if results_dir is None or not results_dir.exists():
        return phases
    for path in results_dir.glob("*results*.jsonl"):
        try:
            with path.open("r", errors="replace") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rid = row.get("request_id")
                    phase = row.get("phase")
                    if rid and phase:
                        phases[str(rid)] = str(phase)
        except OSError:
            continue
    return phases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmcache-log", type=Path, required=True)
    ap.add_argument("--vllm-log", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    lookup: dict[str, dict[str, int]] = {}
    prefetch_queue_wait: list[float] = []
    prefetch_lookup_s: list[float] = []
    prefetch_total_s: list[float] = []
    prefetch_load_s: list[float] = []
    store_service_s: list[float] = []
    native_service: dict[str, list[float]] = {}
    native_pending_total: list[float] = []
    native_pending_store: list[float] = []
    native_pending_lookup: list[float] = []
    native_pending_load: list[float] = []
    lifetime_ages: list[float] = []

    for line in iter_lines(args.lmcache_log):
        if m := KV_LOOKUP_RE.search(line):
            rid, requested, total, l1, l2 = m.groups()
            lookup[rid] = {
                "requested_tokens": int(requested),
                "total_hit_tokens": int(total),
                "l1_hit_tokens": int(l1),
                "l2_hit_tokens": int(l2),
            }
        if m := PREFETCH_ADMIT_RE.search(line):
            prefetch_queue_wait.append(float(m.group(1)))
        if m := PREFETCH_PHASE_RE.search(line):
            prefetch_lookup_s.append(float(m.group(1)))
        if m := PREFETCH_DONE_RE.search(line):
            prefetch_total_s.append(float(m.group(1)))
            prefetch_load_s.append(float(m.group(3)))
        if m := STORE_DONE_RE.search(line):
            store_service_s.append(float(m.group(1)))
        if m := NATIVE_DONE_RE.search(line):
            op = m.group(1)
            native_service.setdefault(op, []).append(float(m.group(2)))
            native_pending_store.append(float(m.group(3)))
            native_pending_lookup.append(float(m.group(4)))
            native_pending_load.append(float(m.group(5)))
            native_pending_total.append(float(m.group(6)))
        if m := LIFETIME_RE.search(line):
            age = float(m.group(2))
            if age >= 0:
                lifetime_ages.append(age)

    admits: dict[str, dict[str, int | float | str]] = {}
    freshness: list[float] = []
    recoveries = 0
    recovered_request_events = 0
    starvation_abandoned = 0
    for line in iter_lines(args.vllm_log):
        if m := KV_ADMIT_RE.search(line):
            rid, decision, prompt, vpc, lmhit, ext, age = m.groups()
            # Keep the first initial admission. Recovery bypasses use a separate
            # decision name and are not emitted by this patch as CHTHM inputs.
            admits.setdefault(
                rid,
                {
                    "decision": decision,
                    "prompt_tokens": int(prompt),
                    "vllm_hit_tokens": int(vpc),
                    "lmcache_hit_tokens": int(lmhit),
                    "external_load_tokens": int(ext),
                    "lookup_age_s": float(age),
                },
            )
        if m := FRESH_RE.search(line):
            freshness.append(float(m.group(2)))
        if m := RECOVERY_RE.search(line):
            recoveries += 1
            recovered_request_events += int(m.group(1))
        if m := STARVE_RE.search(line):
            starvation_abandoned += int(m.group(1))

    phase_map = load_phase_map(args.results_dir)
    chthm: dict[str, dict[str, int]] = {}
    missing_source = 0
    for rid, a in admits.items():
        phase = phase_map.get(rid, "unknown")
        bucket = chthm.setdefault(
            phase,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "local_gpu_cache_tokens": 0,
                "external_l1_tokens": 0,
                "external_l2_tokens": 0,
                "external_unknown_tokens": 0,
                "local_compute_tokens": 0,
                "freshness_misses": 0,
            },
        )
        prompt = int(a["prompt_tokens"])
        vpc = min(prompt, int(a["vllm_hit_tokens"]))
        ext = min(max(0, prompt - vpc), int(a["external_load_tokens"]))
        l1_use = 0
        l2_use = 0
        unknown = 0
        src = lookup.get(rid)
        if ext > 0 and src is not None:
            # External load covers [vpc, vpc + ext).  For this L0-off PREFIX
            # experiment, server-reported L1 occupies [0, l1_hit_tokens) and
            # L2 is the extension beyond that boundary.
            l1_boundary = int(src["l1_hit_tokens"])
            l1_use = max(0, min(vpc + ext, l1_boundary) - vpc)
            l2_use = ext - l1_use
        elif ext > 0:
            unknown = ext
            missing_source += 1

        compute = max(0, prompt - vpc - ext)
        bucket["requests"] += 1
        bucket["prompt_tokens"] += prompt
        bucket["local_gpu_cache_tokens"] += vpc
        bucket["external_l1_tokens"] += l1_use
        bucket["external_l2_tokens"] += l2_use
        bucket["external_unknown_tokens"] += unknown
        bucket["local_compute_tokens"] += compute
        if a["decision"] == "freshness_miss":
            bucket["freshness_misses"] += 1

    chthm_with_pct: dict[str, dict[str, int | float]] = {}
    for phase, b in chthm.items():
        total = b["prompt_tokens"]
        out: dict[str, int | float] = dict(b)
        for name in (
            "local_gpu_cache_tokens",
            "external_l1_tokens",
            "external_l2_tokens",
            "external_unknown_tokens",
            "local_compute_tokens",
        ):
            out[name.replace("_tokens", "_pct")] = (
                100.0 * b[name] / total if total else 0.0
            )
        chthm_with_pct[phase] = out

    result = {
        "chthm": chthm_with_pct,
        "chthm_requests_missing_server_source_log": missing_source,
        "freshness_guard": {
            "trigger_count": len(freshness),
            "lookup_age_s": dist(freshness),
        },
        "failure_signals": {
            "kv_recovery_waves": recoveries,
            "cumulative_recovered_request_events": recovered_request_events,
            "stage1_starvation_abandoned_requests": starvation_abandoned,
            "unsafe_read_failure_age_s": dist(lifetime_ages),
        },
        "congestion": {
            "prefetch_queue_wait_s": dist(prefetch_queue_wait),
            "prefetch_lookup_s": dist(prefetch_lookup_s),
            "prefetch_load_s": dist(prefetch_load_s),
            "prefetch_total_s": dist(prefetch_total_s),
            "store_service_s": dist(store_service_s),
            "native_service_s_by_op": {
                op: dist(vals) for op, vals in sorted(native_service.items())
            },
            "native_pending_total": dist(native_pending_total),
            "native_pending_store": dist(native_pending_store),
            "native_pending_lookup": dist(native_pending_lookup),
            "native_pending_load": dist(native_pending_load),
        },
    }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

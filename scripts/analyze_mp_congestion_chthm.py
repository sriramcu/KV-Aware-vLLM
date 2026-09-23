#!/usr/bin/env python3
"""Analyze MP congestion, raw hierarchy CHTHM, and useful KV contribution.

Raw CHTHM and scheduler-useful contribution are deliberately separate:
  * raw CHTHM: where KV was eventually found in the hierarchy;
  * useful contribution: what the scheduler actually admitted/loaded.

If Stage-1 is abandoned before the server result is observed, raw hierarchy
attribution is censored. The report exposes observation coverage and lower-bound
rates instead of silently treating unresolved work as a disk miss.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Iterable

RAW_RE = re.compile(
    r"\[MP_CHTHM_RAW\] request=(\S+) chunk_size=(\d+) requested_chunks=(\d+) "
    r"total_hit_chunks=(\d+) l0_hit_chunks=(\d+) l1_hit_chunks=(\d+) "
    r"l2_hit_chunks=(\d+) source_tiers=(\S+)"
)
OLD_LOOKUP_RE = re.compile(
    r"\[MP_CHTHM_LOOKUP\] request=(\S+) requested_tokens=(\d+) "
    r"total_hit_tokens=(\d+) l1_hit_tokens=(\d+) l2_hit_tokens=(\d+)"
    r"(?: l0_union=(\S+))?"
)
ADMIT_RE = re.compile(
    r"\[MP_CHTHM_ADMIT\] request=(\S+) decision=(\S+) prompt_tokens=(\d+) "
    r"vllm_hit_tokens=(\d+) lmcache_hit_tokens=(\d+) external_load_tokens=(\d+) "
    r"lookup_age_s=([0-9.]+)"
)
FRESH_RE = re.compile(r"\[MP_STAGE1_FRESHNESS_GUARD\] request=(\S+) lookup_age_s=([0-9.]+)")
STARVE_REQ_RE = re.compile(r"\[MP_STAGE1_STARVATION_FALLBACK\] abandoning pending request (\S+)")
STARVE_WAVE_RE = re.compile(r"\[KV_STAGE1_STARVATION_FALLBACK\].*abandoning (\d+)")
RECOVERY_RE = re.compile(r"Recovered from KV load failure:\s*(\d+) request")
PREFETCH_ADMIT_RE = re.compile(r"\[MP_PREFETCH_ADMIT\].*queue_wait_s=([0-9.]+).*")
PREFETCH_PHASE_RE = re.compile(r"\[MP_PREFETCH_PHASE\].*lookup_s=([0-9.]+).*")
PREFETCH_DONE_RE = re.compile(
    r"\[MP_PREFETCH_DONE\].*total_s=([0-9.]+).*lookup_s=([0-9.]+) load_s=([0-9.]+).*"
)
STORE_DONE_RE = re.compile(r"\[MP_STORE_DONE\].*service_s=([0-9.]+).*")
NATIVE_DONE_RE = re.compile(
    r"\[MP_NATIVE_IO_DONE\].*op=(\S+).*service_s=([0-9.]+) "
    r"pending_store=(\d+) pending_lookup=(\d+) pending_load=(\d+) pending_total=(\d+)"
)
LIFETIME_RE = re.compile(r"\[L1_READ_LIFETIME\] unsafe_read (?:missing|unlocked).*reserved_age_s=([-0-9.]+)")


def lines(path: Path | None) -> Iterable[str]:
    if path is None or not path.exists():
        return []
    return path.open("r", errors="replace")


def pctile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
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




def base_request_id(request_id: str) -> str:
    parts = request_id.split("-")
    if len(parts) >= 4 and parts[0] == "cmpl":
        return "-".join(parts[:2])
    return request_id

def phase_map(results: Path | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not results or not results.exists():
        return out
    for p in results.glob("*results*.jsonl"):
        for line in p.open(errors="replace"):
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("request_id") and row.get("phase"):
                out[str(row["request_id"])] = str(row["phase"])
    return out


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def source_token_counts(raw: dict, start: int, end: int) -> dict[str, int]:
    out = {"L0": 0, "L1": 0, "L2": 0}
    cs = raw["chunk_size"]
    for i, tier in enumerate(raw["source_tiers"]):
        out[tier] += overlap(start, end, i * cs, (i + 1) * cs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmcache-log", type=Path, required=True)
    ap.add_argument("--vllm-log", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    raw: dict[str, dict] = {}
    old: dict[str, dict] = {}
    qwait: list[float] = []
    lookup_s: list[float] = []
    total_s: list[float] = []
    load_s: list[float] = []
    store_s: list[float] = []
    native: dict[str, list[float]] = {}
    pending = {k: [] for k in ("store", "lookup", "load", "total")}
    lifetimes: list[float] = []

    for line in lines(args.lmcache_log):
        if m := RAW_RE.search(line):
            rid, cs, req, hit, l0, l1, l2, tiers = m.groups()
            tier_list = [] if tiers == "-" else tiers.split(",")
            raw[rid] = {
                "chunk_size": int(cs),
                "requested_chunks": int(req),
                "total_hit_chunks": int(hit),
                "l0_hit_chunks": int(l0),
                "l1_hit_chunks": int(l1),
                "l2_hit_chunks": int(l2),
                "source_tiers": tier_list,
            }
        if m := OLD_LOOKUP_RE.search(line):
            old[m.group(1)] = {
                "requested": int(m.group(2)),
                "total": int(m.group(3)),
                "l1": int(m.group(4)),
                "l2": int(m.group(5)),
                "l0_union": str(m.group(6) or "False").lower() == "true",
            }
        if m := PREFETCH_ADMIT_RE.search(line):
            qwait.append(float(m.group(1)))
        if m := PREFETCH_PHASE_RE.search(line):
            lookup_s.append(float(m.group(1)))
        if m := PREFETCH_DONE_RE.search(line):
            total_s.append(float(m.group(1)))
            load_s.append(float(m.group(3)))
        if m := STORE_DONE_RE.search(line):
            store_s.append(float(m.group(1)))
        if m := NATIVE_DONE_RE.search(line):
            native.setdefault(m.group(1), []).append(float(m.group(2)))
            for k, g in zip(("store", "lookup", "load", "total"), (3, 4, 5, 6)):
                pending[k].append(float(m.group(g)))
        if m := LIFETIME_RE.search(line):
            age = float(m.group(1))
            if age >= 0:
                lifetimes.append(age)

    admits: dict[str, dict] = {}
    freshness: list[tuple[str, float]] = []
    starvation_requests: list[str] = []
    starvation_wave_sum = 0
    recovery_waves = 0
    recovered_request_events = 0
    for line in lines(args.vllm_log):
        if m := ADMIT_RE.search(line):
            rid, decision, prompt, vpc, lmhit, ext, age = m.groups()
            admits.setdefault(
                rid,
                {
                    "decision": decision,
                    "prompt": int(prompt),
                    "vpc": int(vpc),
                    "lmhit": int(lmhit),
                    "ext": int(ext),
                    "age": float(age),
                },
            )
        if m := FRESH_RE.search(line):
            freshness.append((m.group(1), float(m.group(2))))
        if m := STARVE_REQ_RE.search(line):
            starvation_requests.append(m.group(1))
        if m := STARVE_WAVE_RE.search(line):
            starvation_wave_sum += int(m.group(1))
        if m := RECOVERY_RE.search(line):
            recovery_waves += 1
            recovered_request_events += int(m.group(1))

    # Backward-compatible raw-source reconstruction for older L0-off logs.
    # Their MP_CHTHM_LOOKUP records describe a monotonic L1 prefix followed by
    # an L2 extension. Do not attempt this for legacy L0-union records because
    # those logs intentionally collapsed mixed L0/L1/L2 attribution.
    for rid, rec in old.items():
        if rid in raw or rec.get("l0_union"):
            continue
        chunk_size = 512
        l1_chunks = rec["l1"] // chunk_size
        l2_chunks = rec["l2"] // chunk_size
        raw[rid] = {
            "chunk_size": chunk_size,
            "requested_chunks": rec["requested"] // chunk_size,
            "total_hit_chunks": rec["total"] // chunk_size,
            "l0_hit_chunks": 0,
            "l1_hit_chunks": l1_chunks,
            "l2_hit_chunks": l2_chunks,
            "source_tiers": ["L1"] * l1_chunks + ["L2"] * l2_chunks,
            "reconstructed_from_legacy_lookup": True,
        }

    phases = phase_map(args.results_dir)
    raw_b: dict[str, dict] = {}
    useful_b: dict[str, dict] = {}

    for rid, admit in admits.items():
        phase = phases.get(rid, phases.get(base_request_id(rid), "unknown"))
        prompt = admit["prompt"]
        vpc = min(prompt, admit["vpc"])
        rr = raw.get(rid)
        chunk_size = rr["chunk_size"] if rr is not None else 512
        chunk_addressable = prompt - (prompt % chunk_size)
        external_opportunity = max(0, chunk_addressable - min(vpc, chunk_addressable))

        rb = raw_b.setdefault(
            phase,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "vpc_hit_tokens": 0,
                "l0_hit_tokens": 0,
                "l1_hit_tokens": 0,
                "l2_hit_tokens": 0,
                "known_hierarchy_miss_tokens": 0,
                "raw_external_opportunity_tokens": 0,
                "observed_external_opportunity_tokens": 0,
                "unobserved_external_opportunity_tokens": 0,
            },
        )
        ub = useful_b.setdefault(
            phase,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "vpc_tokens": 0,
                "l0_tokens": 0,
                "l1_tokens": 0,
                "l2_tokens": 0,
                "external_unknown_tokens": 0,
                "recompute_tokens": 0,
                "freshness_misses": 0,
            },
        )

        rb["requests"] += 1
        rb["prompt_tokens"] += prompt
        rb["vpc_hit_tokens"] += vpc
        rb["raw_external_opportunity_tokens"] += external_opportunity
        ub["requests"] += 1
        ub["prompt_tokens"] += prompt
        ub["vpc_tokens"] += vpc

        if rr is not None:
            rb["observed_external_opportunity_tokens"] += external_opportunity
            src = source_token_counts(rr, vpc, chunk_addressable)
            rb["l0_hit_tokens"] += src["L0"]
            rb["l1_hit_tokens"] += src["L1"]
            rb["l2_hit_tokens"] += src["L2"]
            known_hit = src["L0"] + src["L1"] + src["L2"]
            rb["known_hierarchy_miss_tokens"] += max(0, external_opportunity - known_hit)
        else:
            rb["unobserved_external_opportunity_tokens"] += external_opportunity

        # Non-chunk-aligned tail tokens can never be served by LMCache and are
        # known hierarchy misses unless already covered by a vLLM/VPC hit.
        tail_start = chunk_addressable
        if vpc < prompt:
            rb["known_hierarchy_miss_tokens"] += max(0, prompt - max(vpc, tail_start))

        ext = min(max(0, prompt - vpc), admit["ext"])
        ub["recompute_tokens"] += max(0, prompt - vpc - ext)
        if ext and rr is not None:
            src = source_token_counts(rr, vpc, vpc + ext)
            ub["l0_tokens"] += src["L0"]
            ub["l1_tokens"] += src["L1"]
            ub["l2_tokens"] += src["L2"]
            attributed = src["L0"] + src["L1"] + src["L2"]
            ub["external_unknown_tokens"] += max(0, ext - attributed)
        else:
            ub["external_unknown_tokens"] += ext
        if admit["decision"] == "freshness_miss":
            ub["freshness_misses"] += 1

    for bucket in raw_b.values():
        total = bucket["prompt_tokens"]
        gpu = bucket["vpc_hit_tokens"] + bucket["l0_hit_tokens"]
        cpu = bucket["l1_hit_tokens"]
        disk = bucket["l2_hit_tokens"]
        opportunity = bucket["raw_external_opportunity_tokens"]
        observed = bucket["observed_external_opportunity_tokens"]
        bucket["gpu_hit_tokens"] = gpu
        bucket["raw_external_observation_coverage_pct"] = (
            100.0 * observed / opportunity if opportunity else 100.0
        )
        bucket["gpu_hit_pct_all_prompt_lower_bound"] = 100.0 * gpu / total if total else 0.0
        bucket["cpu_hit_conditional_gpu_miss_lower_bound_pct"] = (
            100.0 * cpu / (total - gpu) if total > gpu else 0.0
        )
        bucket["disk_hit_conditional_gpu_cpu_miss_lower_bound_pct"] = (
            100.0 * disk / (total - gpu - cpu) if total > gpu + cpu else 0.0
        )
        bucket["known_hierarchy_miss_pct_all_prompt_lower_bound"] = (
            100.0 * bucket["known_hierarchy_miss_tokens"] / total if total else 0.0
        )
        bucket["rates_exact"] = bucket["unobserved_external_opportunity_tokens"] == 0

    for bucket in useful_b.values():
        total = bucket["prompt_tokens"]
        for key in (
            "vpc_tokens",
            "l0_tokens",
            "l1_tokens",
            "l2_tokens",
            "external_unknown_tokens",
            "recompute_tokens",
        ):
            bucket[key.replace("_tokens", "_pct")] = 100.0 * bucket[key] / total if total else 0.0

    phase_counts = {phase: sum(1 for x in phases.values() if x == phase) for phase in set(phases.values())}
    starve_by: dict[str, set[str]] = {}
    fresh_by: dict[str, set[str]] = {}
    for rid in starvation_requests:
        starve_by.setdefault(phases.get(rid, phases.get(base_request_id(rid), "unknown")), set()).add(rid)
    for rid, _ in freshness:
        fresh_by.setdefault(phases.get(rid, phases.get(base_request_id(rid), "unknown")), set()).add(rid)

    result = {
        "raw_chthm": raw_b,
        "useful_contribution": useful_b,
        "chthm_definition": (
            "GPU hit is over all prompt tokens; CPU hit is conditional on GPU miss; "
            "disk hit is conditional on GPU+CPU miss; hierarchy miss/recompute is over all prompt tokens."
        ),
        "observation_note": (
            "Raw tier rates are lower bounds when raw_external_observation_coverage_pct < 100. "
            "Useful contribution is scheduler-consumed KV and remains exact from MP_CHTHM_ADMIT."
        ),
        "freshness_guard": {
            "trigger_count": len(freshness),
            "lookup_age_s": dist([x for _, x in freshness]),
            "unique_requests_by_phase": {p: len(v) for p, v in fresh_by.items()},
            "rate_pct_by_phase": {
                p: 100.0 * len(v) / phase_counts.get(p, len(v)) for p, v in fresh_by.items()
            },
        },
        "failure_signals": {
            "kv_recovery_waves": recovery_waves,
            "cumulative_recovered_request_events": recovered_request_events,
            "stage1_starvation_markers": len(starvation_requests),
            "stage1_starvation_wave_sum_fallback": starvation_wave_sum,
            "unique_starvation_requests_by_phase": {p: len(v) for p, v in starve_by.items()},
            "starvation_rate_pct_by_phase": {
                p: 100.0 * len(v) / phase_counts.get(p, len(v)) for p, v in starve_by.items()
            },
            "unsafe_read_failure_age_s": dist(lifetimes),
        },
        "congestion": {
            "prefetch_queue_wait_s": dist(qwait),
            "prefetch_lookup_s": dist(lookup_s),
            "prefetch_load_s": dist(load_s),
            "prefetch_total_s": dist(total_s),
            "store_service_s": dist(store_s),
            "native_service_s_by_op": {k: dist(v) for k, v in sorted(native.items())},
            "native_pending_store": dist(pending["store"]),
            "native_pending_lookup": dist(pending["lookup"]),
            "native_pending_load": dist(pending["load"]),
            "native_pending_total": dist(pending["total"]),
        },
        "legacy_lookup_records": len(old),
        "raw_source_records": len(raw),
        "scheduler_admit_records": len(admits),
    }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze MP congestion, raw L1/L2 opportunity, and useful KV contribution.

Raw hierarchy opportunity and scheduler-useful contribution are deliberately separate:
  * raw opportunity: where external KV was eventually found in L1/L2;
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

LOOKUP_RE = re.compile(
    r"\[MP_CHTHM_LOOKUP\] request=(\S+) requested_tokens=(\d+) "
    r"total_hit_tokens=(\d+) l1_hit_tokens=(\d+) l2_hit_tokens=(\d+)"
)
RAW_RE = re.compile(
    r"\[MP_CHTHM_RAW\] request=(\S+) chunk_size=(\d+) "
    r"requested_chunks=(\d+) reachable_prefix_chunks=(\d+) "
    r"l1_hit_chunks=(\d+) l2_hit_chunks=(\d+) miss_chunks=(\d+) "
    r"source_map=([A-Za-z]*)"
)
ADMIT_RE = re.compile(
    r"\[MP_CHTHM_ADMIT\] request=(\S+) decision=(\S+) prompt_tokens=(\d+) "
    r"vllm_hit_tokens=(\d+) lmcache_hit_tokens=(\d+) external_load_tokens=(\d+) "
    r"(?:lookup_start_tokens=\d+ )?lookup_age_s=([0-9.]+)"
)
MUTUAL_PREFIX_RE = re.compile(
    r"\[MP_MUTUAL_PREFIX_LOOKUP\] request=(\S+) prompt_tokens=(\d+) "
    r"vllm_hit_tokens=(\d+) lookup_start_tokens=(\d+)"
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
LIFETIME_RE = re.compile(
    r"\[L1_READ_LIFETIME\] unsafe_read (?:missing|unlocked).*reserved_age_s=([-0-9.]+)"
)


def lines(path: Path | None) -> Iterable[str]:
    if path is None or not path.exists():
        return
    with path.open("r", errors="replace") as stream:
        yield from stream


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


# The workload driver writes cmpl-kvaware-cold-000000, while vLLM/LMCache
# append a per-attempt suffix such as -0-ba3f55a1 to the same request.
# Do not truncate it to cmpl-kvaware (which destroys the phase and identity).
_ATTEMPT_SUFFIX_RE = re.compile(r"-\d+-[A-Za-z0-9]{6,}$")
_PHASE_ID_RE = re.compile(r"(?:^|-)kvaware-(cold|warm)-\d+(?:-|$)")


def base_request_id(request_id: str) -> str:
    return _ATTEMPT_SUFFIX_RE.sub("", request_id)


def phase_map(results: Path | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not results or not results.exists():
        return out
    for p in results.glob("*results*.jsonl"):
        for line in lines(p):
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            rid, phase = row.get("request_id"), row.get("phase")
            if rid and phase:
                rid, phase = str(rid), str(phase)
                if rid in out and out[rid] != phase:
                    raise ValueError(f"Conflicting phase for {rid}: {out[rid]} vs {phase}")
                out[rid] = phase
    return out


def request_phase(request_id: str, phases: dict[str, str]) -> str:
    canonical = base_request_id(request_id)
    for candidate in (request_id, canonical):
        if candidate in phases:
            return phases[candidate]
    # Request IDs are deterministic in this workload and explicitly encode
    # their phase. Use this only when the results JSONL is unavailable.
    match = _PHASE_ID_RE.search(canonical)
    return match.group(1) if match else "unknown"


def validate_raw_geometry(request_id: str, rec: dict) -> None:
    source_map = rec["source_map"]
    count = rec["requested_chunks"]
    if rec["chunk_size"] <= 0:
        raise ValueError(f"Invalid chunk size for {request_id}")
    if len(source_map) != count or set(source_map) - {"C", "D", "M"}:
        raise ValueError(f"Invalid source_map geometry for {request_id}: "
                         f"{len(source_map)} vs requested={count}, map={source_map}")
    if (source_map.count("C") != rec["l1_hit_chunks"] or
        source_map.count("D") != rec["l2_hit_chunks"] or
        source_map.count("M") != rec["miss_chunks"]):
        raise ValueError(f"Source-map counts disagree for {request_id}")
    if not 0 <= rec["reachable_prefix_chunks"] <= count:
        raise ValueError(f"Invalid reachable prefix for {request_id}")
    # Prefix reachability ends at the first miss, not at the last present chunk.
    first_miss = source_map.find("M")
    expected_prefix = count if first_miss < 0 else first_miss
    if rec["reachable_prefix_chunks"] != expected_prefix:
        raise ValueError(f"Reachable prefix disagrees with physical source map "
                         f"for {request_id}: {rec['reachable_prefix_chunks']} vs "
                         f"{expected_prefix}")


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def source_token_counts(raw: dict, start: int, end: int) -> dict[str, int]:
    """Attribute an interval to the per-chunk physical hierarchy source map."""
    out = {"L1": 0, "L2": 0, "MISS": 0}
    cs = raw["chunk_size"]
    offset = raw.get("start_token", 0)
    for i, tier in enumerate(raw["source_tiers"]):
        chunk_start = offset + i * cs
        out[tier] += overlap(start, end, chunk_start, chunk_start + cs)
    return out


def reachable_prefix_tokens(raw: dict, start: int, end: int) -> int:
    """Overlap ``[start,end)`` with the hierarchy's contiguous reachable prefix."""
    prefix_start = raw.get("start_token", 0)
    prefix_end = prefix_start + raw.get("reachable_prefix_chunks", 0) * raw["chunk_size"]
    return overlap(start, end, prefix_start, prefix_end)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmcache-log", type=Path, required=True)
    ap.add_argument("--vllm-log", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--strict-phase", action="store_true",
                    help="Reject any unknown phase or missing expected result ID")
    args = ap.parse_args()

    lookups: dict[str, dict] = {}
    raw_lookups: dict[str, dict] = {}
    qwait: list[float] = []
    lookup_s: list[float] = []
    total_s: list[float] = []
    load_s: list[float] = []
    store_s: list[float] = []
    native: dict[str, list[float]] = {}
    pending = {k: [] for k in ("store", "lookup", "load", "total")}
    lifetimes: list[float] = []

    raw_geometry_warning_count = 0
    duplicate_raw_records = 0
    duplicate_admit_records = 0
    changed_admit_request_ids: set[str] = set()
    for line in lines(args.lmcache_log):
        if "Raw CHTHM geometry mismatch request=" in line:
            raw_geometry_warning_count += 1
        if m := RAW_RE.search(line):
            source_map = m.group(8)
            if m.group(1) in raw_lookups:
                duplicate_raw_records += 1
                continue  # First physical snapshot, before any subsequent work.
            raw_lookups[m.group(1)] = {
                "chunk_size": int(m.group(2)),
                "requested_chunks": int(m.group(3)),
                "reachable_prefix_chunks": int(m.group(4)),
                "l1_hit_chunks": int(m.group(5)),
                "l2_hit_chunks": int(m.group(6)),
                "miss_chunks": int(m.group(7)),
                "source_map": source_map,
            }
        if m := LOOKUP_RE.search(line):
            lookups[m.group(1)] = {
                "requested": int(m.group(2)),
                "total": int(m.group(3)),
                "l1": int(m.group(4)),
                "l2": int(m.group(5)),
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
            for key, group in zip(("store", "lookup", "load", "total"), (3, 4, 5, 6)):
                pending[key].append(float(m.group(group)))
        if m := LIFETIME_RE.search(line):
            age = float(m.group(1))
            if age >= 0:
                lifetimes.append(age)

    admits: dict[str, dict] = {}
    mutual_lookup_starts: dict[str, int] = {}
    freshness: list[tuple[str, float]] = []
    starvation_requests: list[str] = []
    starvation_wave_sum = 0
    recovery_waves = 0
    recovered_request_events = 0
    for line in lines(args.vllm_log):
        if m := ADMIT_RE.search(line):
            rid, decision, prompt, vpc, lmhit, ext, age = m.groups()
            parsed_admit = {
                "decision": decision,
                "prompt": int(prompt),
                "vpc": int(vpc),
                "lmhit": int(lmhit),
                "ext": int(ext),
                "age": float(age),
            }
            if rid in admits:
                duplicate_admit_records += 1
                if any(admits[rid][key] != parsed_admit[key]
                       for key in ("decision", "prompt", "vpc", "lmhit", "ext")):
                    changed_admit_request_ids.add(rid)
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
        if m := MUTUAL_PREFIX_RE.search(line):
            mutual_lookup_starts[m.group(1)] = int(m.group(4))
        if m := FRESH_RE.search(line):
            freshness.append((m.group(1), float(m.group(2))))
        if m := STARVE_REQ_RE.search(line):
            starvation_requests.append(m.group(1))
        if m := STARVE_WAVE_RE.search(line):
            starvation_wave_sum += int(m.group(1))
        if m := RECOVERY_RE.search(line):
            recovery_waves += 1
            recovered_request_events += int(m.group(1))

    # New MP_CHTHM_RAW records describe physical per-chunk L1/L2 availability
    # immediately after lookup/EXISTS, before staging/load.  They intentionally
    # include hits after a prefix hole.  Fall back to the legacy post-load
    # monotonic MP_CHTHM_LOOKUP record only for old archives.
    raw: dict[str, dict] = {}
    chunk_size = 512
    for rid, rec in raw_lookups.items():
        validate_raw_geometry(rid, rec)
        cs = rec["chunk_size"]
        tiers = {"C": "L1", "D": "L2", "M": "MISS"}
        source_tiers = [tiers[ch] for ch in rec["source_map"]]
        raw[rid] = {
            "chunk_size": cs,
            "requested_chunks": rec["requested_chunks"],
            "total_hit_chunks": rec["l1_hit_chunks"] + rec["l2_hit_chunks"],
            "l1_hit_chunks": rec["l1_hit_chunks"],
            "l2_hit_chunks": rec["l2_hit_chunks"],
            "miss_chunks": rec["miss_chunks"],
            "reachable_prefix_chunks": rec["reachable_prefix_chunks"],
            "source_tiers": source_tiers,
            "start_token": mutual_lookup_starts.get(rid, 0),
            "source": "MP_CHTHM_RAW",
            "exact_physical_map": True,
        }

    for rid, rec in lookups.items():
        if rid in raw:
            continue
        l1_chunks = rec["l1"] // chunk_size
        l2_chunks = rec["l2"] // chunk_size
        requested_chunks = rec["requested"] // chunk_size
        miss_chunks = max(0, requested_chunks - l1_chunks - l2_chunks)
        raw[rid] = {
            "chunk_size": chunk_size,
            "requested_chunks": requested_chunks,
            "total_hit_chunks": rec["total"] // chunk_size,
            "l1_hit_chunks": l1_chunks,
            "l2_hit_chunks": l2_chunks,
            "miss_chunks": miss_chunks,
            "reachable_prefix_chunks": l1_chunks + l2_chunks,
            # Legacy LOOKUP knows only the reachable, post-lookup prefix.
            # Do NOT invent physical misses outside that prefix: later CPU/L2
            # chunks might exist behind a gap but were never observed.
            "source_tiers": ["L1"] * l1_chunks + ["L2"] * l2_chunks,
            "start_token": mutual_lookup_starts.get(rid, 0),
            "source": "MP_CHTHM_LOOKUP_LEGACY",
            "exact_physical_map": False,
        }

    if args.strict_phase and raw_geometry_warning_count:
        raise ValueError(f"Found {raw_geometry_warning_count} runtime CHTHM geometry warnings")
    phases = phase_map(args.results_dir)
    if args.strict_phase and not phases:
        raise ValueError("--strict-phase requires result JSONLs with request IDs and phases")
    raw_b: dict[str, dict] = {}
    useful_b: dict[str, dict] = {}
    unknown_request_ids: list[str] = []
    phase_admit_counts: dict[str, int] = {}

    for rid, admit in admits.items():
        phase = request_phase(rid, phases)
        phase_admit_counts[phase] = phase_admit_counts.get(phase, 0) + 1
        if phase == "unknown":
            unknown_request_ids.append(rid)
        if args.strict_phase and (phase == "unknown" or
                                  (phases and base_request_id(rid) not in phases
                                   and rid not in phases)):
            raise ValueError(f"Unmatched results phase/request for {rid}")
        prompt = admit["prompt"]
        vpc = min(prompt, admit["vpc"])
        rr = raw.get(rid)
        cs = rr["chunk_size"] if rr is not None else chunk_size
        chunk_addressable = prompt - (prompt % cs)
        external_opportunity = max(0, chunk_addressable - min(vpc, chunk_addressable))

        rb = raw_b.setdefault(
            phase,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "vpc_hit_tokens": 0,
                "l1_hit_tokens": 0,
                "l2_hit_tokens": 0,
                "known_hierarchy_miss_tokens": 0,
                "reachable_external_prefix_tokens": 0,
                "raw_external_opportunity_tokens": 0,
                "observed_external_opportunity_tokens": 0,
                "unobserved_external_opportunity_tokens": 0,
                "physical_raw_requests": 0,
                "legacy_lookup_requests": 0,
                "no_raw_lookup_requests": 0,
                "partial_raw_observation_requests": 0,
            },
        )
        ub = useful_b.setdefault(
            phase,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "vpc_tokens": 0,
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
            physical = rr["exact_physical_map"]
            rb["physical_raw_requests" if physical else "legacy_lookup_requests"] += 1
            # A logged source map is authoritative only over its actual token
            # range. Missing leading/suffix coverage is censored, not a miss.
            raw_start = rr["start_token"]
            raw_end = raw_start + len(rr["source_tiers"]) * rr["chunk_size"]
            observed = overlap(vpc, chunk_addressable, raw_start, raw_end)
            rb["observed_external_opportunity_tokens"] += observed
            rb["unobserved_external_opportunity_tokens"] += (external_opportunity - observed)
            if observed < external_opportunity:
                rb["partial_raw_observation_requests"] += 1
            src = source_token_counts(rr, vpc, chunk_addressable)
            rb["l1_hit_tokens"] += src["L1"]
            rb["l2_hit_tokens"] += src["L2"]
            rb["known_hierarchy_miss_tokens"] += src["MISS"]
            rb["reachable_external_prefix_tokens"] += reachable_prefix_tokens(
                rr, vpc, chunk_addressable
            )
        else:
            rb["no_raw_lookup_requests"] += 1
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
            ub["l1_tokens"] += src["L1"]
            ub["l2_tokens"] += src["L2"]
            attributed = src["L1"] + src["L2"]
            ub["external_unknown_tokens"] += max(0, ext - attributed)
        else:
            ub["external_unknown_tokens"] += ext
        if admit["decision"] == "freshness_miss":
            ub["freshness_misses"] += 1

    if args.strict_phase:
        expected_by_phase: dict[str, int] = {}
        for ph in phases.values():
            expected_by_phase[ph] = expected_by_phase.get(ph, 0) + 1
        if phase_admit_counts != expected_by_phase:
            raise ValueError("Phase admit counts do not match result request IDs: "
                             f"observed={phase_admit_counts}, "
                             f"expected={expected_by_phase}")

    for bucket in raw_b.values():
        total = bucket["prompt_tokens"]
        gpu = bucket["vpc_hit_tokens"]
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
        bucket["reachable_external_prefix_pct_all_prompt_lower_bound"] = (
            100.0 * bucket["reachable_external_prefix_tokens"] / total
            if total
            else 0.0
        )
        bucket["rates_exact"] = bucket["unobserved_external_opportunity_tokens"] == 0

    for bucket in useful_b.values():
        total = bucket["prompt_tokens"]
        for key in (
            "vpc_tokens",
            "l1_tokens",
            "l2_tokens",
            "external_unknown_tokens",
            "recompute_tokens",
        ):
            bucket[key.replace("_tokens", "_pct")] = 100.0 * bucket[key] / total if total else 0.0

    phase_counts = {
        phase: sum(1 for value in phases.values() if value == phase)
        for phase in set(phases.values())
    }
    starve_by: dict[str, set[str]] = {}
    fresh_by: dict[str, set[str]] = {}
    for rid in starvation_requests:
        phase = request_phase(rid, phases)
        starve_by.setdefault(phase, set()).add(rid)
    for rid, _ in freshness:
        phase = request_phase(rid, phases)
        fresh_by.setdefault(phase, set()).add(rid)

    result = {
        "raw_chthm": raw_b,
        "phase_attribution": {
            "result_request_ids": len(phases),
            "scheduler_admit_by_phase": phase_admit_counts,
            "unknown_scheduler_admit_ids": unknown_request_ids[:20],
            "unknown_scheduler_admit_count": len(unknown_request_ids),
        },
        "useful_contribution": useful_b,
        "chthm_definition": (
            "GPU/VPC hit is over all prompt tokens; CPU/L1 hit is conditional on GPU miss; "
            "disk/L2 hit is conditional on GPU+CPU miss; hierarchy miss/recompute is over all prompt tokens."
        ),
        "observation_note": (
            "MP_CHTHM_RAW is pre-load physical per-chunk hierarchy availability; "
            "reachable_external_prefix is a separate contiguous-prefix metric. "
            "Physical raw rates are exact only where pre-load MP_CHTHM_RAW covers "
            "the lookup; legacy post-lookup prefix counters never establish "
            "a physical miss beyond that prefix. Any unobserved coverage "
            "makes raw tier rates lower bounds. "
            "Useful contribution is scheduler-consumed KV and remains separate."
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
        "raw_lookup_records": len(raw_lookups),
        "raw_geometry_warning_count": raw_geometry_warning_count,
        "legacy_lookup_records": len(lookups),
        "lookup_records_used": len(raw),
        "scheduler_admit_records": len(admits),
        "duplicate_raw_records": duplicate_raw_records,
        "duplicate_scheduler_admit_records": duplicate_admit_records,
        "changed_scheduler_admit_requests": len(changed_admit_request_ids),
        "changed_scheduler_admit_request_examples": sorted(changed_admit_request_ids)[:20],
        "scheduler_admit_observation_note": (
            "First admission record per request is used for raw and contribution "
            "summary compatibility. Repeated admission logs can change VPC/LOAD "
            "fields; changed_scheduler_admit_requests identifies such cases. "
            "Do not treat repeated records as independent requests."
        ),
    }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

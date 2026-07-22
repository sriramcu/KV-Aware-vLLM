#!/usr/bin/env python3
"""
Summarize and compare KV-Aware vLLM/LMCache run logs.

Usage:
    python summarize_kvaware_performance.py run1.out
    python summarize_kvaware_performance.py with_gnn.out wo_gnn.out

The script uses only Python's standard library.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

REQ_RE = re.compile(
    r"Reqid:\s*([^,]+),\s*"
    r"Total tokens\s*(\d+),\s*"
    r"Inference Engine computed tokens:\s*(\d+),\s*"
    r"LMCache hit tokens:\s*(\d+),\s*"
    r"need to load:\s*(\d+)",
    re.IGNORECASE,
)

PROGRESS_RE = re.compile(
    r"Processed prompts:\s*100%.*?"
    r"est\. speed input:\s*([0-9.]+)\s*toks/s,\s*"
    r"output:\s*([0-9.]+)\s*toks/s",
    re.IGNORECASE,
)

ENGINE_TPUT_RE = re.compile(
    r"Avg prompt throughput:\s*([0-9.]+)\s*tokens/s,\s*"
    r"Avg generation throughput:\s*([0-9.]+)\s*tokens/s",
    re.IGNORECASE,
)

STORE_RE = re.compile(
    r"\[req_id=([^\]]+)\]\s*Stored\s+(\d+)\s+out of total\s+(\d+)\s+tokens\."
    r".*?cost\s+([0-9.]+)\s+ms,\s*throughput:\s*([0-9.]+)\s+GB/s",
    re.IGNORECASE,
)

RETRIEVE_RE = re.compile(
    r"\[req_id=([^\]]+)\]\s*Retrieved\s+(\d+)\s+out of\s+(\d+)\s+required tokens"
    r".*?cost\s+([0-9.]+)\s+ms,\s*throughput:\s*([0-9.]+)\s+GB/s",
    re.IGNORECASE,
)

PLACEMENT_RE = re.compile(r"\[KV placement\]\s*(\{.*?\})")

CACHE_SIZE_RE = re.compile(
    r"^([0-9.]+[KMGTP]?)\s+(.*/lmcache_vllm/\S+)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

MODE_RE = re.compile(r"^\+\s*MODE=(\S+)", re.MULTILINE)
MAX_Q_RE = re.compile(r"^\+\s*MAX_QUESTIONS=(\d+)", re.MULTILINE)
JOB_ID_RE = re.compile(r"(\d+)(?=\.out$)")


@dataclass
class PhaseStats:
    name: str
    wall_seconds: Optional[float] = None
    final_input_tps: Optional[float] = None
    final_output_tps: Optional[float] = None
    requests: int = 0
    requests_with_hit: int = 0
    zero_hit_requests: int = 0
    average_hit_tokens: float = 0.0
    maximum_hit_tokens: int = 0
    total_hit_tokens: int = 0
    total_prompt_tokens: int = 0
    hit_token_ratio: float = 0.0
    average_computed_tokens: float = 0.0
    average_need_to_load: float = 0.0
    rolling_prompt_tps_mean: Optional[float] = None
    rolling_generation_tps_mean: Optional[float] = None
    store_events: int = 0
    stored_tokens: int = 0
    average_store_ms: Optional[float] = None
    average_store_gbps: Optional[float] = None
    retrieve_events: int = 0
    retrieved_tokens: int = 0
    average_retrieve_ms: Optional[float] = None
    average_retrieve_gbps: Optional[float] = None
    warnings: Counter = field(default_factory=Counter)
    placements: Counter = field(default_factory=Counter)


@dataclass
class RunStats:
    path: Path
    mode: str
    job_id: str
    max_questions: Optional[int]
    cache_size: Optional[str]
    cache_path: Optional[str]
    cold: PhaseStats
    warm: PhaseStats
    full_warnings: Counter


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def split_phases(text: str) -> tuple[str, str]:
    cold_marker = "Cold run starting..."
    warm_marker = "Warm run starting..."

    if cold_marker not in text:
        return text, ""

    after_cold = text.split(cold_marker, 1)[1]
    if warm_marker not in after_cold:
        return after_cold, ""

    cold, warm = after_cold.split(warm_marker, 1)
    return cold, warm


def parse_wall_time(segment: str, phase: str) -> Optional[float]:
    if phase == "cold":
        pattern = r"first generation took\s+([0-9.]+)\s+seconds"
    else:
        pattern = r"Second generation took\s+([0-9.]+)\s+seconds"
    matches = re.findall(pattern, segment, flags=re.IGNORECASE)
    return float(matches[-1]) if matches else None


def parse_request_stats(segment: str) -> dict[str, tuple[int, int, int, int]]:
    # A request can appear multiple times in the log. Keep the line with the
    # largest hit/load values, which represents the final scheduler decision.
    records: dict[str, tuple[int, int, int, int]] = {}
    for req_id, total, computed, hit, need in REQ_RE.findall(segment):
        values = (int(total), int(computed), int(hit), int(need))
        old = records.get(req_id)
        if old is None or (values[2], values[3], values[1]) > (old[2], old[3], old[1]):
            records[req_id] = values
    return records


def tp0_or_all_lines(segment: str) -> list[str]:
    lines = segment.splitlines()
    tp0 = [line for line in lines if "Worker_TP0" in line]
    return tp0 if tp0 else lines


def parse_transfer_events(segment: str, regex: re.Pattern[str]) -> list[tuple]:
    # Prefer TP0 so tensor-parallel workers do not double-count the same KV.
    lines = tp0_or_all_lines(segment)
    out = []
    for line in lines:
        match = regex.search(line)
        if match:
            out.append(match.groups())
    return out


def count_warnings(segment: str) -> Counter:
    warning_patterns = {
        "async_lookup_timeout": r"still waiting for async lookup after",
        "cpu_memory_pressure": r"Local cpu memory under pressure",
        "allocation_failure": r"Failed to allocate memory block",
        "disk_load_allocation_failure": r"Memory allocation failed during async disk load",
        "no_space_left": r"No space left on device",
        "quota_exceeded": r"Disk quota exceeded|quota exceeded",
        "store_failure": r"Failed to store|store.*failed",
        "retrieve_failure": r"Failed to retrieve|retrieve.*failed",
    }
    return Counter(
        {
            name: len(re.findall(pattern, segment, flags=re.IGNORECASE))
            for name, pattern in warning_patterns.items()
        }
    )


def parse_placements(segment: str) -> Counter:
    counter: Counter = Counter()
    for line in tp0_or_all_lines(segment):
        match = PLACEMENT_RE.search(line)
        if not match:
            continue
        try:
            placement = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            continue
        if isinstance(placement, dict):
            for backend, count in placement.items():
                try:
                    counter[str(backend)] += int(count)
                except (TypeError, ValueError):
                    pass
    return counter


def safe_mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def parse_phase(segment: str, name: str) -> PhaseStats:
    stats = PhaseStats(name=name)
    stats.wall_seconds = parse_wall_time(segment, name)

    progress = PROGRESS_RE.findall(segment)
    if progress:
        stats.final_input_tps = float(progress[-1][0])
        stats.final_output_tps = float(progress[-1][1])

    engine_tputs = [(float(a), float(b)) for a, b in ENGINE_TPUT_RE.findall(segment)]
    if engine_tputs:
        stats.rolling_prompt_tps_mean = safe_mean([x[0] for x in engine_tputs])
        stats.rolling_generation_tps_mean = safe_mean([x[1] for x in engine_tputs])

    records = parse_request_stats(segment)
    stats.requests = len(records)
    if records:
        totals = [v[0] for v in records.values()]
        computed = [v[1] for v in records.values()]
        hits = [v[2] for v in records.values()]
        needs = [v[3] for v in records.values()]

        stats.requests_with_hit = sum(hit > 0 for hit in hits)
        stats.zero_hit_requests = stats.requests - stats.requests_with_hit
        stats.average_hit_tokens = sum(hits) / len(hits)
        stats.maximum_hit_tokens = max(hits)
        stats.total_hit_tokens = sum(hits)
        stats.total_prompt_tokens = sum(totals)
        stats.hit_token_ratio = (
            stats.total_hit_tokens / stats.total_prompt_tokens
            if stats.total_prompt_tokens
            else 0.0
        )
        stats.average_computed_tokens = sum(computed) / len(computed)
        stats.average_need_to_load = sum(needs) / len(needs)

    store_events = parse_transfer_events(segment, STORE_RE)
    stats.store_events = len(store_events)
    if store_events:
        stats.stored_tokens = sum(int(x[1]) for x in store_events)
        stats.average_store_ms = safe_mean([float(x[3]) for x in store_events])
        stats.average_store_gbps = safe_mean([float(x[4]) for x in store_events])

    retrieve_events = parse_transfer_events(segment, RETRIEVE_RE)
    stats.retrieve_events = len(retrieve_events)
    if retrieve_events:
        stats.retrieved_tokens = sum(int(x[1]) for x in retrieve_events)
        stats.average_retrieve_ms = safe_mean([float(x[3]) for x in retrieve_events])
        stats.average_retrieve_gbps = safe_mean([float(x[4]) for x in retrieve_events])

    stats.warnings = count_warnings(segment)
    stats.placements = parse_placements(segment)
    return stats


def parse_run(path: Path) -> RunStats:
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = strip_ansi(raw)

    mode_match = MODE_RE.search(text)
    mode = mode_match.group(1) if mode_match else path.stem

    job_match = JOB_ID_RE.search(path.name)
    job_id = job_match.group(1) if job_match else "-"

    max_q_match = MAX_Q_RE.search(text)
    max_questions = int(max_q_match.group(1)) if max_q_match else None

    cache_matches = CACHE_SIZE_RE.findall(text)
    cache_size = cache_matches[-1][0] if cache_matches else None
    cache_path = cache_matches[-1][1] if cache_matches else None

    cold_text, warm_text = split_phases(text)

    return RunStats(
        path=path,
        mode=mode,
        job_id=job_id,
        max_questions=max_questions,
        cache_size=cache_size,
        cache_path=cache_path,
        cold=parse_phase(cold_text, "cold"),
        warm=parse_phase(warm_text, "warm"),
        full_warnings=count_warnings(text),
    )


def fmt_num(value: Optional[float], digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def fmt_int(value: int) -> str:
    return f"{value:,}"


def print_phase(phase: PhaseStats) -> None:
    print(f"\n{phase.name.upper()}")
    print("-" * 66)
    print(f"Wall time:                 {fmt_num(phase.wall_seconds, 2)} s")
    print(f"Final effective input TPS: {fmt_num(phase.final_input_tps, 2)}")
    print(f"Final effective output TPS:{fmt_num(phase.final_output_tps, 2)}")
    print(f"Requests:                  {phase.requests}")
    print(
        f"Requests with LMCache hit: {phase.requests_with_hit} "
        f"({(100.0 * phase.requests_with_hit / phase.requests):.1f}%)"
        if phase.requests
        else "Requests with LMCache hit: -"
    )
    print(f"Average hit tokens/request:{phase.average_hit_tokens:.1f}")
    print(f"Maximum hit tokens:        {phase.maximum_hit_tokens}")
    print(f"Total hit-token ratio:     {100.0 * phase.hit_token_ratio:.2f}%")
    print(f"Average computed tokens:   {phase.average_computed_tokens:.1f}")
    print(f"Average tokens to load:    {phase.average_need_to_load:.1f}")
    print(
        f"Store events (TP0):        {phase.store_events}, "
        f"tokens={fmt_int(phase.stored_tokens)}, "
        f"avg={fmt_num(phase.average_store_ms, 2)} ms, "
        f"{fmt_num(phase.average_store_gbps, 2)} GB/s"
    )
    print(
        f"Retrieve events (TP0):     {phase.retrieve_events}, "
        f"tokens={fmt_int(phase.retrieved_tokens)}, "
        f"avg={fmt_num(phase.average_retrieve_ms, 2)} ms, "
        f"{fmt_num(phase.average_retrieve_gbps, 2)} GB/s"
    )
    if phase.placements:
        placements = ", ".join(
            f"{name}={count:,}" for name, count in phase.placements.items()
        )
        print(f"Placement chunks (TP0):    {placements}")

    nonzero_warnings = {k: v for k, v in phase.warnings.items() if v}
    if nonzero_warnings:
        print("Warning/error log lines (all ranks; TP may duplicate):")
        for name, count in nonzero_warnings.items():
            print(f"  {name}: {count}")


def print_run(run: RunStats) -> None:
    print("\n" + "=" * 72)
    print(f"{run.mode} | job={run.job_id} | file={run.path}")
    print("=" * 72)
    if run.max_questions is not None:
        print(f"Configured questions: {run.max_questions}")
    if run.cache_size:
        print(f"Final LMCache directory size: {run.cache_size}")
    if run.cache_path:
        print(f"LMCache path: {run.cache_path}")

    print_phase(run.cold)
    print_phase(run.warm)

    if run.cold.wall_seconds and run.warm.wall_seconds:
        reduction = (
            100.0 * (run.cold.wall_seconds - run.warm.wall_seconds) / run.cold.wall_seconds
        )
        print(
            f"\nWarm-vs-cold wall-time reduction: "
            f"{run.cold.wall_seconds - run.warm.wall_seconds:.2f} s ({reduction:.2f}%)"
        )


def pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old in (None, 0):
        return None
    return 100.0 * (new - old) / old


def print_comparison(a: RunStats, b: RunStats) -> None:
    print("\n" + "#" * 72)
    print(f"PAIRWISE COMPARISON: {a.mode} relative to {b.mode}")
    print("#" * 72)

    for phase_name in ("cold", "warm"):
        pa = getattr(a, phase_name)
        pb = getattr(b, phase_name)
        print(f"\n{phase_name.upper()}")
        if pa.wall_seconds is not None and pb.wall_seconds is not None:
            delta = pa.wall_seconds - pb.wall_seconds
            change = pct_change(pa.wall_seconds, pb.wall_seconds)
            print(
                f"Wall time: {pa.wall_seconds:.2f} s vs {pb.wall_seconds:.2f} s "
                f"(delta {delta:+.2f} s, {change:+.2f}%)"
            )
        if pa.final_input_tps is not None and pb.final_input_tps is not None:
            change = pct_change(pa.final_input_tps, pb.final_input_tps)
            print(
                f"Input TPS: {pa.final_input_tps:.2f} vs {pb.final_input_tps:.2f} "
                f"({change:+.2f}%)"
            )
        if pa.final_output_tps is not None and pb.final_output_tps is not None:
            change = pct_change(pa.final_output_tps, pb.final_output_tps)
            print(
                f"Output TPS: {pa.final_output_tps:.2f} vs {pb.final_output_tps:.2f} "
                f"({change:+.2f}%)"
            )
        print(
            f"LMCache-hit requests: {pa.requests_with_hit}/{pa.requests} vs "
            f"{pb.requests_with_hit}/{pb.requests}"
        )
        print(
            f"Average hit tokens: {pa.average_hit_tokens:.1f} vs "
            f"{pb.average_hit_tokens:.1f}"
        )

    print("\nInterpretation checks")
    print("-" * 66)
    if (
        a.warm.requests
        and b.warm.requests
        and abs(a.warm.requests_with_hit - b.warm.requests_with_hit)
        > 0.20 * max(a.warm.requests, b.warm.requests)
    ):
        print(
            "WARNING: Warm hit rates differ substantially. The warm throughput "
            "comparison is not a clean placement-policy comparison."
        )
    for run in (a, b):
        timeout_count = run.warm.warnings["async_lookup_timeout"]
        disk_alloc_count = run.warm.warnings["disk_load_allocation_failure"]
        pressure_count = run.warm.warnings["cpu_memory_pressure"]
        if timeout_count or disk_alloc_count or pressure_count:
            print(
                f"WARNING: {run.mode} warm run had lookup/storage pressure: "
                f"timeouts={timeout_count}, disk-load-allocation-failures="
                f"{disk_alloc_count}, cpu-pressure={pressure_count}."
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize KV-Aware vLLM/LMCache performance logs."
    )
    parser.add_argument("logs", nargs="+", type=Path, help="One or more .out log files")
    args = parser.parse_args()

    runs: list[RunStats] = []
    for path in args.logs:
        if not path.is_file():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            return 2
        try:
            runs.append(parse_run(path))
        except OSError as exc:
            print(f"ERROR reading {path}: {exc}", file=sys.stderr)
            return 2

    for run in runs:
        print_run(run)

    if len(runs) == 2:
        print_comparison(runs[0], runs[1])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 JOB_ID" >&2
  exit 2
fi

JOB_ID="$1"
LOG_DIR=/mnt/shared/gpfs/home/sriramc2/runs/kvaware_repro/logs
OUT="$LOG_DIR/kvaware_p0ab_${JOB_ID}.out"
ERR="$LOG_DIR/kvaware_p0ab_${JOB_ID}.err"

[[ -f "$OUT" ]] || { echo "Missing $OUT" >&2; exit 1; }
[[ -f "$ERR" ]] || { echo "Missing $ERR" >&2; exit 1; }

python - "$OUT" "$ERR" <<'PY'
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
import sys

paths = [Path(p) for p in sys.argv[1:]]
text = "\n".join(p.read_text(errors="replace") for p in paths)
text = re.sub(r"\x1b\[[0-9;]*m", "", text)
lines = text.splitlines()

markers = [
    "P0_LOOKUP_ADMISSION_WAIT",
    "P0_LOOKUP_ADMISSION_ACQUIRE",
    "P0_LOOKUP_ADMISSION_RELEASE",
    "P0_LOOKUP_ADMISSION_ABORT",
    "P0_PUT_ADMISSION_WAIT",
    "P0_PUT_ADMISSION_ACQUIRE",
    "P0_PUT_ADMISSION_RELEASE",
    "P0_PUT_ADMISSION_STALLED",
    "P0_PUT_ADMISSION_REJECT_LOOP_THREAD",
]
counts = Counter()
for line in lines:
    for marker in markers:
        if f"[{marker}]" in line:
            counts[marker] += 1


def field_values(marker: str, field: str, cast=int):
    pattern = re.compile(rf"\[{re.escape(marker)}\].*?\b{re.escape(field)}=([^\s]+)")
    values = []
    for line in lines:
        match = pattern.search(line)
        if match:
            raw = match.group(1).rstrip(",")
            try:
                values.append(cast(raw))
            except ValueError:
                pass
    return values


def per_pid(marker: str):
    result = Counter()
    pattern = re.compile(rf"\[{re.escape(marker)}\].*?\bpid=(\d+)")
    for line in lines:
        match = pattern.search(line)
        if match:
            result[match.group(1)] += 1
    return result

lookup_inflight = field_values("P0_LOOKUP_ADMISSION_ACQUIRE", "inflight")
lookup_limits = field_values("P0_LOOKUP_ADMISSION_ACQUIRE", "limit")
lookup_wait = field_values("P0_LOOKUP_ADMISSION_ACQUIRE", "waited", float)
put_inflight = field_values("P0_PUT_ADMISSION_ACQUIRE", "inflight")
put_limits = field_values("P0_PUT_ADMISSION_ACQUIRE", "limit")
put_wait = field_values("P0_PUT_ADMISSION_ACQUIRE", "waited", float)
put_queue = field_values("P0_PUT_ADMISSION_ACQUIRE", "queue_depth")

print("=== P0 FIRST-HALF SUMMARY ===")
for marker in markers:
    print(f"{marker}: {counts[marker]}")
print()
print("lookup max inflight:", max(lookup_inflight, default=None))
print("lookup configured limits:", sorted(set(lookup_limits)))
print("lookup max admission wait seconds:", max(lookup_wait, default=None))
print("put max inflight:", max(put_inflight, default=None))
print("put configured limits:", sorted(set(put_limits)))
print("put max admission wait seconds:", max(put_wait, default=None))
print("put max executor queue depth at acquire:", max(put_queue, default=None))

lookup_acq = per_pid("P0_LOOKUP_ADMISSION_ACQUIRE")
lookup_rel = per_pid("P0_LOOKUP_ADMISSION_RELEASE")
put_acq = per_pid("P0_PUT_ADMISSION_ACQUIRE")
put_rel = per_pid("P0_PUT_ADMISSION_RELEASE")

print("\nlookup acquire/release by pid:")
for pid in sorted(set(lookup_acq) | set(lookup_rel)):
    print(f"  pid={pid} acquire={lookup_acq[pid]} release={lookup_rel[pid]}")
print("put acquire/release by pid:")
for pid in sorted(set(put_acq) | set(put_rel)):
    print(f"  pid={pid} acquire={put_acq[pid]} release={put_rel[pid]}")

failure_patterns = {
    "job did not reach success marker": "=== P0 FIRST HALF JOB PASSED ===" not in text,
    "lookup admission abort": counts["P0_LOOKUP_ADMISSION_ABORT"] > 0,
    "put rejected on event-loop thread": counts["P0_PUT_ADMISSION_REJECT_LOOP_THREAD"] > 0,
    "put completion failed/cancelled": bool(
        re.search(
            r"\[P0_PUT_ADMISSION_RELEASE\].*?reason=(?:failed|cancelled)",
            text,
        )
    ),
    "lookup limit exceeded": bool(lookup_inflight and lookup_limits)
    and max(lookup_inflight) > max(lookup_limits),
    "put limit exceeded": bool(put_inflight and put_limits)
    and max(put_inflight) > max(put_limits),
    "lookup acquire/release imbalance": lookup_acq != lookup_rel,
    "put acquire/release imbalance": put_acq != put_rel,
    "traceback/import/extension/OOM error": bool(
        re.search(
            r"Traceback \(most recent call last\)|ModuleNotFoundError|"
            r"ImportError|undefined symbol|CUDA out of memory|OutOfMemoryError|"
            r"P0 .*admission underflow",
            text,
            re.IGNORECASE,
        )
    ),
}

failed = [name for name, present in failure_patterns.items() if present]
print("\nchecks:")
if failed:
    for name in failed:
        print("  FAIL:", name)
else:
    print("  PASS: admission limits held, completions balanced, and job finished")

interesting = [
    line
    for line in lines
    if "P0_PUT_ADMISSION_REJECT" in line
    or re.search(r"\[P0_PUT_ADMISSION_RELEASE\].*?reason=(?:failed|cancelled)", line)
    or "P0_LOOKUP_ADMISSION_ABORT" in line
    or "Traceback (most recent call last)" in line
    or "undefined symbol" in line
    or "out of memory" in line.lower()
]
if interesting:
    print("\nanomaly lines (first 30):")
    for line in interesting[:30]:
        print(line)

if failed:
    raise SystemExit(1)
PY

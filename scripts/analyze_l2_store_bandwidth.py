#!/usr/bin/env python3
"""Summarize LMCache StoreController arrival/completion throughput.

Designed for the current KV-Aware MP logs containing [MP_STORE_SUBMIT] and
[MP_STORE_DONE]. It splits nonzero submit activity into epochs at large time
gaps and treats the epoch with the most submitted bytes as the primary (cold)
store epoch. Completion throughput is then measured from that epoch's first
submit until the next submit epoch begins (or end of log).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]")
SUBMIT_RE = re.compile(
    r"\[MP_STORE_SUBMIT\].*?task=(\d+).*?keys=(\d+).*?bytes=(\d+).*?"
    r"inflight_after=(\d+).*?inflight_bytes=(\d+)"
)
DONE_RE = re.compile(
    r"\[MP_STORE_DONE\].*?task=(\d+).*?success=(True|False).*?keys=(\d+).*?"
    r"bytes=(\d+).*?service_s=([0-9.]+).*?inflight_after=(\d+).*?inflight_bytes=(\d+)"
)


def parse_ts(line: str) -> float | None:
    m = TS_RE.search(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()


def rate_mib(total_bytes: int, start: float, end: float) -> float | None:
    span = end - start
    if span <= 0:
        return None
    return total_bytes / (1024**2) / span


def fnum(x):
    return None if x is None or not math.isfinite(x) else round(x, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmcache-log", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--kv-bytes-per-token", type=float, required=True)
    ap.add_argument("--epoch-gap-s", type=float, default=30.0)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-txt", required=True)
    args = ap.parse_args()

    submits = []
    dones = []
    for raw in Path(args.lmcache_log).read_text(errors="replace").splitlines():
        line = ANSI_RE.sub("", raw)
        ts = parse_ts(line)
        if ts is None:
            continue
        m = SUBMIT_RE.search(line)
        if m:
            task, keys, nbytes, inflight, inflight_bytes = map(int, m.groups())
            if nbytes > 0:
                submits.append({
                    "ts": ts, "task": task, "keys": keys, "bytes": nbytes,
                    "inflight_after": inflight, "inflight_bytes": inflight_bytes,
                })
            continue
        m = DONE_RE.search(line)
        if m:
            task = int(m.group(1)); success = m.group(2) == "True"
            keys = int(m.group(3)); nbytes = int(m.group(4)); service_s = float(m.group(5))
            inflight = int(m.group(6)); inflight_bytes = int(m.group(7))
            dones.append({
                "ts": ts, "task": task, "success": success, "keys": keys,
                "bytes": nbytes, "service_s": service_s,
                "inflight_after": inflight, "inflight_bytes": inflight_bytes,
            })

    submits.sort(key=lambda x: x["ts"])
    dones.sort(key=lambda x: x["ts"])
    if not submits:
        raise SystemExit("No nonzero [MP_STORE_SUBMIT] events found")

    epochs = []
    cur = [submits[0]]
    for ev in submits[1:]:
        if ev["ts"] - cur[-1]["ts"] > args.epoch_gap_s:
            epochs.append(cur); cur = [ev]
        else:
            cur.append(ev)
    epochs.append(cur)
    primary_idx = max(range(len(epochs)), key=lambda i: sum(x["bytes"] for x in epochs[i]))
    primary = epochs[primary_idx]
    t0 = primary[0]["ts"]
    submit_last = primary[-1]["ts"]
    next_epoch_start = epochs[primary_idx + 1][0]["ts"] if primary_idx + 1 < len(epochs) else float("inf")

    cold_dones = [d for d in dones if t0 <= d["ts"] < next_epoch_start and d["success"]]
    positive_dones = [d for d in cold_dones if d["bytes"] > 0]

    submitted_bytes = sum(x["bytes"] for x in primary)
    completed_bytes = sum(x["bytes"] for x in positive_dones)
    submit_rate = rate_mib(submitted_bytes, t0, submit_last) if len(primary) >= 2 else None
    completion_rate = (
        rate_mib(completed_bytes, positive_dones[0]["ts"], positive_dones[-1]["ts"])
        if len(positive_dones) >= 2 else None
    )

    saturated = {}
    for threshold in (10, 25, 50, 100):
        evs = [d for d in positive_dones if d["inflight_after"] >= threshold]
        if len(evs) >= 2:
            b = sum(x["bytes"] for x in evs)
            saturated[str(threshold)] = {
                "events": len(evs),
                "bytes": b,
                "mib_per_s": fnum(rate_mib(b, evs[0]["ts"], evs[-1]["ts"])),
                "span_s": round(evs[-1]["ts"] - evs[0]["ts"], 3),
            }
        else:
            saturated[str(threshold)] = {"events": len(evs), "bytes": sum(x["bytes"] for x in evs), "mib_per_s": None, "span_s": None}

    summary = json.loads(Path(args.summary).read_text())
    cold = summary.get("cold", {})
    prompt_tps = float(cold.get("prompt_tokens_per_second", 0.0) or 0.0)
    raw_kv_mib_s = prompt_tps * args.kv_bytes_per_token / (1024**2)

    result = {
        "primary_store_epoch_index": primary_idx,
        "submit_epochs": len(epochs),
        "epoch_gap_s": args.epoch_gap_s,
        "primary_submit_events": len(primary),
        "primary_submitted_bytes": submitted_bytes,
        "primary_submit_span_s": round(submit_last - t0, 3),
        "l2_submitted_mib_per_s": fnum(submit_rate),
        "primary_successful_nonzero_completions": len(positive_dones),
        "primary_completed_bytes": completed_bytes,
        "primary_completion_span_s": round(positive_dones[-1]["ts"] - positive_dones[0]["ts"], 3) if len(positive_dones) >= 2 else None,
        "l2_whole_window_completion_mib_per_s": fnum(completion_rate),
        "saturated_completion_by_storecontroller_inflight": saturated,
        "max_storecontroller_inflight_after": max((d["inflight_after"] for d in cold_dones), default=0),
        "cold_prompt_tokens_per_second": prompt_tps,
        "kv_bytes_per_token": args.kv_bytes_per_token,
        "raw_prompt_kv_creation_mib_per_s": round(raw_kv_mib_s, 3),
        "submission_over_raw_kv_ratio": round(submit_rate / raw_kv_mib_s, 4) if submit_rate and raw_kv_mib_s else None,
        "arrival_over_completion_ratio": round(submit_rate / completion_rate, 4) if submit_rate and completion_rate else None,
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "L2 STORE BANDWIDTH SUMMARY",
        f"raw prompt-KV creation:          {result['raw_prompt_kv_creation_mib_per_s']:.3f} MiB/s",
        f"L2 submitted (primary epoch):    {result['l2_submitted_mib_per_s']} MiB/s",
        f"L2 completed whole window:       {result['l2_whole_window_completion_mib_per_s']} MiB/s",
        f"arrival/completion ratio:         {result['arrival_over_completion_ratio']}",
        f"primary submitted bytes:          {submitted_bytes / (1024**3):.3f} GiB",
        f"primary completed unique bytes:   {completed_bytes / (1024**3):.3f} GiB",
        f"max StoreController inflight:     {result['max_storecontroller_inflight_after']}",
        "saturated completion estimates:",
    ]
    for k, v in saturated.items():
        lines.append(f"  inflight >= {k:>3}: events={v['events']:>5} rate={v['mib_per_s']} MiB/s span={v['span_s']} s")
    Path(args.output_txt).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()

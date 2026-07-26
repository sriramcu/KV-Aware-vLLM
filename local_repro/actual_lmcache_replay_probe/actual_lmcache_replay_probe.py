#!/usr/bin/env python3
"""
Read-only replay benchmark over an existing LMCache disk cache.

The benchmark never writes, renames, truncates, or deletes anything beneath
--cache-root. Synthetic writer traffic is created under a separate, uniquely
marked scratch directory on the same filesystem.

Typical usage:

  # Login node: build a manifest without reading file contents.
  python3 actual_lmcache_replay_probe.py inventory \
      --cache-root /mnt/shared/.../lmcache_vllm/hotpotqa_wo_gnn_17383 \
      --source-log /path/to/kvaware_h100_wo_gnn_17383.out \
      --output-root /mnt/shared/.../actual_lmcache_replay_probe

  # H100 node: baseline, six-writer stress, then recovery.
  python3 actual_lmcache_replay_probe.py run \
      --manifest /mnt/shared/.../actual_lmcache_replay_probe/manifest.json \
      --output-root /mnt/shared/.../actual_lmcache_replay_probe \
      --scratch-root /mnt/shared/.../actual_lmcache_replay_scratch
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import multiprocessing as mp
import os
import platform
import queue
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

MIB = 1024 * 1024
GIB = 1024 * MIB
MARKER_NAME = ".actual_lmcache_replay_scratch"


# ---------------------------------------------------------------------------
# Logging and statistics
# ---------------------------------------------------------------------------

def append_json(log_path: str | Path, event: str, **fields: object) -> None:
    record = {
        "event": event,
        "wall": time.time(),
        "mono": time.monotonic(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        **fields,
    }
    with open(log_path, "a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def read_proc_io(pid: int | None = None) -> dict[str, int]:
    target = os.getpid() if pid is None else pid
    result: dict[str, int] = {}
    try:
        with open(f"/proc/{target}/io", encoding="utf-8") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                result[key.strip()] = int(value.strip())
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return result


def delta_dict(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in set(before) | set(after)
    }


def read_meminfo() -> dict[str, int]:
    wanted = {
        "MemTotal",
        "MemAvailable",
        "Cached",
        "Dirty",
        "Writeback",
        "WritebackTmp",
        "SReclaimable",
    }
    result: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, rest = line.split(":", 1)
                if key in wanted:
                    result[key] = int(rest.strip().split()[0]) * 1024
    except OSError:
        pass
    return result


def read_vmstat() -> dict[str, int]:
    wanted = {
        "nr_dirty",
        "nr_writeback",
        "nr_writeback_temp",
        "pgpgin",
        "pgpgout",
        "pswpin",
        "pswpout",
    }
    result: dict[str, int] = {}
    try:
        with open("/proc/vmstat", encoding="utf-8") as handle:
            for line in handle:
                key, value = line.split()
                if key in wanted:
                    result[key] = int(value)
    except OSError:
        pass
    return result


def read_netdev() -> dict[str, Any]:
    total_rx = 0
    total_tx = 0
    interfaces: dict[str, dict[str, int]] = {}
    try:
        with open("/proc/net/dev", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                interface, rest = line.split(":", 1)
                interface = interface.strip()
                if interface == "lo":
                    continue
                columns = rest.split()
                if len(columns) < 16:
                    continue
                rx_bytes = int(columns[0])
                tx_bytes = int(columns[8])
                total_rx += rx_bytes
                total_tx += tx_bytes
                interfaces[interface] = {
                    "rx_bytes": rx_bytes,
                    "tx_bytes": tx_bytes,
                    "rx_errors": int(columns[2]),
                    "rx_dropped": int(columns[3]),
                    "tx_errors": int(columns[10]),
                    "tx_dropped": int(columns[11]),
                }
    except OSError:
        pass
    return {
        "net_rx_bytes": total_rx,
        "net_tx_bytes": total_tx,
        "interfaces": interfaces,
    }


def parse_named_counter_file(path: str, section: str) -> dict[str, int]:
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return {}
    for index in range(len(lines) - 1):
        if not lines[index].startswith(section + ":"):
            continue
        if not lines[index + 1].startswith(section + ":"):
            continue
        names = lines[index].split()[1:]
        values = lines[index + 1].split()[1:]
        if len(names) != len(values):
            return {}
        result: dict[str, int] = {}
        for name, value in zip(names, values):
            try:
                result[name] = int(value)
            except ValueError:
                pass
        return result
    return {}


def read_tcp_counters() -> dict[str, int]:
    tcp = parse_named_counter_file("/proc/net/snmp", "Tcp")
    ext = parse_named_counter_file("/proc/net/netstat", "TcpExt")
    tcp_wanted = {
        "ActiveOpens",
        "PassiveOpens",
        "AttemptFails",
        "EstabResets",
        "InSegs",
        "OutSegs",
        "RetransSegs",
        "InErrs",
        "OutRsts",
    }
    ext_wanted = {
        "TCPLostRetransmit",
        "TCPFastRetrans",
        "TCPTimeouts",
        "TCPSynRetrans",
        "TCPSpuriousRTOs",
        "TCPDSACKOldSent",
        "TCPDSACKRecv",
    }
    result = {
        f"tcp_{key}": value for key, value in tcp.items() if key in tcp_wanted
    }
    result.update(
        {f"tcpext_{key}": value for key, value in ext.items() if key in ext_wanted}
    )
    return result


def read_nfs_client_stats() -> dict[str, Any]:
    try:
        lines = Path("/proc/net/rpc/nfs").read_text().splitlines()
    except OSError as exc:
        return {"available": False, "error": repr(exc)}
    records: dict[str, list[int]] = {}
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        values: list[int] = []
        for value in parts[1:]:
            try:
                values.append(int(value))
            except ValueError:
                pass
        records[parts[0]] = values
    return {"available": True, "records": records}


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "output": completed.stdout[-30000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": repr(exc)}


def sampler_worker(stop_event: Any, log_path: str, interval_s: float) -> None:
    previous_net = read_netdev()
    previous_vm = read_vmstat()
    previous_tcp = read_tcp_counters()
    append_json(
        log_path,
        "sampler_start",
        meminfo=read_meminfo(),
        vmstat=previous_vm,
        net=previous_net,
        tcp=previous_tcp,
        nfs=read_nfs_client_stats(),
    )
    while not stop_event.wait(interval_s):
        net = read_netdev()
        vm = read_vmstat()
        tcp = read_tcp_counters()
        append_json(
            log_path,
            "system_sample",
            meminfo=read_meminfo(),
            vmstat=vm,
            vmstat_delta=delta_dict(vm, previous_vm),
            net=net,
            net_rx_delta=net.get("net_rx_bytes", 0)
            - previous_net.get("net_rx_bytes", 0),
            net_tx_delta=net.get("net_tx_bytes", 0)
            - previous_net.get("net_tx_bytes", 0),
            tcp=tcp,
            tcp_delta=delta_dict(tcp, previous_tcp),
            nfs=read_nfs_client_stats(),
        )
        previous_net = net
        previous_vm = vm
        previous_tcp = tcp
    append_json(log_path, "sampler_done")


def summarize_sampler(path: Path) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("event") in {"sampler_start", "system_sample"}:
                    samples.append(record)
    except OSError:
        return {}
    if len(samples) < 2:
        return {}

    first = samples[0]
    last = samples[-1]
    elapsed = max(float(last["mono"]) - float(first["mono"]), 1e-9)
    first_net = first.get("net", {})
    last_net = last.get("net", {})
    first_tcp = first.get("tcp", {})
    last_tcp = last.get("tcp", {})

    dirty_values = [
        int(sample.get("meminfo", {}).get("Dirty", 0)) for sample in samples
    ]
    writeback_values = [
        int(sample.get("meminfo", {}).get("Writeback", 0)) for sample in samples
    ]

    first_rpc = first.get("nfs", {}).get("records", {}).get("rpc", [])
    last_rpc = last.get("nfs", {}).get("records", {}).get("rpc", [])
    nfs_calls_delta = None
    nfs_retrans_delta = None
    if len(first_rpc) >= 2 and len(last_rpc) >= 2:
        nfs_calls_delta = last_rpc[0] - first_rpc[0]
        nfs_retrans_delta = last_rpc[1] - first_rpc[1]

    return {
        "elapsed_s": elapsed,
        "average_net_rx_mib_per_s": (
            last_net.get("net_rx_bytes", 0) - first_net.get("net_rx_bytes", 0)
        )
        / MIB
        / elapsed,
        "average_net_tx_mib_per_s": (
            last_net.get("net_tx_bytes", 0) - first_net.get("net_tx_bytes", 0)
        )
        / MIB
        / elapsed,
        "peak_dirty_mib": max(dirty_values, default=0) / MIB,
        "peak_writeback_mib": max(writeback_values, default=0) / MIB,
        "tcp_retrans_segments_delta": (
            last_tcp.get("tcp_RetransSegs", 0)
            - first_tcp.get("tcp_RetransSegs", 0)
        ),
        "tcp_timeouts_delta": (
            last_tcp.get("tcpext_TCPTimeouts", 0)
            - first_tcp.get("tcpext_TCPTimeouts", 0)
        ),
        "nfs_rpc_calls_delta": nfs_calls_delta,
        "nfs_rpc_retrans_delta": nfs_retrans_delta,
    }


# ---------------------------------------------------------------------------
# LMCache filename parsing and inventory
# ---------------------------------------------------------------------------

def parse_lmcache_filename(path: Path) -> dict[str, Any] | None:
    """
    Parse:
      model@tp_size@rank@chunk_hash@dtype.pt

    rsplit() is used so the model portion may contain punctuation.
    """
    if path.suffix != ".pt":
        return None
    parts = path.stem.rsplit("@", 4)
    if len(parts) != 5:
        return None
    model, tp_text, rank_text, chunk_hash, dtype = parts
    try:
        tp_size = int(tp_text)
        rank = int(rank_text)
    except ValueError:
        return None
    return {
        "model": model,
        "tp_size": tp_size,
        "rank": rank,
        "chunk_hash": chunk_hash,
        "dtype": dtype,
    }


def pair_identity(parsed: dict[str, Any], relative_parent: str) -> str:
    return "|".join(
        [
            relative_parent,
            str(parsed["model"]),
            str(parsed["tp_size"]),
            str(parsed["chunk_hash"]),
            str(parsed["dtype"]),
        ]
    )


def parse_source_log_order(source_log: Path) -> list[str]:
    path_pattern = re.compile(r"\bpath=(\S+?\.pt)(?:\s|$)")
    ordered: list[str] = []
    seen: set[str] = set()
    with open(source_log, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "KVIO_FILE_READ" not in line:
                continue
            match = path_pattern.search(line)
            if not match:
                continue
            basename = Path(match.group(1)).name
            parsed = parse_lmcache_filename(Path(basename))
            if parsed is None:
                continue
            # Source logs use a flat cache directory in the observed run.
            identity = pair_identity(parsed, ".")
            if identity not in seen:
                seen.add(identity)
                ordered.append(identity)
    return ordered


def cache_snapshot(cache_root: Path) -> dict[str, int]:
    file_count = 0
    total_bytes = 0
    latest_mtime_ns = 0
    for entry in os.scandir(cache_root):
        try:
            if not entry.is_file(follow_symlinks=False) or not entry.name.endswith(".pt"):
                continue
            stat = entry.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        file_count += 1
        total_bytes += stat.st_size
        latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
    return {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "latest_mtime_ns": latest_mtime_ns,
    }


def inventory_command(args: argparse.Namespace) -> int:
    cache_root = Path(args.cache_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    inventory_log = output_root / "inventory.jsonl"

    if not cache_root.is_dir():
        raise FileNotFoundError(f"LMCache directory does not exist: {cache_root}")

    expected_size = args.expected_file_mib * MIB
    append_json(
        inventory_log,
        "inventory_start",
        cache_root=str(cache_root),
        expected_size=expected_size,
        source_log=args.source_log,
    )

    groups: dict[str, dict[int, dict[str, Any]]] = {}
    unparsed = 0
    wrong_size = 0

    for directory, _, filenames in os.walk(cache_root):
        directory_path = Path(directory)
        relative_parent = str(directory_path.relative_to(cache_root)) or "."
        for filename in filenames:
            path = directory_path / filename
            parsed = parse_lmcache_filename(path)
            if parsed is None:
                unparsed += 1
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            if stat.st_size != expected_size:
                wrong_size += 1
                continue
            identity = pair_identity(parsed, relative_parent)
            groups.setdefault(identity, {})[int(parsed["rank"])] = {
                "path": str(path),
                "relative_path": str(path.relative_to(cache_root)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "inode": stat.st_ino,
                "rank": int(parsed["rank"]),
                "tp_size": int(parsed["tp_size"]),
                "model": parsed["model"],
                "chunk_hash": parsed["chunk_hash"],
                "dtype": parsed["dtype"],
                "relative_parent": relative_parent,
            }

    pairs: list[dict[str, Any]] = []
    incomplete = 0
    for identity, ranks in groups.items():
        if 0 not in ranks or 1 not in ranks:
            incomplete += 1
            continue
        pair = {
            "identity": identity,
            "rank0": ranks[0],
            "rank1": ranks[1],
            "pair_mtime_ns": min(ranks[0]["mtime_ns"], ranks[1]["mtime_ns"]),
        }
        pairs.append(pair)

    pairs.sort(key=lambda item: (item["pair_mtime_ns"], item["identity"]))

    observed_order: list[str] = []
    source_log = Path(args.source_log).resolve() if args.source_log else None
    if source_log is not None:
        if not source_log.is_file():
            raise FileNotFoundError(f"Source log does not exist: {source_log}")
        observed_order = parse_source_log_order(source_log)

    observed_index = {identity: index for index, identity in enumerate(observed_order)}
    observed_pairs = [
        pair for pair in pairs if pair["identity"] in observed_index
    ]
    observed_pairs.sort(key=lambda pair: observed_index[pair["identity"]])
    unobserved_pairs = [
        pair for pair in pairs if pair["identity"] not in observed_index
    ]
    ordered_pairs = observed_pairs + unobserved_pairs
    for ordinal, pair in enumerate(ordered_pairs):
        pair["ordinal"] = ordinal
        pair["observed_in_source_log"] = pair["identity"] in observed_index
        pair["source_log_ordinal"] = observed_index.get(pair["identity"])

    manifest = {
        "version": 1,
        "created_wall": time.time(),
        "cache_root": str(cache_root),
        "expected_file_size": expected_size,
        "source_log": str(source_log) if source_log else None,
        "source_log_unique_pairs": len(observed_pairs),
        "pair_count": len(ordered_pairs),
        "unparsed_files": unparsed,
        "wrong_size_files": wrong_size,
        "incomplete_pair_groups": incomplete,
        "cache_snapshot": cache_snapshot(cache_root),
        "pairs": ordered_pairs,
    }

    manifest_path = Path(args.manifest).resolve() if args.manifest else (
        output_root / "manifest.json"
    )
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, manifest_path)

    append_json(
        inventory_log,
        "inventory_done",
        manifest=str(manifest_path),
        pair_count=len(ordered_pairs),
        observed_pairs=len(observed_pairs),
        incomplete_pair_groups=incomplete,
        unparsed_files=unparsed,
        wrong_size_files=wrong_size,
    )
    print(json.dumps({
        "manifest": str(manifest_path),
        "pair_count": len(ordered_pairs),
        "observed_source_log_pairs": len(observed_pairs),
        "total_paired_gib": len(ordered_pairs) * expected_size * 2 / GIB,
        "incomplete_pair_groups": incomplete,
        "unparsed_files": unparsed,
        "wrong_size_files": wrong_size,
    }, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Read and write workers
# ---------------------------------------------------------------------------

def fadvise_dontneed(fd: int) -> bool:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False


def reader_worker(
    rank: int,
    entries: list[dict[str, Any]],
    expected_size: int,
    use_cold_hint: bool,
    start_event: Any,
    stop_event: Any,
    summary_queue: Any,
    log_path: str,
) -> None:
    buffer = bytearray(expected_size)
    latencies: list[float] = []
    physical_latencies: list[float] = []
    physical_bandwidths: list[float] = []
    cache_latencies: list[float] = []
    bytes_read = 0
    operations = 0
    physical_operations = 0
    cache_operations = 0
    short_reads = 0
    errors = 0

    append_json(
        log_path,
        "reader_ready",
        rank=rank,
        entry_count=len(entries),
        expected_size=expected_size,
        use_cold_hint=use_cold_hint,
    )
    start_event.wait()
    wall_start = time.monotonic()
    append_json(log_path, "reader_start", rank=rank)

    for entry in entries:
        if stop_event.is_set():
            break
        path = Path(entry["path"])
        handle = None
        try:
            stat_before = path.stat()
            if stat_before.st_size != expected_size:
                raise RuntimeError(
                    f"File changed size: {path}: {stat_before.st_size} != {expected_size}"
                )

            io_before = read_proc_io()
            open_start = time.monotonic()
            handle = open(path, "rb")
            open_s = time.monotonic() - open_start

            cold_hint_before = False
            if use_cold_hint:
                cold_hint_before = fadvise_dontneed(handle.fileno())

            read_start = time.monotonic()
            nread = handle.readinto(buffer)
            read_s = time.monotonic() - read_start

            cold_hint_after = False
            if use_cold_hint:
                cold_hint_after = fadvise_dontneed(handle.fileno())

            close_start = time.monotonic()
            handle.close()
            handle = None
            close_s = time.monotonic() - close_start

            io_delta = delta_dict(read_proc_io(), io_before)
            physical_bytes = int(io_delta.get("read_bytes", 0))
            physical = physical_bytes >= expected_size // 2
            bandwidth = nread / MIB / read_s if read_s else None

            operations += 1
            bytes_read += nread
            short_reads += int(nread != expected_size)
            latencies.append(read_s)
            if physical:
                physical_operations += 1
                physical_latencies.append(read_s)
                if bandwidth is not None:
                    physical_bandwidths.append(bandwidth)
            else:
                cache_operations += 1
                cache_latencies.append(read_s)

            append_json(
                log_path,
                "reader_file",
                rank=rank,
                operation=operations,
                ordinal=entry.get("ordinal"),
                source_log_ordinal=entry.get("source_log_ordinal"),
                observed_in_source_log=entry.get("observed_in_source_log"),
                identity=entry.get("identity"),
                chunk_hash=entry.get("chunk_hash"),
                path=str(path),
                bytes_expected=expected_size,
                bytes_read=nread,
                short_read=nread != expected_size,
                open_s=open_s,
                read_s=read_s,
                close_s=close_s,
                bandwidth_mib_per_s=bandwidth,
                physical=physical,
                proc_read_bytes_delta=physical_bytes,
                proc_rchar_delta=io_delta.get("rchar", 0),
                proc_syscr_delta=io_delta.get("syscr", 0),
                cold_hint_before=cold_hint_before,
                cold_hint_after=cold_hint_after,
                original_mtime_ns=entry.get("mtime_ns"),
                current_mtime_ns=stat_before.st_mtime_ns,
            )
        except BaseException as exc:
            errors += 1
            append_json(
                log_path,
                "reader_error",
                rank=rank,
                path=str(path),
                error=repr(exc),
                traceback=traceback.format_exc(),
            )
        finally:
            if handle is not None:
                handle.close()

    elapsed = time.monotonic() - wall_start
    summary = {
        "kind": "reader",
        "rank": rank,
        "operations": operations,
        "bytes": bytes_read,
        "elapsed_s": elapsed,
        "aggregate_mib_per_s": bytes_read / MIB / elapsed if elapsed else None,
        "short_reads": short_reads,
        "errors": errors,
        "physical_operations": physical_operations,
        "cache_operations": cache_operations,
        "median_read_s": statistics.median(latencies) if latencies else None,
        "p95_read_s": percentile(latencies, 0.95),
        "p99_read_s": percentile(latencies, 0.99),
        "max_read_s": max(latencies) if latencies else None,
        "physical_median_read_s": (
            statistics.median(physical_latencies) if physical_latencies else None
        ),
        "physical_median_mib_per_s": (
            statistics.median(physical_bandwidths)
            if physical_bandwidths
            else None
        ),
        "physical_p95_read_s": percentile(physical_latencies, 0.95),
        "cache_median_read_s": (
            statistics.median(cache_latencies) if cache_latencies else None
        ),
    }
    append_json(log_path, "reader_done", **summary)
    summary_queue.put(summary)


def write_all(fd: int, payload: bytes) -> int:
    view = memoryview(payload)
    total = 0
    while total < len(view):
        written = os.write(fd, view[total:])
        if written <= 0:
            raise OSError(f"os.write returned {written}")
        total += written
    return total


def writer_worker(
    writer_id: int,
    scratch_dir: str,
    file_size: int,
    max_files: int,
    start_event: Any,
    stop_event: Any,
    summary_queue: Any,
    log_path: str,
) -> None:
    payload = os.urandom(file_size)
    operations = 0
    bytes_written = 0
    errors = 0
    latencies: list[float] = []
    write_latencies: list[float] = []
    close_latencies: list[float] = []
    last_path: str | None = None

    append_json(
        log_path,
        "writer_ready",
        writer_id=writer_id,
        scratch_dir=scratch_dir,
        file_size=file_size,
        max_files=max_files,
    )
    start_event.wait()
    wall_start = time.monotonic()
    append_json(log_path, "writer_start", writer_id=writer_id)

    for index in range(max_files):
        if stop_event.is_set():
            break
        path = Path(scratch_dir) / f"writer_{writer_id:02d}_{index:06d}.bin"
        fd = -1
        try:
            io_before = read_proc_io()

            open_start = time.monotonic()
            fd = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            open_s = time.monotonic() - open_start

            write_start = time.monotonic()
            written = write_all(fd, payload)
            write_s = time.monotonic() - write_start

            close_start = time.monotonic()
            os.close(fd)
            fd = -1
            close_s = time.monotonic() - close_start

            elapsed = open_s + write_s + close_s
            io_delta = delta_dict(read_proc_io(), io_before)
            operations += 1
            bytes_written += written
            latencies.append(elapsed)
            write_latencies.append(write_s)
            close_latencies.append(close_s)
            last_path = str(path)

            append_json(
                log_path,
                "writer_file",
                writer_id=writer_id,
                operation=operations,
                path=str(path),
                bytes=written,
                open_s=open_s,
                write_s=write_s,
                close_s=close_s,
                elapsed_s=elapsed,
                application_mib_per_s=written / MIB / elapsed if elapsed else None,
                proc_write_bytes_delta=io_delta.get("write_bytes", 0),
                proc_wchar_delta=io_delta.get("wchar", 0),
                proc_syscw_delta=io_delta.get("syscw", 0),
            )
        except BaseException as exc:
            errors += 1
            append_json(
                log_path,
                "writer_error",
                writer_id=writer_id,
                path=str(path),
                error=repr(exc),
                traceback=traceback.format_exc(),
            )
            time.sleep(0.05)
        finally:
            if fd >= 0:
                os.close(fd)

    if operations >= max_files and not stop_event.is_set():
        append_json(
            log_path,
            "writer_cap_reached",
            writer_id=writer_id,
            max_files=max_files,
            total_gib=bytes_written / GIB,
        )
        stop_event.wait()

    elapsed = time.monotonic() - wall_start
    summary = {
        "kind": "writer",
        "writer_id": writer_id,
        "operations": operations,
        "bytes": bytes_written,
        "elapsed_s": elapsed,
        "aggregate_mib_per_s": bytes_written / MIB / elapsed if elapsed else None,
        "errors": errors,
        "median_operation_s": statistics.median(latencies) if latencies else None,
        "p95_operation_s": percentile(latencies, 0.95),
        "max_operation_s": max(latencies) if latencies else None,
        "median_write_s": (
            statistics.median(write_latencies) if write_latencies else None
        ),
        "median_close_s": (
            statistics.median(close_latencies) if close_latencies else None
        ),
        "last_path": last_path,
    }
    append_json(log_path, "writer_done", **summary)
    summary_queue.put(summary)


# ---------------------------------------------------------------------------
# Phase orchestration
# ---------------------------------------------------------------------------

def collect_summaries(summary_queue: Any, expected: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    deadline = time.monotonic() + 20
    while len(results) < expected and time.monotonic() < deadline:
        try:
            results.append(summary_queue.get(timeout=0.5))
        except queue.Empty:
            continue
    return results


def build_rank_entries(pairs: list[dict[str, Any]], rank: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    rank_key = f"rank{rank}"
    for pair in pairs:
        file_entry = dict(pair[rank_key])
        file_entry.update(
            {
                "identity": pair["identity"],
                "ordinal": pair.get("ordinal"),
                "source_log_ordinal": pair.get("source_log_ordinal"),
                "observed_in_source_log": pair.get("observed_in_source_log"),
            }
        )
        entries.append(file_entry)
    return entries


def run_phase(
    ctx: Any,
    run_dir: Path,
    phase_index: int,
    name: str,
    pairs: list[dict[str, Any]],
    expected_size: int,
    use_cold_hint: bool,
    timeout_s: float,
    writer_count: int,
    writer_headstart_s: float,
    scratch_phase_dir: Path | None,
    writer_max_files_each: int,
    sample_interval_s: float,
) -> dict[str, Any]:
    phase_dir = run_dir / f"phase_{phase_index:02d}_{name}"
    phase_dir.mkdir()
    parent_log = phase_dir / "parent.jsonl"

    start_event = ctx.Event()
    stop_event = ctx.Event()
    sampler_stop = ctx.Event()
    summary_queue = ctx.Queue()

    sampler = ctx.Process(
        target=sampler_worker,
        args=(sampler_stop, str(phase_dir / "sampler.jsonl"), sample_interval_s),
        name=f"sampler-{phase_index}",
    )
    sampler.start()

    processes: list[Any] = []
    writers: list[Any] = []

    if writer_count:
        if scratch_phase_dir is None:
            raise ValueError("scratch_phase_dir is required when writers are enabled")
        scratch_phase_dir.mkdir(parents=True, exist_ok=False)
        for writer_id in range(writer_count):
            process = ctx.Process(
                target=writer_worker,
                args=(
                    writer_id,
                    str(scratch_phase_dir),
                    expected_size,
                    writer_max_files_each,
                    start_event,
                    stop_event,
                    summary_queue,
                    str(phase_dir / f"writer_{writer_id}.jsonl"),
                ),
                name=f"writer-{phase_index}-{writer_id}",
            )
            process.start()
            writers.append(process)
            processes.append(process)

    readers: list[Any] = []
    reader_start_event = ctx.Event()
    for rank in (0, 1):
        entries = build_rank_entries(pairs, rank)
        process = ctx.Process(
            target=reader_worker,
            args=(
                rank,
                entries,
                expected_size,
                use_cold_hint,
                reader_start_event,
                stop_event,
                summary_queue,
                str(phase_dir / f"reader_rank{rank}.jsonl"),
            ),
            name=f"reader-{phase_index}-rank{rank}",
        )
        process.start()
        readers.append(process)
        processes.append(process)

    append_json(
        parent_log,
        "phase_start",
        phase_index=phase_index,
        name=name,
        pair_count=len(pairs),
        timeout_s=timeout_s,
        writer_count=writer_count,
        writer_headstart_s=writer_headstart_s,
        writer_max_files_each=writer_max_files_each,
        use_cold_hint=use_cold_hint,
        first_pair_ordinal=pairs[0].get("ordinal") if pairs else None,
        last_pair_ordinal=pairs[-1].get("ordinal") if pairs else None,
        observed_source_log_pairs=sum(
            int(bool(pair.get("observed_in_source_log"))) for pair in pairs
        ),
    )

    start_event.set()
    if writer_count and writer_headstart_s > 0:
        time.sleep(writer_headstart_s)

    reader_wall_start = time.monotonic()
    reader_start_event.set()
    deadline = reader_wall_start + timeout_s

    timed_out = False
    while True:
        if all(not reader.is_alive() for reader in readers):
            break
        if time.monotonic() >= deadline:
            timed_out = True
            stop_event.set()
            break
        time.sleep(0.2)

    for reader in readers:
        reader.join(timeout=30)
        if reader.is_alive():
            append_json(
                parent_log,
                "forced_reader_terminate",
                process=reader.name,
                pid=reader.pid,
            )
            reader.terminate()
            reader.join(timeout=5)

    reader_wall_elapsed = time.monotonic() - reader_wall_start
    stop_event.set()

    for writer in writers:
        writer.join(timeout=60)
        if writer.is_alive():
            append_json(
                parent_log,
                "forced_writer_terminate",
                process=writer.name,
                pid=writer.pid,
            )
            writer.terminate()
            writer.join(timeout=5)

    summaries = collect_summaries(summary_queue, len(readers) + len(writers))
    sampler_stop.set()
    sampler.join(timeout=10)
    if sampler.is_alive():
        sampler.terminate()
        sampler.join(timeout=5)

    reader_summaries = [
        summary for summary in summaries if summary.get("kind") == "reader"
    ]
    writer_summaries = [
        summary for summary in summaries if summary.get("kind") == "writer"
    ]
    total_reader_bytes = sum(int(item.get("bytes", 0)) for item in reader_summaries)
    total_writer_bytes = sum(int(item.get("bytes", 0)) for item in writer_summaries)

    summary = {
        "phase_index": phase_index,
        "name": name,
        "timed_out": timed_out,
        "pair_count_requested": len(pairs),
        "reader_wall_elapsed_s": reader_wall_elapsed,
        "writer_count": writer_count,
        "reader_total_bytes": total_reader_bytes,
        "reader_aggregate_mib_per_s": (
            total_reader_bytes / MIB / reader_wall_elapsed
            if reader_wall_elapsed
            else None
        ),
        "writer_total_bytes": total_writer_bytes,
        "writer_total_gib": total_writer_bytes / GIB,
        "reader_short_reads": sum(
            int(item.get("short_reads", 0)) for item in reader_summaries
        ),
        "reader_errors": sum(
            int(item.get("errors", 0)) for item in reader_summaries
        ),
        "writer_errors": sum(
            int(item.get("errors", 0)) for item in writer_summaries
        ),
        "physical_read_operations": sum(
            int(item.get("physical_operations", 0)) for item in reader_summaries
        ),
        "cache_read_operations": sum(
            int(item.get("cache_operations", 0)) for item in reader_summaries
        ),
        "reader_summaries": reader_summaries,
        "writer_summaries": writer_summaries,
        "child_exitcodes": {
            process.name: process.exitcode for process in processes
        },
        "sampler_summary": summarize_sampler(phase_dir / "sampler.jsonl"),
    }
    append_json(parent_log, "phase_done", **summary)
    return summary


def select_phase_pairs(
    pairs: list[dict[str, Any]],
    pairs_per_phase: int,
    phase_count: int,
    partition: str,
) -> list[list[dict[str, Any]]]:
    available_per_phase = len(pairs) // phase_count
    count = min(pairs_per_phase, available_per_phase)
    if count < 4:
        raise RuntimeError(
            f"Only {len(pairs)} paired files are available; at least "
            f"{phase_count * 4} pairs are required."
        )
    selected = pairs[: count * phase_count]
    if partition == "round_robin":
        return [selected[index::phase_count] for index in range(phase_count)]
    if partition == "contiguous":
        return [
            selected[index * count : (index + 1) * count]
            for index in range(phase_count)
        ]
    raise ValueError(f"Unknown partition mode: {partition}")


def cache_is_stable(cache_root: Path, wait_s: float) -> tuple[bool, dict[str, Any]]:
    before = cache_snapshot(cache_root)
    time.sleep(wait_s)
    after = cache_snapshot(cache_root)
    return before == after, {"before": before, "after": after, "wait_s": wait_s}


def ensure_safe_paths(cache_root: Path, scratch_root: Path, output_root: Path) -> None:
    cache_root = cache_root.resolve()
    scratch_root = scratch_root.resolve()
    output_root = output_root.resolve()

    if scratch_root == cache_root:
        raise RuntimeError("scratch-root must not equal cache-root")
    if cache_root in scratch_root.parents:
        raise RuntimeError("scratch-root must not be inside cache-root")
    if scratch_root in cache_root.parents:
        raise RuntimeError("cache-root must not be inside scratch-root")
    if output_root == cache_root or cache_root in output_root.parents:
        raise RuntimeError("output-root must not be inside cache-root")


def fsync_last_writer_files(phases: list[dict[str, Any]], log_path: Path) -> dict[str, Any]:
    paths: list[str] = []
    for phase in phases:
        for writer in phase.get("writer_summaries", []):
            if writer.get("last_path"):
                paths.append(str(writer["last_path"]))
    latencies: list[float] = []
    errors = 0
    for path in paths:
        fd = -1
        try:
            start = time.monotonic()
            fd = os.open(path, os.O_RDONLY)
            os.fsync(fd)
            latency = time.monotonic() - start
            latencies.append(latency)
            append_json(log_path, "final_sample_fsync", path=path, fsync_s=latency)
        except OSError as exc:
            errors += 1
            append_json(
                log_path,
                "final_sample_fsync_error",
                path=path,
                error=repr(exc),
            )
        finally:
            if fd >= 0:
                os.close(fd)
    return {
        "files": len(paths),
        "errors": errors,
        "total_s": sum(latencies),
        "median_s": statistics.median(latencies) if latencies else None,
        "max_s": max(latencies) if latencies else None,
    }


def print_summary(phases: list[dict[str, Any]]) -> None:
    print(
        "phase name                    timeout read_MiB/s physical cached "
        "phys_med_MiB/s phys_med_s p95_s max_s writer_GiB "
        "netRX netTX peakWB tcpRet nfsRet"
    )
    for phase in phases:
        readers = phase.get("reader_summaries", [])
        physical_rates = [
            float(reader["physical_median_mib_per_s"])
            for reader in readers
            if reader.get("physical_median_mib_per_s") is not None
        ]
        physical_times = [
            float(reader["physical_median_read_s"])
            for reader in readers
            if reader.get("physical_median_read_s") is not None
        ]
        p95s = [
            float(reader["p95_read_s"])
            for reader in readers
            if reader.get("p95_read_s") is not None
        ]
        maxes = [
            float(reader["max_read_s"])
            for reader in readers
            if reader.get("max_read_s") is not None
        ]
        sampler = phase.get("sampler_summary", {})
        print(
            f'{phase["phase_index"]:>5} '
            f'{phase["name"]:<23.23} '
            f'{str(phase.get("timed_out", False)):>7} '
            f'{float(phase.get("reader_aggregate_mib_per_s") or 0):>10.2f} '
            f'{int(phase.get("physical_read_operations", 0)):>8} '
            f'{int(phase.get("cache_read_operations", 0)):>6} '
            f'{statistics.median(physical_rates) if physical_rates else 0:>14.2f} '
            f'{statistics.median(physical_times) if physical_times else 0:>10.3f} '
            f'{max(p95s) if p95s else 0:>6.3f} '
            f'{max(maxes) if maxes else 0:>6.3f} '
            f'{float(phase.get("writer_total_gib") or 0):>10.2f} '
            f'{float(sampler.get("average_net_rx_mib_per_s") or 0):>5.0f} '
            f'{float(sampler.get("average_net_tx_mib_per_s") or 0):>5.0f} '
            f'{float(sampler.get("peak_writeback_mib") or 0):>6.0f} '
            f'{int(sampler.get("tcp_retrans_segments_delta") or 0):>6} '
            f'{int(sampler.get("nfs_rpc_retrans_delta") or 0):>6}'
        )


def run_command(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    cache_root = Path(manifest["cache_root"]).resolve()
    output_root = Path(args.output_root).resolve()
    scratch_root = Path(args.scratch_root).resolve()

    ensure_safe_paths(cache_root, scratch_root, output_root)
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache root no longer exists: {cache_root}")

    stable, stability = cache_is_stable(cache_root, args.stability_check_s)
    if not stable and not args.allow_changing_cache:
        raise RuntimeError(
            "The LMCache directory changed during the stability check. "
            "Do not run this probe while vLLM/LMCache is using the directory. "
            f"Snapshot: {stability}"
        )

    expected_size = int(manifest["expected_file_size"])
    pairs = list(manifest["pairs"])
    phase_pair_sets = select_phase_pairs(
        pairs,
        args.pairs_per_phase,
        phase_count=3,
        partition=args.partition,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch_marker = scratch_root / MARKER_NAME
    if not scratch_marker.exists():
        scratch_marker.write_text(
            "Created by actual_lmcache_replay_probe.py. "
            "Only run-specific child directories are deleted.\n"
        )

    run_id = f"{socket.gethostname()}_{int(time.time())}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True)
    scratch_run_dir = scratch_root / run_id
    scratch_run_dir.mkdir(parents=True, exist_ok=False)
    (scratch_run_dir / MARKER_NAME).write_text(run_id + "\n")

    run_log = run_dir / "run.jsonl"
    environment = {
        "run_id": run_id,
        "created_wall": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "manifest": str(manifest_path),
        "cache_root": str(cache_root),
        "output_root": str(output_root),
        "scratch_root": str(scratch_root),
        "scratch_run_dir": str(scratch_run_dir),
        "cache_stability": stability,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "df_cache": command_output(["df", "-hT", str(cache_root)]),
        "mount": command_output(["mount"]),
        "nfsstat_m": command_output(["nfsstat", "-m"]),
        "nfsstat_c": command_output(["nfsstat", "-c"]),
        "ip_link": command_output(["ip", "-s", "link"]),
        "meminfo": read_meminfo(),
        "tcp": read_tcp_counters(),
        "nfs": read_nfs_client_stats(),
    }
    (run_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, default=str) + "\n"
    )

    total_writer_bytes_cap = args.writer_max_scratch_gib * GIB
    max_files_total = max(1, int(total_writer_bytes_cap // expected_size))
    writer_max_files_each = max(1, math.ceil(max_files_total / args.writers))

    append_json(
        run_log,
        "run_start",
        run_id=run_id,
        cache_root=str(cache_root),
        pair_count_manifest=len(pairs),
        pairs_per_phase=len(phase_pair_sets[0]),
        partition=args.partition,
        expected_size=expected_size,
        use_cold_hint=not args.no_cold_hint,
        writer_count=args.writers,
        writer_max_scratch_gib=args.writer_max_scratch_gib,
        writer_max_files_each=writer_max_files_each,
        phase_timeout_s=args.phase_timeout_s,
    )

    ctx = mp.get_context("spawn")
    phases: list[dict[str, Any]] = []

    phase_specs = [
        ("actual_baseline", 0, None),
        (
            "actual_6w_unique",
            args.writers,
            scratch_run_dir / "stress_unique",
        ),
        ("actual_recovery", 0, None),
    ]

    try:
        for phase_index, ((name, writer_count, scratch_dir), phase_pairs) in enumerate(
            zip(phase_specs, phase_pair_sets, strict=True)
        ):
            print(
                f"Starting phase {phase_index}: {name}; "
                f"{len(phase_pairs)} actual LMCache pairs",
                flush=True,
            )
            summary = run_phase(
                ctx=ctx,
                run_dir=run_dir,
                phase_index=phase_index,
                name=name,
                pairs=phase_pairs,
                expected_size=expected_size,
                use_cold_hint=not args.no_cold_hint,
                timeout_s=args.phase_timeout_s,
                writer_count=writer_count,
                writer_headstart_s=args.writer_headstart_s,
                scratch_phase_dir=scratch_dir,
                writer_max_files_each=writer_max_files_each,
                sample_interval_s=args.sample_interval_s,
            )
            phases.append(summary)
            print_summary([summary])

            bad_exitcodes = {
                name: code
                for name, code in summary["child_exitcodes"].items()
                if code not in (0, None)
            }
            if bad_exitcodes:
                raise RuntimeError(f"Child process failures: {bad_exitcodes}")

        final_fsync = fsync_last_writer_files(phases, run_log)
        summary_document = {
            "run_id": run_id,
            "manifest": str(manifest_path),
            "cache_root": str(cache_root),
            "scratch_run_dir": str(scratch_run_dir),
            "phases": phases,
            "final_sample_fsync": final_fsync,
        }
        summary_path = run_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary_document, indent=2, default=str) + "\n"
        )
        append_json(
            run_log,
            "run_done",
            summary=str(summary_path),
            final_sample_fsync=final_fsync,
        )
        print("\n=== FINAL SUMMARY ===")
        print_summary(phases)
        print(f"\nRun directory: {run_dir}")
        print(f"Summary: {summary_path}")
    finally:
        if args.keep_scratch:
            append_json(
                run_log,
                "scratch_kept",
                scratch_run_dir=str(scratch_run_dir),
            )
            print(f"Scratch retained: {scratch_run_dir}", file=sys.stderr)
        else:
            marker = scratch_run_dir / MARKER_NAME
            if scratch_run_dir.exists() and marker.is_file():
                cleanup_start = time.monotonic()
                shutil.rmtree(scratch_run_dir)
                append_json(
                    run_log,
                    "scratch_deleted",
                    scratch_run_dir=str(scratch_run_dir),
                    elapsed_s=time.monotonic() - cleanup_start,
                )
            elif scratch_run_dir.exists():
                append_json(
                    run_log,
                    "scratch_cleanup_refused",
                    scratch_run_dir=str(scratch_run_dir),
                    reason="run marker missing",
                )
                print(
                    f"Refusing to delete unmarked scratch directory: {scratch_run_dir}",
                    file=sys.stderr,
                )

    return 0


def summarize_command(args: argparse.Namespace) -> int:
    document = json.loads(Path(args.summary).read_text())
    print_summary(document["phases"])
    print("\nFinal sampled fsync:")
    print(json.dumps(document.get("final_sample_fsync", {}), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser(
        "inventory",
        help="Scan actual LMCache files and build TP0/TP1 pairs.",
    )
    inventory.add_argument("--cache-root", required=True)
    inventory.add_argument("--output-root", required=True)
    inventory.add_argument("--source-log")
    inventory.add_argument("--manifest")
    inventory.add_argument("--expected-file-mib", type=int, default=80)
    inventory.set_defaults(func=inventory_command)

    run = subparsers.add_parser(
        "run",
        help="Run baseline, six-writer stress, and recovery phases.",
    )
    run.add_argument("--manifest", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--scratch-root", required=True)
    run.add_argument("--pairs-per-phase", type=int, default=400)
    run.add_argument(
        "--partition",
        choices=("round_robin", "contiguous"),
        default="round_robin",
        help=(
            "round_robin balances file age/order across phases while preserving "
            "relative order within each phase."
        ),
    )
    run.add_argument("--phase-timeout-s", type=float, default=240)
    run.add_argument("--writers", type=int, default=6)
    run.add_argument("--writer-headstart-s", type=float, default=10)
    run.add_argument(
        "--writer-max-scratch-gib",
        type=float,
        default=160,
        help="Maximum unique synthetic data created across all writers.",
    )
    run.add_argument("--sample-interval-s", type=float, default=1)
    run.add_argument("--stability-check-s", type=float, default=5)
    run.add_argument(
        "--no-cold-hint",
        action="store_true",
        help=(
            "Disable POSIX_FADV_DONTNEED around buffered readinto(). "
            "The read call itself remains ordinary buffered I/O either way."
        ),
    )
    run.add_argument(
        "--allow-changing-cache",
        action="store_true",
        help="Unsafe: permit execution while the LMCache tree is changing.",
    )
    run.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep synthetic writer files after the benchmark.",
    )
    run.set_defaults(func=run_command)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("summary")
    summarize.set_defaults(func=summarize_command)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

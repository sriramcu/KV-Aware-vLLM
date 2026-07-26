#!/usr/bin/env python3
"""
NFS/LMCache topology probe.

Two-stage usage:

  1. Login node:
       python3 nfs_lmcache_topology_probe.py prepare --root /mnt/shared/.../probe

  2. H100 compute node:
       python3 nfs_lmcache_topology_probe.py run --root /mnt/shared/.../probe

The default run approximates the storage topology observed in the LMCache job:

  * two serialized readers;
  * six concurrent writers;
  * 80 MiB files;
  * ordinary buffered writer calls without fsync in the hot loop;
  * buffered reader using one Python readinto() call per file;
  * O_DIRECT control phase;
  * fixed-file overwrite phases;
  * optional same-file read/write overlap phase.

All detailed records are JSON Lines. Each process writes a separate log file.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import math
import mmap
import multiprocessing as mp
import os
import platform
import queue
import shutil
import socket
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

MIB = 1024 * 1024
GIB = 1024 * MIB


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def append_json(log_path: str | Path, event: str, **fields: object) -> None:
    record = {
        "event": event,
        "wall": time.time(),
        "mono": time.monotonic(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        **fields,
    }
    line = json.dumps(record, sort_keys=True, default=str)
    with open(log_path, "a", encoding="utf-8", buffering=1) as handle:
        handle.write(line + "\n")


def write_all(fd: int, payload: bytes | bytearray | memoryview) -> int:
    view = memoryview(payload)
    total = 0
    while total < len(view):
        written = os.write(fd, view[total:])
        if written <= 0:
            raise OSError(f"os.write returned {written}")
        total += written
    return total


def read_proc_io(pid: int | None = None) -> dict[str, int]:
    target = os.getpid() if pid is None else pid
    values: dict[str, int] = {}
    try:
        with open(f"/proc/{target}/io", encoding="utf-8") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                values[key.strip()] = int(value.strip())
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return values


def delta_dict(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: after.get(key, 0) - before.get(key, 0) for key in set(before) | set(after)}


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
                key, remainder = line.split(":", 1)
                if key in wanted:
                    result[key] = int(remainder.strip().split()[0]) * 1024
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
                fields = rest.split()
                if len(fields) < 16:
                    continue
                rx = int(fields[0])
                tx = int(fields[8])
                total_rx += rx
                total_tx += tx
                interfaces[interface] = {"rx_bytes": rx, "tx_bytes": tx}
    except OSError:
        pass
    return {
        "net_rx_bytes": total_rx,
        "net_tx_bytes": total_tx,
        "net_interfaces": interfaces,
    }


def parse_named_counter_file(path: str, section_name: str) -> dict[str, int]:
    """Parse /proc/net/snmp or /proc/net/netstat two-line named counter sections."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return {}
    for index in range(len(lines) - 1):
        if not lines[index].startswith(section_name + ":"):
            continue
        if not lines[index + 1].startswith(section_name + ":"):
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
                continue
        return result
    return {}


def read_tcp_counters() -> dict[str, int]:
    tcp = parse_named_counter_file("/proc/net/snmp", "Tcp")
    tcpext = parse_named_counter_file("/proc/net/netstat", "TcpExt")
    wanted_tcp = {
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
    wanted_ext = {
        "TCPLostRetransmit",
        "TCPFastRetrans",
        "TCPTimeouts",
        "TCPSynRetrans",
        "TCPSpuriousRTOs",
        "TCPDSACKOldSent",
        "TCPDSACKRecv",
    }
    result = {f"tcp_{key}": value for key, value in tcp.items() if key in wanted_tcp}
    result.update(
        {f"tcpext_{key}": value for key, value in tcpext.items() if key in wanted_ext}
    )
    return result


def read_nfs_client_stats() -> dict[str, Any]:
    """Read raw NFS client counters without assuming a kernel-specific layout."""
    path = Path("/proc/net/rpc/nfs")
    try:
        lines = path.read_text().splitlines()
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


def read_mount_info(target: Path) -> list[str]:
    target_text = str(target.resolve())
    matches: list[str] = []
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            if f" {target_text} " in line or target_text.startswith(line.split()[4].rstrip("/") + "/"):
                matches.append(line)
    except OSError:
        pass
    return matches


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "output": result.stdout[-20000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": repr(exc)}


def fadvise_dontneed(fd: int) -> bool:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------

def create_synced_file(
    path: Path,
    file_size: int,
    chunk: bytes,
    overwrite: bool,
    log_path: Path,
    group: str,
    index: int,
) -> None:
    if path.exists() and not overwrite:
        actual = path.stat().st_size
        if actual != file_size:
            raise RuntimeError(
                f"{path} exists with {actual} bytes; expected {file_size}. "
                "Pass --overwrite to replace it."
            )
        append_json(log_path, "prepare_reuse", group=group, index=index, path=str(path))
        return

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    fd = -1
    try:
        open_start = time.monotonic()
        fd = os.open(tmp, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o644)
        open_s = time.monotonic() - open_start

        write_start = time.monotonic()
        written = 0
        while written < file_size:
            remaining = min(len(chunk), file_size - written)
            written += write_all(fd, memoryview(chunk)[:remaining])
        write_s = time.monotonic() - write_start

        sync_start = time.monotonic()
        os.fdatasync(fd)
        fdatasync_s = time.monotonic() - sync_start

        close_start = time.monotonic()
        os.close(fd)
        fd = -1
        close_s = time.monotonic() - close_start

        os.replace(tmp, path)
        append_json(
            log_path,
            "prepare_file_done",
            group=group,
            index=index,
            path=str(path),
            bytes=written,
            open_s=open_s,
            write_s=write_s,
            fdatasync_s=fdatasync_s,
            close_s=close_s,
        )
    finally:
        if fd >= 0:
            os.close(fd)
        tmp.unlink(missing_ok=True)


def prepare_command(args: argparse.Namespace) -> int:
    root = Path(args.root)
    data_root = root / "datasets"
    groups = {
        "immutable": args.immutable_files,
        "writer_fixed": args.writer_files,
        "overlap_fixed": args.overlap_files,
    }
    for group in groups:
        (data_root / group).mkdir(parents=True, exist_ok=True)

    prep_log = root / "prepare.jsonl"
    file_size = args.file_mib * MIB
    chunk_size = args.chunk_mib * MIB
    if file_size <= 0 or chunk_size <= 0:
        raise ValueError("file and chunk sizes must be positive")
    chunk = os.urandom(chunk_size)

    append_json(
        prep_log,
        "prepare_start",
        root=str(root),
        file_size=file_size,
        groups=groups,
        total_bytes=sum(groups.values()) * file_size,
        overwrite=args.overwrite,
    )

    manifest_groups: dict[str, list[str]] = {}
    for group, count in groups.items():
        paths: list[str] = []
        for index in range(count):
            path = data_root / group / f"{group}_{index:05d}.bin"
            create_synced_file(
                path,
                file_size,
                chunk,
                args.overwrite,
                prep_log,
                group,
                index,
            )
            paths.append(str(path))
        manifest_groups[group] = paths

    manifest = {
        "version": 2,
        "created_wall": time.time(),
        "file_size": file_size,
        "groups": manifest_groups,
    }
    tmp = root / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, root / "manifest.json")
    append_json(prep_log, "prepare_done", manifest=str(root / "manifest.json"))
    print(root / "manifest.json")
    return 0


# ---------------------------------------------------------------------------
# Worker processes
# ---------------------------------------------------------------------------

def buffered_readinto_file(
    path: str,
    expected_size: int,
    buffer: bytearray,
    cold_hint: bool,
) -> dict[str, Any]:
    io_before = read_proc_io()

    open_start = time.monotonic()
    handle = open(path, "rb")
    open_s = time.monotonic() - open_start

    pre_hint = False
    post_hint = False
    try:
        if cold_hint:
            pre_hint = fadvise_dontneed(handle.fileno())

        read_start = time.monotonic()
        nread = handle.readinto(buffer)
        read_s = time.monotonic() - read_start

        if cold_hint:
            post_hint = fadvise_dontneed(handle.fileno())
    finally:
        close_start = time.monotonic()
        handle.close()
        close_s = time.monotonic() - close_start

    io_delta = delta_dict(read_proc_io(), io_before)
    return {
        "bytes": nread,
        "expected_bytes": expected_size,
        "short_read": nread != expected_size,
        "open_s": open_s,
        "read_s": read_s,
        "close_s": close_s,
        "cold_hint_pre": pre_hint,
        "cold_hint_post": post_hint,
        "proc_read_bytes_delta": io_delta.get("read_bytes", 0),
        "proc_rchar_delta": io_delta.get("rchar", 0),
        "proc_syscr_delta": io_delta.get("syscr", 0),
    }


def direct_read_file(path: str, expected_size: int, chunk_size: int) -> dict[str, Any]:
    if not hasattr(os, "O_DIRECT"):
        raise OSError(errno.EOPNOTSUPP, "O_DIRECT unavailable")

    io_before = read_proc_io()
    open_start = time.monotonic()
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    open_s = time.monotonic() - open_start

    aligned = mmap.mmap(-1, chunk_size)
    view = memoryview(aligned)
    total = 0
    calls = 0
    read_start = time.monotonic()
    try:
        while total < expected_size:
            nread = os.readv(fd, [view])
            calls += 1
            if nread == 0:
                break
            total += nread
    finally:
        read_s = time.monotonic() - read_start
        view.release()
        aligned.close()

    close_start = time.monotonic()
    os.close(fd)
    close_s = time.monotonic() - close_start
    io_delta = delta_dict(read_proc_io(), io_before)
    return {
        "bytes": total,
        "expected_bytes": expected_size,
        "short_read": total != expected_size,
        "open_s": open_s,
        "read_s": read_s,
        "close_s": close_s,
        "direct_calls": calls,
        "proc_read_bytes_delta": io_delta.get("read_bytes", 0),
        "proc_rchar_delta": io_delta.get("rchar", 0),
        "proc_syscr_delta": io_delta.get("syscr", 0),
    }


def reader_worker(
    reader_id: int,
    paths: list[str],
    file_size: int,
    mode: str,
    direct_chunk_size: int,
    cold_hint: bool,
    short_read_backoff_s: float,
    start_event: Any,
    stop_event: Any,
    summary_queue: Any,
    ready_queue: Any,
    log_path: str,
) -> None:
    latencies: list[float] = []
    open_latencies: list[float] = []
    close_latencies: list[float] = []
    bytes_read = 0
    operations = 0
    short_reads = 0
    errors = 0
    direct_fallback: str | None = None
    buffer = bytearray(file_size) if mode == "buffered" else None

    try:
        append_json(
            log_path,
            "reader_ready",
            reader_id=reader_id,
            mode=mode,
            path_count=len(paths),
            file_size=file_size,
            cold_hint=cold_hint,
            short_read_backoff_s=short_read_backoff_s,
        )
        ready_queue.put({"kind": "reader_ready", "reader_id": reader_id})
        start_event.wait()
        worker_start = time.monotonic()
        append_json(log_path, "reader_start", reader_id=reader_id, mode=mode)

        index = 0
        while not stop_event.is_set():
            path = paths[index % len(paths)]
            index += 1
            try:
                if mode == "direct" and direct_fallback is None:
                    try:
                        result = direct_read_file(path, file_size, direct_chunk_size)
                    except OSError as exc:
                        if exc.errno not in {errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP}:
                            raise
                        direct_fallback = f"{type(exc).__name__}: {exc}"
                        buffer = bytearray(file_size)
                        append_json(
                            log_path,
                            "reader_direct_fallback",
                            reader_id=reader_id,
                            reason=direct_fallback,
                        )
                        result = buffered_readinto_file(
                            path, file_size, buffer, cold_hint
                        )
                elif mode == "direct" and direct_fallback is not None:
                    assert buffer is not None
                    result = buffered_readinto_file(path, file_size, buffer, cold_hint)
                else:
                    assert buffer is not None
                    result = buffered_readinto_file(path, file_size, buffer, cold_hint)

                operations += 1
                bytes_read += int(result["bytes"])
                short_reads += int(bool(result["short_read"]))
                latencies.append(float(result["read_s"]))
                open_latencies.append(float(result["open_s"]))
                close_latencies.append(float(result["close_s"]))
                append_json(
                    log_path,
                    "reader_operation",
                    reader_id=reader_id,
                    operation=operations,
                    path=path,
                    effective_mode=(
                        "buffered_fallback"
                        if mode == "direct" and direct_fallback is not None
                        else mode
                    ),
                    read_mib_per_s=(
                        result["bytes"] / MIB / result["read_s"]
                        if result["read_s"]
                        else None
                    ),
                    **result,
                )
                if result["short_read"] and short_read_backoff_s > 0:
                    time.sleep(short_read_backoff_s)
            except BaseException as exc:
                errors += 1
                append_json(
                    log_path,
                    "reader_operation_error",
                    reader_id=reader_id,
                    path=path,
                    error=repr(exc),
                    traceback=traceback.format_exc(),
                )
                time.sleep(0.01)

        elapsed = time.monotonic() - worker_start
        summary = {
            "kind": "reader",
            "reader_id": reader_id,
            "requested_mode": mode,
            "effective_mode": (
                "buffered_fallback"
                if mode == "direct" and direct_fallback is not None
                else mode
            ),
            "direct_fallback": direct_fallback,
            "operations": operations,
            "bytes": bytes_read,
            "short_reads": short_reads,
            "errors": errors,
            "elapsed_s": elapsed,
            "aggregate_mib_per_s": bytes_read / MIB / elapsed if elapsed else None,
            "median_read_s": statistics.median(latencies) if latencies else None,
            "p95_read_s": percentile(latencies, 0.95),
            "p99_read_s": percentile(latencies, 0.99),
            "max_read_s": max(latencies) if latencies else None,
            "median_open_s": statistics.median(open_latencies) if open_latencies else None,
            "median_close_s": statistics.median(close_latencies) if close_latencies else None,
        }
        append_json(log_path, "reader_done", **summary)
        summary_queue.put(summary)
    except BaseException as exc:
        summary_queue.put(
            {
                "kind": "reader_error",
                "reader_id": reader_id,
                "error": repr(exc),
            }
        )
        append_json(
            log_path,
            "reader_fatal",
            reader_id=reader_id,
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        raise


def writer_worker(
    writer_id: int,
    paths: list[str],
    file_size: int,
    start_event: Any,
    stop_event: Any,
    summary_queue: Any,
    ready_queue: Any,
    log_path: str,
) -> None:
    payload = os.urandom(file_size)
    latencies: list[float] = []
    write_latencies: list[float] = []
    close_latencies: list[float] = []
    bytes_written = 0
    operations = 0
    errors = 0

    try:
        append_json(
            log_path,
            "writer_ready",
            writer_id=writer_id,
            path_count=len(paths),
            file_size=file_size,
        )
        ready_queue.put({"kind": "writer_ready", "writer_id": writer_id})
        start_event.wait()
        worker_start = time.monotonic()
        append_json(log_path, "writer_start", writer_id=writer_id)

        index = 0
        while not stop_event.is_set():
            path = paths[index % len(paths)]
            index += 1
            fd = -1
            try:
                io_before = read_proc_io()

                open_start = time.monotonic()
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
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
                append_json(
                    log_path,
                    "writer_operation",
                    writer_id=writer_id,
                    operation=operations,
                    path=path,
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
                    "writer_operation_error",
                    writer_id=writer_id,
                    path=path,
                    error=repr(exc),
                    traceback=traceback.format_exc(),
                )
                time.sleep(0.01)
            finally:
                if fd >= 0:
                    os.close(fd)

        elapsed = time.monotonic() - worker_start
        summary = {
            "kind": "writer",
            "writer_id": writer_id,
            "operations": operations,
            "bytes": bytes_written,
            "errors": errors,
            "elapsed_s": elapsed,
            "aggregate_mib_per_s": bytes_written / MIB / elapsed if elapsed else None,
            "median_operation_s": statistics.median(latencies) if latencies else None,
            "p95_operation_s": percentile(latencies, 0.95),
            "max_operation_s": max(latencies) if latencies else None,
            "median_write_s": (
                statistics.median(write_latencies) if write_latencies else None
            ),
            "median_close_s": (
                statistics.median(close_latencies) if close_latencies else None
            ),
        }
        append_json(log_path, "writer_done", **summary)
        summary_queue.put(summary)
    except BaseException as exc:
        summary_queue.put(
            {
                "kind": "writer_error",
                "writer_id": writer_id,
                "error": repr(exc),
            }
        )
        append_json(
            log_path,
            "writer_fatal",
            writer_id=writer_id,
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        raise


def sampler_worker(
    stop_event: Any,
    log_path: str,
    interval_s: float,
) -> None:
    previous_vm = read_vmstat()
    previous_net = read_netdev()
    previous_tcp = read_tcp_counters()
    previous_nfs = read_nfs_client_stats()
    previous_mono = time.monotonic()

    append_json(
        log_path,
        "sampler_start",
        meminfo=read_meminfo(),
        vmstat=previous_vm,
        net=previous_net,
        tcp=previous_tcp,
        nfs=previous_nfs,
    )

    while not stop_event.wait(interval_s):
        now = time.monotonic()
        vm = read_vmstat()
        net = read_netdev()
        tcp = read_tcp_counters()
        nfs = read_nfs_client_stats()
        elapsed = max(now - previous_mono, 1e-9)

        interface_deltas: dict[str, dict[str, int]] = {}
        previous_interfaces = previous_net.get("net_interfaces", {})
        for interface, current in net.get("net_interfaces", {}).items():
            old = previous_interfaces.get(interface, {})
            interface_deltas[interface] = {
                "rx_delta": current.get("rx_bytes", 0) - old.get("rx_bytes", 0),
                "tx_delta": current.get("tx_bytes", 0) - old.get("tx_bytes", 0),
            }

        append_json(
            log_path,
            "system_sample",
            interval_s=elapsed,
            meminfo=read_meminfo(),
            vmstat=vm,
            vmstat_delta=delta_dict(vm, previous_vm),
            net=net,
            net_rx_delta=net.get("net_rx_bytes", 0)
            - previous_net.get("net_rx_bytes", 0),
            net_tx_delta=net.get("net_tx_bytes", 0)
            - previous_net.get("net_tx_bytes", 0),
            net_interface_deltas=interface_deltas,
            tcp=tcp,
            tcp_delta=delta_dict(tcp, previous_tcp),
            nfs=nfs,
        )
        previous_vm = vm
        previous_net = net
        previous_tcp = tcp
        previous_nfs = nfs
        previous_mono = now

    append_json(log_path, "sampler_done")


# ---------------------------------------------------------------------------
# Phase orchestration and summarization
# ---------------------------------------------------------------------------

def partition_round_robin(paths: list[str], workers: int) -> list[list[str]]:
    result = [[] for _ in range(workers)]
    for index, path in enumerate(paths):
        result[index % workers].append(path)
    if any(not group for group in result):
        raise ValueError(f"Not enough paths ({len(paths)}) for {workers} workers")
    return result


def fsync_paths(paths: list[str], log_path: Path) -> dict[str, Any]:
    latencies: list[float] = []
    failures = 0
    start = time.monotonic()
    for index, path in enumerate(sorted(set(paths))):
        fd = -1
        try:
            file_start = time.monotonic()
            fd = os.open(path, os.O_RDONLY)
            os.fsync(fd)
            latency = time.monotonic() - file_start
            latencies.append(latency)
            append_json(
                log_path,
                "post_phase_fsync",
                index=index,
                path=path,
                fsync_s=latency,
            )
        except OSError as exc:
            failures += 1
            append_json(
                log_path,
                "post_phase_fsync_error",
                index=index,
                path=path,
                error=repr(exc),
            )
        finally:
            if fd >= 0:
                os.close(fd)
    elapsed = time.monotonic() - start
    return {
        "files": len(set(paths)),
        "failures": failures,
        "elapsed_s": elapsed,
        "median_s": statistics.median(latencies) if latencies else None,
        "p95_s": percentile(latencies, 0.95),
        "max_s": max(latencies) if latencies else None,
    }


def collect_summaries(summary_queue: Any, expected: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    deadline = time.monotonic() + 15
    while len(results) < expected and time.monotonic() < deadline:
        try:
            results.append(summary_queue.get(timeout=0.5))
        except queue.Empty:
            continue
    return results


def run_phase(
    ctx: Any,
    run_dir: Path,
    phase_index: int,
    name: str,
    duration_s: float,
    writer_headstart_s: float,
    reader_mode: str,
    reader_paths: list[str],
    writer_paths: list[str],
    readers: int,
    writers: int,
    file_size: int,
    direct_chunk_size: int,
    cold_hint: bool,
    short_read_backoff_s: float,
    sample_interval_s: float,
) -> dict[str, Any]:
    phase_dir = run_dir / f"phase_{phase_index:02d}_{name}"
    phase_dir.mkdir()
    parent_log = phase_dir / "parent.jsonl"

    reader_groups = partition_round_robin(reader_paths, readers)
    writer_groups = partition_round_robin(writer_paths, writers) if writers else []

    reader_start = ctx.Event()
    writer_start = ctx.Event()
    stop_event = ctx.Event()
    sampler_stop = ctx.Event()
    summary_queue = ctx.Queue()
    ready_queue = ctx.Queue()

    sampler = ctx.Process(
        target=sampler_worker,
        args=(sampler_stop, str(phase_dir / "sampler.jsonl"), sample_interval_s),
        name=f"sampler-{phase_index}",
    )
    sampler.start()

    processes: list[Any] = []
    writer_processes: list[Any] = []
    for writer_id in range(writers):
        process = ctx.Process(
            target=writer_worker,
            args=(
                writer_id,
                writer_groups[writer_id],
                file_size,
                writer_start,
                stop_event,
                summary_queue,
                ready_queue,
                str(phase_dir / f"writer_{writer_id}.jsonl"),
            ),
            name=f"writer-{phase_index}-{writer_id}",
        )
        process.start()
        writer_processes.append(process)
        processes.append(process)

    reader_processes: list[Any] = []
    for reader_id in range(readers):
        process = ctx.Process(
            target=reader_worker,
            args=(
                reader_id,
                reader_groups[reader_id],
                file_size,
                reader_mode,
                direct_chunk_size,
                cold_hint,
                short_read_backoff_s,
                reader_start,
                stop_event,
                summary_queue,
                ready_queue,
                str(phase_dir / f"reader_{reader_id}.jsonl"),
            ),
            name=f"reader-{phase_index}-{reader_id}",
        )
        process.start()
        reader_processes.append(process)
        processes.append(process)

    append_json(
        parent_log,
        "phase_start",
        phase_index=phase_index,
        name=name,
        duration_s=duration_s,
        writer_headstart_s=writer_headstart_s,
        reader_mode=reader_mode,
        readers=readers,
        writers=writers,
        reader_path_count=len(reader_paths),
        writer_path_count=len(writer_paths),
        cold_hint=cold_hint,
        meminfo=read_meminfo(),
        tcp=read_tcp_counters(),
        nfs=read_nfs_client_stats(),
    )

    expected_ready = readers + writers
    ready_messages: list[dict[str, Any]] = []
    ready_deadline = time.monotonic() + 120
    while len(ready_messages) < expected_ready and time.monotonic() < ready_deadline:
        try:
            ready_messages.append(ready_queue.get(timeout=0.5))
        except queue.Empty:
            failed_early = [p.name for p in processes if p.exitcode not in (None, 0)]
            if failed_early:
                raise RuntimeError(f"Workers failed before readiness: {failed_early}")
    if len(ready_messages) != expected_ready:
        raise TimeoutError(
            f"Only {len(ready_messages)}/{expected_ready} workers became ready"
        )
    append_json(parent_log, "all_workers_ready", messages=ready_messages)

    writer_wall_start = time.monotonic()
    writer_start.set()
    if writers and writer_headstart_s > 0:
        time.sleep(writer_headstart_s)

    reader_wall_start = time.monotonic()
    reader_start.set()
    stop_event.wait(duration_s)
    stop_event.set()
    stop_signal_elapsed_s = time.monotonic() - reader_wall_start

    join_start = time.monotonic()
    for process in processes:
        process.join(timeout=120)
        if process.is_alive():
            append_json(
                parent_log,
                "process_terminate",
                process_name=process.name,
                pid=process.pid,
            )
            process.terminate()
            process.join(timeout=10)

    join_elapsed_s = time.monotonic() - join_start
    reader_wall_elapsed_s = time.monotonic() - reader_wall_start
    writer_wall_elapsed_s = time.monotonic() - writer_wall_start
    child_exitcodes = {process.name: process.exitcode for process in processes}

    summaries = collect_summaries(summary_queue, readers + writers)
    flush_summary = fsync_paths(writer_paths, parent_log) if writers else {
        "files": 0,
        "failures": 0,
        "elapsed_s": 0.0,
        "median_s": None,
        "p95_s": None,
        "max_s": None,
    }

    sampler_stop.set()
    sampler.join(timeout=10)
    if sampler.is_alive():
        sampler.terminate()
        sampler.join(timeout=5)

    reader_summaries = [item for item in summaries if item.get("kind") == "reader"]
    writer_summaries = [item for item in summaries if item.get("kind") == "writer"]

    total_reader_bytes = sum(int(item.get("bytes", 0)) for item in reader_summaries)
    total_writer_bytes = sum(int(item.get("bytes", 0)) for item in writer_summaries)

    summary = {
        "phase_index": phase_index,
        "name": name,
        "duration_requested_s": duration_s,
        "stop_signal_elapsed_s": stop_signal_elapsed_s,
        "join_elapsed_s": join_elapsed_s,
        "reader_wall_elapsed_s": reader_wall_elapsed_s,
        "writer_wall_elapsed_s": writer_wall_elapsed_s,
        "reader_mode": reader_mode,
        "readers": readers,
        "writers": writers,
        "reader_total_bytes": total_reader_bytes,
        "reader_aggregate_mib_per_s": (
            total_reader_bytes / MIB / reader_wall_elapsed_s
            if reader_wall_elapsed_s
            else None
        ),
        "writer_total_bytes": total_writer_bytes,
        "writer_aggregate_mib_per_s": (
            total_writer_bytes / MIB / writer_wall_elapsed_s
            if writer_wall_elapsed_s
            else None
        ),
        "reader_short_reads": sum(
            int(item.get("short_reads", 0)) for item in reader_summaries
        ),
        "reader_errors": sum(int(item.get("errors", 0)) for item in reader_summaries),
        "writer_errors": sum(int(item.get("errors", 0)) for item in writer_summaries),
        "reader_summaries": reader_summaries,
        "writer_summaries": writer_summaries,
        "all_child_summaries": summaries,
        "child_exitcodes": child_exitcodes,
        "post_phase_fsync": flush_summary,
    }
    append_json(parent_log, "phase_done", **summary)
    return summary


def build_default_phases(
    manifest: dict[str, Any],
    stress_duration_s: float,
    baseline_duration_s: float,
) -> list[dict[str, Any]]:
    immutable = manifest["groups"]["immutable"]
    writer_fixed = manifest["groups"]["writer_fixed"]
    overlap_fixed = manifest["groups"]["overlap_fixed"]
    return [
        {
            "name": "buffered_baseline",
            "duration_s": baseline_duration_s,
            "reader_mode": "buffered",
            "reader_paths": immutable,
            "writer_paths": [],
            "writers": 0,
        },
        {
            "name": "buffered_6w_separate",
            "duration_s": stress_duration_s,
            "reader_mode": "buffered",
            "reader_paths": immutable,
            "writer_paths": writer_fixed,
            "writers": 6,
        },
        {
            "name": "buffered_6w_same_files",
            "duration_s": stress_duration_s,
            "reader_mode": "buffered",
            "reader_paths": overlap_fixed,
            "writer_paths": overlap_fixed,
            "writers": 6,
        },
        {
            "name": "direct_6w_separate",
            "duration_s": stress_duration_s,
            "reader_mode": "direct",
            "reader_paths": immutable,
            "writer_paths": writer_fixed,
            "writers": 6,
        },
        {
            "name": "buffered_recovery",
            "duration_s": baseline_duration_s,
            "reader_mode": "buffered",
            "reader_paths": immutable,
            "writer_paths": [],
            "writers": 0,
        },
    ]


def print_summary_table(phases: list[dict[str, Any]]) -> None:
    header = (
        "phase name                         mode      R W "
        "read_MiB/s write_MiB/s short errR errW "
        "fsync_s reader_p95_s reader_max_s"
    )
    print(header)
    for phase in phases:
        reader_p95_values = [
            float(item["p95_read_s"])
            for item in phase.get("reader_summaries", [])
            if item.get("p95_read_s") is not None
        ]
        reader_max_values = [
            float(item["max_read_s"])
            for item in phase.get("reader_summaries", [])
            if item.get("max_read_s") is not None
        ]
        print(
            f'{phase["phase_index"]:>5} '
            f'{phase["name"]:<28.28} '
            f'{phase["reader_mode"]:<9} '
            f'{phase["readers"]:>1} {phase["writers"]:>1} '
            f'{float(phase.get("reader_aggregate_mib_per_s") or 0):>10.2f} '
            f'{float(phase.get("writer_aggregate_mib_per_s") or 0):>11.2f} '
            f'{phase.get("reader_short_reads", 0):>5} '
            f'{phase.get("reader_errors", 0):>4} '
            f'{phase.get("writer_errors", 0):>4} '
            f'{float(phase.get("post_phase_fsync", {}).get("elapsed_s") or 0):>7.3f} '
            f'{max(reader_p95_values) if reader_p95_values else 0:>12.3f} '
            f'{max(reader_max_values) if reader_max_values else 0:>12.3f}'
        )


def run_command(args: argparse.Namespace) -> int:
    root = Path(args.root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run the prepare subcommand first."
        )
    manifest = json.loads(manifest_path.read_text())
    file_size = int(manifest["file_size"])
    expected_size = args.file_mib * MIB
    if file_size != expected_size:
        raise ValueError(
            f"Manifest file size is {file_size}; CLI expects {expected_size}."
        )

    run_id = f"{socket.gethostname()}_{int(time.time())}_{os.getpid()}"
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    environment_path = run_dir / "environment.json"
    environment = {
        "run_id": run_id,
        "created_wall": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": os.getcwd(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mount_info": read_mount_info(root),
        "df": command_output(["df", "-hT", str(root)]),
        "mount": command_output(["mount"]),
        "nfsstat_m": command_output(["nfsstat", "-m"]),
        "nfsstat_c": command_output(["nfsstat", "-c"]),
        "ip_link": command_output(["ip", "-s", "link"]),
        "meminfo": read_meminfo(),
        "tcp": read_tcp_counters(),
        "nfs": read_nfs_client_stats(),
    }
    environment_path.write_text(json.dumps(environment, indent=2, default=str) + "\n")

    phases = build_default_phases(
        manifest,
        stress_duration_s=args.stress_duration_s,
        baseline_duration_s=args.baseline_duration_s,
    )

    ctx = mp.get_context("spawn")
    summaries: list[dict[str, Any]] = []
    for phase_index, phase in enumerate(phases):
        print(f"Starting phase {phase_index}: {phase['name']}", flush=True)
        summary = run_phase(
            ctx=ctx,
            run_dir=run_dir,
            phase_index=phase_index,
            name=phase["name"],
            duration_s=float(phase["duration_s"]),
            writer_headstart_s=args.writer_headstart_s,
            reader_mode=phase["reader_mode"],
            reader_paths=list(phase["reader_paths"]),
            writer_paths=list(phase["writer_paths"]),
            readers=args.readers,
            writers=int(phase["writers"]),
            file_size=file_size,
            direct_chunk_size=args.direct_chunk_mib * MIB,
            cold_hint=not args.no_cold_hint,
            short_read_backoff_s=args.short_read_backoff_ms / 1000.0,
            sample_interval_s=args.sample_interval_s,
        )
        summaries.append(summary)
        print_summary_table([summary])

        if any(code not in (0, None) for code in summary["child_exitcodes"].values()):
            print("A child process failed; aborting remaining phases.", file=sys.stderr)
            break

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, default=str) + "\n")
    print("\n=== FINAL SUMMARY ===")
    print_summary_table(summaries)
    print(f"\nRun directory: {run_dir}")
    print(f"Summary: {summary_path}")
    return 0


def summarize_command(args: argparse.Namespace) -> int:
    phases = json.loads(Path(args.summary).read_text())
    print_summary_table(phases)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--root",
        default="/mnt/shared/gpfs/home/sriramc2/runs/nfs_lmcache_topology_probe",
    )
    prepare.add_argument("--file-mib", type=int, default=80)
    prepare.add_argument("--chunk-mib", type=int, default=8)
    prepare.add_argument(
        "--immutable-files",
        type=int,
        default=256,
        help="20 GiB at the default 80 MiB size.",
    )
    prepare.add_argument(
        "--writer-files",
        type=int,
        default=48,
        help="Six writers receive eight fixed files each.",
    )
    prepare.add_argument(
        "--overlap-files",
        type=int,
        default=48,
        help="Fixed files that readers and writers deliberately share.",
    )
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(func=prepare_command)

    run = subparsers.add_parser("run")
    run.add_argument(
        "--root",
        default="/mnt/shared/gpfs/home/sriramc2/runs/nfs_lmcache_topology_probe",
    )
    run.add_argument("--file-mib", type=int, default=80)
    run.add_argument("--readers", type=int, default=2)
    run.add_argument("--stress-duration-s", type=float, default=120)
    run.add_argument("--baseline-duration-s", type=float, default=30)
    run.add_argument("--writer-headstart-s", type=float, default=10)
    run.add_argument("--direct-chunk-mib", type=int, default=8)
    run.add_argument("--sample-interval-s", type=float, default=1)
    run.add_argument(
        "--short-read-backoff-ms",
        type=float,
        default=10,
        help="Sleep after a truncate-race short read to avoid a tight log-flood loop.",
    )
    run.add_argument(
        "--no-cold-hint",
        action="store_true",
        help=(
            "Do not call POSIX_FADV_DONTNEED around buffered reads. "
            "The default hint keeps repeated 20 GiB scans from becoming pure "
            "compute-node page-cache tests."
        ),
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

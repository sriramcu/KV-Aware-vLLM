#!/usr/bin/env python3
"""Foreground, resumable Slurm runner for the SC LMCache Round-3 tuning grid."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

GRID_VERSION = "sc-round3-tuning-grid-v1"
SCHEMA_VERSION = 1
TERMINAL_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}
ACTIVE_SLURM_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "RUNNING",
    "RESIZING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "SIGNALING",
    "STAGE_OUT",
    "SUSPENDED",
}
DONE_MARKER = "=== DONE ==="


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def normalize_slurm_state(value: str) -> str:
    value = value.strip().upper()
    value = value.split("+", 1)[0]
    value = value.split(" ", 1)[0]
    return value


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        rendered = shlex.join(command)
        raise RuntimeError(
            f"Command failed ({exc.returncode}): {rendered}\n"
            f"stdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
        ) from exc


def base_environment(
    max_questions: int,
    lmcache_data_dir: Path | None = None,
) -> dict[str, str]:
    """Round-3 baseline: Round-2 run 02 behavior, except for tuned fields."""
    env = {
        "MAX_QUESTIONS": str(max_questions),
        "DATASET_NAME": "medical",
        # Round-2 run 02 is the baseline for Round 3. Keep submission batching
        # fixed at 250 and tune only scheduler max, max_num_seqs, and dataset.
        "SUBMISSION_BATCH_SIZE": "250",
        "VLLM_MAX_NUM_SEQS": "12",
        "VLLM_KV_IMPORTANCE_ENABLE": "0",
        "SC_LMCACHE_SOURCE_DEFAULT_KNOBS": "1",
        "SC_LMCACHE_PROFILE": "custom",
        "SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE": "1",
        "SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT": "4",
        "SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_WAIT_MS": "0",
        # Round 2 showed worker admission is redundant behind scheduler
        # admission for the performance question we are studying here.
        "SC_LMCACHE_WORKER_LOOKUP_ADMISSION_ENABLE": "0",
        "SC_LMCACHE_WORKER_LOOKUP_MAX_INFLIGHT": "4",
        "SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE": "0",
        "SC_LMCACHE_DISK_PUT_MAX_PENDING": "8",
        # Keep the normal grid observability, but explicitly disable the very
        # verbose lifecycle/GPU-assert debugging used for job 18996. Explicit
        # values matter because sbatch uses --export=ALL and the launching shell
        # may still contain variables from lifecycle_debug_run00_env.sh.
        "SC_LMCACHE_IO_TRACE_ENABLE": "1",
        "SC_LMCACHE_LOAD_TRACE_ENABLE": "1",
        "SC_LMCACHE_MEMORY_TRACE_ENABLE": "1",
        "SC_LMCACHE_MEMORY_SNAPSHOT_ENABLE": "0",
        "SC_LMCACHE_LOOKUP_TRACE_ENABLE": "0",
        "SC_LMCACHE_REQUEST_TRACE_ENABLE": "0",
        "SC_LMCACHE_TIER_TRACE_ENABLE": "0",
        "SC_LMCACHE_GPU_ASSERT_SNAPSHOT_ENABLE": "0",
        "SC_LMCACHE_LIFECYCLE_TRACE_ENABLE": "0",
        "SC_LMCACHE_MEMORY_TRACE_INTERVAL_S": "5",
        "SC_LMCACHE_LONG_PIN_THRESHOLD_S": "10",
        "SC_LMCACHE_LONG_REF_THRESHOLD_S": "10",
        "SC_DRIVER_RESOURCE_MONITOR_ENABLE": "1",
        "SC_DRIVER_RESOURCE_MONITOR_INTERVAL_S": "30",
        "CLEAN_OLD_LMCACHE": "1",
    }
    if lmcache_data_dir is not None:
        # Jobs run sequentially. Reusing one dedicated cache directory is
        # intentional: run_driver.sh deletes/recreates it at the start of each
        # job, so storage usage does not accumulate across grid entries.
        env["SC_LMCACHE_DATA_DIR"] = str(lmcache_data_dir)
    return env


def make_grid(
    max_questions: int,
    lmcache_data_dir: Path | None = None,
) -> list[dict[str, Any]]:
    base = base_environment(max_questions, lmcache_data_dir)

    # Round 3 deliberately changes multiple parameters in some rows. We are no
    # longer doing one-factor-at-a-time characterization; this is a compact
    # search around Round-2 run 02 (medical, sched=4/wait=0, seqs=12, batch=250)
    # plus workload-sensitivity probes. Worker and disk-put admission stay off.
    rows: list[tuple[str, str, dict[str, str]]] = [
        (
            "00_med_sched2_seq12",
            "Medical: tighter scheduler admission (2) at max_num_seqs 12",
            {
                "DATASET_NAME": "medical",
                "SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT": "2",
                "VLLM_MAX_NUM_SEQS": "12",
            },
        ),
        (
            "01_med_sched8_seq12",
            "Medical: looser scheduler admission (8) at max_num_seqs 12",
            {
                "DATASET_NAME": "medical",
                "SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT": "8",
                "VLLM_MAX_NUM_SEQS": "12",
            },
        ),
        (
            "02_med_sched4_seq8",
            "Medical: scheduler max 4 with lower max_num_seqs 8",
            {
                "DATASET_NAME": "medical",
                "SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT": "4",
                "VLLM_MAX_NUM_SEQS": "8",
            },
        ),
        (
            "03_med_sched4_seq16",
            "Medical: scheduler max 4 with higher max_num_seqs 16",
            {
                "DATASET_NAME": "medical",
                "SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT": "4",
                "VLLM_MAX_NUM_SEQS": "16",
            },
        ),
        (
            "04_hotpot_sched2_seq8",
            "HotpotQA conservative corner: scheduler max 2, max_num_seqs 8",
            {
                "DATASET_NAME": "hotpotqa",
                "SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT": "2",
                "VLLM_MAX_NUM_SEQS": "8",
            },
        ),
        (
            "05_hotpot_sched8_seq16",
            "HotpotQA aggressive corner: scheduler max 8, max_num_seqs 16",
            {
                "DATASET_NAME": "hotpotqa",
                "SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT": "8",
                "VLLM_MAX_NUM_SEQS": "16",
            },
        ),
        (
            "06_2wiki_sched4_seq12",
            "2WikiMultihop center point: scheduler max 4, max_num_seqs 12",
            {
                "DATASET_NAME": "2wikimultihop",
                "SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT": "4",
                "VLLM_MAX_NUM_SEQS": "12",
            },
        ),
        (
            "07_musique_sched4_seq12",
            "MuSiQue center point: scheduler max 4, max_num_seqs 12",
            {
                "DATASET_NAME": "musique",
                "SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT": "4",
                "VLLM_MAX_NUM_SEQS": "12",
            },
        ),
    ]

    grid: list[dict[str, Any]] = []
    for index, (run_id, description, overrides) in enumerate(rows):
        env = dict(base)
        env.update(overrides)
        config_core = {
            "index": index,
            "run_id": run_id,
            "description": description,
            "environment": env,
        }
        grid.append(
            {
                **config_core,
                "config_sha256": sha256_text(canonical_json(config_core)),
            }
        )
    return grid

def default_state_dir(max_questions: int) -> Path:
    return (
        Path.home()
        / "runs"
        / "kvaware_repro"
        / "pressure_grid"
        / f"{GRID_VERSION}_maxq_{max_questions}"
    )


def git_metadata(repo: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"repo": str(repo)}
    for key, command in {
        "commit": ["git", "-C", str(repo), "rev-parse", "HEAD"],
        "branch": ["git", "-C", str(repo), "branch", "--show-current"],
        "status": ["git", "-C", str(repo), "status", "--short"],
    }.items():
        try:
            metadata[key] = run_checked(command).stdout.strip()
        except RuntimeError as exc:
            metadata[key] = f"unavailable: {exc}"
    return metadata


def initialize_state(
    state_dir: Path,
    repo: Path,
    sbatch_script: Path,
    max_questions: int,
    grid: list[dict[str, Any]],
) -> dict[str, Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = state_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    grid_sha256 = sha256_text(canonical_json(grid))
    run_states = []
    for config in grid:
        run_dir = runs_dir / config["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(run_dir / "config.json", config)
        run_states.append(
            {
                "run_id": config["run_id"],
                "status": "pending",
                "job_id": None,
                "slurm_state": None,
                "attempts": [],
                "submitted_at": None,
                "last_seen_at": None,
                "terminal_observed_at": None,
                "finished_at": None,
                "summary_path": None,
                "failure_reason": None,
            }
        )

    state = {
        "schema_version": SCHEMA_VERSION,
        "grid_version": GRID_VERSION,
        "grid_sha256": grid_sha256,
        "max_questions": max_questions,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "repo_metadata": git_metadata(repo),
        "sbatch_script": str(sbatch_script),
        "state_dir": str(state_dir),
        "runs": run_states,
        "archive_path": None,
        "archive_sha256": None,
    }
    atomic_write_json(state_dir / "grid_manifest.json", {"grid": grid, "grid_sha256": grid_sha256})
    atomic_write_json(state_dir / "grid_state.json", state)
    atomic_write_text(state_dir / "active_job_id.txt", "")
    return state


def validate_loaded_state(
    state: dict[str, Any], max_questions: int, grid: list[dict[str, Any]]
) -> None:
    expected_hash = sha256_text(canonical_json(grid))
    checks = {
        "schema_version": SCHEMA_VERSION,
        "grid_version": GRID_VERSION,
        "max_questions": max_questions,
        "grid_sha256": expected_hash,
    }
    mismatches = {
        key: {"expected": expected, "actual": state.get(key)}
        for key, expected in checks.items()
        if state.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "Existing grid state does not match this runner invocation:\n"
            + json.dumps(mismatches, indent=2, sort_keys=True)
        )
    if len(state.get("runs", [])) != len(grid):
        raise RuntimeError("Existing state has a different number of grid runs")


def save_state(state_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_write_json(state_dir / "grid_state.json", state)


def status_counts(state: dict[str, Any]) -> Counter[str]:
    return Counter(str(run.get("status", "unknown")) for run in state["runs"])


def first_unfinished_index(state: dict[str, Any]) -> int | None:
    for index, run in enumerate(state["runs"]):
        if run.get("status") not in {"completed", "failed"}:
            return index
    return None


def write_progress(
    state_dir: Path,
    state: dict[str, Any],
    grid: list[dict[str, Any]],
    note: str = "",
) -> None:
    counts = status_counts(state)
    total = len(grid)
    terminal = counts["completed"] + counts["failed"]
    current_index = first_unfinished_index(state)
    lines = [
        "SC LMCache pressure grid",
        f"Grid version: {GRID_VERSION}",
        f"MAX_QUESTIONS: {state['max_questions']}",
        f"Progress: {terminal} / {total} terminal",
        f"Completed: {counts['completed']}",
        f"Failed: {counts['failed']}",
        f"Pending/in progress: {total - terminal}",
        f"Last update: {utc_now()}",
    ]
    if note:
        lines.append(f"Note: {note}")
    if current_index is None:
        lines.append("Current: none; grid is terminal")
    else:
        config = grid[current_index]
        run = state["runs"][current_index]
        env = config["environment"]
        lines.extend(
            [
                "",
                f"Current index: {current_index + 1} / {total}",
                f"Current run: {config['run_id']}",
                f"Description: {config['description']}",
                f"State: {run.get('status')}",
                f"Slurm job ID: {run.get('job_id') or '-'}",
                f"Slurm state: {run.get('slurm_state') or '-'}",
                f"Dataset: {env['DATASET_NAME']}",
                f"Submission batch size: {env['SUBMISSION_BATCH_SIZE']}",
                f"max_num_seqs: {env['VLLM_MAX_NUM_SEQS']}",
                "Behavioral knobs:",
                f"  scheduler_enable={env['SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE']}",
                f"  scheduler_max={env['SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT']}",
                f"  scheduler_wait_ms={env['SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_WAIT_MS']}",
                f"  worker_enable={env['SC_LMCACHE_WORKER_LOOKUP_ADMISSION_ENABLE']}",
                f"  worker_max={env['SC_LMCACHE_WORKER_LOOKUP_MAX_INFLIGHT']}",
                f"  disk_put_enable={env['SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE']}",
                f"  disk_put_max={env['SC_LMCACHE_DISK_PUT_MAX_PENDING']}",
            ]
        )
        next_index = current_index + 1
        if next_index < total:
            lines.append(f"Next run: {grid[next_index]['run_id']}")
    if state.get("archive_path"):
        lines.append(f"Archive: {state['archive_path']}")
    atomic_write_text(state_dir / "grid_progress.txt", "\n".join(lines) + "\n")


def managed_active_job_file(state_dir: Path, job_id: str | None) -> None:
    try:
        atomic_write_text(
            state_dir / "active_job_id.txt",
            f"{job_id}\n" if job_id else "",
        )
    except OSError as exc:
        # This file is only a convenience mirror. Recovery always uses the
        # atomically written grid_state.json, so an unreadable or unwritable
        # active-job file must not stop the grid.
        print(
            f"Warning: could not update active_job_id.txt: {exc}; "
            "continuing with grid_state.json",
            flush=True,
        )


def check_active_file(state_dir: Path, state_job_id: str | None) -> str | None:
    path = state_dir / "active_job_id.txt"
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError, UnicodeError):
        return "active_job_id.txt is missing or unreadable; grid_state.json remains authoritative"
    if not raw:
        if state_job_id:
            return "active_job_id.txt is empty; restoring it from grid_state.json"
        return None
    tokens = raw.split()
    if len(tokens) != 1 or not tokens[0].isdigit():
        return "active_job_id.txt is malformed; ignoring it and using grid_state.json"
    if state_job_id and tokens[0] != str(state_job_id):
        return (
            f"active_job_id.txt contains {tokens[0]}, but state records {state_job_id}; "
            "ignoring the file"
        )
    if not state_job_id:
        return "active_job_id.txt has a job ID but no run is active in state; ignoring it"
    return None


class DeferredInterrupts:
    def __init__(self) -> None:
        self.received: list[int] = []
        self.old_handlers: dict[int, Any] = {}

    def _handler(self, signum: int, _frame: Any) -> None:
        self.received.append(signum)
        print(
            f"\nReceived signal {signum}; deferring it until the protected submission section ends.",
            flush=True,
        )

    def __enter__(self) -> "DeferredInterrupts":
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            self.old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handler)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        for signum, handler in self.old_handlers.items():
            signal.signal(signum, handler)

    def raise_if_deferred(self) -> None:
        if self.received:
            raise KeyboardInterrupt


def countdown_before_submission() -> None:
    print("", flush=True)
    print("Entering non-interruptible submission section in:", flush=True)
    for remaining in range(4, 0, -1):
        print(f"{remaining}...", flush=True)
        time.sleep(1)
    print("Do not interrupt now.", flush=True)


def slurm_job_name(max_questions: int, index: int, run_id: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9_]", "_", run_id)
    return f"scg_q{max_questions}_{index:02d}_{compact}"[:120]


def build_export_argument(config: dict[str, Any]) -> str:
    env = dict(config["environment"])
    env.update(
        {
            "GRID_RUN_ID": config["run_id"],
            "GRID_RUN_LABEL": config["run_id"],
            "GRID_CONFIG_SHA256": config["config_sha256"],
        }
    )
    for key, value in env.items():
        if any(char in str(value) for char in (",", "\n", "\x00")):
            raise ValueError(f"Unsafe value for sbatch --export: {key}={value!r}")
    return "ALL," + ",".join(f"{key}={value}" for key, value in sorted(env.items()))


def submit_run(
    *,
    state_dir: Path,
    state: dict[str, Any],
    grid: list[dict[str, Any]],
    index: int,
    sbatch_script: Path,
) -> None:
    config = grid[index]
    run = state["runs"][index]
    run_dir = state_dir / "runs" / config["run_id"]
    countdown_before_submission()

    deferred = DeferredInterrupts()
    with deferred:
        run["status"] = "submitting"
        run["failure_reason"] = None
        run["slurm_state"] = None
        run["terminal_observed_at"] = None
        save_state(state_dir, state)
        write_progress(state_dir, state, grid, "Inside protected submission section")

        job_name = slurm_job_name(state["max_questions"], index, config["run_id"])
        output_pattern = run_dir / "slurm-%j.out"
        error_pattern = run_dir / "slurm-%j.err"
        command = [
            "sbatch",
            "--parsable",
            f"--job-name={job_name}",
            f"--output={output_pattern}",
            f"--error={error_pattern}",
            f"--export={build_export_argument(config)}",
            str(sbatch_script),
        ]

        try:
            result = run_checked(command)
            raw_job_id = result.stdout.strip().splitlines()[-1]
            job_id = raw_job_id.split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"Could not parse sbatch job ID from: {raw_job_id!r}")
        except Exception as exc:
            run["status"] = "submission_error"
            run["failure_reason"] = str(exc)
            save_state(state_dir, state)
            write_progress(state_dir, state, grid, f"Submission failed: {exc}")
            raise

        submitted_at = utc_now()
        attempt = {
            "attempt_number": len(run["attempts"]) + 1,
            "job_id": job_id,
            "job_name": job_name,
            "submitted_at": submitted_at,
            "output_path": str(run_dir / f"slurm-{job_id}.out"),
            "error_path": str(run_dir / f"slurm-{job_id}.err"),
            "terminal_state": None,
            "exit_code": None,
            "finished_at": None,
            "result": None,
        }
        run["attempts"].append(attempt)
        run["job_id"] = job_id
        run["status"] = "submitted"
        run["submitted_at"] = submitted_at
        run["last_seen_at"] = None
        save_state(state_dir, state)
        managed_active_job_file(state_dir, job_id)
        write_progress(state_dir, state, grid, f"Submitted Slurm job {job_id}")

    print("Leaving non-interruptible section. You may press Ctrl-C safely.", flush=True)
    print(f"Submitted {config['run_id']} as Slurm job {run['job_id']}.", flush=True)
    deferred.raise_if_deferred()


def query_squeue(job_id: str) -> str | None:
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Required command not found: squeue") from exc

    if result.returncode != 0:
        combined = f"{result.stdout}\n{result.stderr}"
        if "Invalid job id specified" in combined:
            return None
        raise RuntimeError(
            f"squeue failed ({result.returncode}) for job {job_id}:\n{combined}"
        )

    states = [
        normalize_slurm_state(line)
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    return states[0] if states else None


def query_sacct(job_id: str) -> dict[str, str] | None:
    try:
        result = subprocess.run(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                job_id,
                "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        # This cluster currently has Slurm accounting disabled. Do not infer
        # failure and do not blindly resubmit when sacct is unavailable.
        return None

    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 7:
            continue
        if fields[0].strip() != job_id:
            continue
        return {
            "job_id": fields[0].strip(),
            "job_name": fields[1].strip(),
            "state": normalize_slurm_state(fields[2]),
            "state_raw": fields[2].strip(),
            "exit_code": fields[3].strip(),
            "elapsed": fields[4].strip(),
            "start": fields[5].strip(),
            "end": fields[6].strip(),
        }
    return None


def age_seconds(iso_value: str | None) -> float | None:
    if not iso_value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(iso_value)
        return (dt.datetime.now(dt.timezone.utc) - parsed).total_seconds()
    except ValueError:
        return None


def read_log_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def parse_simple_summary(log_path: Path) -> dict[str, Any]:
    text = read_log_text(log_path)
    summary: dict[str, Any] = {
        "log_path": str(log_path),
        "log_exists": log_path.exists(),
        "log_size_bytes": log_path.stat().st_size if log_path.exists() else 0,
        "done_marker_present": DONE_MARKER in text,
        "traceback_count": text.count("Traceback (most recent call last):"),
        "lookup_timeout_count": text.count("SC_IO_LOOKUP_TIMEOUT"),
        "resource_tracker_warning_count": text.count("resource_tracker:"),
        "phase_summaries": [],
        "output_consistency": None,
        "batch_elapsed_seconds": {"cold": [], "warm": []},
    }

    for match in re.finditer(r"^\[SC_DRIVER_PHASE_SUMMARY\]\s+(\{.*\})$", text, re.MULTILINE):
        try:
            summary["phase_summaries"].append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass
    matches = list(
        re.finditer(r"^\[SC_DRIVER_OUTPUT_CONSISTENCY\]\s+(\{.*\})$", text, re.MULTILINE)
    )
    if matches:
        try:
            summary["output_consistency"] = json.loads(matches[-1].group(1))
        except json.JSONDecodeError:
            pass

    for match in re.finditer(
        r"\[SC_IO_DRIVER_BATCH_DONE\].*?phase=(cold|warm).*?elapsed=([0-9.]+)", text
    ):
        summary["batch_elapsed_seconds"][match.group(1)].append(float(match.group(2)))

    cold = re.findall(r"first generation took ([0-9.]+) seconds", text, re.IGNORECASE)
    warm = re.findall(r"Second generation took ([0-9.]+) seconds", text, re.IGNORECASE)
    summary["cold_elapsed_seconds_fallback"] = float(cold[-1]) if cold else None
    summary["warm_elapsed_seconds_fallback"] = float(warm[-1]) if warm else None
    return summary


def finalize_run(
    *,
    state_dir: Path,
    state: dict[str, Any],
    grid: list[dict[str, Any]],
    index: int,
    sacct_record: dict[str, str] | None,
    success: bool,
    failure_reason: str | None,
) -> None:
    config = grid[index]
    run = state["runs"][index]
    attempt = run["attempts"][-1]
    job_id = str(run["job_id"])
    log_path = Path(attempt["output_path"])
    simple = parse_simple_summary(log_path)
    finished_at = utc_now()
    slurm_state = sacct_record["state"] if sacct_record else run.get("slurm_state")
    attempt.update(
        {
            "terminal_state": slurm_state,
            "exit_code": sacct_record.get("exit_code") if sacct_record else None,
            "finished_at": finished_at,
            "result": "completed" if success else "failed",
        }
    )
    run["status"] = "completed" if success else "failed"
    run["slurm_state"] = slurm_state
    run["finished_at"] = finished_at
    run["failure_reason"] = failure_reason
    run["terminal_observed_at"] = None
    summary_path = state_dir / "runs" / config["run_id"] / "summary.json"
    run["summary_path"] = str(summary_path)
    summary = {
        "grid_version": GRID_VERSION,
        "run": config,
        "result": run["status"],
        "failure_reason": failure_reason,
        "job_id": job_id,
        "slurm": sacct_record,
        "simple_metrics": simple,
        "finished_at": finished_at,
    }
    atomic_write_json(summary_path, summary)
    save_state(state_dir, state)
    managed_active_job_file(state_dir, None)
    write_progress(state_dir, state, grid, f"Run {config['run_id']} is {run['status']}")


def monitor_current_run(
    *,
    state_dir: Path,
    state: dict[str, Any],
    grid: list[dict[str, Any]],
    index: int,
    poll_seconds: float,
    missing_job_grace_seconds: float,
    completed_log_grace_seconds: float,
) -> None:
    config = grid[index]
    run = state["runs"][index]
    job_id = str(run.get("job_id") or "")
    if not job_id.isdigit():
        run["status"] = "pending"
        run["job_id"] = None
        save_state(state_dir, state)
        managed_active_job_file(state_dir, None)
        write_progress(
            state_dir,
            state,
            grid,
            "No persisted job ID; current run will be resubmitted",
        )
        return

    warning = check_active_file(state_dir, job_id)
    if warning:
        print(f"Recovery note: {warning}", flush=True)
        managed_active_job_file(state_dir, job_id)

    while True:
        squeue_state = query_squeue(job_id)
        if squeue_state is not None:
            run["status"] = "running" if squeue_state == "RUNNING" else "submitted"
            run["slurm_state"] = squeue_state
            run["last_seen_at"] = utc_now()
            run["terminal_observed_at"] = None
            save_state(state_dir, state)
            write_progress(
                state_dir,
                state,
                grid,
                f"Job {job_id} is {squeue_state}; polling every {poll_seconds:g} seconds",
            )
            print(
                f"[{utc_now()}] {config['run_id']} job {job_id}: {squeue_state}",
                flush=True,
            )
            time.sleep(poll_seconds)
            continue

        attempt = run["attempts"][-1]
        log_path = Path(attempt["output_path"])
        log_has_done = DONE_MARKER in read_log_text(log_path)

        if log_has_done:
            run["slurm_state"] = "COMPLETED_LOG_MARKER"
            finalize_run(
                state_dir=state_dir,
                state=state,
                grid=grid,
                index=index,
                sacct_record=None,
                success=True,
                failure_reason=None,
            )
            print(
                f"Completed {config['run_id']} with job {job_id} "
                f"via {DONE_MARKER!r}.",
                flush=True,
            )
            return

        sacct_record = query_sacct(job_id)
        if sacct_record is not None:
            slurm_state = sacct_record["state"]
            run["slurm_state"] = slurm_state
            run["last_seen_at"] = utc_now()
            save_state(state_dir, state)

            if slurm_state not in TERMINAL_SLURM_STATES:
                write_progress(
                    state_dir,
                    state,
                    grid,
                    f"Accounting reports {slurm_state}",
                )
                print(
                    f"[{utc_now()}] {config['run_id']} job {job_id}: {slurm_state}",
                    flush=True,
                )
                time.sleep(poll_seconds)
                continue

            if slurm_state == "COMPLETED":
                if not run.get("terminal_observed_at"):
                    run["terminal_observed_at"] = utc_now()
                    save_state(state_dir, state)
                terminal_age = age_seconds(run.get("terminal_observed_at")) or 0.0
                if terminal_age < completed_log_grace_seconds:
                    note = (
                        f"Job {job_id} is COMPLETED but {DONE_MARKER!r} is not visible yet; "
                        f"waiting {terminal_age:.0f}/{completed_log_grace_seconds:.0f}s"
                    )
                    write_progress(state_dir, state, grid, note)
                    print(note, flush=True)
                    time.sleep(min(poll_seconds, 10.0))
                    continue
                reason = (
                    f"Slurm completed, but {DONE_MARKER!r} is missing from {log_path}"
                )
            else:
                reason = (
                    f"Slurm terminal state is {slurm_state} "
                    f"(exit {sacct_record['exit_code']})"
                )

            finalize_run(
                state_dir=state_dir,
                state=state,
                grid=grid,
                index=index,
                sacct_record=sacct_record,
                success=False,
                failure_reason=reason,
            )
            print(f"Recorded {config['run_id']} as failed: {reason}", flush=True)
            return

        if not run.get("terminal_observed_at"):
            run["terminal_observed_at"] = utc_now()
            save_state(state_dir, state)

        missing_age = age_seconds(run.get("terminal_observed_at")) or 0.0
        if missing_age < missing_job_grace_seconds:
            note = (
                f"Job {job_id} is absent from squeue; no accounting record and "
                f"no {DONE_MARKER!r} yet. Waiting "
                f"{missing_age:.0f}/{missing_job_grace_seconds:.0f}s."
            )
            write_progress(state_dir, state, grid, note)
            print(note, flush=True)
            time.sleep(min(poll_seconds, 10.0))
            continue

        run["slurm_state"] = "UNKNOWN_NO_ACCOUNTING"
        reason = (
            f"Job {job_id} disappeared from squeue, accounting is unavailable, "
            f"and {DONE_MARKER!r} was absent from {log_path} after "
            f"{missing_job_grace_seconds:g}s"
        )
        finalize_run(
            state_dir=state_dir,
            state=state,
            grid=grid,
            index=index,
            sacct_record=None,
            success=False,
            failure_reason=reason,
        )
        print(f"Recorded {config['run_id']} as failed: {reason}", flush=True)
        return


def create_archive(state_dir: Path, state: dict[str, Any], grid: list[dict[str, Any]]) -> Path:
    archive_path = state_dir.parent / f"{state_dir.name}.zip"
    temporary_path = archive_path.with_suffix(".zip.tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    atomic_write_json(
        state_dir / "archive_manifest.json",
        {
            "created_at": utc_now(),
            "grid_version": GRID_VERSION,
            "max_questions": state["max_questions"],
            "grid_sha256": state["grid_sha256"],
            "terminal_counts": dict(status_counts(state)),
            "archive_path": str(archive_path),
        },
    )

    with zipfile.ZipFile(
        temporary_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path in sorted(state_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(state_dir.parent)
            archive.write(path, arcname=str(relative))
    os.replace(temporary_path, archive_path)

    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    state["archive_path"] = str(archive_path)
    state["archive_sha256"] = digest.hexdigest()
    save_state(state_dir, state)
    write_progress(state_dir, state, grid, "All runs terminal; archive created")
    return archive_path


def print_grid(grid: list[dict[str, Any]]) -> None:
    for config in grid:
        env = config["environment"]
        print(
            f"{config['index'] + 1:02d}. {config['run_id']}: {config['description']}\n"
            f"    dataset={env['DATASET_NAME']} batch={env['SUBMISSION_BATCH_SIZE']} "
            f"max_num_seqs={env['VLLM_MAX_NUM_SEQS']} "
            f"sched={env['SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_ENABLE']}/"
            f"{env['SC_LMCACHE_SCHEDULER_LOOKUP_MAX_INFLIGHT']}/"
            f"{env['SC_LMCACHE_SCHEDULER_LOOKUP_ADMISSION_WAIT_MS']}ms "
            f"worker={env['SC_LMCACHE_WORKER_LOOKUP_ADMISSION_ENABLE']}/"
            f"{env['SC_LMCACHE_WORKER_LOOKUP_MAX_INFLIGHT']} "
            f"disk_put={env['SC_LMCACHE_DISK_PUT_ADMISSION_ENABLE']}/"
            f"{env['SC_LMCACHE_DISK_PUT_MAX_PENDING']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-questions",
        type=int,
        default=1,
        help="Questions per dataset. Defaults to 1 for the Round-3 smoke grid; pass 250 for the full grid.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Persistent state/output directory. Defaults under ~/runs/kvaware_repro/pressure_grid.",
    )
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--missing-job-grace-seconds", type=float, default=180.0)
    parser.add_argument("--completed-log-grace-seconds", type=float, default=60.0)
    parser.add_argument(
        "--lmcache-data-dir",
        type=Path,
        default=None,
        help=(
            "Optional exact LMCache local-disk directory shared by these "
            "sequential jobs. Useful for node-local scratch, e.g. "
            "/scratch/$USER/lmcache_round3_grid. run_driver.sh clears it "
            "at the start of every job."
        ),
    )
    parser.add_argument(
        "--print-grid",
        action="store_true",
        help="Print the Round-3 tuning configurations and exit without creating state or submitting jobs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_questions <= 0:
        raise SystemExit("--max-questions must be greater than zero")
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be greater than zero")

    if args.lmcache_data_dir is not None:
        args.lmcache_data_dir = args.lmcache_data_dir.expanduser().resolve()

    script_path = Path(__file__).resolve()
    repo = script_path.parents[2]
    sbatch_script = script_path.with_name("submit_round3_tuning_grid.sbatch")
    if not sbatch_script.is_file():
        raise SystemExit(f"Missing sbatch script: {sbatch_script}")

    grid = make_grid(args.max_questions, args.lmcache_data_dir)
    if args.print_grid:
        print_grid(grid)
        return 0

    state_dir = (args.state_dir or default_state_dir(args.max_questions)).expanduser().resolve()
    state_path = state_dir / "grid_state.json"
    if state_path.exists():
        state = read_json(state_path)
        validate_loaded_state(state, args.max_questions, grid)
    else:
        state = initialize_state(state_dir, repo, sbatch_script, args.max_questions, grid)

    write_progress(state_dir, state, grid, "Master started or resumed")
    print(f"Persistent state: {state_path}", flush=True)
    print(f"Human-readable progress: {state_dir / 'grid_progress.txt'}", flush=True)
    print(f"Managed active job ID: {state_dir / 'active_job_id.txt'}", flush=True)

    try:
        while True:
            index = first_unfinished_index(state)
            if index is None:
                archive_path = create_archive(state_dir, state, grid)
                print(f"All {len(grid)} Round-3 tuning configurations are terminal.", flush=True)
                print(f"Archive: {archive_path}", flush=True)
                print(f"SHA256: {state['archive_sha256']}", flush=True)
                return 0

            run = state["runs"][index]
            status = run.get("status")
            if status in {"pending", "submission_error", "submitting"} and not run.get("job_id"):
                submit_run(
                    state_dir=state_dir,
                    state=state,
                    grid=grid,
                    index=index,
                    sbatch_script=sbatch_script,
                )
                continue

            if status in {"submitted", "running", "submitting"} or run.get("job_id"):
                monitor_current_run(
                    state_dir=state_dir,
                    state=state,
                    grid=grid,
                    index=index,
                    poll_seconds=args.poll_seconds,
                    missing_job_grace_seconds=args.missing_job_grace_seconds,
                    completed_log_grace_seconds=args.completed_log_grace_seconds,
                )
                continue

            raise RuntimeError(
                f"Unhandled state for {grid[index]['run_id']}: {json.dumps(run, indent=2)}"
            )
    except KeyboardInterrupt:
        index = first_unfinished_index(state)
        current = grid[index]["run_id"] if index is not None else "none"
        write_progress(state_dir, state, grid, "Master stopped by user; active Slurm job was not cancelled")
        print("\nMaster stopped cleanly. No active Slurm job was cancelled.", flush=True)
        print(f"Current run: {current}", flush=True)
        print(f"Resume with the same command. State: {state_path}", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())


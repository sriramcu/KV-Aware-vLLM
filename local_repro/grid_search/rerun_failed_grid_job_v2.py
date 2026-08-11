#!/usr/bin/env python3
"""Rerun one failed pressure-grid configuration.

The selector is the failed run directory:

    python rerun_failed_grid_job.py \
      /path/to/sc-pressure-grid-v1_maxq_250/runs/03_scheduler_max_12

Normal Ctrl-C behavior is preserved. Before the final commit, all replacement
artifacts are built in a separate staging directory, so interrupting submission,
monitoring, metric parsing, or ZIP creation leaves the original grid untouched.

Only the short final commit defers Ctrl-C. That commit replaces the old run
directory, grid-state entry, progress file, archive manifest, and final ZIP.
There is deliberately no resume mechanism. Re-running this script submits a new
Slurm job.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

DONE_MARKER = "=== DONE ==="
TERMINAL_GRID_STATES = {"completed", "failed"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(
    command: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required command not found: {command[0]}") from exc

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {shlex.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def normalize_slurm_state(value: str) -> str:
    value = value.strip().upper()
    value = value.split("+", 1)[0]
    value = value.split(" ", 1)[0]
    return value


def query_squeue(job_id: str) -> str | None:
    result = run_command(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        check=False,
    )
    if result.returncode != 0:
        combined = f"{result.stdout}\n{result.stderr}"
        if "Invalid job id specified" in combined:
            return None
        raise RuntimeError(
            f"squeue failed ({result.returncode}) for job {job_id}:\n"
            f"{combined}"
        )

    states = [
        normalize_slurm_state(line)
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    return states[0] if states else None


def read_log_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def parse_simple_summary(
    staged_log_path: Path,
    final_log_path: Path,
) -> dict[str, Any]:
    text = read_log_text(staged_log_path)
    summary: dict[str, Any] = {
        "log_path": str(final_log_path),
        "log_exists": staged_log_path.exists(),
        "log_size_bytes": (
            staged_log_path.stat().st_size
            if staged_log_path.exists()
            else 0
        ),
        "done_marker_present": DONE_MARKER in text,
        "traceback_count": text.count(
            "Traceback (most recent call last):"
        ),
        "lookup_timeout_count": text.count("SC_IO_LOOKUP_TIMEOUT"),
        "resource_tracker_warning_count": text.count(
            "resource_tracker:"
        ),
        "phase_summaries": [],
        "output_consistency": None,
        "batch_elapsed_seconds": {"cold": [], "warm": []},
    }

    for match in re.finditer(
        r"^\[SC_DRIVER_PHASE_SUMMARY\]\s+(\{.*\})$",
        text,
        re.MULTILINE,
    ):
        try:
            summary["phase_summaries"].append(
                json.loads(match.group(1))
            )
        except json.JSONDecodeError:
            pass

    consistency_matches = list(
        re.finditer(
            r"^\[SC_DRIVER_OUTPUT_CONSISTENCY\]\s+(\{.*\})$",
            text,
            re.MULTILINE,
        )
    )
    if consistency_matches:
        try:
            summary["output_consistency"] = json.loads(
                consistency_matches[-1].group(1)
            )
        except json.JSONDecodeError:
            pass

    for match in re.finditer(
        r"\[SC_IO_DRIVER_BATCH_DONE\].*?"
        r"phase=(cold|warm).*?elapsed=([0-9.]+)",
        text,
    ):
        summary["batch_elapsed_seconds"][match.group(1)].append(
            float(match.group(2))
        )

    cold = re.findall(
        r"first generation took ([0-9.]+) seconds",
        text,
        re.IGNORECASE,
    )
    warm = re.findall(
        r"Second generation took ([0-9.]+) seconds",
        text,
        re.IGNORECASE,
    )
    summary["cold_elapsed_seconds_fallback"] = (
        float(cold[-1]) if cold else None
    )
    summary["warm_elapsed_seconds_fallback"] = (
        float(warm[-1]) if warm else None
    )
    return summary


def build_export_argument(config: dict[str, Any]) -> str:
    environment = dict(config["environment"])
    environment.update(
        {
            "GRID_RUN_ID": config["run_id"],
            "GRID_RUN_LABEL": f"{config['run_id']}_failed_rerun",
            "GRID_CONFIG_SHA256": config["config_sha256"],
        }
    )

    for key, value in environment.items():
        rendered = str(value)
        if any(character in rendered for character in (",", "\n", "\x00")):
            raise RuntimeError(
                f"Unsafe value for sbatch --export: {key}={rendered!r}"
            )

    return "ALL," + ",".join(
        f"{key}={value}"
        for key, value in sorted(environment.items())
    )


def build_job_name(
    max_questions: int,
    index: int,
    run_id: str,
) -> str:
    compact = re.sub(r"[^A-Za-z0-9_]", "_", run_id)
    return (
        f"scgr_q{max_questions}_{index:02d}_{compact}"
    )[:120]


def load_grid(state_dir: Path) -> list[dict[str, Any]]:
    manifest = read_json(state_dir / "grid_manifest.json")
    grid = manifest.get("grid")
    if not isinstance(grid, list) or not all(
        isinstance(item, dict) for item in grid
    ):
        raise RuntimeError(
            "grid_manifest.json does not contain a valid grid"
        )
    return grid


def find_run_index(
    state: dict[str, Any],
    grid: list[dict[str, Any]],
    run_id: str,
) -> int:
    state_matches = [
        index
        for index, run in enumerate(state.get("runs", []))
        if run.get("run_id") == run_id
    ]
    grid_matches = [
        index
        for index, config in enumerate(grid)
        if config.get("run_id") == run_id
    ]

    if len(state_matches) != 1 or len(grid_matches) != 1:
        raise RuntimeError(
            f"Could not uniquely locate {run_id!r} "
            "in grid_state.json and grid_manifest.json"
        )
    if state_matches[0] != grid_matches[0]:
        raise RuntimeError(
            f"State/manifest index mismatch for {run_id!r}"
        )
    return state_matches[0]


def validate_grid(
    state_dir: Path,
    run_dir: Path,
    state: dict[str, Any],
    grid: list[dict[str, Any]],
    index: int,
) -> None:
    nonterminal = [
        run.get("run_id")
        for run in state["runs"]
        if run.get("status") not in TERMINAL_GRID_STATES
    ]
    if nonterminal:
        raise RuntimeError(
            "The main grid is not fully terminal. "
            "Refusing to modify it while runs remain active: "
            f"{nonterminal}"
        )

    selected = state["runs"][index]
    if selected.get("status") != "failed":
        raise RuntimeError(
            f"{run_dir.name} has status "
            f"{selected.get('status')!r}; expected 'failed'"
        )

    active_file = state_dir / "active_job_id.txt"
    try:
        active_text = active_file.read_text(
            encoding="utf-8"
        ).strip()
    except FileNotFoundError:
        active_text = ""
    if active_text:
        raise RuntimeError(
            f"{active_file} is not empty ({active_text!r}); "
            "a grid job may still be active"
        )

    disk_config = read_json(run_dir / "config.json")
    if canonical_json(disk_config) != canonical_json(grid[index]):
        raise RuntimeError(
            f"{run_dir / 'config.json'} does not match "
            "grid_manifest.json"
        )

    configured_state_dir = Path(
        str(state.get("state_dir", ""))
    ).expanduser().resolve()
    if configured_state_dir != state_dir:
        raise RuntimeError(
            "grid_state.json points at a different state directory: "
            f"{configured_state_dir}"
        )


def submit_replacement(
    state: dict[str, Any],
    config: dict[str, Any],
    candidate_dir: Path,
) -> tuple[str, str, str]:
    sbatch_script = Path(
        state["sbatch_script"]
    ).expanduser().resolve()
    if not sbatch_script.is_file():
        raise RuntimeError(
            f"Missing sbatch script: {sbatch_script}"
        )

    candidate_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(candidate_dir / "config.json", config)

    name = build_job_name(
        int(state["max_questions"]),
        int(config["index"]),
        str(config["run_id"]),
    )
    output_pattern = candidate_dir / "slurm-%j.out"
    error_pattern = candidate_dir / "slurm-%j.err"

    command = [
        "sbatch",
        "--parsable",
        f"--job-name={name}",
        f"--output={output_pattern}",
        f"--error={error_pattern}",
        f"--export={build_export_argument(config)}",
        str(sbatch_script),
    ]

    print(
        f"Submitting replacement for {config['run_id']}...",
        flush=True,
    )
    result = run_command(command)
    raw_job_id = result.stdout.strip().splitlines()[-1]
    job_id = raw_job_id.split(";", 1)[0]

    if not job_id.isdigit():
        raise RuntimeError(
            f"Could not parse sbatch job ID from {raw_job_id!r}"
        )

    submitted_at = utc_now()
    print(
        f"Submitted replacement as Slurm job {job_id}.",
        flush=True,
    )
    return job_id, name, submitted_at


def monitor_replacement(
    job_id: str,
    output_path: Path,
    *,
    poll_seconds: float,
    missing_job_grace_seconds: float,
) -> tuple[bool, str | None, str | None]:
    missing_since: dt.datetime | None = None
    last_seen_at: str | None = None

    while True:
        slurm_state = query_squeue(job_id)

        if slurm_state is not None:
            last_seen_at = utc_now()
            missing_since = None
            print(
                f"[{last_seen_at}] job {job_id}: "
                f"{slurm_state}",
                flush=True,
            )
            time.sleep(poll_seconds)
            continue

        if DONE_MARKER in read_log_text(output_path):
            return True, None, last_seen_at

        now = dt.datetime.now(dt.timezone.utc)
        if missing_since is None:
            missing_since = now

        missing_age = (now - missing_since).total_seconds()
        if missing_age >= missing_job_grace_seconds:
            reason = (
                f"Job {job_id} disappeared from squeue and "
                f"{DONE_MARKER!r} was absent from {output_path} "
                f"after {missing_job_grace_seconds:g} seconds"
            )
            return False, reason, last_seen_at

        print(
            f"[{utc_now()}] job {job_id} is absent from "
            "squeue and the completion marker is not visible; "
            f"waiting {missing_age:.0f}/"
            f"{missing_job_grace_seconds:g}s",
            flush=True,
        )
        time.sleep(min(poll_seconds, 10.0))


def build_replacement_metadata(
    *,
    state: dict[str, Any],
    grid: list[dict[str, Any]],
    index: int,
    run_dir: Path,
    candidate_dir: Path,
    job_id: str,
    job_name: str,
    submitted_at: str,
    last_seen_at: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = grid[index]
    finished_at = utc_now()
    final_output_path = run_dir / f"slurm-{job_id}.out"
    final_error_path = run_dir / f"slurm-{job_id}.err"
    staged_output_path = candidate_dir / f"slurm-{job_id}.out"

    attempt = {
        "attempt_number": 1,
        "job_id": job_id,
        "job_name": job_name,
        "submitted_at": submitted_at,
        "output_path": str(final_output_path),
        "error_path": str(final_error_path),
        "terminal_state": "COMPLETED_LOG_MARKER",
        "exit_code": None,
        "finished_at": finished_at,
        "result": "completed",
    }

    replacement_entry = {
        "run_id": config["run_id"],
        "status": "completed",
        "job_id": job_id,
        "slurm_state": "COMPLETED_LOG_MARKER",
        "attempts": [attempt],
        "submitted_at": submitted_at,
        "last_seen_at": last_seen_at,
        "terminal_observed_at": None,
        "finished_at": finished_at,
        "summary_path": str(run_dir / "summary.json"),
        "failure_reason": None,
    }

    summary = {
        "grid_version": state["grid_version"],
        "run": config,
        "result": "completed",
        "failure_reason": None,
        "job_id": job_id,
        "slurm": None,
        "simple_metrics": parse_simple_summary(
            staged_output_path,
            final_output_path,
        ),
        "finished_at": finished_at,
        "failed_run_replacement": True,
    }

    return replacement_entry, summary


def render_progress(
    state: dict[str, Any],
    grid: list[dict[str, Any]],
    note: str,
) -> str:
    counts = Counter(
        str(run.get("status", "unknown"))
        for run in state["runs"]
    )
    terminal = counts["completed"] + counts["failed"]

    lines = [
        "SC LMCache pressure grid",
        f"Grid version: {state['grid_version']}",
        f"MAX_QUESTIONS: {state['max_questions']}",
        f"Progress: {terminal} / {len(grid)} terminal",
        f"Completed: {counts['completed']}",
        f"Failed: {counts['failed']}",
        f"Pending/in progress: {len(grid) - terminal}",
        f"Last update: {utc_now()}",
        f"Note: {note}",
    ]

    unfinished = [
        index
        for index, run in enumerate(state["runs"])
        if run.get("status") not in TERMINAL_GRID_STATES
    ]
    if unfinished:
        index = unfinished[0]
        lines.extend(
            [
                "",
                f"Current index: {index + 1} / {len(grid)}",
                f"Current run: {grid[index]['run_id']}",
                f"State: {state['runs'][index].get('status')}",
            ]
        )
    else:
        lines.append("Current: none; grid is terminal")

    if state.get("archive_path"):
        lines.append(f"Archive: {state['archive_path']}")

    return "\n".join(lines) + "\n"


def build_staged_zip(
    *,
    state_dir: Path,
    run_dir: Path,
    candidate_dir: Path,
    staged_state_path: Path,
    staged_progress_path: Path,
    staged_archive_manifest_path: Path,
    output_zip: Path,
) -> None:
    state_root_name = state_dir.name
    skipped_metadata = {
        Path("grid_state.json"),
        Path("grid_progress.txt"),
        Path("archive_manifest.json"),
    }
    selected_prefix = Path("runs") / run_dir.name

    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(
        output_zip,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path in sorted(state_dir.rglob("*")):
            if not path.is_file():
                continue

            relative = path.relative_to(state_dir)
            if relative in skipped_metadata:
                continue
            if (
                len(relative.parts) >= 2
                and Path(*relative.parts[:2]) == selected_prefix
            ):
                continue

            archive.write(
                path,
                arcname=str(Path(state_root_name) / relative),
            )

        for path in sorted(candidate_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(candidate_dir)
            archive.write(
                path,
                arcname=str(
                    Path(state_root_name)
                    / "runs"
                    / run_dir.name
                    / relative
                ),
            )

        archive.write(
            staged_state_path,
            arcname=str(Path(state_root_name) / "grid_state.json"),
        )
        archive.write(
            staged_progress_path,
            arcname=str(
                Path(state_root_name) / "grid_progress.txt"
            ),
        )
        archive.write(
            staged_archive_manifest_path,
            arcname=str(
                Path(state_root_name) / "archive_manifest.json"
            ),
        )


class FinalCommitSigintDeferral:
    """Defer Ctrl-C only while the final multi-file commit executes."""

    def __init__(self) -> None:
        self._previous: Any = None
        self.interrupted = False

    def _handler(self, signum: int, frame: Any) -> None:
        del signum, frame
        self.interrupted = True

    def __enter__(self) -> "FinalCommitSigintDeferral":
        self._previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> bool:
        signal.signal(signal.SIGINT, self._previous)
        return False


def commit_replacement(
    *,
    state_dir: Path,
    run_dir: Path,
    candidate_dir: Path,
    staging_root: Path,
    original_entry_sha256: str,
    index: int,
    staged_state_path: Path,
    staged_progress_path: Path,
    staged_archive_manifest_path: Path,
    staged_zip_path: Path,
    final_zip_path: Path,
) -> bool:
    state_path = state_dir / "grid_state.json"
    progress_path = state_dir / "grid_progress.txt"
    archive_manifest_path = state_dir / "archive_manifest.json"
    backup_dir = staging_root / "original_run_backup"

    current_state = read_json(state_path)
    current_entry = current_state["runs"][index]
    if sha256_json(current_entry) != original_entry_sha256:
        raise RuntimeError(
            "The selected grid-state entry changed while the "
            "replacement job was running. Nothing was committed."
        )

    deferral = FinalCommitSigintDeferral()

    with deferral:
        print(
            "Replacement completed. Entering short final commit; "
            "Ctrl-C will be deferred until it finishes.",
            flush=True,
        )

        if backup_dir.exists():
            raise RuntimeError(
                f"Unexpected backup directory already exists: "
                f"{backup_dir}"
            )

        os.replace(run_dir, backup_dir)
        candidate_installed = False

        try:
            os.replace(candidate_dir, run_dir)
            candidate_installed = True

            os.replace(staged_state_path, state_path)
            os.replace(staged_progress_path, progress_path)
            os.replace(
                staged_archive_manifest_path,
                archive_manifest_path,
            )
            os.replace(staged_zip_path, final_zip_path)

            shutil.rmtree(backup_dir)

        except Exception:
            # Best-effort rollback. Ctrl-C cannot enter this block because it
            # is deferred, but filesystem errors are still handled safely.
            if candidate_installed and run_dir.exists():
                failed_candidate = (
                    staging_root / "candidate_after_failed_commit"
                )
                if failed_candidate.exists():
                    shutil.rmtree(failed_candidate)
                os.replace(run_dir, failed_candidate)

            if backup_dir.exists() and not run_dir.exists():
                os.replace(backup_dir, run_dir)

            raise

        print(
            "Final commit completed.",
            flush=True,
        )

    return deferral.interrupted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help=(
            "Path to one failed run directory, for example "
            ".../sc-pressure-grid-v1_maxq_250/"
            "runs/03_scheduler_max_12"
        ),
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--missing-job-grace-seconds",
        type=float,
        default=180.0,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.poll_seconds <= 0:
        raise SystemExit(
            "--poll-seconds must be greater than zero"
        )
    if args.missing_job_grace_seconds < 0:
        raise SystemExit(
            "--missing-job-grace-seconds must be nonnegative"
        )

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(
            f"Run directory does not exist: {run_dir}"
        )
    if run_dir.parent.name != "runs":
        raise SystemExit(
            "The selected path must be a direct child of the "
            f"grid's runs/ directory: {run_dir}"
        )

    state_dir = run_dir.parent.parent.resolve()
    state_path = state_dir / "grid_state.json"
    if not state_path.is_file():
        raise SystemExit(f"Missing grid state: {state_path}")

    lock_root = (
        state_dir.parent
        / f".{state_dir.name}.failed-rerun-control"
    )
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / ".lock"

    invocation_id = (
        dt.datetime.now(dt.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        + "-"
        + str(os.getpid())
        + "-"
        + uuid.uuid4().hex[:8]
    )
    staging_root = (
        state_dir.parent
        / f".{state_dir.name}.failed-rerun-staging"
        / run_dir.name
        / invocation_id
    )
    candidate_dir = staging_root / "candidate_run"

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(
                lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise SystemExit(
                "Another failed-run replacement script is "
                f"currently active for {state_dir}"
            ) from exc

        grid = load_grid(state_dir)
        state = read_json(state_path)
        index = find_run_index(
            state,
            grid,
            run_dir.name,
        )
        validate_grid(
            state_dir,
            run_dir,
            state,
            grid,
            index,
        )

        original_entry_sha256 = sha256_json(
            state["runs"][index]
        )
        config = grid[index]

        staging_root.mkdir(parents=True, exist_ok=False)
        print(
            f"Staging directory: {staging_root}",
            flush=True,
        )
        print(
            "Ctrl-C is allowed while submitting, waiting, "
            "parsing, and building the replacement archive.",
            flush=True,
        )
        print(
            "An interrupt before the final commit leaves the "
            "original grid artifacts unchanged.",
            flush=True,
        )

        job_id, job_name, submitted_at = submit_replacement(
            state,
            config,
            candidate_dir,
        )
        staged_output_path = (
            candidate_dir / f"slurm-{job_id}.out"
        )

        success, failure_reason, last_seen_at = (
            monitor_replacement(
                job_id,
                staged_output_path,
                poll_seconds=args.poll_seconds,
                missing_job_grace_seconds=(
                    args.missing_job_grace_seconds
                ),
            )
        )

        if not success:
            print(
                "Replacement job did not complete successfully.",
                flush=True,
            )
            print(f"Reason: {failure_reason}", flush=True)
            print(
                "The original run, grid state, progress file, "
                "and final ZIP were not modified.",
                flush=True,
            )
            print(
                f"Replacement logs were retained in: "
                f"{staging_root}",
                flush=True,
            )
            return 1

        replacement_entry, summary = (
            build_replacement_metadata(
                state=state,
                grid=grid,
                index=index,
                run_dir=run_dir,
                candidate_dir=candidate_dir,
                job_id=job_id,
                job_name=job_name,
                submitted_at=submitted_at,
                last_seen_at=last_seen_at,
            )
        )
        atomic_write_json(
            candidate_dir / "summary.json",
            summary,
        )

        staged_state = copy.deepcopy(state)
        staged_state["runs"][index] = replacement_entry
        staged_state["archive_path"] = None
        staged_state["archive_sha256"] = None
        staged_state["updated_at"] = utc_now()

        staged_state_path = staging_root / "grid_state.json"
        staged_progress_path = (
            staging_root / "grid_progress.txt"
        )
        staged_archive_manifest_path = (
            staging_root / "archive_manifest.json"
        )
        staged_zip_path = (
            staging_root / f"{state_dir.name}.zip"
        )
        final_zip_path = (
            state_dir.parent / f"{state_dir.name}.zip"
        )

        atomic_write_json(
            staged_state_path,
            staged_state,
        )
        atomic_write_text(
            staged_progress_path,
            render_progress(
                staged_state,
                grid,
                (
                    f"Failed run {run_dir.name} replaced "
                    f"by Slurm job {job_id}"
                ),
            ),
        )

        counts = Counter(
            str(run.get("status", "unknown"))
            for run in staged_state["runs"]
        )
        atomic_write_json(
            staged_archive_manifest_path,
            {
                "created_at": utc_now(),
                "grid_version": staged_state[
                    "grid_version"
                ],
                "max_questions": staged_state[
                    "max_questions"
                ],
                "grid_sha256": staged_state[
                    "grid_sha256"
                ],
                "terminal_counts": dict(counts),
                "archive_path": str(final_zip_path),
                "failed_run_replacement": True,
                "replacement_run_id": run_dir.name,
                "replacement_job_id": job_id,
            },
        )

        print(
            "Building replacement ZIP before touching the "
            "original grid...",
            flush=True,
        )
        build_staged_zip(
            state_dir=state_dir,
            run_dir=run_dir,
            candidate_dir=candidate_dir,
            staged_state_path=staged_state_path,
            staged_progress_path=staged_progress_path,
            staged_archive_manifest_path=(
                staged_archive_manifest_path
            ),
            output_zip=staged_zip_path,
        )

        archive_sha256 = sha256_file(staged_zip_path)
        staged_state["archive_path"] = str(
            final_zip_path
        )
        staged_state["archive_sha256"] = archive_sha256
        staged_state["updated_at"] = utc_now()
        atomic_write_json(
            staged_state_path,
            staged_state,
        )

        # The ZIP intentionally contains the pre-checksum form of
        # grid_state.json, matching the original grid runner's behavior.
        interrupted_during_commit = commit_replacement(
            state_dir=state_dir,
            run_dir=run_dir,
            candidate_dir=candidate_dir,
            staging_root=staging_root,
            original_entry_sha256=original_entry_sha256,
            index=index,
            staged_state_path=staged_state_path,
            staged_progress_path=staged_progress_path,
            staged_archive_manifest_path=(
                staged_archive_manifest_path
            ),
            staged_zip_path=staged_zip_path,
            final_zip_path=final_zip_path,
        )

        try:
            shutil.rmtree(staging_root)
        except FileNotFoundError:
            pass

        print(
            f"Replaced {run_dir.name} with successful "
            f"Slurm job {job_id}.",
            flush=True,
        )
        print(
            f"Updated archive: {final_zip_path}",
            flush=True,
        )

        if interrupted_during_commit:
            print(
                "Ctrl-C was received during the final commit. "
                "It was deferred, and the commit completed "
                "safely.",
                flush=True,
            )
            return 130

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterrupted. The original grid artifacts were "
            "not modified unless the script had already entered "
            "and completed its printed final-commit section.",
            file=sys.stderr,
            flush=True,
        )
        print(
            "Any submitted Slurm job was not cancelled. "
            "Running this script again will submit a new job.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(130)
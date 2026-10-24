"""Small, opt-in wave-level KV mapper. No torch/vLLM/LMCache imports.

A score source yields ordered (chunk_hash, score) pairs. One fixed-weight
prior-regularized histogram and tier-cost means produce percentile cutoffs.
Placement metadata uses the existing hash -> gpu/cpu/disk/drop format.
"""
from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
import re
from typing import Any

TIERS = ("gpu", "cpu", "disk")


def atomic_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_score_rows(path: str | Path) -> dict[int, list[tuple[str, float]]]:
    rows: dict[int, list[tuple[str, float]]] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["source_index"])
            if idx in rows:
                raise ValueError(f"duplicate score request {idx}")
            chunks = [(str(r["chunk_hash"]).lower(), float(r["score"])) for r in row["chunks"]]
            if any(not math.isfinite(v) for _, v in chunks):
                raise ValueError(f"non-finite score at request {idx}")
            rows[idx] = chunks
    if not rows:
        raise ValueError("score file has no requests")
    return rows


class WaveBayesianMapper:
    def __init__(self, prior_file: str | Path, weight: float, gpu_gb: float,
                 cpu_gb: float, disk_gb: float, score_file: str | Path,
                 metadata_file: str | Path, vpc_file: str | Path,
                 *, posterior_mode: str = "fixed", score_prior_count: float = 10.0,
                 cost_prior_count: float = 5.0, readmission: bool = False,
                 random_score_seed: int | None = None):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("prior weight must be finite and >= 0")
        self.weight = float(weight)
        if posterior_mode not in {"fixed", "pseudo_count"}:
            raise ValueError("posterior_mode must be fixed or pseudo_count")
        if any(not math.isfinite(x) or x <= 0 for x in (score_prior_count, cost_prior_count)):
            raise ValueError("pseudo-count strengths must be finite and > 0")
        self.posterior_mode = posterior_mode
        self.score_prior_count = float(score_prior_count)
        self.cost_prior_count = float(cost_prior_count)
        self.readmission = bool(readmission)
        self.random_score_seed = random_score_seed
        self.prior = json.loads(Path(prior_file).read_text(encoding="utf-8"))
        self.edges = [float(x) for x in self.prior["score_bin_edges"]]
        self.prior_hist = [float(x) for x in self.prior["score_bin_probabilities"]]
        if len(self.edges) != len(self.prior_hist) + 1 or len(self.edges) < 3:
            raise ValueError("score histogram bins/edges mismatch")
        if any(not math.isfinite(x) for x in self.edges + self.prior_hist):
            raise ValueError("nonfinite histogram prior")
        if any(b <= a for a, b in zip(self.edges, self.edges[1:])):
            raise ValueError("histogram edges must be strictly increasing")
        if any(x < 0 for x in self.prior_hist) or sum(self.prior_hist) <= 0:
            raise ValueError("histogram prior must be nonnegative with positive mass")
        s = sum(self.prior_hist)
        self.prior_hist = [v / s for v in self.prior_hist]
        self.prior_ms = {t: float(self.prior["cost_prior_ms"][t]) for t in ("cpu", "disk", "recompute")}
        cost_switch = os.environ.get("KV_BAYES_ADAPT_COST", "1")
        if cost_switch not in ("0", "1"):
            raise ValueError("KV_BAYES_ADAPT_COST must be 0 or 1")
        self.adapt_cost = cost_switch == "1"
        if any(not math.isfinite(x) or x <= 0 for x in self.prior_ms.values()):
            raise ValueError("cost_prior_ms values must be finite and > 0")
        self.capacities = {"gpu": float(gpu_gb), "cpu": float(cpu_gb), "disk": float(disk_gb)}
        if any(not math.isfinite(v) or v < 0 for v in self.capacities.values()) or sum(self.capacities.values()) <= 0:
            raise ValueError("capacities must be finite/nonnegative and at least one positive")
        self.rows = read_score_rows(score_file)
        if random_score_seed is not None:
            # Shuffle unique-hash scores; keep the observed score population and
            # duplicate hashes' score consistency. This is a controlled random
            # ranking, not random independent tier assignment.
            import random
            first = {}
            for rows in self.rows.values():
                for h, score in rows:
                    first.setdefault(h, score)
            hashes = sorted(first)
            scores = [first[h] for h in hashes]
            random.Random(int(random_score_seed)).shuffle(scores)
            shuffled = dict(zip(hashes, scores, strict=True))
            self.rows = {source: [(h, shuffled[h]) for h, _ in rows]
                         for source, rows in self.rows.items()}
        self.metadata_file = Path(metadata_file)
        self.vpc_file = Path(vpc_file)
        self.assigned: dict[str, str] = {}
        self.request_labels: dict[str, dict[str, str]] = {}
        self.seen_scores: set[str] = set()
        self.hist_counts = [0] * len(self.prior_hist)
        self.cost_sum_ms = {"cpu": 0.0, "disk": 0.0, "recompute": 0.0}
        self.cost_counts = {"cpu": 0.0, "disk": 0.0, "recompute": 0.0}
        self.out_of_range_scores = 0
        self.epoch = 0
        self.profiles: list[dict[str, Any]] = []

    def _bin(self, score: float) -> int:
        # Clamp outliers, count/report them via diagnostics; never silently wrap.
        import bisect
        return max(0, min(len(self.hist_counts) - 1, bisect.bisect_right(self.edges, score) - 1))

    def observe_scores(self, source_indices: list[int]) -> None:
        for source in source_indices:
            if source not in self.rows:
                raise KeyError(f"GNN score missing for source_index={source}")
            for hash_hex, score in self.rows[source]:
                if hash_hex not in self.seen_scores:
                    self.seen_scores.add(hash_hex)
                    self.out_of_range_scores += int(score < self.edges[0] or score > self.edges[-1])
                    self.hist_counts[self._bin(score)] += 1

    def observe_cost(self, tier: str, milliseconds_per_chunk: float, count: float = 1) -> None:
        if tier not in self.cost_counts:
            raise ValueError(tier)
        if not math.isfinite(milliseconds_per_chunk) or milliseconds_per_chunk <= 0 or count <= 0:
            raise ValueError("invalid tier-cost observation")
        if not self.adapt_cost:
            return  # Instrumentation remains ON; freeze CPU/Disk posterior.
        self.cost_sum_ms[tier] += milliseconds_per_chunk * count
        self.cost_counts[tier] += count

    def estimated_cost(self, tier: str) -> float:
        n = self.cost_counts[tier]
        if n == 0:
            return self.prior_ms[tier]
        if self.posterior_mode == "pseudo_count":
            return ((self.cost_prior_count * self.prior_ms[tier]) + self.cost_sum_ms[tier]) / (self.cost_prior_count + n)
        return ((self.weight * self.prior_ms[tier]) + self.cost_sum_ms[tier] / n) / (self.weight + 1.0)

    def probabilities(self) -> list[float]:
        n = sum(self.hist_counts)
        if n == 0:
            return list(self.prior_hist)
        obs = [v / n for v in self.hist_counts]
        if self.posterior_mode == "pseudo_count":
            return [(self.score_prior_count * a + c) / (self.score_prior_count + n)
                    for a, c in zip(self.prior_hist, self.hist_counts)]
        return [(self.weight * a + b) / (self.weight + 1.0) for a, b in zip(self.prior_hist, obs)]

    def percentile(self, score: float) -> float:
        bins = self.probabilities()
        idx = self._bin(score)
        lo, hi = self.edges[idx], self.edges[idx + 1]
        fraction = min(1.0, max(0.0, (score - lo) / (hi - lo)))
        return min(1.0, max(0.0, sum(bins[:idx]) + bins[idx] * fraction))

    def score_at_percentile(self, percentile: float) -> float:
        """Invert current piecewise-uniform posterior CDF for diagnostics.

        These raw cutoffs are approximate within histogram bins, not exact
        empirical percentiles; nothing here changes placement decisions.
        """
        if not math.isfinite(percentile) or not 0 <= percentile <= 1:
            raise ValueError("percentile must be finite and in [0,1]")
        if percentile <= 0:
            return self.edges[0]
        if percentile >= 1:
            return self.edges[-1]
        acc = 0.0
        for i, mass in enumerate(self.probabilities()):
            end = acc + mass
            if mass > 0 and percentile <= end:
                return self.edges[i] + (self.edges[i+1] - self.edges[i]) * ((percentile-acc)/mass)
            acc = end
        return self.edges[-1]

    def bands(self) -> dict[str, float]:
        csum = sum(self.capacities.values())
        tau = self.estimated_cost("recompute")
        fractions = {
            "gpu": self.capacities["gpu"] / csum,
            "cpu": self.capacities["cpu"] / csum * min(1.0, tau / self.estimated_cost("cpu")),
            "disk": self.capacities["disk"] / csum * min(1.0, tau / self.estimated_cost("disk")),
        }
        fractions["drop"] = max(0.0, 1.0 - sum(fractions.values()))
        return {
            "gpu": 1.0 - fractions["gpu"],
            "cpu": 1.0 - fractions["gpu"] - fractions["cpu"],
            "disk": fractions["drop"],
            "fractions": fractions,
        }

    @staticmethod
    def _tier(percentile: float, bands: dict[str, float]) -> str:
        if percentile >= bands["gpu"]:
            return "gpu"
        if percentile >= bands["cpu"]:
            return "cpu"
        if percentile >= bands["disk"]:
            return "disk"
        return "drop"

    @staticmethod
    def _prefix_closure(candidates: list[str], percentiles: list[float],
                        hashes: list[str], fixed: dict[str, str],
                        bands: dict[str, float]) -> tuple[list[str], int, bool]:
        """Constrained O(n) minimum deviation. A fixed hash cannot migrate.

        When fixed labels conflict (a fixed store following a fixed DROP), keep
        them and suppress NEW admissions after the earliest fixed DROP.
        Existing historical stranding is reported as infeasible, not hidden.
        """
        n = len(candidates)
        discard = [max(0.0, p - bands[t]) if t != "drop" else 0.0
                   for t, p in zip(candidates, percentiles)]
        repair = [max(0.0, bands["disk"] - p) if t == "drop" else 0.0
                  for t, p in zip(candidates, percentiles)]
        loss = sum(discard)
        best_loss = float("inf")
        best_m = 0
        has_feasible = False
        # Fixed stores require m > their last position. Fixed DROPs require
        # m <= their first position. Constraints are computed once: O(n).
        min_m = max((i + 1 for i, h in enumerate(hashes)
                     if h in fixed and fixed[h] != "drop"), default=0)
        max_m = min((i for i, h in enumerate(hashes)
                     if fixed.get(h) == "drop"), default=n)
        # Candidate m denotes number of retained chunks, 0 .. n.
        for m in range(n + 1):
            feasible = min_m <= m <= max_m
            if feasible and loss < best_loss - 1e-12:
                has_feasible, best_loss, best_m = True, loss, m
            if m < n:
                loss += repair[m] - discard[m]
        if not has_feasible:
            first_fixed_drop = next((i for i, h in enumerate(hashes)
                                     if fixed.get(h) == "drop"), n)
            best_m = first_fixed_drop
        result = []
        for i, (h, tier) in enumerate(zip(hashes, candidates)):
            if h in fixed:
                result.append(fixed[h])
            elif i < best_m:
                result.append(tier if tier != "drop" else "disk")
            else:
                result.append("drop")
        return result, best_m, has_feasible

    def publish_wave(self, source_indices: list[int], phase: str) -> dict[str, Any]:
        # Wave boundaries freeze tier preferences until all requests finish.
        # Existing *physical* hits are fetched normally regardless of their
        # current admission preference. A changed preference is used only if
        # that KV must be computed and newly stored in this wave.
        self.epoch += 1
        b = self.bands()
        new_count = 0
        preference_changes = 0
        transitions: Counter[str] = Counter()
        contradictions = 0
        previous = dict(self.assigned)
        wave_fixed: dict[str, str] = {} if self.readmission else dict(self.assigned)
        for source in source_indices:
            seq = self.rows[source]
            hashes = [h for h, _ in seq]
            ps = [self.percentile(s) for _, s in seq]
            proposals = [self._tier(p, b) for p in ps]
            finalized, cutoff, feasible = self._prefix_closure(
                proposals, ps, hashes, wave_fixed, b)
            contradictions += int(not feasible)
            block_labels = {}
            for i, (h, tier) in enumerate(zip(hashes, finalized)):
                if h not in wave_fixed:
                    old = previous.get(h)
                    if old is None:
                        new_count += 1
                    elif old != tier:
                        preference_changes += 1
                        transitions[f"{old}->{tier}"] += 1
                    wave_fixed[h] = tier
                label = wave_fixed[h]
                for native in range(i * 32, (i + 1) * 32):
                    block_labels[str(native)] = "disk" if label == "drop" else label
            self.request_labels[str(source)] = block_labels
        self.assigned.update(wave_fixed)
        atomic_json(self.metadata_file, self.assigned)
        atomic_json(self.vpc_file, self.request_labels)
        status = {
            "epoch": self.epoch, "phase": phase, "requests": len(source_indices),
            "new_hashes": new_count, "total_hashes": len(self.assigned),
            "readmission_preferences_changed": preference_changes,
            "readmission_preference_transitions": dict(transitions),
            "note": "preference changes are NOT proof of a physical hierarchy miss; runtime store logs report actual admissions",
            "score_samples": len(self.seen_scores),
            "score_out_of_prior_range": self.out_of_range_scores,
            "posterior_mode": self.posterior_mode,
            "score_prior_count": self.score_prior_count,
            "cost_prior_count": self.cost_prior_count,
            "cost_adaptation_enabled": self.adapt_cost,
            "cost_sample_counts": dict(self.cost_counts),
            "cost_estimate_ms": {t: self.estimated_cost(t) for t in ("cpu", "disk", "recompute")},
            "thresholds": {k: b[k] for k in TIERS},
            "score_format": self.prior.get("score_format", "unknown"),
            "raw_score_cutoffs": {k: self.score_at_percentile(b[k]) for k in TIERS},
            "raw_score_cutoffs_note": "piecewise-uniform posterior histogram inverse, not exact quantiles",
            "nominal_shares": b["fractions"],
            "fixed_conflict_requests": contradictions,
            "mapped_counts": dict(Counter(self.assigned.values())),
        }
        self.profiles.append(status)
        return status


class LmcacheLogCostReader:
    """Experimental LOG PROXIES ONLY; not exclusive device retrieval timings.

    Disk: MP_PREFETCH_DONE load_s / loaded_keys (includes scheduling).
    CPU: 'Retrieved N tokens in S seconds' / 512-chunk equivalents
    (H2D submission-side wall time; can include prefetched disk chunks).
    To avoid unjustified placement tuning, no cost is fed back unless
    the user explicitly enables proxy observations.
    """
    _load = re.compile(r"\[MP_PREFETCH_DONE\].*?load_s=([0-9.]+).*?loaded_keys=(\d+)")
    _retrieve = re.compile(r"Retrieved (\d+) tokens in ([0-9.]+) seconds")

    def __init__(self, logfile: str | Path):
        self.logfile = Path(logfile)
        self.offset = 0

    def collect(self, mapper: WaveBayesianMapper) -> dict[str, int]:
        counts = {"cpu_proxy": 0, "disk_proxy": 0}
        if not self.logfile.exists():
            return counts
        with self.logfile.open(encoding="utf-8", errors="replace") as f:
            f.seek(self.offset)
            data = f.read()
            if not data.endswith("\n"):
                at = data.rfind("\n")
                if at < 0:
                    return counts
                data, tail = data[:at + 1], data[at + 1:]
                f.seek(f.tell() - len(tail.encode('utf-8', errors='replace')))
            self.offset = f.tell()
        for line in data.splitlines():
            match = self._load.search(line)
            if match and int(match[2]) and float(match[1]) > 0:
                mapper.observe_cost("disk", float(match[1]) * 1000 / int(match[2]), count=int(match[2]))
                counts["disk_proxy"] += 1
            match = self._retrieve.search(line)
            if match and int(match[1]) >= 512 and float(match[2]) > 0:
                chunks = int(match[1]) // 512
                mapper.observe_cost("cpu", float(match[2]) * 1000 / (int(match[1]) / 512), count=chunks)
                counts["cpu_proxy"] += 1
        return counts


class LmcacheCompletionCostReader(LmcacheLogCostReader):
    """Completed runtime H2D and L2 load stages; associate by external request.

    L2 stage and H2D are not exclusive service costs and may overlap. A matched
    H2D request may contain both L1 and L2 chunks; its per-chunk H2D value is
    an approximation, not a per-key physical transfer trace. A wave-local H2D
    average is used ONLY if no matched request was observed, and is reported.
    """
    # [SC CE28G] Permanently consume ONLY completed CUDA transfer-region
    # observations. The old BAYES_H2D_COMPLETE interval is diagnostic only.
    _h2d = re.compile(
        r"\[BAYES_H2D_COST_V2\].*?request=(\S+).*?chunks=(\d+).*?elapsed_ms=([0-9.]+)")
    _old_h2d = re.compile(r"\[BAYES_H2D_COMPLETE\]")
    _disk = re.compile(
        r"\[BAYES_L2_LOAD_COMPLETE\].*?external_request=(\S+).*?chunks=(\d+).*?elapsed_ms=([0-9.]+)")
    _legacy_disk = re.compile(
        r"\[BAYES_L2_LOAD_COMPLETE\].*?chunks=(\d+).*?elapsed_ms=([0-9.]+)")

    def collect(self, mapper: WaveBayesianMapper) -> dict[str, int]:
        counts = {"h2d_completed": 0, "h2d_legacy_diagnostic": 0,
                  "disk_load_completed": 0, "disk_matched": 0,
                  "disk_wave_fallback": 0, "disk_legacy_unmatched": 0}
        if not self.logfile.exists():
            return counts
        with self.logfile.open(encoding="utf-8", errors="replace") as f:
            f.seek(self.offset)
            data = f.read()
            if not data.endswith("\n"):
                cut = data.rfind("\n")
                if cut < 0:
                    return counts
                data = data[:cut + 1]
                f.seek(self.offset + len(data.encode("utf-8", errors="replace")))
            self.offset = f.tell()
        h2d_by_request: dict[str, list[float]] = {}
        disk_rows: list[tuple[str | None, int, float]] = []
        wave_h2d_sum = 0.0
        wave_h2d_n = 0
        for line in data.splitlines():
            if self._old_h2d.search(line):
                counts["h2d_legacy_diagnostic"] += 1
            m = self._h2d.search(line)
            if m and int(m[2]) > 0 and float(m[3]) > 0:
                key, n, elapsed = m[1], int(m[2]), float(m[3])
                per_chunk = elapsed / n
                mapper.observe_cost("cpu", per_chunk, count=n)
                h2d_by_request.setdefault(key, []).append(per_chunk)
                wave_h2d_sum += elapsed
                wave_h2d_n += n
                counts["h2d_completed"] += 1
            m = self._disk.search(line)
            if m and int(m[2]) > 0 and float(m[3]) > 0:
                disk_rows.append((m[1], int(m[2]), float(m[3])))
                counts["disk_load_completed"] += 1
            elif "[BAYES_L2_LOAD_COMPLETE]" in line:
                legacy = self._legacy_disk.search(line)
                if legacy and int(legacy[1]) > 0 and float(legacy[2]) > 0:
                    disk_rows.append((None, int(legacy[1]), float(legacy[2])))
                    counts["disk_load_completed"] += 1
                    counts["disk_legacy_unmatched"] += 1
        # Never silently fall back to the old inflated CPU measurement.
        # Guard a completely missing V2 stream whenever legacy completions
        # were emitted during this wave. Some late V2 callbacks may cross
        # wave boundaries; compare per-wave counts with that limitation.
        if counts["h2d_legacy_diagnostic"] > 0 and not counts["h2d_completed"]:
            raise RuntimeError(
                "CE28G: old H2D completions exist but no BAYES_H2D_COST_V2 "
                "events were observed; verify CUDA events and LMCache logs")
        fallback = (wave_h2d_sum / wave_h2d_n if wave_h2d_n else None)
        for key, n, elapsed in disk_rows:
            if key is not None and key in h2d_by_request:
                # TP2 logical readiness is controlled by the slower rank.
                h2d = max(h2d_by_request[key])
                counts["disk_matched"] += 1
            elif fallback is not None:
                h2d = fallback
                counts["disk_wave_fallback"] += 1
            else:
                # Do not mix a lifetime CPU H2D mean with the current L2
                # sample; retain the prior until representative H2D exists.
                continue
            mapper.observe_cost("disk", elapsed / n + h2d, count=n)
        return counts


class PhaseRecomputeCostReader:
    """Wave-level effective recomputation cost per 512 prompt tokens.

    Synchronous steps: model-executor result minus submit timestamp.
    Asynchronous steps: elapsed between consecutive *completed* engine steps,
    clipped at the current step's submit time to exclude idle gaps. This is a
    completion-throughput proxy, not the isolated GPU duration or the latency
    of a single request. It adapts to actual engine contention without forcing
    synchronous scheduling. Decode-heavy and undersized steps are excluded.
    No GPU synchronization is introduced.
    """
    def __init__(self, trace_dir: str | Path, *, min_prefill_fraction: float = 0.9,
                 min_prompt_tokens: int = 512):
        self.trace_dir = Path(trace_dir)
        self.offsets: dict[Path, int] = {}
        # Per-engine-process completion time, persisted across wave reads.
        self._last_completion_ns: dict[Path, int] = {}
        self.min_prefill_fraction = min_prefill_fraction
        self.min_prompt_tokens = min_prompt_tokens

    def collect(self, mapper: WaveBayesianMapper) -> dict[str, int | float]:
        status: dict[str, int | float] = {"recompute_steps": 0,
            "recompute_skipped": 0, "recompute_chunk_equivalents": 0.0,
            "recompute_async_steps": 0, "recompute_sync_steps": 0}
        if not self.trace_dir.is_dir():
            return status
        for path in sorted(self.trace_dir.glob("phase_engine.pid*.jsonl")):
            with path.open(encoding="utf-8", errors="replace") as f:
                f.seek(self.offsets.get(path, 0))
                data = f.read()
                if not data.endswith("\n"):
                    cut = data.rfind("\n")
                    if cut < 0:
                        continue
                    data = data[:cut + 1]
                    f.seek(self.offsets.get(path, 0) + len(data.encode("utf-8", errors="replace")))
                self.offsets[path] = f.tell()
            for line in data.splitlines():
                try:
                    row = json.loads(line)
                    if row.get("kind") != "engine_step":
                        continue
                    prompt = int(row.get("prompt_compute_tokens", 0))
                    decode = int(row.get("decode_position_tokens", 0))
                    end_ns = int(row["result_end_ns"])
                    submit_ns = int(row["submit_end_ns"])
                    if row.get("async_scheduling", False):
                        # Futures overlap in async mode, so an individual
                        # result_end-submit_end interval includes prior queued
                        # work. Attribute the *incremental completion interval*
                        # to this batch instead. Preserve continuity even when
                        # this step is a zero-token/drain/decode step.
                        previous_end = self._last_completion_ns.get(path)
                        elapsed_ns = end_ns - max(submit_ns,
                                                 previous_end if previous_end is not None
                                                 else submit_ns)
                        mode = "async"
                    else:
                        elapsed_ns = end_ns - submit_ns
                        mode = "sync"
                    if end_ns > 0:
                        self._last_completion_ns[path] = max(
                            end_ns, self._last_completion_ns.get(path, 0))
                    if (prompt < self.min_prompt_tokens or elapsed_ns <= 0 or
                            prompt / max(1, prompt + decode) < self.min_prefill_fraction):
                        status["recompute_skipped"] += 1
                        continue
                    chunks = prompt / 512.0
                    mapper.observe_cost("recompute", (elapsed_ns / 1e6) / chunks,
                                        count=chunks)
                    status["recompute_steps"] += 1
                    status[f"recompute_{mode}_steps"] += 1
                    status["recompute_chunk_equivalents"] += chunks
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    status["recompute_skipped"] += 1
        return status

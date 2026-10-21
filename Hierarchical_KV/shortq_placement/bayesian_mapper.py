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
        self.cost_sum_ms = {"cpu": 0.0, "disk": 0.0}
        self.cost_counts = {"cpu": 0, "disk": 0}
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

    def observe_cost(self, tier: str, milliseconds_per_chunk: float, count: int = 1) -> None:
        if tier not in self.cost_counts:
            raise ValueError(tier)
        if not math.isfinite(milliseconds_per_chunk) or milliseconds_per_chunk <= 0 or count <= 0:
            raise ValueError("invalid retrieval observation")
        self.cost_sum_ms[tier] += milliseconds_per_chunk * count
        self.cost_counts[tier] += count

    def estimated_cost(self, tier: str) -> float:
        if tier == "recompute":
            return self.prior_ms[tier]  # No reliable GPU recompute timer in this patch.
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
            "cost_sample_counts": dict(self.cost_counts),
            "cost_estimate_ms": {t: self.estimated_cost(t) for t in ("cpu", "disk", "recompute")},
            "thresholds": {k: b[k] for k in TIERS},
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
    """Completed-device H2D + finished L2 load stage, no blocking synchronize.

    H2D measures elapsed time until a stream-ordered callback runs. L2 stage
    measures from load start to completion. Their sum is an approximate serial
    GET-path cost, not an exclusive service time nor a causal E2E cost. The
    components are not matched per request; report as a calibrated proxy.
    """
    _h2d = re.compile(r"\[BAYES_H2D_COMPLETE\].*?chunks=(\d+).*?elapsed_ms=([0-9.]+)")
    _disk = re.compile(r"\[BAYES_L2_LOAD_COMPLETE\].*?chunks=(\d+).*?elapsed_ms=([0-9.]+)")

    def collect(self, mapper: WaveBayesianMapper) -> dict[str, int]:
        counts = {"h2d_completed": 0, "disk_load_completed": 0}
        if not self.logfile.exists():
            return counts
        with self.logfile.open(encoding="utf-8", errors="replace") as f:
            f.seek(self.offset)
            data = f.read()
            # Preserve complete lines only; never double-count partially written logs.
            if not data.endswith("\n"):
                cut = data.rfind("\n")
                if cut < 0:
                    return counts
                data = data[:cut + 1]
                f.seek(self.offset + len(data.encode("utf-8", errors="replace")))
            self.offset = f.tell()
        new_disk_sum, new_disk_n = 0.0, 0
        for line in data.splitlines():
            m = self._h2d.search(line)
            if m and int(m[1]) > 0 and float(m[2]) > 0:
                n = int(m[1])
                mapper.observe_cost("cpu", float(m[2]) / n, count=n)
                counts["h2d_completed"] += 1
            m = self._disk.search(line)
            if m and int(m[1]) > 0 and float(m[2]) > 0:
                n = int(m[1])
                new_disk_sum += float(m[2])
                new_disk_n += n
                counts["disk_load_completed"] += 1
        if new_disk_n:
            # Disk GET includes loading from disk into host plus H2D. Use the
            # best observed H2D component. Unlike the old log proxy this H2D
            # measurement is collected after device-stream completion.
            h2d_per_chunk = (mapper.cost_sum_ms["cpu"] / mapper.cost_counts["cpu"]
                             if mapper.cost_counts["cpu"] else mapper.prior_ms["cpu"])
            mapper.observe_cost("disk", new_disk_sum / new_disk_n + h2d_per_chunk,
                                count=new_disk_n)
        return counts

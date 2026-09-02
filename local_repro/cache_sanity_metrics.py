#!/usr/bin/env python3

import time
from prometheus_client import REGISTRY


def _counter_value(base_name: str) -> float:
    """
    Read one vLLM Prometheus Counter from the current process.

    prometheus_client exposes a Counter named:
        vllm:prefix_cache_hits

    as a sample named:
        vllm:prefix_cache_hits_total

    We accept both spellings defensively.
    """
    wanted = {
        base_name,
        f"{base_name}_total",
    }

    values = []

    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name not in wanted:
                continue

            # This experiment has exactly one vLLM engine (engine 0).
            engine = sample.labels.get("engine")
            if engine is not None and str(engine) != "0":
                continue

            values.append(float(sample.value))

    if not values:
        available = sorted(
            {
                sample.name
                for metric in REGISTRY.collect()
                for sample in metric.samples
                if "prefix_cache" in sample.name
            }
        )
        raise RuntimeError(
            f"Could not find Prometheus counter {base_name!r}. "
            f"Available prefix-cache samples: {available}"
        )

    return sum(values)


def counter_int(base_name: str) -> int:
    return int(round(_counter_value(base_name)))


def wait_for_counter(
    base_name: str,
    minimum: int,
    timeout_s: float = 5.0,
) -> int:
    deadline = time.monotonic() + timeout_s
    last = None

    while time.monotonic() < deadline:
        try:
            last = counter_int(base_name)
            if last >= minimum:
                return last
        except RuntimeError:
            pass

        time.sleep(0.05)

    raise RuntimeError(
        f"Counter {base_name!r} did not reach {minimum} "
        f"within {timeout_s}s; last={last}"
    )


def snapshot_local_prefix() -> tuple[int, int]:
    return (
        counter_int("vllm:prefix_cache_queries"),
        counter_int("vllm:prefix_cache_hits"),
    )


def snapshot_external_prefix() -> tuple[int, int]:
    return (
        counter_int("vllm:external_prefix_cache_queries"),
        counter_int("vllm:external_prefix_cache_hits"),
    )
